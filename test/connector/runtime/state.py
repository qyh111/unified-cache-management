"""MRV1 Mamba decisions and copy specs stay native; CPU memcpy preserves effects."""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from .environment import require


class StateLifecycle:
    def __init__(self, worker):
        import torch
        from vllm.v1.utils import CpuGpuBuffer
        from vllm.v1.worker import mamba_utils

        class CPUControlBuffer(CpuGpuBuffer):
            def copy_to_gpu(self, n=None):
                # Both buffers are CPU tensors in this runner adapter.
                src = self.cpu if n is None else self.cpu[:n]
                dst = self.gpu if n is None else self.gpu[:n]
                return dst.copy_(src)

        def make_buffer(n, dtype):
            # Ascend remaps pointer buffers for NPU int64 zero kernels. CPU
            # control buffers retain uint64 addresses without that constraint.
            if dtype == torch.int64:
                dtype = torch.uint64
            return CPUControlBuffer(
                n, dtype=dtype, device=torch.device("cpu"), pin_memory=False
            )

        self.worker = worker
        self.indices = {}
        runner = worker.runner
        runner._mamba_state_copy_funcs = None
        self.funcs = runner._get_mamba_state_copy_funcs()
        self.buffers = mamba_utils.MambaCopyBuffers.create(
            max_num_reqs=2,
            kv_cache_config=worker.native,
            copy_funcs=self.funcs,
            make_buffer=make_buffer,
        )
        self.names = [
            name for name, spec in worker.specs.items() if spec.tokens_per_state < 0
        ]
        self.copy_count = 0

    def prepare(self, output):
        import numpy as np
        from vllm.v1.worker import mamba_utils

        # The native routine consumes only these batch fields on its scalar
        # path. Request states and block tables came from SchedulerOutput.
        ids = list(output.num_scheduled_tokens)
        batch = SimpleNamespace(
            req_ids=ids, num_accepted_tokens_cpu=np.ones(len(ids), dtype=np.int32)
        )
        requests = {
            req: SimpleNamespace(
                block_ids=self.worker.tables[req],
                num_computed_tokens=self.worker.positions[req],
            )
            for req in ids
        }
        # Keep the installed engine/UCM ordering, including the mainline
        # UCM pre-copy patch. Only the hardware memcpy launch is adapted.
        with self.cpu_copy_context():
            mamba_utils.preprocess_mamba(
                output,
                self.worker.native,
                self.worker.config.cache_config,
                self.indices,
                batch,
                requests,
                self.worker.config.compilation_config.static_forward_context,
                self.funcs,
                self.buffers,
            )

    def cpu_memcpy(self, src, dst, sizes):
        memory = self.worker.oracle.memory
        for source, target, count in zip(src.tolist(), dst.tolist(), sizes.tolist()):
            source, target, count = int(source), int(target), int(count)
            if not count:
                continue
            require(source != 0 and target != 0, "Null native State copy")
            recipe = (
                memory.recipe(source, count)
                if self.worker.args.store == "mock" and memory.recipes
                else None
            )
            data = bytes(memory.read(source, count))
            memory.write(target, data)
            if recipe is not None:
                memory.assign(target, count, recipe)
            self.copy_count += 1

    def cpu_copy_context(self):
        from vllm.v1.worker import mamba_utils

        context = ExitStack()
        context.enter_context(
            patch.object(mamba_utils, "batch_memcpy", self.cpu_memcpy)
        )
        if self.worker.args.backend == "ascend":
            from vllm_ascend.patch.worker import patch_mamba_utils

            context.enter_context(
                patch.object(patch_mamba_utils, "_batch_memcpy_triton", self.cpu_memcpy)
            )
        return context

    def copy_after_load(self):
        from vllm.v1.worker import mamba_utils

        for name in self.names:
            self.worker.connector.wait_for_layer_load(name)
        if self.worker.args.backend == "ascend":
            # Native MRV1 issues its deferred copy here. The mainline UCM
            # patch may already have consumed the buffer during prepare.
            with self.cpu_copy_context():
                mamba_utils.do_mamba_copy_block(self.buffers)
        self.worker.info["native_state_copy_count"] = self.copy_count
