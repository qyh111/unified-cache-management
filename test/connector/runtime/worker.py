"""Rank process: native spec/allocation, MRV1 lifecycle, deterministic KV writes."""

import json
import os
import pickle
from pathlib import Path
import traceback
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from dataclasses import asdict
from unittest.mock import patch

from .environment import (
    initialize_runtime,
    cpu_worker_context,
    require,
    loaded_ucm_extensions,
    capture_engine_modules,
)


class CacheWorker:
    def __init__(self, args, rank, ledger, lock, control, guard):
        initialize_runtime(args)
        from .environment import observe_owned_shared_memory

        guard.enter_context(observe_owned_shared_memory(Path(args.output) / args.phase))
        import torch
        from ..cache_setup import make_config, prepare
        from vllm.config import set_current_vllm_config
        import ucm.integration.vllm.ucm_connector as module

        guard.enter_context(
            patch.object(
                torch.Tensor,
                "pin_memory",
                side_effect=AssertionError("CPU test requested pinned device memory"),
            )
        )
        self.args, self.rank = args, rank
        self.ledger, self.lock, self.control = ledger, lock, control
        self.accelerator = torch.cuda if args.backend == "cuda" else torch.npu
        guard.enter_context(
            patch.object(
                self.accelerator,
                "_lazy_init",
                side_effect=AssertionError("CPU test initialized accelerator"),
            )
        )
        from vllm.platforms import current_platform

        guard.enter_context(
            patch.object(
                type(current_platform),
                "get_device_communicator_cls",
                return_value="vllm.distributed.device_communicators.cpu_communicator.CpuCommunicator",
            )
        )
        self.config = make_config(args, rank)
        guard.enter_context(set_current_vllm_config(self.config))
        self.runner, self.specs = prepare(args, self.config, rank)
        if self.runner is None:
            from vllm.distributed import (
                init_distributed_environment,
                initialize_model_parallel,
            )

            init_distributed_environment(
                world_size=args.tp,
                rank=rank,
                local_rank=rank,
                distributed_init_method="file://"
                + str(Path(args.output) / ("gloo-" + args.phase)),
                backend="gloo",
            )
            initialize_model_parallel(
                tensor_model_parallel_size=args.tp, backend="gloo"
            )
        from vllm.distributed import get_world_group, get_tp_group
        import torch.distributed as distributed

        world, tp = get_world_group(), get_tp_group()
        require(
            world.rank == rank and world.world_size == args.tp,
            "Native rank topology differs",
        )
        self.distributed_info = {
            "rank": world.rank,
            "world_size": world.world_size,
            "tp_rank": tp.rank_in_group,
            "tp_size": tp.world_size,
            "backend": distributed.get_backend(),
            "group_class": type(world).__module__ + "." + type(world).__name__,
        }
        guard.enter_context(cpu_worker_context(module, rank))
        self.tables, self.positions, self.prompts = {}, {}, {}
        self.snapshot = None
        self.expected = None
        self.compared = (0, 0)
        self.info = {}

    def get_kv_cache_spec(self):
        return self.specs

    def get_supported_kv_cache_layouts(self):
        if self.runner is None:
            return ["LBNHC"]  # explicitly synthetic GQA geometry
        from vllm.distributed.kv_transfer.kv_connector.utils import (
            get_current_attn_backends,
        )
        from vllm.v1.attention.backends.utils import get_supported_kv_cache_layouts

        return [
            x.name
            for x in get_supported_kv_cache_layouts(
                get_current_attn_backends(self.config)
            )
        ]

    def set_kv_cache_layout(self, layout_name):
        from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

        resolve_kv_cache_layout(self.config, [[layout_name]], list(self.specs.values()))

    def determine_available_memory(self):
        return self.args.kv_budget_mib << 20

    def initialize_from_config(self, configs):
        from ..cache_setup import allocate, allocation_upper_bound
        from ..data_check.comparison import KVComparison, components
        from ..data_check.mock_store import bind
        from vllm.distributed.kv_transfer.kv_transfer_state import (
            ensure_kv_transfer_initialized,
            get_kv_transfer_group,
        )

        self.native = configs[self.rank]
        size = allocation_upper_bound(self.args, self.native)
        require(
            size <= self.args.kv_budget_mib << 20,
            f"Native allocation {size} exceeds budget {self.args.kv_budget_mib << 20}",
        )
        self.views, self.raw, kernels, axes = allocate(
            self.args, self.runner, self.native, self.specs
        )
        self.oracle = KVComparison(self.native, self.views, self.raw, axes)
        allocated = sum(t.numel() for t in self.oracle.memory.buffers)
        require(
            allocated <= self.args.kv_budget_mib << 20,
            "Actual native storage exceeds KV budget",
        )
        self.replicated = (
            self.config.model_config.use_mla
            or self.args.tp > self.config.model_config.get_total_num_kv_heads()
        )
        if self.args.store == "mock":
            bind(self.oracle.memory, self.ledger, self.lock, self.control, self.rank)
        else:
            from ..data_check.real_store import bind as bind_real

            bind_real(self.control, self.lock)
        ensure_kv_transfer_initialized(self.config, self.native)
        self.connector = get_kv_transfer_group()
        self.connector.register_kv_caches(self.views)
        self.info = {
            "pid": os.getpid(),
            "distributed": self.distributed_info,
            "rank": self.rank,
            "native_config": asdict(self.native),
            "kernel_block_sizes": kernels,
            "model_class": getattr(self.args, "native_model_class", "synthetic-GQA"),
            "runner_class": getattr(self.args, "native_runner_class", "synthetic-GQA"),
            "allocated_bytes": allocated,
            "descriptor_bytes_upper_bound": size,
            "native_token_axes": axes,
            "spec_module": type(next(iter(self.specs.values()))).__module__,
            "views": {
                name: [
                    {
                        "shape": list(t.shape),
                        "stride": list(t.stride()),
                        "storage_offset": t.storage_offset(),
                        "dtype": str(t.dtype),
                        "device": str(t.device),
                    }
                    for t in components(value)
                ]
                for name, value in self.views.items()
            },
        }
        self.state = None
        if any(spec.tokens_per_state < 0 for spec in self.specs.values()):
            from .state import StateLifecycle

            self.state = StateLifecycle(self)
        self.write_info()

    def write_info(self):
        import torch

        self.info.update(capture_engine_modules())
        self.info.update(
            compared_segments=self.compared[0], compared_bytes=self.compared[1]
        )
        self.info.update(
            cuda_initialized=torch.cuda.is_initialized(),
            accelerator_initialized=self.accelerator.is_initialized(),
            loaded_ucm_extensions=loaded_ucm_extensions(),
        )
        require(not self.info["accelerator_initialized"], "CPU mode initialized device")
        require(
            self.args.store != "mock" or not self.info["loaded_ucm_extensions"],
            "Mock loaded UCM extension",
        )
        directory = Path(self.args.output) / self.args.phase
        (directory / f"rank-{self.rank}-layout.json").write_text(
            json.dumps(self.info, indent=2, default=str)
        )

    def compile_or_warm_up_model(self):
        from vllm.v1.worker.worker_base import CompilationTimes

        return CompilationTimes(0, 0)

    def get_kv_connector_handshake_metadata(self):
        return None

    def arm_reader(self, hashes, hit_tokens):
        self.expected = hashes[self.rank]
        self.hit_tokens = hit_tokens
        self.oracle.memory.poison()
        self.info["poisoned_bytes"] = sum(t.numel() for t in self.oracle.memory.buffers)
        require(
            all(bool((t == 255).all()) for t in self.oracle.memory.buffers),
            "Allocation poison failed",
        )
        self.tables.clear()
        self.positions.clear()
        self.prompts.clear()
        self.compared = (0, 0)
        if self.state is not None:
            self.state.indices.clear()

    def audit(self):
        return {"snapshot": self.snapshot, "compared": self.compared, "info": self.info}

    def execute_model(self, output):
        from vllm.forward_context import set_forward_context
        from vllm.v1.worker.kv_connector_model_runner_mixin import (
            KVConnectorModelRunnerMixin as Lifecycle,
        )
        from vllm.v1.outputs import ModelRunnerOutput

        self.connector.handle_preemptions(output.kv_connector_metadata)
        for new in output.scheduled_new_reqs:
            self.tables[new.req_id] = tuple(list(b) for b in new.block_ids)
            self.positions[new.req_id] = new.num_computed_tokens
            require(
                new.prompt_token_ids is not None,
                "Core fixture requires token-ID requests",
            )
            self.prompts[new.req_id] = new.prompt_token_ids
        cached = output.scheduled_cached_reqs
        for index, req_id in enumerate(cached.req_ids):
            ids = cached.new_block_ids[index]
            if ids is not None:
                if req_id in cached.resumed_req_ids:
                    self.tables[req_id] = tuple(list(b) for b in ids)
                else:
                    for bank, new_ids in zip(self.tables[req_id], ids):
                        bank.extend(new_ids)
            self.positions[req_id] = cached.num_computed_tokens[index]
        if output.block_table_updates:
            for req_id, ids in output.block_table_updates.items():
                self.tables[req_id] = tuple(list(b) for b in ids)
        if output.new_block_ids_to_zero:
            from ..data_check.comparison import components

            # CPU equivalent of native zeroing over each real component view.
            # No computation kernel is invoked; the data effect is preserved.
            for value in self.views.values():
                for tensor in components(value):
                    factor, remainder = divmod(tensor.shape[0], self.native.num_blocks)
                    require(remainder == 0, "Invalid native block geometry for zeroing")
                    for block in output.new_block_ids_to_zero:
                        tensor[int(block) * factor : (int(block) + 1) * factor].zero_()
        if not output.total_num_scheduled_tokens:
            result = Lifecycle.kv_connector_no_forward(output, self.config)
            self.write_info()
            return result
        require(
            not output.kv_cache_block_copies,
            "KV copy path is not implemented in core scope",
        )
        comparison_error = None
        if self.state is not None:
            self.state.prepare(output)
        with (
            set_forward_context(None, self.config),
            Lifecycle.maybe_get_kv_connector_output(output) as kv_result,
        ):
            if self.state is not None:
                self.state.copy_after_load()
            for req_id in output.num_scheduled_tokens:
                blocks = self.tables[req_id]
                # Byte identity comes from the actual request prefix on the
                # independent test channel, never from a store key/IO plan.
                import hashlib

                self.oracle.prefix_identity = hashlib.sha256(
                    json.dumps(self.prompts[req_id][: self.args.tokens]).encode()
                ).hexdigest()
                if self.expected is not None and not self.oracle.targets:
                    require(
                        self.positions[req_id] == self.hit_tokens,
                        f"Actual restored prefix {self.positions[req_id]} != expected {self.hit_tokens}",
                    )
                    self.info["external_hit_tokens"] = self.positions[req_id]
                    self.oracle.target(
                        blocks, self.hit_tokens, self.expected, poison=False
                    )
                if self.expected is None and self.snapshot is None:
                    self.snapshot = self.oracle.source(
                        blocks,
                        self.replicated,
                        self.rank,
                        self.state.indices[req_id] if self.state is not None else None,
                    )
                    self.info["source_block_table"] = blocks
                for name, value in self.views.items():
                    # Only native Attention owners receive Attention hooks.
                    # State specs must be handled through native bookkeeping.
                    spec = self.specs[name]

                    self.connector.wait_for_layer_load(name)
                    if self.expected is not None and self.compared == (0, 0):
                        try:
                            self.oracle.compare_layer(name)
                        except AssertionError as error:
                            comparison_error = error
                            continue
                    # Historical payload is checked before deterministic
                    # writes to this step's newly computed token rows.
                    self.oracle.write_new_layer(
                        name,
                        blocks,
                        self.positions[req_id],
                        self.positions[req_id] + output.num_scheduled_tokens[req_id],
                        self.rank,
                        self.replicated,
                    )
                    if spec.tokens_per_state < 0:
                        self.oracle.write_state_layer(
                            name, blocks, self.state.indices[req_id], self.rank
                        )
                        # Native Ascend KDA emits its empty-name state save
                        # hook, then MRV1 wait_for_save drains outstanding rows.
                        if self.args.backend == "ascend":
                            self.connector.save_kv_layer("", [], None)
                    else:
                        self.connector.save_kv_layer(name, value, None)
                if (
                    self.expected is not None
                    and self.compared == (0, 0)
                    and comparison_error is None
                ):
                    self.compared = self.oracle.finish()
        # The native lifecycle drains load errors exactly once on exit. Never
        # consume get_block_ids_with_load_errors before it reaches Scheduler.
        if comparison_error is not None and not (
            kv_result.invalid_block_ids or kv_result.failed_recving
        ):
            raise comparison_error
        req_ids = list(output.num_scheduled_tokens)
        self.write_info()
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={r: i for i, r in enumerate(req_ids)},
            sampled_token_ids=[[7] for _ in req_ids],
            kv_connector_output=kv_result,
        )

    def shutdown(self):
        from vllm.distributed.kv_transfer.kv_transfer_state import (
            ensure_kv_transfer_shutdown,
        )
        from vllm.distributed.parallel_state import (
            destroy_model_parallel,
            destroy_distributed_environment,
        )

        ensure_kv_transfer_shutdown()
        destroy_model_parallel()
        destroy_distributed_environment()
        self.write_info()


def worker_main(pipe, handle, args, rank, ledger, lock, control):
    directory = Path(args.output) / args.phase
    with (
        (directory / f"rank-{rank}.log").open("w", buffering=1) as log,
        redirect_stdout(log),
        redirect_stderr(log),
        ExitStack() as guard,
    ):
        request_mq = response_mq = worker = None
        try:
            worker = CacheWorker(args, rank, ledger, lock, control, guard)
            from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

            request_mq = MessageQueue.create_from_handle(pickle.loads(handle), rank)
            response_mq = MessageQueue(1, 1, max_chunk_bytes=1 << 20, max_chunks=2)
            pipe.send(("ready", response_mq.export_handle()))
            request_mq.wait_until_ready()
            response_mq.wait_until_ready()
            while True:
                method, values, kwargs = request_mq.dequeue(timeout=args.timeout)
                try:
                    result = getattr(worker, method)(*values, **kwargs)
                    response_mq.enqueue(("ok", result), timeout=args.timeout)
                except BaseException:
                    error = traceback.format_exc()
                    traceback.print_exc()
                    response_mq.enqueue(("error", error), timeout=args.timeout)
                    break
                if method == "shutdown":
                    worker = None
                    break
        except BaseException:
            error = traceback.format_exc()
            traceback.print_exc()
            try:
                pipe.send(("error", error))
            except (BrokenPipeError, EOFError):
                pass
            raise
        finally:
            if worker is not None:
                try:
                    worker.shutdown()
                except BaseException:
                    traceback.print_exc()
            for mq in (request_mq, response_mq):
                if mq is not None:
                    from .environment import close_message_queue

                    close_message_queue(mq)
            pipe.close()
