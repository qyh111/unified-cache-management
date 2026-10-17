"""Worker lifecycle and public connector callbacks; metadata stays opaque."""

import copy
import json
import multiprocessing as mp
import os
import sys
import traceback
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ..cache_setup import construct_cache
from .environment import (
    cpu_worker_context,
    initialize_runtime,
    loaded_ucm_extensions,
    require,
)


def connector_output(worker, finished_req_ids):
    from vllm.v1.outputs import KVConnectorOutput

    results = worker.get_transfer_results(finished_req_ids)
    return KVConnectorOutput(
        finished_sending=results.finished_sending,
        finished_recving=results.finished_recving,
        failed_recving=results.failed_recving,
        invalid_block_ids=worker.get_block_ids_with_load_errors(),
        kv_connector_worker_meta=worker.build_connector_worker_meta(),
    )


def worker_main(connection, args, rank, ledger, lock):
    from ..data_check.comparison import KVComparison, components

    directory = Path(args.output) / args.phase
    with (
        (directory / f"rank-{rank}.log").open("a", buffering=1) as log,
        redirect_stdout(log),
        redirect_stderr(log),
        ExitStack() as guard,
    ):
        failure = None
        try:
            initialize_runtime(args)
            import torch
            from vllm.config import set_current_vllm_config
            from vllm.distributed.kv_transfer.kv_connector.v1.base import (
                KVConnectorRole,
            )
            from vllm.platforms import current_platform

            import ucm.integration.vllm.ucm_connector as module

            accelerator = torch.cuda if args.backend == "cuda" else torch.npu
            guard.enter_context(
                patch.object(
                    accelerator,
                    "_lazy_init",
                    side_effect=AssertionError(
                        "Accelerator initialization is forbidden"
                    ),
                )
            )
            require(not accelerator.is_initialized(), "Accelerator already initialized")
            native, views, raw, config, kernel_sizes = construct_cache(args, rank)
            oracle = KVComparison(native, views, raw)
            if args.store == "mock":
                from ..data_check.mock_store import bind

                bind(oracle.memory, ledger, lock)
            info = {
                "rank": rank,
                "pid": os.getpid(),
                "phase": args.phase,
                "platform": type(current_platform).__module__
                + "."
                + type(current_platform).__name__,
                "resolved_block_size": config.cache_config.block_size,
                "model_class": getattr(args, "native_model_class", None),
                "runner_class": getattr(args, "native_runner_class", None),
                "native_config": asdict(native),
                "kernel_block_sizes": kernel_sizes,
                "cuda_initialized": torch.cuda.is_initialized(),
                "accelerator_initialized": accelerator.is_initialized(),
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
                    for name, value in views.items()
                },
            }
            (directory / f"rank-{rank}-layout.json").write_text(
                json.dumps(info, indent=2, default=str)
            )
            with set_current_vllm_config(config), cpu_worker_context(module, rank):
                worker = module.UCMConnector(config, KVConnectorRole.WORKER, native)
                worker.register_kv_caches(views)
                from .environment import capture_engine_modules

                info.update(capture_engine_modules())
                info["loaded_ucm_extensions"] = loaded_ucm_extensions()
                (directory / f"rank-{rank}-layout.json").write_text(
                    json.dumps(info, indent=2, default=str)
                )
                connection.send(("ready", (native, info)))
                while True:
                    command, data = connection.recv()
                    if command == "stop":
                        connection.send(("stopped", None))
                        break
                    output, phase, blocks, hit_tokens, expected = data
                    snapshot, count, size = None, 0, 0
                    if phase == "source":
                        snapshot = oracle.source(
                            blocks, config.model_config.is_deepseek_mla, rank
                        )
                    elif phase == "target":
                        oracle.target(blocks, hit_tokens, expected)
                    metadata = output.kv_connector_metadata  # opaque SPI token
                    worker.bind_connector_metadata(metadata)
                    worker.handle_preemptions(metadata)
                    worker.start_load_kv(SimpleNamespace(no_compile_layers={}))
                    for name, value in views.items():
                        worker.wait_for_layer_load(name)
                        if phase == "target":
                            oracle.compare_layer(name)
                        worker.save_kv_layer(name, value, None)
                    worker.wait_for_save()
                    if phase == "target":
                        count, size = oracle.finish()
                    result = connector_output(worker, output.finished_req_ids)
                    worker.clear_connector_metadata()
                    require(
                        not accelerator.is_initialized(),
                        "Accelerator initialized during IO",
                    )
                    native_modules = loaded_ucm_extensions()
                    require(
                        args.store != "mock" or not native_modules,
                        "Mock imported UCM native code",
                    )
                    info["loaded_ucm_extensions"] = native_modules
                    (directory / f"rank-{rank}-layout.json").write_text(
                        json.dumps(info, indent=2, default=str)
                    )
                    connection.send(("result", (result, count, size, snapshot)))
        except BaseException:
            failure = traceback.format_exc()
            traceback.print_exc()
        finally:
            if args.model_config and "torch" in sys.modules:
                import torch.distributed as distributed

                if distributed.is_initialized():
                    from vllm.distributed.parallel_state import (
                        destroy_distributed_environment,
                        destroy_model_parallel,
                    )

                    destroy_model_parallel()
                    destroy_distributed_environment()
            if failure:
                try:
                    connection.send(("error", failure))
                except (BrokenPipeError, EOFError):
                    pass
            connection.close()
        if failure:
            raise RuntimeError(failure)


class Workers:
    """A TP worker group; a successful scope exits only after every rank stops."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self.stop()
        finally:
            self.close()

    def __init__(self, args, phase, ledger, lock):
        self.args = copy.copy(args)
        self.args.phase = phase
        (Path(args.output) / phase).mkdir()
        self.processes, self.connections, self.info, self.configs = [], [], [], []
        context = mp.get_context("spawn")
        try:
            for rank in range(args.tp):
                parent, child = context.Pipe()
                process = context.Process(
                    target=worker_main, args=(child, self.args, rank, ledger, lock)
                )
                process.start()
                child.close()
                self.processes.append(process)
                self.connections.append(parent)
            for rank in range(args.tp):
                native, info = self.receive(rank, "ready")
                self.configs.append(native)
                self.info.append(info)
            require(
                all(asdict(c) == asdict(self.configs[0]) for c in self.configs),
                "TP native configs differ",
            )
        except BaseException:
            self.close()
            raise

    def receive(self, rank, expected):
        connection = self.connections[rank]
        location = f"{self.args.phase}/rank-{rank}.log"
        if not connection.poll(self.args.timeout):
            raise TimeoutError(
                f"Worker {rank}: timed out waiting for {expected}; see {location}"
            )
        try:
            kind, result = connection.recv()
        except EOFError as error:
            raise RuntimeError(f"Worker {rank} exited; see {location}") from error
        if kind != expected:
            raise RuntimeError(f"Worker {rank}: {kind}: {result}")
        return result

    def step(self, output, phase="drain", blocks=None, hit_tokens=0, snapshots=None):
        for rank, connection in enumerate(self.connections):
            connection.send(
                (
                    "step",
                    (
                        output,
                        phase,
                        blocks,
                        hit_tokens,
                        snapshots[rank] if snapshots else None,
                    ),
                )
            )
        return [self.receive(rank, "result") for rank in range(self.args.tp)]

    def close(self):
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(10)
            if process.is_alive():
                process.kill()
                process.join(10)
        for connection in self.connections:
            connection.close()

    def stop(self):
        for connection in self.connections:
            connection.send(("stop", None))
        for rank in range(self.args.tp):
            self.receive(rank, "stopped")
        for process in self.processes:
            process.join(min(self.args.timeout, 60))
            require(process.exitcode == 0, f"Worker exited with {process.exitcode}")
