"""Small CPU control executor; EngineCore owns all scheduling and KV grouping."""

import multiprocessing as mp
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time

from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.v1.executor.abstract import Executor

from .environment import require

# Set only inside an isolated driver before EngineCore constructs the executor.
CONTEXT = None
ACTIVE_EXECUTOR = None


class TestExecutor(Executor):
    __test__ = False

    def _init_executor(self):
        global ACTIVE_EXECUTOR
        ACTIVE_EXECUTOR = self
        from .worker import worker_main

        self.args, self.ledger, self.lock, self.control = CONTEXT
        self.processes, self.pipes, self.responses = [], [], []
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.broadcast = None
        self.closed = False
        self.steps = []
        self.forced_shutdown = False
        self.shutdown_errors = []
        try:
            from .environment import observe_owned_shared_memory

            with observe_owned_shared_memory(Path(self.args.output) / self.args.phase):
                self.broadcast = MessageQueue(
                    self.args.tp, self.args.tp, max_chunk_bytes=1 << 20, max_chunks=2
                )
            context = mp.get_context("spawn")
            for rank in range(self.args.tp):
                parent, child = context.Pipe()
                proc = context.Process(
                    target=worker_main,
                    args=(
                        child,
                        pickle.dumps(self.broadcast.export_handle()),
                        self.args,
                        rank,
                        self.ledger,
                        self.lock,
                        self.control,
                    ),
                )
                proc.start()
                child.close()
                self.processes.append(proc)
                self.pipes.append(parent)
            for rank, pipe in enumerate(self.pipes):
                if not pipe.poll(self.args.timeout):
                    raise TimeoutError(f"Rank {rank} bootstrap timed out")
                status, handle = pipe.recv()
                if status != "ready":
                    raise RuntimeError(handle)
                self.responses.append(MessageQueue.create_from_handle(handle, 0))
            self.broadcast.wait_until_ready()
            for mq in self.responses:
                mq.wait_until_ready()
        except BaseException:
            self.shutdown()
            raise

    def collective_rpc(
        self, method, timeout=None, args=(), kwargs=None, non_block=False
    ):
        require(not self.closed, "Executor is closed")
        require(isinstance(method, str), "Only used named RPCs are implemented")
        timeout = self.args.timeout if timeout is None else timeout

        def invoke():
            deadline = time.monotonic() + timeout
            self.broadcast.enqueue((method, args, kwargs or {}), timeout=timeout)
            results = []
            for rank, mq in enumerate(self.responses):
                status, result = mq.dequeue(
                    timeout=max(0.001, deadline - time.monotonic())
                )
                if status != "ok":
                    raise RuntimeError(f"Rank {rank} {method}: {result}")
                results.append(result)
            return results

        future = self.pool.submit(invoke)
        return future if non_block else future.result()

    def get_kv_cache_specs(self):
        self.native_specs = super().get_kv_cache_specs()
        return self.native_specs

    def determine_available_memory(self):
        import copy
        from vllm.v1.core.kv_cache_utils import get_kv_cache_configs

        budget = self.args.kv_budget_mib << 20
        budgets = [budget] * self.args.tp
        cache = self.vllm_config.cache_config
        if cache.num_gpu_blocks_override is None:
            # Native grouped layouts can repeat/pad descriptors beyond the
            # profiling estimate. Bound the actual descriptor allocations;
            # EngineCore still generates and distributes the final configs.
            blocks = 1
            while blocks <= 8192:
                cache.num_gpu_blocks_override = blocks
                try:
                    get_kv_cache_configs(
                        self.vllm_config, copy.deepcopy(self.native_specs), budgets
                    )
                    break
                except ValueError as error:
                    if not any(
                        message in str(error)
                        for message in (
                            "KV cache is needed",
                            "No available memory for the cache blocks",
                        )
                    ):
                        raise
                    blocks *= 2
            else:
                raise RuntimeError("No native capacity fits the bounded test planner")
            cache.num_gpu_blocks_override = blocks * 3
        from ..cache_setup import allocation_upper_bound

        planned = get_kv_cache_configs(
            self.vllm_config, copy.deepcopy(self.native_specs), budgets
        )
        require(
            all(allocation_upper_bound(self.args, c) <= budget for c in planned),
            "Native minimum test capacity exceeds --kv-budget-mib",
        )
        return budgets

    def initialize_from_config(self, kv_cache_configs):
        import json
        from dataclasses import asdict
        from pathlib import Path

        (
            Path(self.args.output) / self.args.phase / "engine-kv-configs.json"
        ).write_text(
            json.dumps([asdict(c) for c in kv_cache_configs], indent=2, default=str)
        )
        # Each worker consumes its exact EngineCore-generated config.
        self.collective_rpc("initialize_from_config", args=(kv_cache_configs,))

    def execute_model(self, scheduler_output, non_block=False):
        from concurrent.futures import Future

        result = Future()
        pending = self.collective_rpc(
            "execute_model", args=(scheduler_output,), non_block=True
        )

        def aggregate(completed):
            outputs = completed.result()
            rank_outputs = []
            for output in outputs:
                kv = output.kv_connector_output
                rank_outputs.append(
                    {
                        "sending": sorted(kv.finished_sending or ())
                        if kv is not None
                        else [],
                        "recving": sorted(kv.finished_recving or ())
                        if kv is not None
                        else [],
                        "failed_recving": sorted(kv.failed_recving or ())
                        if kv is not None
                        else [],
                        "invalid_blocks": sorted(kv.invalid_block_ids)
                        if kv is not None
                        else [],
                        "has_connector_output": kv is not None,
                    }
                )
            merged = self.kv_output_aggregator.aggregate(outputs, output_rank=0)
            self.steps.append(
                {
                    "scheduled_tokens": scheduler_output.total_num_scheduled_tokens,
                    "has_sync_kv_loads": scheduler_output.has_sync_kv_loads,
                    "finished_req_ids": sorted(scheduler_output.finished_req_ids),
                    "rank_outputs": rank_outputs,
                    "aggregated_sending": sorted(
                        merged.kv_connector_output.finished_sending or ()
                    ),
                    "aggregated_recving": sorted(
                        merged.kv_connector_output.finished_recving or ()
                    ),
                }
            )
            return merged

        def resolve(completed):
            try:
                result.set_result(aggregate(completed))
            except BaseException as error:
                result.set_exception(error)

        pending.add_done_callback(resolve)
        return result if non_block else result.result()

    def check_health(self):
        for rank, proc in enumerate(self.processes):
            require(proc.is_alive(), f"Rank {rank} exited ({proc.exitcode})")

    def shutdown(self):
        if self.closed:
            return
        failed = False
        try:
            if (
                self.processes
                and self.broadcast is not None
                and len(self.responses) == len(self.processes)
            ):
                # A failed rank may already have exited after reporting its
                # error. Still stop the surviving readers of the native queue.
                timeout = min(self.args.timeout, 10)
                self.broadcast.enqueue(("shutdown", (), {}), timeout=timeout)
                deadline = time.monotonic() + timeout
                for rank, proc in enumerate(self.processes):
                    if proc.is_alive():
                        status, detail = self.responses[rank].dequeue(
                            timeout=max(0.001, deadline - time.monotonic())
                        )
                        require(
                            status == "ok", f"Rank {rank} shutdown failed: {detail}"
                        )
        except BaseException as error:
            failed = True
            self.shutdown_errors.append(f"{type(error).__name__}: {error}")
        finally:
            self.closed = True
            deadline = time.monotonic() + min(self.args.timeout, 60)
            for proc in self.processes:
                proc.join(max(0, deadline - time.monotonic()))
                if proc.is_alive():
                    self.forced_shutdown = True
                    proc.terminate()
                    proc.join(5)
                if proc.is_alive():
                    proc.kill()
                    proc.join(5)
            for pipe in self.pipes:
                pipe.close()
            for mq in [*self.responses, self.broadcast]:
                from .environment import close_message_queue

                if mq is not None:
                    close_message_queue(mq)
            self.pool.shutdown(wait=False, cancel_futures=True)
        if failed:
            require(
                all(not p.is_alive() for p in self.processes), "Worker cleanup failed"
            )
