"""Driver process. Scheduling, requests and completion belong to EngineCore."""

import copy
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

from .environment import initialize_runtime, require


def request(engine, name, tokens):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request

    return Request(
        request_id=name,
        prompt_token_ids=list(range(10, tokens + 10)),
        sampling_params=SamplingParams(
            max_tokens=2 if name == "source" else 1, ignore_eos=True
        ),
        pooling_params=None,
        block_hasher=engine.request_block_hasher,
    )


def drain(engine, timeout):
    from unittest.mock import patch

    native_take = engine.scheduler.kv_cache_manager.take_boundary_state_offloads
    if not hasattr(engine, "test_boundary_events"):
        engine.test_boundary_events = []

    def observe_boundaries():
        result = native_take()
        engine.test_boundary_events.append(result)
        return result

    deadline = time.monotonic() + timeout
    steps = 0
    while engine.scheduler.has_requests():
        require(time.monotonic() < deadline, "EngineCore step deadline exceeded")
        with patch.object(
            engine.scheduler.kv_cache_manager,
            "take_boundary_state_offloads",
            observe_boundaries,
        ):
            outputs, ran = engine.step()
        engine.post_step(ran)
        trace = engine.model_executor.steps[-1]
        receiving = {
            req
            for rank_output in trace["rank_outputs"]
            for req in rank_output["recving"]
        }
        if "reader" in receiving and "reader" not in trace["aggregated_recving"]:
            from vllm.v1.request import RequestStatus

            pending = engine.scheduler.requests["reader"]
            require(
                pending.status == RequestStatus.WAITING_FOR_REMOTE_KVS,
                "Scheduler advanced before every rank completed",
            )
            blocks = engine.scheduler.kv_cache_manager.get_block_ids("reader")
            require(
                any(block for group in blocks for block in group),
                "Scheduler released pending load blocks",
            )
            engine.test_partial_completion_checks = (
                getattr(engine, "test_partial_completion_checks", 0) + 1
            )
        steps += 1
    require(steps > 0, "EngineCore performed no steps")
    return steps


def make_engine(args, phase, ledger, lock, control):
    from ..cache_setup import make_config
    from vllm.config import set_current_vllm_config
    from vllm.v1.engine.core import EngineCore
    from vllm.platforms import current_platform
    from . import executor

    phase_args = copy.copy(args)
    phase_args.phase = phase
    (Path(args.output) / phase).mkdir()
    config = make_config(phase_args)
    config.scheduler_config.async_scheduling = False
    current_platform.update_block_size_for_backend(config)
    executor.CONTEXT = phase_args, ledger, lock, control
    # EngineCore, not this test, calls get_kv_cache_configs and Scheduler.
    executor.ACTIVE_EXECUTOR = None
    try:
        with set_current_vllm_config(config):
            engine = EngineCore(config, executor.TestExecutor, log_stats=False)
    except BaseException:
        if executor.ACTIVE_EXECUTOR is not None:
            executor.ACTIVE_EXECUTOR.shutdown()
        raise
    args.tokens = phase_args.tokens
    return engine


def prepare_materials(args):
    root = Path(args.output)
    (root / "store").mkdir()
    if args.model_config:
        (root / "model-config.json").write_bytes(
            (Path(args.model_config) / "config.json").read_bytes()
        )
    else:
        args.tokens = args.tokens or 512
        (root / "fixture").mkdir()
        (root / "fixture/config.json").write_text(
            json.dumps(
                {
                    "model_type": "llama",
                    "architectures": ["UCMConnectorTestModel"],
                    "hidden_size": 512,
                    "intermediate_size": 1024,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 4,
                    "head_dim": 128,
                    "vocab_size": 65536,
                    "max_position_embeddings": 32768,
                    "rms_norm_eps": 1e-5,
                    "torch_dtype": "float16",
                }
            )
        )


class EngineSession:
    """The scenario controls save/close/open/load through public engine APIs."""

    def __init__(self, args, ledger, lock, control, fault):
        self.args, self.ledger, self.lock, self.control = args, ledger, lock, control
        self.fault = fault
        self.source = self.reader = None
        self.source_closed = False
        self.result = None

    def save_prefix(self):
        from vllm.config import set_current_vllm_config

        self.source = make_engine(
            self.args, "source", self.ledger, self.lock, self.control
        )
        source_request = request(self.source, "source", self.args.tokens)
        with set_current_vllm_config(self.source.vllm_config):
            self.source.add_request(source_request)
            self.source_steps = drain(self.source, self.args.timeout)
        require(source_request.is_finished(), "Source did not finish")
        self.audit = self.source.model_executor.collective_rpc("audit")
        self.hashes = [x["snapshot"] for x in self.audit]
        require(
            all(self.hashes),
            "A source rank generated no native payload hashes",
        )
        offers = [
            tuple(offer)
            for event in self.source.test_boundary_events
            for offer in event.get("source", [])
        ]
        for rank_audit in self.audit:
            native = rank_audit["info"]["native_config"]
            tables = rank_audit["info"]["source_block_table"]
            for gid, group in enumerate(native["kv_cache_groups"]):
                spec = group["kv_cache_spec"]
                nested = spec.get("kv_cache_specs", {})
                spec = next(iter(nested.values())) if nested else spec
                if spec["tokens_per_state"] >= 0:
                    continue
                indices = {
                    identity[1]
                    for identity in rank_audit["snapshot"]
                    if identity[0] == gid
                }
                require(
                    len(indices) == 1,
                    "State source has no exact native running checkpoint",
                )
                index = next(iter(indices))
                require(
                    (gid, tables[gid][index], self.args.tokens) in offers,
                    f"Native bookkeeping did not confirm group {gid} checkpoint at {self.args.tokens}",
                )
        self.source_pids = [x["info"]["pid"] for x in self.audit]
        self.source_trace = list(self.source.model_executor.steps)
        self.source_step_count = len(self.source_trace)
        return self.hashes

    def close_source(self):
        self.source.shutdown()
        require(
            all(
                not p.is_alive() and p.exitcode == 0
                for p in self.source.model_executor.processes
            ),
            "Source worker did not exit cleanly",
        )
        require(
            not self.source.model_executor.shutdown_errors,
            str(self.source.model_executor.shutdown_errors),
        )
        self.source_closed = True

    def open_reader(self):
        require(self.source_closed, "Cold reader requires exited source workers")
        self.reader = make_engine(
            self.args, "reader", self.ledger, self.lock, self.control
        )

    def reset_local_prefix(self):
        self.reader = self.source
        require(self.reader.reset_prefix_cache(), "Engine prefix reset refused")

    def load_and_compare(self, hashes):
        from vllm.config import set_current_vllm_config
        from .report import provenance
        from .executor import TestExecutor
        from vllm.v1.engine.core import EngineCore
        from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator

        fault = self.fault
        require(
            hashes is self.hashes, "Expected hashes must come from source native views"
        )
        if fault:
            require(self.args.store == "mock", "Fault injection requires --store mock")
            self.control.update(
                fault=fault,
                fault_rank=self.args.tp - 1,
                observation_directory=self.args.output,
            )
        initial_loads = self.control.get("loads", 0)
        self.reader.model_executor.collective_rpc(
            "arm_reader", args=(self.hashes, self.args.tokens)
        )
        target = request(self.reader, "reader", self.args.tokens + 2)
        detected = None
        try:
            with set_current_vllm_config(self.reader.vllm_config):
                self.reader.add_request(target)
                reader_steps = drain(self.reader, self.args.timeout)
            results = self.reader.model_executor.collective_rpc("audit")
            require(target.is_finished(), "Reader request did not finish")
            if fault == "load-error":
                from vllm.v1.request import RequestStatus

                require(
                    target.status == RequestStatus.FINISHED_ERROR,
                    "Scheduler swallowed load failure",
                )
                detected = (
                    "Scheduler returned FINISHED_ERROR after injected load failure"
                )
            require(
                fault == "load-error"
                or all(x["compared"][0] > 0 and x["compared"][1] > 0 for x in results),
                "Reader rank compared no bytes",
            )
        except (RuntimeError, AssertionError) as error:
            expected = (
                "injected mock load failure"
                if fault == "load-error"
                else "Restored KV differs"
                if fault == "corrupt"
                else None
            )
            if expected is None or expected not in str(error):
                raise
            detected = str(error)
            results = []
            reader_steps = len(self.reader.model_executor.steps)
        require(
            fault not in ("load-error", "corrupt") or detected,
            "Injected fault went undetected",
        )
        require(
            self.control.get("loads", 0) > initial_loads,
            "No external store load occurred",
        )
        require(
            self.control.get("lookup_hits", 0) > 0, "No external lookup hit occurred"
        )
        result = {
            "status": "PASS",
            "scenario": "cold" if self.source_closed else "reuse",
            "fault": fault,
            "fault_detected": detected,
            "source_exited_before_reader": self.source_closed,
            "prefix_reset": not self.source_closed,
            "source_pids": self.source_pids,
            "reader_pids": [x["info"]["pid"] for x in results],
            "compared_segments_by_rank": [x["compared"][0] for x in results],
            "compared_bytes_by_rank": [x["compared"][1] for x in results],
            "external_loads": self.control.get("loads", 0) - initial_loads,
            "lookup_hits": self.control.get("lookup_hits", 0),
            "delayed_checks": self.control.get("delayed_checks", 0),
            "partial_completion_checks": getattr(
                self.reader, "test_partial_completion_checks", 0
            ),
            "source_steps": self.source_steps,
            "reader_steps": reader_steps,
            "source_trace": self.source_trace,
            "native_boundary_events": self.source.test_boundary_events,
            "reader_trace": self.reader.model_executor.steps[self.source_step_count :]
            if self.reader is self.source
            else self.reader.model_executor.steps,
            "classes": {
                "engine": EngineCore.__module__,
                "step": EngineCore.step.__code__.co_filename,
                "scheduler": type(self.reader.scheduler).__module__,
                "executor": TestExecutor.__module__,
                "aggregator": KVOutputAggregator.__module__,
            },
        }
        if self.args.store == "mock":
            from ..data_check.mock_store import summarize

            result["store_records"] = summarize(self.ledger)
        result.update(provenance(self.args))
        require(
            not result["cuda_initialized"] and not result.get("npu_initialized"),
            "Driver initialized accelerator",
        )
        self.result = result
        return result

    def close(self):
        if self.reader is not None and self.reader is not self.source:
            self.reader.shutdown()
        if self.source is not None:
            self.source.shutdown()
        if self.result is not None:
            executors = [self.source.model_executor]
            if self.reader is not self.source:
                executors.append(self.reader.model_executor)
            codes = [[p.exitcode for p in ex.processes] for ex in executors]
            forced = any(ex.forced_shutdown for ex in executors)
            errors = [error for ex in executors for error in ex.shutdown_errors]
            self.result.update(
                worker_exit_codes=codes,
                forced_worker_shutdown=forced,
                shutdown_errors=errors,
            )
            if self.fault != "corrupt":
                require(not errors, f"Executor shutdown errors: {errors}")
            require(
                not forced and all(code == 0 for group in codes for code in group),
                "Worker required forced cleanup",
            )


def run(args, scenario, fault):
    args.async_load = fault == "delay"
    initialize_runtime(args)
    prepare_materials(args)
    from ..test_roundtrip import cold_roundtrip, reuse_roundtrip

    with mp.get_context("spawn").Manager() as manager:
        ledger, lock, control = manager.dict(), manager.RLock(), manager.dict()
        if args.store == "mock":
            from ..data_check.mock_store import bind

            bind(None, ledger, lock, control)
        else:
            from ..data_check.real_store import bind

            bind(control, lock)
        session = EngineSession(args, ledger, lock, control, fault)
        try:
            callback = {"cold": cold_roundtrip, "reuse": reuse_roundtrip}[scenario]
            return callback(session)
        finally:
            session.close()


def main():
    args = SimpleNamespace(**json.loads(Path(sys.argv[1]).read_text()))
    result = {
        "status": "FAIL",
        "store": args.store,
        "mode": args.mode,
        "tp": args.tp,
        "pp": args.pp,
        "runner": args.runner,
    }
    code = 1
    try:
        result.update(
            run(args, sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
        )
        code = 0
    except BaseException:
        result["error"] = traceback.format_exc()
        traceback.print_exc()
    if "vllm" in sys.modules:
        try:
            from .report import provenance

            result.update(provenance(args))
            require(
                not result["cuda_initialized"] and not result.get("npu_initialized"),
                "Driver initialized accelerator",
            )
        except BaseException:
            result["provenance_error"] = traceback.format_exc()
            result["status"] = "FAIL"
            code = 1
    result["tokens"] = args.tokens
    result["exit_code"] = code
    (Path(args.output) / "result.json").write_text(
        json.dumps(result, indent=2, default=str)
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
