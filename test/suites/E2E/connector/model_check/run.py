# SPDX-License-Identifier: MIT
"""Unified card-less MRV2 parallel model-check for UCM v1 and v2 connectors.

Drives real vLLM MRV2 KV allocation, real TP x PCP x PP Gloo process groups,
real Scheduler + KVOutputAggregator lifecycles, and engine-native poison +
byte-roundtrip verification without relying on UCM internal layout rules.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4


def token_count(value: str) -> int:
    if value.lower().endswith("k"):
        return int(value[:-1]) * 1024
    return int(value)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--connector",
        choices=("v1", "v2"),
        default="v1",
        help="UCM connector generation to test (default: v1)",
    )
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="local model directory or its config.json",
    )
    parser.add_argument(
        "--platform",
        choices=("cuda-sim", "ascend-sim", "cuda", "ascend"),
        required=True,
    )
    parser.add_argument(
        "--kv-device",
        choices=("cpu", "cuda", "npu"),
        default="cpu",
        help="memory used for KV tensors (default: cpu)",
    )
    parser.add_argument("--cuda-capability", metavar="MAJOR.MINOR")
    parser.add_argument("--ascend-soc", help="installed vllm-ascend SoC family, e.g. A2")
    parser.add_argument(
        "--tokens",
        type=token_count,
        default=4096,
        help="prompt length in tokens (default: 4096; supports K suffix)",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=token_count,
        default=2048,
        help="scheduler token budget per step (default: 2048)",
    )
    parser.add_argument(
        "--prefix-percent",
        type=int,
        default=75,
        metavar="1..99",
        help="shared prefix percentage for hot-load verification (default: 75)",
    )
    parser.add_argument("--block-size", type=int, help="native vLLM block size")
    parser.add_argument("--kv-mib", type=int, help="per-rank KV pool budget in MiB")
    parser.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    parser.add_argument("--pp", type=int, default=1, help="pipeline parallel size")
    parser.add_argument(
        "--pcp", type=int, default=1, help="prefill context parallel size"
    )
    parser.add_argument(
        "--dcp", type=int, default=1, help="decode context parallel size"
    )
    parser.add_argument(
        "--cp-kv-cache-interleave-size",
        type=int,
        default=None,
        help="optional vLLM cp_kv_cache_interleave_size override",
    )
    parser.add_argument(
        "--use-layerwise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable or disable UCM layerwise mode (default: True)",
    )
    parser.add_argument("--chunk-size", type=int, help="optional v2 UCM chunk size")
    parser.add_argument("--ucm-config", type=Path, help="optional custom UCM YAML")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--quantization", help="model weight quantization")
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--indexer-kv-dtype", default="auto")
    parser.add_argument("--mamba-ssm-cache-dtype", default="auto")
    parser.add_argument("--speculative-config", type=json.loads)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--allow-duplicate-writes",
        action="store_true",
        help="allow identical duplicate cross-rank writes instead of failing audit",
    )
    parser.add_argument(
        "--expect-reject",
        metavar="REGEX",
        help="expect startup/connector construction to fail with a matching error",
    )

    args = parser.parse_args(argv)
    if args.model.name == "config.json":
        args.model = args.model.parent
    if not (args.model / "config.json").is_file():
        parser.error("--model must point to a local model directory with config.json")
    if not 1 <= args.prefix_percent <= 99:
        parser.error("--prefix-percent must be in [1, 99]")
    if args.platform == "cuda-sim" and not args.cuda_capability:
        parser.error("cuda-sim requires --cuda-capability, e.g. 9.0")
    if args.platform == "ascend-sim" and not args.ascend_soc:
        parser.error("ascend-sim requires --ascend-soc, e.g. A2")
    if args.platform in ("cuda", "ascend") and (
        args.cuda_capability or args.ascend_soc
    ):
        parser.error("Real platforms discover hardware from the device")
    return args


def _execute_worker_step(
    engine,
    worker,
    metadata,
    total_tokens: int,
    views: dict,
    finished_req_ids: set[str],
):
    """Run the exact vLLM KVConnectorBase_V1 worker callback sequence for one step."""
    from vllm.forward_context import get_forward_context, set_forward_context
    from vllm.v1.outputs import KVConnectorOutput

    if metadata is not None:
        worker.bind_connector_metadata(metadata)
        with set_forward_context(None, engine, num_tokens=total_tokens):
            worker.start_load_kv(get_forward_context())
            for layer_name, view in views.items():
                worker.wait_for_layer_load(layer_name)
                worker.save_kv_layer(layer_name, view, None)
            worker.wait_for_save()

    transfers = worker.get_transfer_results(set(finished_req_ids))
    worker_meta = worker.build_connector_worker_meta()
    invalid_blocks = set(worker.get_block_ids_with_load_errors() or ())
    failed_recving = set(getattr(transfers, "failed_recving", ()) or ())
    if metadata is not None:
        worker.clear_connector_metadata()

    return KVConnectorOutput(
        finished_sending=set(transfers.finished_sending or ()) or None,
        finished_recving=set(transfers.finished_recving or ()) or None,
        failed_recving=failed_recving,
        invalid_block_ids=invalid_blocks,
        kv_connector_worker_meta=worker_meta,
    )


def _step_all_ranks(engine, scheduler, aggregator, worker, views, *, is_root: bool):
    """Execute one synchronized Scheduler -> Workers -> KVOutputAggregator step."""
    from parallel import broadcast, gather
    from vllm.v1.outputs import make_empty_encoder_model_runner_output

    output = None
    step_payload = None
    if is_root:
        output = scheduler.schedule()
        step_payload = (
            output.kv_connector_metadata,
            int(output.total_num_scheduled_tokens),
            dict(output.num_scheduled_tokens),
            set(output.finished_req_ids or ()),
        )
    metadata, total_tokens, num_scheduled, finished_req_ids = broadcast(step_payload)

    local_kv_out = _execute_worker_step(
        engine, worker, metadata, total_tokens, views, finished_req_ids
    )
    all_kv_outs = gather(local_kv_out)

    if is_root:
        runner_outputs = []
        for kv_out in all_kv_outs:
            runner_out = make_empty_encoder_model_runner_output(output)
            runner_out.kv_connector_output = kv_out
            runner_outputs.append(runner_out)
        aggregated = aggregator.aggregate(runner_outputs, output_rank=0)
        scheduler.update_from_output(output, aggregated)
        return output, aggregated
    return None, None


def _drain_scheduler(engine, scheduler, aggregator, worker, views, *, is_root: bool):
    """Advance empty steps until all pending async dumps/offloads finish."""
    from parallel import broadcast

    for _ in range(32):
        has_work = broadcast(scheduler.has_requests() if is_root else False)
        if not has_work:
            return
        _step_all_ranks(engine, scheduler, aggregator, worker, views, is_root=is_root)
    if is_root:
        raise AssertionError("Scheduler still has unfinished requests after drain limit")


def _make_request(request_id: str, token_ids: list[int], hash_size: int, engine):
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import get_hash_fn_by_name
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.request import Request

    hash_fn = get_hash_fn_by_name(engine.cache_config.prefix_caching_hash_algo)
    init_none_hash(hash_fn)
    return Request(
        request_id=request_id,
        prompt_token_ids=list(token_ids),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        block_hasher=get_request_block_hasher(hash_size, hash_fn),
    )


def _run_request_to_prefill_end(
    engine,
    scheduler,
    aggregator,
    worker,
    views,
    req,
    *,
    is_root: bool,
    target_tokens: int,
    on_load_step_complete=None,
):
    """Run one request through chunked prefill and return final tables & hit stats."""
    from parallel import broadcast
    from vllm.v1.request import RequestStatus

    if is_root:
        scheduler.add_request(req)

    local_hit = 0
    external_hit = 0
    first_step = True
    steps = 0
    final_tables = None
    last_computed = 0
    load_checked = False

    for _ in range(max(64, target_tokens // 16 + 16)):
        output, _ = _step_all_ranks(
            engine, scheduler, aggregator, worker, views, is_root=is_root
        )
        load_event = None
        done = False
        if is_root:
            steps += 1
            if first_step and getattr(req, "prefill_stats", None) is not None:
                local_hit = int(req.prefill_stats.num_local_cached_tokens)
                external_hit = int(req.prefill_stats.num_external_cached_tokens)
                first_step = False
            last_computed = int(req.num_computed_tokens)
            # Fire load-complete callback once external load has finished (not
            # still WAITING_FOR_REMOTE_KVS) and before subsequent prefill steps
            # can move running Mamba/state slots.
            if (
                on_load_step_complete is not None
                and not load_checked
                and external_hit > 0
                and req.status != RequestStatus.WAITING_FOR_REMOTE_KVS
                and last_computed > external_hit
            ):
                load_tables = scheduler.kv_cache_manager.get_block_ids(req.request_id)
                load_event = (load_tables, local_hit, external_hit, last_computed)
                load_checked = True
            if last_computed >= target_tokens:
                final_tables = scheduler.kv_cache_manager.get_block_ids(req.request_id)
                done = True
        load_event = broadcast(load_event)
        if load_event is not None and on_load_step_complete is not None:
            on_load_step_complete(*load_event)
        done = broadcast(done)
        if done:
            break
    else:
        raise AssertionError("Request did not complete prefill within step budget")

    info = broadcast((final_tables, local_hit, external_hit, last_computed, steps))
    return info


def run(args, cleanup, scratch_dir: str, kv_device):
    import torch.distributed as dist
    from mock_store import SharedFileByteStore, install_mock_store
    from native_cache import allocate, make_engine
    from parallel import barrier, broadcast, collect_handshakes, inspect_local_coords
    from verifier import (
        audit_writes,
        fill_native_kv,
        poison_native_kv,
        register_kv_buffers,
        snapshot_cold_blocks,
        verify_cold_vs_hot,
    )

    shared_root = Path(scratch_dir) / "shared_store"
    shared_audit = Path(scratch_dir) / "audit"
    store_root = Path(shared_root)
    audit_dir = Path(shared_audit)

    engine = make_engine(args, store_root)
    worker_config, sched_config, block_size, hash_size, views = allocate(
        engine, args.kv_mib, scratch_dir, cleanup, kv_device=kv_device
    )

    # Share rank 0's scratch paths across all torchrun workers.
    store_root_str, audit_dir_str = broadcast((str(store_root), str(audit_dir)))
    store_root = Path(store_root_str)
    audit_dir = Path(audit_dir_str)
    if args.connector == "v1" and not args.ucm_config:
        ucm_cfg = engine.kv_transfer_config.kv_connector_extra_config["ucm_connectors"][0][
            "ucm_connector_config"
        ]
        ucm_cfg["storage_backends"] = str(store_root)

    coords = inspect_local_coords(args)
    is_root = coords.world_rank == 0
    buffers = register_kv_buffers(views)
    backing = SharedFileByteStore(store_root, audit_dir, coords.world_rank)

    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
    from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    from vllm.v1.request import RequestStatus
    from vllm.v1.structured_output import StructuredOutputManager

    with install_mock_store(
        args.connector,
        backing,
        buffers,
        use_host_device=(kv_device.type == "cpu"),
    ):
        scheduler = None
        aggregator = None
        worker = KVConnectorFactory.create_connector(
            engine, KVConnectorRole.WORKER, worker_config
        )
        cleanup.callback(lambda: getattr(worker, "shutdown", lambda: None)())
        if args.connector == "v2" and kv_device.type == "cpu":
            worker.connector_worker._device = None
        worker.register_kv_caches(views)

        handshakes = collect_handshakes(worker, coords, args)
        if is_root:
            scheduler = engine.scheduler_config.get_scheduler_cls()(
                vllm_config=engine,
                kv_cache_config=sched_config,
                structured_output_manager=StructuredOutputManager(engine),
                block_size=block_size,
                hash_block_size=hash_size,
            )
            cleanup.callback(scheduler.shutdown)
            if handshakes:
                scheduler.connector.set_xfer_handshake_metadata_pp_aware(handshakes)
            aggregator = KVOutputAggregator.from_connector(
                scheduler.connector, dist.get_world_size()
            )
        barrier()

        if args.expect_reject:
            raise AssertionError(
                f"Expected startup rejection matching {args.expect_reject!r}, "
                "but connector initialization succeeded"
            )

        # ---- Phase 1: Cold prefill + Dump ----------------------------------
        fill_native_kv(views, worker_config, coords)
        vocab_size = max(1024, int(engine.model_config.get_vocab_size()))
        prompt_a = [(idx * 17 + 3) % vocab_size for idx in range(args.tokens)]

        req_a = (
            _make_request(f"cold-{uuid4().hex[:8]}", prompt_a, hash_size, engine)
            if is_root
            else None
        )
        tables_a, local_a, ext_a, _, steps_a = _run_request_to_prefill_end(
            engine,
            scheduler,
            aggregator,
            worker,
            views,
            req_a,
            is_root=is_root,
            target_tokens=args.tokens,
        )
        if local_a != 0 or ext_a != 0:
            raise AssertionError(
                f"Cold request unexpectedly hit cache: local={local_a}, external={ext_a}"
            )
        if args.tokens > args.max_num_batched_tokens and steps_a < 2:
            raise AssertionError(
                f"Expected multi-step chunked prefill for tokens={args.tokens}, "
                f"budget={args.max_num_batched_tokens}, got steps_cold={steps_a}"
            )
        # Snapshot ONLY the blocks allocated to prompt_a right after cold prefill.
        cold_snapshots = snapshot_cold_blocks(
            worker_config,
            views,
            tables_a,
            device_type=engine.device_config.device_type,
        )
        if is_root:
            scheduler.finish_requests(req_a.request_id, RequestStatus.FINISHED_ABORTED)
        _drain_scheduler(
            engine, scheduler, aggregator, worker, views, is_root=is_root
        )
        barrier()

        # Verify write audit on rank 0 before checking reads.
        audit_summary = None
        if is_root:
            audit_summary = audit_writes(
                audit_dir, allow_duplicate_writes=args.allow_duplicate_writes
            )
            print(f"[rank 0] Cold Dump audit passed: {audit_summary}", flush=True)

        # ---- Phase 2: Negative isolation check (mutated first token) -------
        if is_root:
            if not scheduler.reset_prefix_cache():
                raise AssertionError("Failed to reset local HBM prefix cache")
        barrier()

        prompt_neg = list(prompt_a)
        prompt_neg[0] = (prompt_neg[0] + 1) % vocab_size
        req_neg = (
            _make_request(f"neg-{uuid4().hex[:8]}", prompt_neg, hash_size, engine)
            if is_root
            else None
        )
        _, local_neg, ext_neg, _, _ = _run_request_to_prefill_end(
            engine,
            scheduler,
            aggregator,
            worker,
            views,
            req_neg,
            is_root=is_root,
            target_tokens=args.tokens,
        )
        if ext_neg != 0 or local_neg != 0:
            raise AssertionError(
                f"Mutated prompt falsely hit cache: local={local_neg}, external={ext_neg}"
            )
        if is_root:
            scheduler.finish_requests(req_neg.request_id, RequestStatus.FINISHED_ABORTED)
        _drain_scheduler(
            engine, scheduler, aggregator, worker, views, is_root=is_root
        )
        barrier()

        # ---- Phase 3: Poison KV + Hot reload + Engine-native byte check ----
        from math import lcm

        if args.connector == "v2":
            connector_block = int(worker.connector_worker.layout.spec.chunk_size)
        else:
            connector_block = broadcast(
                int(scheduler.connector.get_block_size() or block_size)
                if is_root
                else 1
            )
        hit_alignment = lcm(
            int(block_size),
            int(getattr(sched_config, "cache_hit_alignment_tokens", 1) or 1),
            int(connector_block),
        )
        prefix_len = max(hit_alignment, args.tokens * args.prefix_percent // 100)
        expected_hit = (min(prefix_len, args.tokens - 1) // hit_alignment) * hit_alignment
        if expected_hit <= 0:
            raise ValueError(
                f"Prompt tokens={args.tokens} too short for hit_alignment={hit_alignment}; "
                "increase --tokens"
            )
        prompt_hot = prompt_a[:prefix_len] + [
            (idx * 31 + 11) % vocab_size for idx in range(args.tokens - prefix_len)
        ]

        if is_root:
            if not scheduler.reset_prefix_cache():
                raise AssertionError(
                    "Failed to reset local HBM prefix cache before reload"
                )
        barrier()

        poison_native_kv(views, poison_byte=0xA5)
        barrier()

        verify_result = {"slices": 0, "bytes": 0, "checked": False}

        def check_after_load(load_tables, local_hit_now, ext_hit_now, step_end_now):
            if local_hit_now != 0:
                raise AssertionError(
                    f"Hot reload unexpectedly hit local HBM ({local_hit_now} tokens) "
                    "instead of external store"
                )
            slices_count, bytes_count = verify_cold_vs_hot(
                worker_config,
                cold_snapshots,
                tables_a,
                views,
                load_tables,
                ext_hit_now,
                hot_load_step_end=step_end_now,
                dcp_size=args.dcp,
                device_type=engine.device_config.device_type,
                world_rank=coords.world_rank,
                poison_byte=0xA5,
            )
            verify_result["slices"] = slices_count
            verify_result["bytes"] = bytes_count
            verify_result["checked"] = True

        req_hot = (
            _make_request(f"hot-{uuid4().hex[:8]}", prompt_hot, hash_size, engine)
            if is_root
            else None
        )
        _, local_hot, ext_hot, _, steps_hot = _run_request_to_prefill_end(
            engine,
            scheduler,
            aggregator,
            worker,
            views,
            req_hot,
            is_root=is_root,
            target_tokens=args.tokens,
            on_load_step_complete=check_after_load,
        )
        if local_hot != 0:
            raise AssertionError(
                f"Hot reload unexpectedly hit local HBM ({local_hot} tokens) "
                "instead of external store"
            )
        if ext_hot != expected_hit:
            raise AssertionError(
                f"Hot reload hit mismatch: expected_hit={expected_hit}, "
                f"actual_external_hit={ext_hot} "
                f"(prefix_len={prefix_len}, hit_alignment={hit_alignment})"
            )
        if not verify_result["checked"]:
            raise AssertionError(
                "Hot reload never triggered post-load native slice verification"
            )

        if is_root:
            scheduler.finish_requests(req_hot.request_id, RequestStatus.FINISHED_ABORTED)
        _drain_scheduler(
            engine, scheduler, aggregator, worker, views, is_root=is_root
        )
        barrier()

        return {
            "steps_cold": steps_a,
            "steps_hot": steps_hot,
            "external_hit_tokens": ext_hot,
            "verified_bytes": verify_result["bytes"],
            "verified_slices": verify_result["slices"],
            "audit": audit_summary,
        }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(f"Model-check arguments: {vars(args)}", flush=True)

    # Make local sibling modules importable when invoked as a script.
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    from parallel import launch

    if launch(args):
        return

    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    import torch

    kv_device = torch.device("cpu")
    if args.platform == "cuda-sim":
        from host_runtime import install

        install("cuda", args.cuda_capability)
    elif args.platform == "ascend-sim":
        from host_runtime import install

        install("npu", args.ascend_soc.upper())
    else:
        if args.platform == "ascend":
            import torch_npu  # noqa: F401
        from vllm.platforms import current_platform

        device_type = "npu" if args.platform == "ascend" else "cuda"
        if current_platform.device_type != device_type:
            raise RuntimeError(
                f"Expected {device_type}, vLLM selected {current_platform.device_type}"
            )
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        accelerator = torch.device(device_type, local_rank)
        current_platform.set_device(accelerator)
        if args.kv_device != "cpu":
            kv_device = accelerator

    try:
        with TemporaryDirectory() as scratch, ExitStack() as cleanup:
            summary = run(args, cleanup, scratch, kv_device)
    except Exception as exc:
        if args.expect_reject and re.search(args.expect_reject, str(exc)):
            if int(os.environ.get("RANK", "0")) == 0:
                print(
                    f"PASS (expected reject): connector={args.connector}, "
                    f"TP={args.tp} PP={args.pp} PCP={args.pcp} DCP={args.dcp}, "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )
            return
        raise

    if int(os.environ.get("RANK", "0")) != 0:
        return
    print(
        f"PASS: connector={args.connector}, "
        f"TP={args.tp} PP={args.pp} PCP={args.pcp} DCP={args.dcp}, "
        f"layerwise={args.use_layerwise}, hit={summary['external_hit_tokens']}/{args.tokens}, "
        f"verified_slices={summary['verified_slices']}, "
        f"verified_bytes={summary['verified_bytes']}, audit={summary['audit']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
