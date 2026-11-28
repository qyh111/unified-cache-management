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


class _ExpectedRejectNotTriggeredError(AssertionError):
    """Raised when --expect-reject succeeded unexpectedly; never self-matched."""


def token_count(value: str) -> int:
    if value.lower().endswith("k"):
        return int(value[:-1]) * 1024
    return int(value)


def interleave_size_arg(value: str) -> int | str:
    if value.lower() == "auto":
        return "auto"
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            "cp-kv-cache-interleave-size must be a positive integer or 'auto'"
        )
    return parsed


def resolve_schedule_plan(
    *,
    requested_tokens: int,
    requested_batch: int,
    hit_alignment: int,
    prefix_percent: int,
    auto_align_chunks: int,
    max_model_len: int,
) -> tuple[int, int, int, int]:
    """Resolve (effective_tokens, effective_batch, prefix_len, expected_hit).

    When auto_align_chunks > 0, size the prompt to an exact multiple of
    (auto_align_chunks * hit_alignment) and set the per-step batch budget to
    one chunk so cold prefill lands on exact hit_alignment boundaries and
    writes a real Mamba/KDA State checkpoint at prefix_len.
    """
    if auto_align_chunks > 0:
        unit = auto_align_chunks * hit_alignment
        if requested_tokens >= unit:
            effective_tokens = (requested_tokens // unit) * unit
        else:
            effective_tokens = unit
        if effective_tokens + 1 > max_model_len:
            raise ValueError(
                f"Auto-aligned prompt effective_tokens={effective_tokens} exceeds "
                f"max_model_len={max_model_len} "
                f"(hit_alignment={hit_alignment}, auto_align_chunks={auto_align_chunks}); "
                "increase --tokens or reduce --auto-align-chunks"
            )
        effective_batch = effective_tokens // auto_align_chunks
        hit_chunks = max(
            1,
            min(
                auto_align_chunks - 1,
                round(auto_align_chunks * prefix_percent / 100),
            ),
        )
        prefix_len = hit_chunks * effective_batch
        expected_hit = prefix_len
        return effective_tokens, effective_batch, prefix_len, expected_hit

    effective_tokens = requested_tokens
    effective_batch = requested_batch
    prefix_len = max(hit_alignment, effective_tokens * prefix_percent // 100)
    expected_hit = (
        min(prefix_len, effective_tokens - 1) // hit_alignment
    ) * hit_alignment
    if expected_hit <= 0:
        raise ValueError(
            f"Prompt tokens={effective_tokens} too short for "
            f"hit_alignment={hit_alignment}; increase --tokens or pass "
            "--auto-align-chunks 4"
        )
    return effective_tokens, effective_batch, prefix_len, expected_hit


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--connector",
        choices=("v1", "v2"),
        default="v1",
        help="UCM connector generation to test (default: v1)",
    )
    parser.add_argument(
        "--multi-connector",
        choices=("none", "pd-first", "ucm-first"),
        default="none",
        help=(
            "wrap UCMConnector inside vLLM MultiConnector with a sibling P/D "
            "handshake connector (default: none; 'pd-first' mimics Mooncake + UCM)"
        ),
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
    parser.add_argument(
        "--auto-align-chunks",
        type=int,
        default=0,
        metavar="N",
        help=(
            "when >= 2, auto-size prompt tokens and per-step batch budget to N "
            "aligned chunks of the real runtime hit_alignment (0 disables)"
        ),
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
        type=interleave_size_arg,
        default=None,
        help="optional vLLM cp_kv_cache_interleave_size override (positive int or 'auto')",
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
        "--partial-miss",
        choices=("pp-stage", "dcp-shard"),
        help="after a successful v1 reload, hide one committed key and check hit truncation",
    )
    parser.add_argument(
        "--dump-delay-polls",
        type=int,
        default=0,
        help="hold v1 Dump tasks for N polls and schedule a second request before draining",
    )
    parser.add_argument(
        "--load-delay-polls",
        type=int,
        default=0,
        help=(
            "hold v1 async Load tasks for N polls and run an overlapping prefill "
            "request while the hot request waits in WAITING_FOR_REMOTE_KVS"
        ),
    )
    parser.add_argument(
        "--enable-ucm-patch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "enable or disable UCM runtime monkey-patches (ENABLE_UCM_PATCH) "
            "before importing vLLM; default follows the ENABLE_UCM_PATCH env var"
        ),
    )
    parser.add_argument(
        "--verify-load-guards",
        action="store_true",
        help="verify unallocated native KV guard pages after the v1 hot load",
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
    if args.auto_align_chunks != 0 and args.auto_align_chunks < 2:
        parser.error("--auto-align-chunks must be 0 (disabled) or >= 2")
    if args.partial_miss:
        if args.connector != "v1" or args.ucm_config or args.expect_reject:
            parser.error("--partial-miss requires the default v1 mock Store")
        if args.partial_miss == "pp-stage" and (args.pp < 2 or args.dcp != 1):
            parser.error("--partial-miss pp-stage requires PP > 1 and DCP = 1")
        if args.partial_miss == "dcp-shard" and (args.dcp < 2 or args.pp != 1):
            parser.error("--partial-miss dcp-shard requires DCP > 1 and PP = 1")
    if args.dump_delay_polls < 0 or args.dump_delay_polls and args.connector != "v1":
        parser.error("--dump-delay-polls requires a nonnegative v1 poll count")
    if args.load_delay_polls < 0 or args.load_delay_polls and args.connector != "v1":
        parser.error("--load-delay-polls requires a nonnegative v1 poll count")
    if args.verify_load_guards and args.connector != "v1":
        parser.error("--verify-load-guards requires v1")
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
    worker_config,
    worker,
    metadata,
    total_tokens: int,
    views: dict,
    finished_req_ids: set[str],
    *,
    native_v1: bool,
    sync_loads: bool,
    prefill=None,
    scheduled_requests=(),
):
    """Drive connector callbacks around the simulated layer forward."""
    from verifier import apply_mamba_align_precopy
    from vllm.forward_context import get_forward_context, set_forward_context
    from vllm.v1.outputs import KVConnectorOutput

    if metadata is not None and native_v1:
        worker.handle_preemptions(metadata)

    # Mirror MRV2 execute_model: run Mamba align pre-copy before pre_forward /
    # sync start_load_kv so async-loaded checkpoints migrate into running slots
    # on the promotion step without masking missing sync loads.
    if total_tokens and scheduled_requests:
        apply_mamba_align_precopy(
            worker_config,
            views,
            scheduled_requests,
            device_type=engine.device_config.device_type,
        )

    if metadata is not None:
        worker.bind_connector_metadata(metadata)
        with set_forward_context(None, engine, num_tokens=total_tokens):
            if not native_v1 or sync_loads:
                worker.start_load_kv(get_forward_context())
            if total_tokens or not native_v1:
                for layer_names in _forward_layers(views, native_v1):
                    for layer_name in layer_names:
                        worker.wait_for_layer_load(layer_name)
                        if prefill is not None:
                            prefill.write_layer(layer_name, scheduled_requests)
                    for layer_name in layer_names:
                        worker.save_kv_layer(layer_name, views[layer_name], None)
            if native_v1:
                worker.finish_forward()
                if not sync_loads:
                    worker.start_load_kv(get_forward_context())
            worker.wait_for_save()
    elif prefill is not None:
        for layer_name in views:
            prefill.write_layer(layer_name, scheduled_requests)

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


def _forward_layers(views, native_v1):
    if not native_v1:
        return [(name,) for name in views]
    # A model layer may publish paired cache views (e.g. attention + indexer).
    # Both receive their new data before any save hook for that layer runs.
    layers = {}
    for name in views:
        match = re.search(r"(?:^|\.)layers\.(\d+)(?=\.|$)", name)
        key = (int(match[1]), name[: match.end()]) if match else (-1, name)
        layers.setdefault(key, []).append(name)
    return [layers[key] for key in sorted(layers, key=lambda group: group[0])]


def _step_all_ranks(
    engine,
    worker_config,
    scheduler,
    aggregator,
    worker,
    views,
    *,
    is_root: bool,
    native_v1: bool,
    prefill=None,
    dcp_dump_ranks=None,
    protected_blocks=(),
    protected_label: str = "pending Dump source",
):
    """Execute one synchronized Scheduler -> Workers -> KVOutputAggregator step."""
    from parallel import broadcast, gather
    from vllm.v1.outputs import make_empty_encoder_model_runner_output

    output = None
    step_payload = None
    boundaries = {}
    if is_root:
        # EngineCore's Scheduler clears connector-only block state before it
        # returns. Capture the native hand-off at the producer SPI boundary.
        if prefill is not None:
            from unittest.mock import patch

            build_meta = scheduler.connector.build_connector_meta

            def capture_state(output):
                state = getattr(output, "kv_connector_block_state", None)
                if state is not None:
                    boundaries.update(
                        (req_id, tuple(entries))
                        for req_id, entries in state.boundary_state_offloads.items()
                    )
                return build_meta(output)

            with patch.object(
                scheduler.connector, "build_connector_meta", capture_state
            ):
                output = scheduler.schedule()
        else:
            output = scheduler.schedule()
        if dcp_dump_ranks is not None and output.kv_connector_metadata is not None:
            # MLA CP exposes scheduler keys unchanged to the Store. Their
            # semantic DCP order is fixed by the v1 metadata contract.
            for request in output.kv_connector_metadata.request_meta.values():
                keys, blocks = request.dump_block_ids
                dcp_size = engine.parallel_config.decode_context_parallel_size
                if len(keys) != dcp_size * len(blocks):
                    raise AssertionError("Invalid scheduler DCP dump plan geometry")
                for offset, key in enumerate(keys):
                    key_hex = key.hex()
                    expected = offset % dcp_size
                    if dcp_dump_ranks.setdefault(key_hex, expected) != expected:
                        raise AssertionError(
                            f"DCP dump key {key_hex} changed shard rank"
                        )
        scheduled_requests = []
        for request_id, count in output.num_scheduled_tokens.items():
            end = scheduler.requests[request_id].num_computed_tokens
            tables = scheduler.kv_cache_manager.get_block_ids(request_id)
            if protected_blocks:
                reused = set(protected_blocks).intersection(
                    block_id for table in tables for block_id in table
                )
                if reused:
                    raise AssertionError(
                        f"Allocated {protected_label} blocks: {sorted(reused)}"
                    )
            scheduled_requests.append(
                (
                    request_id,
                    end - count,
                    end,
                    tables,
                    boundaries.get(request_id, ()),
                )
            )
        step_payload = (
            output.kv_connector_metadata,
            int(output.total_num_scheduled_tokens),
            set(output.finished_req_ids or ()),
            bool(getattr(output, "has_sync_kv_loads", False)),
            scheduled_requests,
        )
    metadata, total_tokens, finished_req_ids, sync_loads, scheduled_requests = (
        broadcast(step_payload)
    )

    local_kv_out = _execute_worker_step(
        engine,
        worker_config,
        worker,
        metadata,
        total_tokens,
        views,
        finished_req_ids,
        native_v1=native_v1,
        sync_loads=sync_loads,
        prefill=prefill,
        scheduled_requests=scheduled_requests,
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
        return output, boundaries
    return None, None


def _drain_scheduler(
    engine,
    worker_config,
    scheduler,
    aggregator,
    worker,
    views,
    *,
    is_root: bool,
    native_v1: bool,
    max_steps: int = 32,
):
    """Advance empty steps until all pending async dumps/offloads finish."""
    from parallel import broadcast

    for _ in range(max_steps):
        has_work = broadcast(scheduler.has_requests() if is_root else False)
        if not has_work:
            return
        _step_all_ranks(
            engine,
            worker_config,
            scheduler,
            aggregator,
            worker,
            views,
            is_root=is_root,
            native_v1=native_v1,
        )
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
    worker_config,
    scheduler,
    aggregator,
    worker,
    views,
    req,
    *,
    is_root: bool,
    target_tokens: int,
    native_v1: bool,
    prefill=None,
    dcp_dump_ranks=None,
    capture_at: int | None = None,
    capture_boundary=None,
    on_load_step_complete=None,
    already_added: bool = False,
    initial_local_hit: int = 0,
    initial_external_hit: int = 0,
    initial_steps: int = 0,
    extra_step_budget: int = 0,
):
    """Run one request through chunked prefill and return final tables & hit stats."""
    from parallel import broadcast
    from vllm.v1.request import RequestStatus

    if is_root and not already_added:
        scheduler.add_request(req)

    local_hit = initial_local_hit
    external_hit = initial_external_hit
    first_step = not already_added
    steps = initial_steps
    final_tables = None
    last_computed = 0
    load_checked = False
    cold_captured = False
    cold_state_blocks = {}
    step_ends: list[int] = []

    for _ in range(max(64, target_tokens // 16 + 16) + extra_step_budget):
        output, boundaries = _step_all_ranks(
            engine,
            worker_config,
            scheduler,
            aggregator,
            worker,
            views,
            is_root=is_root,
            native_v1=native_v1,
            prefill=prefill,
            dcp_dump_ranks=dcp_dump_ranks,
        )
        load_event = None
        cold_event = None
        done = False
        if is_root:
            steps += 1
            if first_step and getattr(req, "prefill_stats", None) is not None:
                local_hit = int(req.prefill_stats.num_local_cached_tokens)
                external_hit = int(req.prefill_stats.num_external_cached_tokens)
                first_step = False
            last_computed = int(req.num_computed_tokens)
            if last_computed > 0 and (
                not step_ends or last_computed != step_ends[-1]
            ):
                step_ends.append(last_computed)
            if capture_at is not None and not cold_captured:
                cold_state_blocks.update(
                    (gid, block)
                    for gid, block, boundary in boundaries.get(req.request_id, ())
                    if boundary == capture_at
                )
                if last_computed >= capture_at:
                    cold_event = (
                        scheduler.kv_cache_manager.get_block_ids(req.request_id),
                        cold_state_blocks,
                    )
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
        if capture_at is not None and not cold_captured:
            cold_event = broadcast(cold_event)
            if cold_event is not None:
                capture_boundary(*cold_event)
                cold_captured = True
        done = broadcast(done)
        if done:
            break
    else:
        raise AssertionError("Request did not complete prefill within step budget")

    info = broadcast(
        (final_tables, local_hit, external_hit, last_computed, steps, step_ends)
    )
    return info


def _find_ucm_connector(connector):
    """Unwrap MultiConnector to locate the UCMConnector child."""
    if hasattr(connector, "sub_connectors"):
        for child in connector.sub_connectors:
            if type(child).__name__ == "UCMConnector" or hasattr(
                child, "connector_worker"
            ):
                return child
    return connector


def _set_v1_store_root(kv_transfer_config, store_root: Path) -> None:
    """Update v1 UCM storage_backends on standalone or MultiConnector configs."""
    extra = kv_transfer_config.kv_connector_extra_config or {}
    if kv_transfer_config.kv_connector == "MultiConnector":
        for child in extra.get("connectors", ()):
            if child.get("kv_connector") == "UCMConnector":
                child_extra = child.get("kv_connector_extra_config") or {}
                ucm_connectors = child_extra.get("ucm_connectors")
                if ucm_connectors:
                    ucm_connectors[0]["ucm_connector_config"]["storage_backends"] = (
                        str(store_root)
                    )
        return
    ucm_connectors = extra.get("ucm_connectors")
    if ucm_connectors:
        ucm_connectors[0]["ucm_connector_config"]["storage_backends"] = str(store_root)


def _register_connector_staging_buffers(buffers, worker) -> None:
    """Register live connector-owned staging buffers (e.g. HLA PLE pack buffer)."""
    candidates = (
        list(worker.sub_connectors) if hasattr(worker, "sub_connectors") else [worker]
    )
    for candidate in candidates:
        inner = getattr(candidate, "connector", candidate)
        layout = getattr(inner, "kv_cache_layout", None)
        if layout is None or not hasattr(layout, "_ple_stage"):
            continue

        def _ple_stage_ranges(bound_layout=layout) -> list[tuple[int, int]]:
            stage = getattr(bound_layout, "_ple_stage", None)
            if stage is None or not stage.numel():
                return []
            return [
                (int(stage.data_ptr()), int(stage.numel() * stage.element_size()))
            ]

        buffers.register_dynamic_provider(_ple_stage_ranges)


def _check_partial_miss(
    args,
    backing,
    scheduler,
    engine,
    hash_size,
    prompt,
    hit_alignment,
    connector_block,
):
    """Remove the commit marker of exactly one key from an otherwise warm prefix."""
    shards = args.dcp
    missing_idx = (hit_alignment // connector_block + 1) * shards - 1
    candidates = []
    for namespace, keys in backing.prefix_lookups:
        if args.partial_miss == "pp-stage" and not namespace.name.startswith("pp_1_"):
            continue
        if len(keys) > missing_idx and all(
            backing.is_v1_committed(namespace, key) for key in keys[: missing_idx + 1]
        ):
            candidates.append((namespace, keys[missing_idx]))
    if not candidates:
        raise AssertionError(
            f"No committed {args.partial_miss} key at index {missing_idx}"
        )
    namespace, missing_key = candidates[0]
    if not namespace.resolve().is_relative_to(backing.root.resolve()):
        raise AssertionError(
            f"Partial-miss key is outside the scratch Store: {namespace}"
        )
    marker = namespace / f"{missing_key.hex()}.committed"
    hidden = marker.with_suffix(".hidden")
    backing.prefix_lookups = None
    marker.rename(hidden)
    try:
        req = _make_request(f"partial-{uuid4().hex[:8]}", prompt, hash_size, engine)
        matched, _ = scheduler.connector.get_num_new_matched_tokens(req, 0)
        if matched != hit_alignment:
            raise AssertionError(
                f"Partial {args.partial_miss} miss: expected {hit_alignment} tokens, "
                f"got {matched} with {missing_key.hex()} missing in {namespace.name}"
            )
    finally:
        hidden.rename(marker)


def run(args, cleanup, scratch_dir: str, kv_device):
    import torch.distributed as dist
    from mock_store import SharedFileByteStore, install_mock_store
    from native_cache import allocate, make_engine
    from parallel import barrier, broadcast, collect_handshakes, inspect_local_coords
    from verifier import (
        NativePrefill,
        audit_writes,
        fill_native_kv,
        poison_native_kv,
        register_kv_buffers,
        snapshot_cold_blocks,
        verify_cold_vs_hot,
        verify_unallocated_kv_is_poisoned,
    )

    native_v1 = args.connector == "v1"
    shared_root = Path(scratch_dir) / "shared_store"
    shared_audit = Path(scratch_dir) / "audit"
    store_root = Path(shared_root)
    audit_dir = Path(shared_audit)

    engine = make_engine(args, store_root)
    schedule_resolver = None
    if args.auto_align_chunks > 0 and not args.expect_reject:

        def schedule_resolver(planned_alignment: int, max_model_len: int):
            return resolve_schedule_plan(
                requested_tokens=args.tokens,
                requested_batch=args.max_num_batched_tokens,
                hit_alignment=planned_alignment,
                prefix_percent=args.prefix_percent,
                auto_align_chunks=args.auto_align_chunks,
                max_model_len=max_model_len,
            )

    (
        worker_config,
        sched_config,
        block_size,
        hash_size,
        views,
        preplanned_schedule,
        head_partitions,
    ) = allocate(
        engine,
        args.kv_mib,
        scratch_dir,
        cleanup,
        kv_device=kv_device,
        connector_version=args.connector,
        auto_dcp_interleave=(args.cp_kv_cache_interleave_size == "auto"),
        schedule_resolver=schedule_resolver,
        chunk_size=args.chunk_size,
    )

    # Share rank 0's scratch paths across all torchrun workers.
    store_root_str, audit_dir_str = broadcast((str(store_root), str(audit_dir)))
    store_root = Path(store_root_str)
    audit_dir = Path(audit_dir_str)
    if args.connector == "v1" and not args.ucm_config:
        _set_v1_store_root(engine.kv_transfer_config, store_root)

    coords = inspect_local_coords(args)
    is_root = coords.world_rank == 0
    buffers = register_kv_buffers(views)
    backing = SharedFileByteStore(
        store_root,
        audit_dir,
        coords.world_rank,
        dump_delay_polls=args.dump_delay_polls,
        load_delay_polls=args.load_delay_polls,
    )
    drain_steps = max(32, args.dump_delay_polls + args.load_delay_polls + 16)

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
            _find_ucm_connector(worker).connector_worker._device = None
        worker.register_kv_caches(views)
        _register_connector_staging_buffers(buffers, worker)

        handshakes = collect_handshakes(worker, coords, args, engine)
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
            if native_v1 and not args.expect_reject and args.multi_connector != "none":
                from mock_pd_connector import MockPDHandshakeConnector

                pd_child = next(
                    child
                    for child in scheduler.connector.sub_connectors
                    if isinstance(child, MockPDHandshakeConnector)
                )
                metadata = pd_child.received_handshake_metadata or {}
                if metadata.keys() != handshakes.keys() or {
                    item.world_rank for item in metadata.values()
                } != set(range(dist.get_world_size())):
                    raise AssertionError(
                        "Scheduler P/D sibling missed worker handshakes"
                    )
            aggregator = KVOutputAggregator.from_connector(
                scheduler.connector, dist.get_world_size()
            )
        barrier()

        if args.expect_reject:
            raise _ExpectedRejectNotTriggeredError(
                f"Expected startup rejection matching {args.expect_reject!r}, "
                "but connector initialization succeeded"
            )

        dcp_dump_ranks = None
        dcp_rank_by_worker = None
        if native_v1 and args.dcp > 1 and args.multi_connector == "none":
            from parallel import gather
            from ucm.integration.vllm.ucm_connector import UCMCPConnector

            # Only the MLA CP connector uses these unmodified scheduler keys
            # for Store writes. FAWA and HLA have their own plan formats.
            check_dcp_ranks = False
            if is_root:
                ucm_scheduler = _find_ucm_connector(scheduler.connector).connector
                check_dcp_ranks = (
                    isinstance(ucm_scheduler, UCMCPConnector) and ucm_scheduler.is_mla
                )
            if broadcast(check_dcp_ranks):
                dcp_dump_ranks = {} if is_root else None
                dcp_rank_by_worker = gather(coords.dcp_rank)

        from math import lcm

        if args.connector == "v2":
            connector_block = int(
                _find_ucm_connector(worker).connector_worker.layout.spec.chunk_size
            )
        else:
            connector_block = broadcast(
                int(
                    _find_ucm_connector(scheduler.connector).get_block_size()
                    or block_size
                )
                if is_root
                else 1
            )
        hit_alignment = lcm(
            int(block_size),
            int(getattr(sched_config, "cache_hit_alignment_tokens", 1) or 1),
            int(connector_block),
        )
        if preplanned_schedule is not None:
            (
                effective_tokens,
                effective_batch,
                prefix_len,
                expected_hit,
                planned_alignment,
            ) = preplanned_schedule
            if hit_alignment != planned_alignment:
                raise AssertionError(
                    f"Connector block size {connector_block} increased runtime "
                    f"hit_alignment from pre-planned {planned_alignment} to "
                    f"{hit_alignment}; pass --chunk-size {connector_block} or "
                    "explicit --tokens / --max-num-batched-tokens"
                )
        else:
            effective_tokens, effective_batch, prefix_len, expected_hit = (
                resolve_schedule_plan(
                    requested_tokens=args.tokens,
                    requested_batch=args.max_num_batched_tokens,
                    hit_alignment=hit_alignment,
                    prefix_percent=args.prefix_percent,
                    auto_align_chunks=0,
                    max_model_len=int(engine.model_config.max_model_len),
                )
            )
        if is_root:
            if int(scheduler.max_num_scheduled_tokens) != effective_batch:
                raise AssertionError(
                    f"Scheduler budget mismatch: max_num_scheduled_tokens="
                    f"{scheduler.max_num_scheduled_tokens}, expected={effective_batch}"
                )
            print(
                f"[rank 0] Resolved schedule: tokens={effective_tokens}, "
                f"batch={effective_batch}, prefix={prefix_len}, "
                f"expected_hit={expected_hit}, hit_alignment={hit_alignment}, "
                f"connector_block={connector_block}, "
                f"num_blocks={worker_config.num_blocks}, "
                f"max_model_len={engine.model_config.max_model_len}",
                flush=True,
            )

        # ---- Phase 1: Cold prefill + Dump ----------------------------------
        prefill = None
        cold_reference = {}
        if native_v1:
            poison_native_kv(views)
            prefill = NativePrefill(
                views, worker_config, coords, engine, head_partitions
            )
        else:
            fill_native_kv(views, worker_config, coords)

        def capture_cold_boundary(tables, state_blocks=None):
            cold_reference["tables"] = tables
            cold_reference["state_blocks"] = state_blocks
            cold_reference["snapshots"] = snapshot_cold_blocks(
                worker_config,
                views,
                tables,
                device_type=engine.device_config.device_type,
                state_blocks=state_blocks,
            )

        vocab_size = max(1024, int(engine.model_config.get_vocab_size()))
        prompt_a = [(idx * 17 + 3) % vocab_size for idx in range(effective_tokens)]

        req_a = (
            _make_request(f"cold-{uuid4().hex[:8]}", prompt_a, hash_size, engine)
            if is_root
            else None
        )
        tables_a, local_a, ext_a, _, steps_a, step_ends_a = _run_request_to_prefill_end(
            engine,
            worker_config,
            scheduler,
            aggregator,
            worker,
            views,
            req_a,
            is_root=is_root,
            target_tokens=effective_tokens,
            native_v1=native_v1,
            prefill=prefill,
            dcp_dump_ranks=dcp_dump_ranks,
            capture_at=expected_hit if native_v1 else None,
            capture_boundary=capture_cold_boundary,
        )
        if local_a != 0 or ext_a != 0:
            raise AssertionError(
                f"Cold request unexpectedly hit cache: local={local_a}, external={ext_a}"
            )
        if args.auto_align_chunks > 0:
            expected_step_ends = [
                effective_batch * idx for idx in range(1, args.auto_align_chunks + 1)
            ]
            if step_ends_a != expected_step_ends:
                raise AssertionError(
                    f"Cold prefill step boundaries drifted from auto-aligned chunks: "
                    f"expected={expected_step_ends}, actual={step_ends_a}"
                )
        elif effective_tokens > effective_batch and steps_a < 2:
            raise AssertionError(
                f"Expected multi-step chunked prefill for tokens={effective_tokens}, "
                f"budget={effective_batch}, got steps_cold={steps_a}"
            )
        if not native_v1:
            capture_cold_boundary(tables_a)
        if not cold_reference:
            raise AssertionError(
                f"Cold prefill never reached checkpoint {expected_hit}"
            )
        protected_blocks = set()
        if args.dump_delay_polls:
            from parallel import gather

            local_sources = prefill.pending_source_blocks(
                backing.pending_dump_sources.values(), tables_a
            )
            protected_blocks = set().union(*gather(local_sources))
            if not protected_blocks:
                raise AssertionError(
                    "No cold Dump was still pending before the overlap step"
                )
        if is_root:
            scheduler.finish_requests(req_a.request_id, RequestStatus.FINISHED_ABORTED)
        if args.dump_delay_polls:
            probe = None
            if is_root:
                block_pool = scheduler.kv_cache_manager.block_pool
                released = {
                    block_id
                    for block_id in protected_blocks
                    if block_pool.blocks[block_id].ref_cnt <= 0
                }
                if released:
                    raise AssertionError(
                        f"Freed pending Dump source blocks: {sorted(released)}"
                    )
                probe_prompt = list(prompt_a)
                probe_prompt[0] = (probe_prompt[0] + 2) % vocab_size
                probe = _make_request(
                    f"overlap-{uuid4().hex[:8]}",
                    probe_prompt,
                    hash_size,
                    engine,
                )
                scheduler.add_request(probe)
            output, _ = _step_all_ranks(
                engine,
                worker_config,
                scheduler,
                aggregator,
                worker,
                views,
                is_root=is_root,
                native_v1=True,
                prefill=prefill,
                dcp_dump_ranks=dcp_dump_ranks,
                protected_blocks=protected_blocks,
                protected_label="pending Dump source",
            )
            if is_root:
                if not output.num_scheduled_tokens.get(probe.request_id):
                    raise AssertionError(
                        "The overlap request received no prefill tokens"
                    )
                scheduler.finish_requests(
                    probe.request_id, RequestStatus.FINISHED_ABORTED
                )
        _drain_scheduler(
            engine,
            worker_config,
            scheduler,
            aggregator,
            worker,
            views,
            is_root=is_root,
            native_v1=native_v1,
            max_steps=drain_steps,
        )
        if backing.io_errors:
            raise backing.io_errors[0]
        if backing.pending_dumps:
            raise AssertionError(
                f"Rank {coords.world_rank} still has pending cold Dumps"
            )
        barrier()

        # Verify write audit on rank 0 before checking reads.
        audit_summary = None
        if is_root:
            audit_summary = audit_writes(
                audit_dir,
                allow_duplicate_writes=args.allow_duplicate_writes,
                dcp_dump_ranks=dcp_dump_ranks,
                dcp_rank_by_worker=dcp_rank_by_worker,
                pp_size=args.pp,
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
        _, local_neg, ext_neg, _, _, _ = _run_request_to_prefill_end(
            engine,
            worker_config,
            scheduler,
            aggregator,
            worker,
            views,
            req_neg,
            is_root=is_root,
            target_tokens=effective_tokens,
            native_v1=native_v1,
            prefill=prefill,
        )
        if ext_neg != 0 or local_neg != 0:
            raise AssertionError(
                f"Mutated prompt falsely hit cache: local={local_neg}, external={ext_neg}"
            )
        if is_root:
            scheduler.finish_requests(req_neg.request_id, RequestStatus.FINISHED_ABORTED)
        _drain_scheduler(
            engine,
            worker_config,
            scheduler,
            aggregator,
            worker,
            views,
            is_root=is_root,
            native_v1=native_v1,
            max_steps=drain_steps,
        )
        if backing.io_errors:
            raise backing.io_errors[0]
        barrier()

        # ---- Phase 3: Poison KV + Hot reload + Engine-native byte check ----
        prompt_hot = prompt_a[:prefix_len] + [
            (idx * 31 + 11) % vocab_size
            for idx in range(effective_tokens - prefix_len)
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

        compress_ratios = getattr(
            engine.model_config.hf_text_config, "compress_ratios", None
        )

        def check_after_load(load_tables, local_hit_now, ext_hit_now, step_end_now):
            if local_hit_now != 0:
                raise AssertionError(
                    f"Hot reload unexpectedly hit local HBM ({local_hit_now} tokens) "
                    "instead of external store"
                )
            slices_count, bytes_count = verify_cold_vs_hot(
                worker_config,
                cold_reference["snapshots"],
                cold_reference["tables"],
                views,
                load_tables,
                ext_hit_now,
                hot_load_step_end=step_end_now,
                dcp_size=args.dcp,
                device_type=engine.device_config.device_type,
                world_rank=coords.world_rank,
                poison_byte=0xA5,
                compress_ratios=compress_ratios,
                require_state_targets=native_v1,
                cold_state_blocks=cold_reference["state_blocks"],
            )
            verify_result["slices"] = slices_count
            verify_result["bytes"] = bytes_count
            if args.verify_load_guards:
                verify_result["guards"] = verify_unallocated_kv_is_poisoned(
                    worker_config,
                    views,
                    load_tables,
                    device_type=engine.device_config.device_type,
                )
            verify_result["checked"] = True

        req_hot = (
            _make_request(f"hot-{uuid4().hex[:8]}", prompt_hot, hash_size, engine)
            if is_root
            else None
        )
        if is_root and args.partial_miss:
            backing.prefix_lookups = []
        already_added = False
        initial_local_hot = 0
        initial_ext_hot = 0
        initial_steps_hot = 0
        if args.load_delay_polls:
            from parallel import gather

            if is_root:
                scheduler.add_request(req_hot)
            _step_all_ranks(
                engine,
                worker_config,
                scheduler,
                aggregator,
                worker,
                views,
                is_root=is_root,
                native_v1=native_v1,
            )
            hot_wait_tables = None
            if is_root:
                if getattr(req_hot, "prefill_stats", None) is not None:
                    initial_local_hot = int(
                        req_hot.prefill_stats.num_local_cached_tokens
                    )
                    initial_ext_hot = int(
                        req_hot.prefill_stats.num_external_cached_tokens
                    )
                if req_hot.status != RequestStatus.WAITING_FOR_REMOTE_KVS:
                    raise AssertionError(
                        f"--load-delay-polls expected WAITING_FOR_REMOTE_KVS after "
                        f"step 1, got status={req_hot.status} "
                        f"(external_hit={initial_ext_hot})"
                    )
                hot_wait_tables = scheduler.kv_cache_manager.get_block_ids(
                    req_hot.request_id
                )
            hot_wait_tables = broadcast(hot_wait_tables)
            local_load_targets = prefill.pending_source_blocks(
                backing.pending_load_targets.values(),
                hot_wait_tables,
                label="async Load targets",
            )
            protected_load_blocks = set().union(*gather(local_load_targets))
            if not protected_load_blocks:
                raise AssertionError(
                    "No hot async Load was still pending before the overlap step"
                )
            overlap_load = None
            if is_root:
                block_pool = scheduler.kv_cache_manager.block_pool
                released = {
                    block_id
                    for block_id in protected_load_blocks
                    if block_pool.blocks[block_id].ref_cnt <= 0
                }
                if released:
                    raise AssertionError(
                        f"Freed pending async Load destination blocks: {sorted(released)}"
                    )
                overlap_prompt = list(prompt_a)
                overlap_prompt[0] = (overlap_prompt[0] + 3) % vocab_size
                overlap_load = _make_request(
                    f"load-overlap-{uuid4().hex[:8]}",
                    overlap_prompt,
                    hash_size,
                    engine,
                )
                scheduler.add_request(overlap_load)
            overlap_output, _ = _step_all_ranks(
                engine,
                worker_config,
                scheduler,
                aggregator,
                worker,
                views,
                is_root=is_root,
                native_v1=True,
                protected_blocks=protected_load_blocks,
                protected_label="pending async Load destination",
            )
            if is_root:
                if not overlap_output.num_scheduled_tokens.get(overlap_load.request_id):
                    raise AssertionError(
                        "The async-load overlap request received no prefill tokens"
                    )
                if req_hot.status != RequestStatus.WAITING_FOR_REMOTE_KVS:
                    raise AssertionError(
                        "Hot request left WAITING_FOR_REMOTE_KVS during the overlap step"
                    )
                scheduler.finish_requests(
                    overlap_load.request_id, RequestStatus.FINISHED_ABORTED
                )
            already_added = True
            initial_steps_hot = 2
        _, local_hot, ext_hot, _, steps_hot, _ = _run_request_to_prefill_end(
            engine,
            worker_config,
            scheduler,
            aggregator,
            worker,
            views,
            req_hot,
            is_root=is_root,
            target_tokens=effective_tokens,
            native_v1=native_v1,
            on_load_step_complete=check_after_load,
            already_added=already_added,
            initial_local_hit=initial_local_hot,
            initial_external_hit=initial_ext_hot,
            initial_steps=initial_steps_hot,
            extra_step_budget=args.load_delay_polls + 4,
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
            engine,
            worker_config,
            scheduler,
            aggregator,
            worker,
            views,
            is_root=is_root,
            native_v1=native_v1,
            max_steps=drain_steps,
        )
        if backing.io_errors:
            raise backing.io_errors[0]
        if backing.pending_loads:
            raise AssertionError(
                f"Rank {coords.world_rank} still has pending hot async Loads"
            )
        barrier()
        if args.partial_miss:
            if is_root:
                if expected_hit <= hit_alignment:
                    raise ValueError(
                        "--partial-miss requires at least two warm aligned chunks"
                    )
                _check_partial_miss(
                    args,
                    backing,
                    scheduler,
                    engine,
                    hash_size,
                    prompt_hot,
                    hit_alignment,
                    connector_block,
                )
                print(
                    f"[rank 0] Partial {args.partial_miss} hit: {hit_alignment}",
                    flush=True,
                )
            barrier()

        return {
            "tokens": effective_tokens,
            "batch_tokens": effective_batch,
            "hit_alignment": hit_alignment,
            "steps_cold": steps_a,
            "steps_hot": steps_hot,
            "external_hit_tokens": ext_hot,
            "verified_bytes": verify_result["bytes"],
            "verified_slices": verify_result["slices"],
            "guard_pages": verify_result.get("guards", 0),
            "audit": audit_summary,
        }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(f"Model-check arguments: {vars(args)}", flush=True)

    # Make local sibling modules importable when invoked as a script.
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    if args.enable_ucm_patch is True:
        os.environ["ENABLE_UCM_PATCH"] = "1"
    elif args.enable_ucm_patch is False:
        os.environ["ENABLE_UCM_PATCH"] = "0"

    from parallel import launch

    if launch(args):
        return

    ucm_patch_enabled = os.environ.get("ENABLE_UCM_PATCH", "").lower() in (
        "1",
        "true",
    )
    if ucm_patch_enabled:
        from ucm.integration.vllm.patch import boot_patch

        boot_patch.install_hook()

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
    except _ExpectedRejectNotTriggeredError:
        raise
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
        f"layerwise={args.use_layerwise}, "
        f"ucm_patch={ucm_patch_enabled}, "
        f"hit={summary['external_hit_tokens']}/{summary['tokens']} "
        f"(batch={summary['batch_tokens']}, align={summary['hit_alignment']}), "
        f"verified_slices={summary['verified_slices']}, "
        f"guard_pages={summary['guard_pages']}, "
        f"verified_bytes={summary['verified_bytes']}, audit={summary['audit']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
