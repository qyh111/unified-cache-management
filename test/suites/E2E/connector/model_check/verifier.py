# SPDX-License-Identifier: MIT
"""Engine-native KV truth oracle: deterministic fill, poison, and slice checks.

All verification targets are derived strictly from vLLM's native KVCacheConfig,
KVCacheManager block tables, and allocated PyTorch KV views -- never from UCM
layout or topology helpers.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from mock_store import BufferRegistry


def _unwrap_layer_spec(spec, layer_name: str | None = None):
    """Unwrap UniformTypeKVCacheSpecs to the concrete per-layer KVCacheSpec."""
    nested = getattr(spec, "kv_cache_specs", None)
    if isinstance(nested, dict) and nested:
        if layer_name is not None and layer_name in nested:
            return nested[layer_name]
        return next(iter(nested.values()))
    return spec


def _extract_layer_index(layer_name: str) -> int | None:
    match = re.search(r"\.layers\.(\d+)\.", layer_name)
    return int(match.group(1)) if match else None


def _effective_sliding_window(
    layer_names: list[str],
    sliding_window: int,
    compress_ratios: list[int] | tuple[int, ...] | None,
) -> int:
    """Return the persisted sliding-window tail in tokens for a window group.

    For DeepSeek-V4 compressor state caches, a restore boundary closes each
    compression group so only (sliding_window - compress_ratio) overlap tokens
    remain live (0 when compress_ratio == sliding_window).
    """
    if (
        not compress_ratios
        or not layer_names
        or not all(name.endswith(".compressor.state_cache") for name in layer_names)
    ):
        return sliding_window
    tails = set()
    for name in layer_names:
        layer_idx = _extract_layer_index(name)
        if layer_idx is None or not 0 <= layer_idx < len(compress_ratios):
            return sliding_window
        tails.add(max(0, sliding_window - max(1, int(compress_ratios[layer_idx]))))
    if len(tails) == 1:
        return tails.pop()
    return sliding_window


def _safe_tokens_per_state(spec) -> int:
    """Read concrete tokens_per_state/compress_ratio without hitting abstract stubs."""
    concrete = _unwrap_layer_spec(spec)
    for attr in ("tokens_per_state", "compress_ratio"):
        try:
            value = getattr(concrete, attr, None)
        except NotImplementedError:
            continue
        if value is not None and int(value) > 0:
            return int(value)
    return 1


def view_components(view) -> tuple:
    return tuple(view) if isinstance(view, (tuple, list)) else (view,)


def register_kv_buffers(views: dict[str, Any]) -> BufferRegistry:
    """Register every unique storage backing allocated by the native MRV2 runner."""
    registry = BufferRegistry()
    seen = set()
    for view in views.values():
        for tensor in view_components(view):
            storage = tensor.untyped_storage()
            ptr = int(storage.data_ptr())
            if ptr not in seen:
                seen.add(ptr)
                registry.register(ptr, int(storage.nbytes()))
    return registry


def _native_layer_content_key(
    worker_config,
    layer_name: str,
    coords,
    *,
    num_blocks: int,
    leading_dim: int,
) -> tuple[int, int, int, str]:
    """Return a seed key shared ONLY by workers whose native KV bytes must match.

    Two workers share the same key if and only if they belong to the same PP
    stage, hold the same KV head partition, and hold the same sequence shard
    for this layer. Replicas therefore start with identical bytes (so whichever
    replica UCM picks as dump writer, every replica loads back exact bytes),
    while distinct shards start with distinct bytes (catching cross-rank mixups).
    """
    from vllm.v1.kv_cache_interface import (
        MLAAttentionSpec,
        MambaSpec,
        SlidingWindowMLASpec,
        UniformTypeKVCacheSpecs,
    )

    spec = None
    for group in worker_config.kv_cache_groups:
        if layer_name in group.layer_names:
            spec = _unwrap_layer_spec(group.kv_cache_spec, layer_name)
            break

    is_state = isinstance(spec, MambaSpec)
    is_mla = (
        isinstance(spec, (MLAAttentionSpec, SlidingWindowMLASpec))
        or type(spec).__name__.endswith("MLASpec")
        or bool(getattr(spec, "tp_replicated", False))
    )

    if is_mla:
        tp_shard = 0
    else:
        max_tp_shards = getattr(spec, "max_tp_shards", None) or coords.tp_size
        tp_shards = min(coords.tp_size, int(max_tp_shards))
        tp_shard = coords.tp_rank * tp_shards // coords.tp_size

    # DCP-replicated tensors allocate num_blocks * dcp_size rows per rank,
    # holding the same full logical block across all DCP peers.
    rows_per_block = leading_dim // max(1, num_blocks)
    if is_state:
        # At TP=1 with PCP>1, HLA scopes state per PCP rank.
        seq_shard = coords.pcp_rank if coords.tp_size == 1 else 0
    elif rows_per_block > 1 and rows_per_block == coords.dcp_size:
        seq_shard = 0
    else:
        group_dcp = _group_dcp_size(spec, coords.dcp_size) if spec is not None else 1
        seq_shard = coords.dcp_rank if group_dcp > 1 else 0

    return (coords.pp_rank, tp_shard, seq_shard, layer_name)


def fill_native_kv(views: dict[str, Any], worker_config, coords) -> None:
    """Fill every native KV layer with deterministic content-scoped bytes.

    Fill non-prefix-cacheable (transient) layers before prefix-cacheable layers
    so shared physical KV pools (e.g. DeepSeek-V4.1 long_kv_cache aliased with
    transient compressor.state_cache) always end up with the prefix-cacheable
    layer's replica-consistent bytes.
    """
    import hashlib
    import torch

    num_blocks = int(worker_config.num_blocks)
    cacheable_ids = getattr(worker_config, "prefix_cacheable_group_ids", None)
    if cacheable_ids is None:
        cacheable_ids = tuple(
            gid
            for gid, group in enumerate(worker_config.kv_cache_groups)
            if group.kv_cache_spec.prefix_cacheable
        )
    cacheable_layers = {
        name
        for gid in cacheable_ids
        for name in worker_config.kv_cache_groups[gid].layer_names
    }
    ordered_passes = (
        sorted(name for name in views if name not in cacheable_layers),
        sorted(name for name in views if name in cacheable_layers),
    )

    for layer_pass in ordered_passes:
        filled_ptrs: set[tuple[int, int, int, tuple[int, ...]]] = set()
        for layer_name in layer_pass:
            for part_idx, tensor in enumerate(view_components(views[layer_name])):
                if not tensor.numel():
                    continue
                key_tuple = _native_layer_content_key(
                    worker_config,
                    layer_name,
                    coords,
                    num_blocks=num_blocks,
                    leading_dim=int(tensor.size(0)),
                )
                storage = tensor.untyped_storage()
                elem_bytes = int(tensor.element_size())
                byte_offset = int(tensor.storage_offset()) * elem_bytes
                # Include element_size and stride so distinct aliased views
                # (e.g. FP32 vs BF16 with the same numel) never collide.
                span_key = (
                    int(storage.data_ptr()) + byte_offset,
                    int(tensor.numel()),
                    elem_bytes,
                    tuple(int(s) for s in tensor.stride()),
                )
                if span_key in filled_ptrs:
                    continue
                filled_ptrs.add(span_key)

                seed_material = f"{key_tuple}:{part_idx}".encode("utf-8")
                seed = int.from_bytes(
                    hashlib.sha256(seed_material).digest()[:8], "little"
                )
                rng = np.random.default_rng(seed)

                # Fill in slices along axis 0 so temporary memory stays tiny
                # even for multi-gigabyte KV pools.
                step = max(1, min(64, int(tensor.size(0))))
                row_elems = int(tensor[0].numel()) if tensor.ndim > 1 else 1
                for start in range(0, int(tensor.size(0)), step):
                    stop = min(int(tensor.size(0)), start + step)
                    chunk_view = tensor.narrow(0, start, stop - start)
                    chunk_bytes = (stop - start) * row_elems * elem_bytes
                    data = rng.integers(1, 255, size=chunk_bytes, dtype=np.uint8)
                    data[data == 0xA5] = 0x5A
                    src = (
                        torch.from_numpy(data)
                        .view(tensor.dtype)
                        .reshape(chunk_view.shape)
                    )
                    chunk_view.copy_(src.to(tensor.device))


def poison_native_kv(views: dict[str, Any], poison_byte: int = 0xA5) -> None:
    """Overwrite every allocated KV buffer with a poison byte before hot load."""
    import torch

    seen = set()
    for view in views.values():
        for tensor in view_components(view):
            storage = tensor.untyped_storage()
            ptr = int(storage.data_ptr())
            if ptr in seen:
                continue
            seen.add(ptr)
            raw = torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
                storage, 0, (storage.nbytes(),), (1,)
            )
            raw.fill_(poison_byte)


def _unwrap_parts(view, *, is_state: bool, device_type: str) -> tuple:
    parts = view_components(view)
    if (
        len(parts) == 1
        and not is_state
        and parts[0].ndim == 5
        and parts[0].shape[0] == 2
        and (device_type == "npu" or parts[0].shape[1] != 2)
    ):
        return tuple(parts[0].unbind(0))
    return parts


def _group_dcp_size(spec, global_dcp_size: int) -> int:
    if global_dcp_size <= 1:
        return 1
    try:
        from vllm.v1.core.kv_cache_utils import resolve_dcp_kv_block_size

        if hasattr(spec, "dcp_sharded"):
            return max(1, resolve_dcp_kv_block_size(spec, global_dcp_size) // spec.block_size)
    except ImportError:
        pass
    try:
        from vllm.v1.core.kv_cache_utils import dcp_world_size_for_kv_cache_spec

        return max(1, int(dcp_world_size_for_kv_cache_spec(spec, global_dcp_size)))
    except ImportError:
        return global_dcp_size


def _slice_block_tensor(
    tensor,
    num_blocks: int,
    block_id: int,
    *,
    token_start: int | None = None,
    token_end: int | None = None,
    block_tokens: int | None = None,
):
    """Index one logical block (or partial tail window) in a native tensor view."""
    if not tensor.numel():
        return None
    rows_per_block, remainder = divmod(int(tensor.size(0)), int(num_blocks))
    if remainder != 0 or rows_per_block < 1:
        raise AssertionError(
            f"Native tensor leading dim {tensor.size(0)} is not a multiple of "
            f"num_blocks={num_blocks}"
        )
    if not 0 < block_id < num_blocks:
        raise AssertionError(
            f"Invalid block_id={block_id} for num_blocks={num_blocks}"
        )
    page = tensor.narrow(0, block_id * rows_per_block, rows_per_block)
    if (
        token_start is None
        or token_end is None
        or block_tokens is None
        or (token_start == 0 and token_end == block_tokens)
    ):
        return page

    # Partial sliding-window tail inside a single block.
    # For NHD/HND pages, locate the token axis matching block_tokens (or per_row).
    per_row_tokens = block_tokens // rows_per_block
    token_axis = None
    for axis in range(1, page.ndim):
        if int(page.size(axis)) == per_row_tokens:
            token_axis = axis
            break
    if token_axis is None or rows_per_block != 1:
        # Fallback: verify full page if layout uses compressed/non-standard inner axis.
        return page
    return page.narrow(token_axis, token_start, token_end - token_start)


def snapshot_cold_blocks(
    worker_config,
    views: dict[str, Any],
    block_tables: tuple[list[int], ...],
    *,
    device_type: str,
) -> dict[tuple[str, int, int], Any]:
    """Clone ONLY the blocks allocated to the cold request (avoids full-pool copy)."""
    from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs

    num_blocks = int(worker_config.num_blocks)
    snapshots: dict[tuple[str, int, int], Any] = {}
    for gid, group in enumerate(worker_config.kv_cache_groups):
        if not group.layer_names:
            continue
        spec = group.kv_cache_spec
        members = (
            list(spec.kv_cache_specs.values())
            if isinstance(spec, UniformTypeKVCacheSpecs)
            else [spec]
        )
        is_state = any(isinstance(member, MambaSpec) for member in members)
        unique_blocks = {blk for blk in block_tables[gid] if blk > 0}
        for layer_name in group.layer_names:
            if layer_name not in views:
                continue
            parts = _unwrap_parts(
                views[layer_name], is_state=is_state, device_type=device_type
            )
            for part_idx, tensor in enumerate(parts):
                if not tensor.numel():
                    continue
                rows_per_block, remainder = divmod(int(tensor.size(0)), num_blocks)
                if remainder != 0 or rows_per_block < 1:
                    raise AssertionError(
                        f"Native tensor leading dim {tensor.size(0)} is not a "
                        f"multiple of num_blocks={num_blocks}"
                    )
                for blk in unique_blocks:
                    key = (layer_name, part_idx, blk)
                    if key in snapshots:
                        continue
                    page = tensor.narrow(0, blk * rows_per_block, rows_per_block)
                    snapshots[key] = page.detach().cpu().clone()
    return snapshots


def _slice_single_page(
    page,
    *,
    token_start: int | None,
    token_end: int | None,
    block_tokens: int | None,
):
    if (
        token_start is None
        or token_end is None
        or block_tokens is None
        or (token_start == 0 and token_end == block_tokens)
    ):
        return page
    rows_per_block = int(page.size(0))
    per_row_tokens = block_tokens // rows_per_block
    token_axis = None
    for axis in range(1, page.ndim):
        if int(page.size(axis)) == per_row_tokens:
            token_axis = axis
            break
    if token_axis is None or rows_per_block != 1:
        return page
    return page.narrow(token_axis, token_start, token_end - token_start)


def _tensor_raw_bytes(tensor) -> bytes:
    """Extract bit-exact raw bytes for any dtype (including BF16 and FP8)."""
    import torch

    flat = tensor.detach().cpu().contiguous().reshape(-1)
    if flat.numel() == 0:
        return b""
    return flat.view(torch.uint8).numpy().tobytes()


def verify_cold_vs_hot(
    worker_config,
    cold_block_snapshots: dict[tuple[str, int, int], Any],
    cold_tables: tuple[list[int], ...],
    hot_views: dict[str, Any],
    hot_tables: tuple[list[int], ...],
    hit_tokens: int,
    *,
    hot_load_step_end: int,
    dcp_size: int,
    device_type: str,
    world_rank: int,
    poison_byte: int = 0xA5,
    compress_ratios: list[int] | tuple[int, ...] | None = None,
) -> tuple[int, int]:
    """Compare every engine-native restored hot slice against its cold snapshot."""
    from vllm.v1.kv_cache_interface import (
        MambaSpec,
        SlidingWindowSpec,
        UniformTypeKVCacheSpecs,
    )

    num_blocks = int(worker_config.num_blocks)
    cacheable_ids = getattr(worker_config, "prefix_cacheable_group_ids", None)
    if cacheable_ids is None:
        cacheable_ids = tuple(
            gid
            for gid, group in enumerate(worker_config.kv_cache_groups)
            if group.kv_cache_spec.prefix_cacheable
        )

    verified_slices = 0
    verified_bytes = 0
    for gid in cacheable_ids:
        group = worker_config.kv_cache_groups[gid]
        if not group.layer_names:
            continue
        spec = group.kv_cache_spec
        member_specs = (
            list(spec.kv_cache_specs.values())
            if isinstance(spec, UniformTypeKVCacheSpecs)
            else [spec]
        )
        is_state = any(isinstance(member, MambaSpec) for member in member_specs)
        sliding_window = None
        for member in member_specs:
            win = getattr(member, "sliding_window", None)
            if win is not None or isinstance(member, SlidingWindowSpec):
                sliding_window = int(win) if win is not None else None
                break

        cold_table = cold_tables[gid]
        hot_table = hot_tables[gid]
        group_dcp = 1 if is_state else _group_dcp_size(spec, dcp_size)
        logical_span = int(spec.block_size) * group_dcp

        # List of (cold_block_id, hot_block_id, tok_lo, tok_hi, slot_desc)
        pairs: list[tuple[int, int, int | None, int | None, str]] = []
        if is_state:
            # Both checkpoint and running slots on hot load restore the exact
            # state snapshot saved at the hit_tokens boundary of the cold run.
            checkpoint_idx = max(0, (hit_tokens - 1) // logical_span)
            running_idx = max(0, (hot_load_step_end - 1) // logical_span)
            cold_ckpt_blk = cold_table[checkpoint_idx]
            if cold_ckpt_blk > 0:
                seen_hot = set()
                for role_name, idx in (
                    ("state_ckpt", checkpoint_idx),
                    ("state_run", running_idx),
                ):
                    if idx < len(hot_table) and hot_table[idx] > 0:
                        hot_blk = hot_table[idx]
                        if hot_blk not in seen_hot:
                            seen_hot.add(hot_blk)
                            pairs.append((cold_ckpt_blk, hot_blk, None, None, role_name))
        elif sliding_window is not None:
            effective_window = _effective_sliding_window(
                group.layer_names, sliding_window, compress_ratios
            )
            if effective_window <= 0:
                continue
            win_start = max(0, hit_tokens - effective_window)
            first_idx = win_start // logical_span
            last_idx = (hit_tokens - 1) // logical_span
            ratio = _safe_tokens_per_state(spec)
            for idx in range(first_idx, last_idx + 1):
                c_blk = cold_table[idx]
                h_blk = hot_table[idx]
                if c_blk <= 0 or h_blk <= 0:
                    continue
                if group_dcp == 1 and int(ratio) == 1:
                    blk_lo = max(0, win_start - idx * logical_span)
                    blk_hi = min(logical_span, hit_tokens - idx * logical_span)
                else:
                    blk_lo, blk_hi = None, None
                pairs.append((c_blk, h_blk, blk_lo, blk_hi, f"wa_{idx}"))
        else:
            num_hit_blocks = hit_tokens // logical_span
            for idx in range(num_hit_blocks):
                c_blk = cold_table[idx]
                h_blk = hot_table[idx]
                if c_blk > 0 and h_blk > 0:
                    pairs.append((c_blk, h_blk, None, None, f"fa_{idx}"))

        for layer_name in group.layer_names:
            if layer_name not in hot_views:
                continue
            parts = _unwrap_parts(
                hot_views[layer_name], is_state=is_state, device_type=device_type
            )
            for c_blk, h_blk, tok_lo, tok_hi, slot_desc in pairs:
                for part_idx, hot_tensor in enumerate(parts):
                    if not hot_tensor.numel():
                        continue
                    cold_page = cold_block_snapshots.get((layer_name, part_idx, c_blk))
                    if cold_page is None:
                        raise AssertionError(
                            f"[rank {world_rank}] Missing cold snapshot for "
                            f"{layer_name} part={part_idx} block={c_blk}"
                        )
                    expected_sub = _slice_single_page(
                        cold_page,
                        token_start=tok_lo,
                        token_end=tok_hi,
                        block_tokens=logical_span,
                    )
                    actual_sub = _slice_block_tensor(
                        hot_tensor,
                        num_blocks,
                        h_blk,
                        token_start=tok_lo,
                        token_end=tok_hi,
                        block_tokens=logical_span,
                    )
                    expected = _tensor_raw_bytes(expected_sub)
                    actual = _tensor_raw_bytes(actual_sub)
                    verified_slices += 1
                    verified_bytes += len(expected)
                    if actual != expected:
                        poison_count = actual.count(bytes([poison_byte]))
                        first_diff = next(
                            (
                                i
                                for i, (a, b) in enumerate(zip(actual, expected))
                                if a != b
                            ),
                            min(len(actual), len(expected)),
                        )
                        raise AssertionError(
                            f"[rank {world_rank}] Restored KV mismatch at "
                            f"g{gid}:{layer_name}:p{part_idx}:{slot_desc} "
                            f"(cold_blk={c_blk}, hot_blk={h_blk}): "
                            f"len={len(actual)}/{len(expected)}, "
                            f"first_diff_byte={first_diff}, "
                            f"untouched_poison_bytes={poison_count}/{len(actual)}"
                        )

    if verified_slices == 0:
        raise AssertionError(f"[rank {world_rank}] No native KV slices were verified")
    return verified_slices, verified_bytes


def audit_writes(
    audit_dir: Path,
    *,
    allow_duplicate_writes: bool = False,
) -> dict[str, int]:
    """Verify cross-rank write ownership and detect conflicting/duplicate writes."""
    records = []
    for audit_file in sorted(audit_dir.glob("rank_*.jsonl")):
        for line in audit_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise AssertionError("Cold write produced zero store writes across all ranks")

    seen: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        slot = (
            rec["namespace"],
            rec["key"],
            rec["shard"],
            rec["offset"],
            rec["size"],
        )
        seen[slot].append(rec)

    duplicates = []
    conflicts = []
    for slot, writers in seen.items():
        ranks = [w["rank"] for w in writers]
        hashes = {w["sha256"] for w in writers}
        if len(hashes) > 1:
            conflicts.append((slot, ranks))
        elif len(set(ranks)) > 1:
            duplicates.append((slot, sorted(set(ranks))))

    if conflicts:
        slot, ranks = conflicts[0]
        raise AssertionError(
            f"Conflicting cross-rank writes to the same store slot {slot} "
            f"from ranks {ranks} (total conflicts={len(conflicts)})"
        )
    if duplicates and not allow_duplicate_writes:
        slot, ranks = duplicates[0]
        raise AssertionError(
            f"Duplicate cross-rank writes to the same store slot {slot} "
            f"from ranks {ranks} (total duplicates={len(duplicates)})"
        )

    namespaces = {rec["namespace"] for rec in records}
    unique_keys = {(rec["namespace"], rec["key"]) for rec in records}
    return {
        "write_events": len(records),
        "unique_keys": len(unique_keys),
        "namespaces": len(namespaces),
        "duplicate_slots": len(duplicates),
    }
