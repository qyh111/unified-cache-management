# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
"""Per-group dispatch planning: one dump/load rule over group chains.

Extracted from ``hla_connector.UCMHybridLinearAttentionConnector``
(``_generate_hla_dispatch_meta`` / ``_append_mamba_align_state_block``)
so the unified ``ucm_connector`` and HLA share the planner instead of
duplicating the pass structure. The token geometry, unchanged:

* **load** (caller guards ``external > 0``): full-attention groups slice
  ``[hbm_hit_tokens, total_hit_tokens)``; each mamba state group anchors
  ONE state at ``total_hit_tokens`` (the newest hit LCM boundary), with
  the vLLM block reverse-searched to the last non-null entry;
* **dump**: full-attention groups slice
  ``[token_processed, dump_tok_end)``; state groups anchor at the last
  LCM boundary only when ``dump_tok_end`` lands exactly on it and that
  boundary is beyond the first one;
* full-attention slices skip null vLLM ids (mamba-align placeholders);
* a state anchor's ucm key comes from ``state_hash_fn`` (the group
  manager's ``compute_mamba_align_state_hash``); anchors whose vLLM id
  or hash cannot be resolved are silently skipped, as before.

Outputs are flat per-group :class:`GroupSlice` entries in FA-first,
then State order -- the ordering the MLA rank-scoping depends on
(``load_full_attn_count`` counts full-attention entries only).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional

from ucm.integration.vllm.group_hash import extend_non_null

from ucm.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class GroupSlice:
    """One group's transfer entries for one dispatch step.

    ``is_state`` marks mamba-align anchors (one entry per group at most);
    full-attention slices carry the boundary-aligned block range.
    """

    group_id: int
    is_state: bool
    ucm_block_ids: list[bytes]
    vllm_block_ids: list[int]

    @property
    def size(self) -> int:
        return len(self.ucm_block_ids)


def fa_block_slice(
    group: Any,
    chain_ucm: Sequence[bytes],
    chain_vllm: Sequence[int],
    tok_start: int,
    tok_end: int,
) -> GroupSlice:
    """Slice one full-attention group's chains over a token range.

    Block indices are ``[tok_start // block_size, tok_end // block_size)``
    per the group's own block size; null vLLM ids (mamba-align
    placeholders) are skipped.
    """
    start_blk = tok_start // group.block_size
    end_blk = tok_end // group.block_size
    ucm: list[bytes] = []
    vllm: list[int] = []
    if start_blk < end_blk:
        extend_non_null(
            ucm,
            vllm,
            chain_ucm[start_blk:end_blk],
            chain_vllm[start_blk:end_blk],
        )
    return GroupSlice(group.group_id, False, ucm, vllm)


def state_anchor_slice(
    group: Any,
    group_ucm_ids: Mapping[int, Sequence[bytes]],
    group_vllm_ids: Mapping[int, Sequence[int]],
    state_hash_fn: Callable[[Any, int, Mapping[int, Sequence[bytes]]], Optional[bytes]],
    seq_len: int,
    reason: str,
) -> Optional[GroupSlice]:
    """One mamba-align state anchor at ``seq_len`` for one state group.

    Mirrors ``_append_mamba_align_state_block``: dump indexes the block
    containing the boundary's last token directly; load reverse-searches
    the group's chain for the last non-null vLLM id. Anchors without a
    resolvable vLLM id or hash are skipped (return ``None``).
    """
    state_idx = max((seq_len - 1) // group.block_size, 0)
    vllm_state_idx = state_idx
    chain_vllm = group_vllm_ids[group.group_id]
    if reason == "load":
        for i in range(len(chain_vllm) - 1, -1, -1):
            if chain_vllm[i] != 0:
                vllm_state_idx = i
                break
    try:
        vllm_block_id = chain_vllm[vllm_state_idx]
    except IndexError:
        logger.error(
            "mamba-align state vLLM block missing: "
            f"group_id={group.group_id}, reason={reason}, "
            f"seq_len={seq_len}, state_idx={state_idx}, "
            f"vllm_state_idx={vllm_state_idx}, "
            f"num_vllm_blocks={len(chain_vllm)}"
        )
        return None
    if vllm_block_id == 0:
        return None
    ucm_block_id = state_hash_fn(group, seq_len, group_ucm_ids)
    if ucm_block_id is None:
        logger.error(
            "mamba-align state hash missing: "
            f"group_id={group.group_id}, reason={reason}, seq_len={seq_len}"
        )
        return None
    return GroupSlice(group.group_id, True, [ucm_block_id], [vllm_block_id])


def plan_load(
    groups: Sequence[Any],
    lcm_block_size: int,
    state_hash_fn: Callable[..., Optional[bytes]],
    group_ucm_ids: Mapping[int, Sequence[bytes]],
    group_vllm_ids: Mapping[int, Sequence[int]],
    *,
    hbm_hit_block_num: int,
    total_hit_block_num: int,
) -> list[GroupSlice]:
    """Load-side slices: FA ranges then one state anchor per state group."""
    hbm_hit_tokens = hbm_hit_block_num * lcm_block_size
    total_hit_tokens = total_hit_block_num * lcm_block_size
    slices: list[GroupSlice] = []
    for group in groups:
        if group.is_mamba_align:
            continue
        slices.append(
            fa_block_slice(
                group,
                group_ucm_ids[group.group_id],
                group_vllm_ids[group.group_id],
                hbm_hit_tokens,
                total_hit_tokens,
            )
        )
    for group in groups:
        if not group.is_mamba_align:
            continue
        anchor = state_anchor_slice(
            group,
            group_ucm_ids,
            group_vllm_ids,
            state_hash_fn,
            total_hit_tokens,
            "load",
        )
        if anchor is not None:
            slices.append(anchor)
    return slices


def plan_dump(
    groups: Sequence[Any],
    lcm_block_size: int,
    state_hash_fn: Callable[..., Optional[bytes]],
    group_ucm_ids: Mapping[int, Sequence[bytes]],
    group_vllm_ids: Mapping[int, Sequence[int]],
    *,
    token_processed: int,
    dump_tok_end: int,
) -> list[GroupSlice]:
    """Dump-side slices; state anchors only on exact LCM boundaries."""
    first_lcm_b = (token_processed // lcm_block_size + 1) * lcm_block_size
    last_lcm_b = (dump_tok_end // lcm_block_size) * lcm_block_size
    slices: list[GroupSlice] = []
    for group in groups:
        if group.is_mamba_align:
            continue
        slices.append(
            fa_block_slice(
                group,
                group_ucm_ids[group.group_id],
                group_vllm_ids[group.group_id],
                token_processed,
                dump_tok_end,
            )
        )
    for group in groups:
        if not group.is_mamba_align:
            continue
        if dump_tok_end != last_lcm_b or last_lcm_b < first_lcm_b:
            continue
        anchor = state_anchor_slice(
            group,
            group_ucm_ids,
            group_vllm_ids,
            state_hash_fn,
            last_lcm_b,
            "dump",
        )
        if anchor is not None:
            slices.append(anchor)
    return slices
