# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
"""Group-aware hashing and two-stage lookup for hybrid KV connectors.

Moved verbatim from ``hla_connector.py`` (develop branch, 2026-09-26) so
the planned unified ``ucm_connector`` can share the per-group hash
machinery without importing the HLA connector. Adaptations: the lookup's
error metric is injected as an ``on_lookup_error`` callback instead of
calling ``ucm_connector._record_counter`` directly, keeping this module
free of connector imports.

Semantics (unchanged):

* every group gets an independent hash chain seed derived from the base
  seed (``UCM_GROUP_SEED`` domain tag);
* mamba-align groups pad their block table with null placeholders
  (``b""`` / vLLM block id 0) and carry no per-block hashes -- their
  state keys are derived at LCM boundaries from the primary
  full-attention group's prefix hash (``UCM_MAMBA_ALIGN_STATE``);
* resume boundaries must align to ``lcm_block_size`` of all groups.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

import numpy as np
from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs

from ucm.integration.vllm.request_hasher import RequestHasher
from ucm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
    from vllm.v1.request import Request

logger = init_logger(__name__)


def layer_name_to_kv_cache_spec(
    kv_cache_config: "KVCacheConfig",
) -> dict[str, list["KVCacheSpec"]]:
    """Map each model layer name to its concrete KVCacheSpec.

    Handles merged group specs and UniformTypeKVCacheSpecs (per-layer
    ``kv_cache_specs`` entries).
    """
    out: dict[str, list[KVCacheSpec]] = defaultdict(list)
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            by_name = spec.kv_cache_specs
            for name in group.layer_names:
                out[name].append(by_name[name])
        else:
            for name in group.layer_names:
                out[name].append(spec)
    return out


def block_size_from_kv_cache_spec(spec: "KVCacheSpec") -> int:
    """Token block size used for KV scheduling / hashing for one group spec."""
    block_size = 0
    if isinstance(spec, UniformTypeKVCacheSpecs):
        block_size = next(iter(spec.kv_cache_specs.values())).block_size
    else:
        block_size = spec.block_size

    return block_size


def is_mamba_align_kv_cache_spec(spec: "KVCacheSpec") -> bool:
    if isinstance(spec, UniformTypeKVCacheSpecs):
        sample = next(iter(spec.kv_cache_specs.values()))
        return is_mamba_align_kv_cache_spec(sample)
    return isinstance(spec, MambaSpec) and spec.mamba_cache_mode == "align"


def extend_non_null(
    dst_ucm_block_ids: list[bytes],
    dst_vllm_block_ids: list[int],
    src_ucm_block_ids: list[bytes],
    src_vllm_block_ids: list[int],
) -> None:
    # Skip vLLM null blocks (block_id=0) used as mamba-align placeholders.
    for ucm_block_id, vllm_block_id in zip(src_ucm_block_ids, src_vllm_block_ids):
        if vllm_block_id == 0:
            continue
        dst_ucm_block_ids.append(ucm_block_id)
        dst_vllm_block_ids.append(vllm_block_id)


def _normalize_tensor_size_list(tensor_size_list: Any) -> list[int]:
    if isinstance(tensor_size_list, np.ndarray):
        return [int(v) for v in tensor_size_list.reshape(-1).tolist()]
    if isinstance(tensor_size_list, (list, tuple)):
        return [int(v) for v in tensor_size_list]
    return [int(tensor_size_list)]


@dataclass
class GroupInfo:
    """Per-group metadata used by :class:`KVCacheGroupManager`."""

    group_id: int
    block_size: int
    layer_names: tuple[str, ...]
    # Independent hash chain seed per group (see ``KVCacheGroupManager``).
    seed: bytes
    is_mamba_align: bool = False
    block_hasher: Optional[Callable[["Request"], list[bytes]]] = None

    @property
    def is_full_attention(self) -> bool:
        return not self.is_mamba_align


class KVCacheGroupManager:
    """Group-aware hashing and two-stage lookup for hybrid (HLA) connectors."""

    def __init__(
        self,
        kv_cache_config: "KVCacheConfig",
        request_hasher: "RequestHasher",
        base_seed: bytes,
    ) -> None:
        self.request_hasher = request_hasher
        self.groups_by_id: list[GroupInfo] = []
        self.full_attn_groups: list[GroupInfo] = []
        self.state_groups: list[GroupInfo] = []

        for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
            spec = group.kv_cache_spec
            block_size = block_size_from_kv_cache_spec(spec)
            is_mamba_align = is_mamba_align_kv_cache_spec(spec)
            seed = request_hasher((b"UCM_GROUP_SEED", base_seed, group_id))
            info = GroupInfo(
                group_id=group_id,
                block_size=block_size,
                layer_names=tuple(group.layer_names),
                seed=seed,
                is_mamba_align=is_mamba_align,
            )
            if not is_mamba_align:
                info.block_hasher = request_hasher.make_request_block_hasher(
                    block_size, seed
                )
            self.groups_by_id.append(info)
            if info.is_full_attention:
                self.full_attn_groups.append(info)
            else:
                self.state_groups.append(info)

        assert len(self.full_attn_groups) >= 1, (
            "UCMHybridLinearAttentionConnector expects at least one full-attention group in "
            "kv_cache_config.kv_cache_groups."
        )

        # Resume points must align to the LCM of all group block_sizes.
        all_block_sizes = [g.block_size for g in self.groups_by_id]
        self.lcm_block_size: int = math.lcm(*all_block_sizes)

        for g in self.groups_by_id:
            assert self.lcm_block_size % g.block_size == 0, (
                f"group {g.group_id} block_size={g.block_size} does not "
                f"divide LCM={self.lcm_block_size}"
            )
        for sg in self.state_groups:
            assert sg.is_mamba_align, (
                f"state group {sg.group_id} is not mamba-align; "
                f"UCMHybridLinearAttentionConnector only supports mamba-align "
                f"state groups."
            )

        logger.info(
            "KVCacheGroupManager initialized: "
            f"lcm_block_size={self.lcm_block_size}, "
            f"full_attn_groups="
            f"{[(g.group_id, g.block_size) for g in self.full_attn_groups]}, "
            f"state_groups="
            f"{[(g.group_id, g.block_size, g.is_mamba_align) for g in self.state_groups]}"
        )

    @property
    def num_groups(self) -> int:
        return len(self.groups_by_id)

    def compute_block_hashes(self, group: GroupInfo, request: "Request") -> list[bytes]:
        """Hash a request at one group's block boundaries and chain seed."""
        if group.is_mamba_align:
            # mamba-align pads block table with null blocks; no per-block hash.
            return [b""] * (len(request.all_token_ids) // group.block_size)

        assert group.block_hasher is not None
        return group.block_hasher(request)

    def compute_all_group_block_ids(self, request: "Request") -> list[list[bytes]]:
        """Compute full block hashes for every group, indexed by group_id."""
        return [self.compute_block_hashes(g, request) for g in self.groups_by_id]

    def compute_mamba_align_state_hash(
        self,
        group: GroupInfo,
        seq_len: int,
        group_block_ids: list[list[bytes]],
    ) -> Optional[bytes]:
        """Derive the mamba-align state hash at ``seq_len`` from the prefix hash."""
        if seq_len <= 0 or seq_len % self.lcm_block_size != 0:
            return None
        primary = self.full_attn_groups[0]
        prefix_idx = seq_len // primary.block_size - 1
        if prefix_idx < 0:
            return None
        try:
            prefix_hash = group_block_ids[primary.group_id][prefix_idx]
        except IndexError:
            logger.error(
                "mamba-align state hash missing primary prefix hash: "
                f"group_id={group.group_id}, seq_len={seq_len}, "
                f"primary_group_id={primary.group_id}, "
                f"prefix_idx={prefix_idx}, "
                f"num_primary_hashes="
                f"{len(group_block_ids[primary.group_id])}"
            )
            return None
        if not prefix_hash:
            return None
        return self.request_hasher(
            (group.seed, b"UCM_MAMBA_ALIGN_STATE", seq_len, prefix_hash)
        )

    def lookup_external_hit_tokens(
        self,
        num_computed_tokens: int,
        group_block_ids: list[list[bytes]],
        lookup_on_prefix: Callable[[list[bytes]], int],
        lookup_on_reverse: Callable[[list[bytes]], int],
        on_lookup_error: Optional[Callable[[], None]] = None,
    ) -> tuple[int, int, list[bytes]]:
        """Two-stage HLA lookup using precomputed per-group hashes.

        ``group_block_ids`` must have one entry per group, indexed by the
        original ``group_id`` (see :meth:`compute_all_group_block_ids`).

        Stage 1 — every full-attention group runs ``lookup_on_prefix``
        beyond its own ``hbm_hit_block_num``; the candidate hits are taken
        as a min and rounded down to ``lcm_block_size`` so the final
        external hit is consistent across all full-attn groups and aligns
        to the kv-cache page granularity expected by the scheduler.

        Stage 2 — mamba-align state groups are checked via
        ``lookup_on_reverse``: for each state group, the state hashes at
        all candidate LCM boundary positions (earliest-to-latest) are
        collected and a single reverse scan finds the rightmost hit.
        The min across state groups is the rightmost position where ALL
        state groups' states are present. If any state group has no hit
        at any candidate position, the external hit is downgraded to zero.

        ``on_lookup_error`` is invoked once per failed store lookup so the
        caller can feed its metrics pipeline.

        Returns:
            Tuple of
            - ``external_hit_tokens``: tokens hit beyond ``num_computed_tokens``,
              aligned to ``lcm_block_size``. ``0`` if any check fails.
            - ``external_hit_lcm_blocks``: ``external_hit_tokens //
              lcm_block_size`` (also ``0`` on downgrade).
            - ``mamba_prefetch_hashes``: rank-0 mamba state hashes from
              ``num_computed_tokens + lcm_block_size`` to ``best_pos``,
              for GC heat update (rank-0 un-checked positions + other ranks).
        """
        assert len(group_block_ids) == self.num_groups, (
            f"group_block_ids length {len(group_block_ids)} does not match "
            f"num_groups {self.num_groups}"
        )
        assert num_computed_tokens % self.lcm_block_size == 0, (
            f"num_computed_tokens={num_computed_tokens} is not aligned to "
            f"lcm_block_size={self.lcm_block_size}"
        )

        # Stage 1: each full-attn group contributes a candidate hit count.
        candidates: list[int] = []
        for fa in self.full_attn_groups:
            fa_block_ids = group_block_ids[fa.group_id]
            fa_hbm_blocks = num_computed_tokens // fa.block_size
            fa_external = fa_block_ids[fa_hbm_blocks:]
            if not fa_external:
                candidates.append(0)
                continue
            try:
                fa_hit_blocks = lookup_on_prefix(fa_external) + 1
            except Exception as e:
                logger.error(
                    f"full-attn group {fa.group_id} lookup error. "
                    f"{type(e).__name__}: {e}"
                )
                if on_lookup_error is not None:
                    on_lookup_error()
                candidates.append(0)
                continue
            candidates.append(max(fa_hit_blocks, 0) * fa.block_size)

        # Resume boundary must be a multiple of lcm_block_size so every
        # group's tail/dispatch slicing lands on a real block boundary.
        min_external_hit_tokens = min(candidates)
        external_hit_tokens = (
            min_external_hit_tokens // self.lcm_block_size
        ) * self.lcm_block_size
        if external_hit_tokens <= 0:
            return 0, 0, []

        # Stage 2: reverse scan for mamba state at LCM boundaries.
        # For each state group, collect state hashes at all candidate
        # positions (earliest-to-latest) and use lookup_on_reverse to find
        # the rightmost hit.  The min across state groups is the rightmost
        # position where ALL states are present.
        total_hit_tokens = num_computed_tokens + external_hit_tokens

        if not self.state_groups:
            return (
                external_hit_tokens,
                external_hit_tokens // self.lcm_block_size,
                [],
            )

        positions = list(
            range(
                num_computed_tokens + self.lcm_block_size,
                total_hit_tokens + self.lcm_block_size,
                self.lcm_block_size,
            )
        )

        best_pos = total_hit_tokens
        for sg in self.state_groups:
            # Truncate to positions <= best_pos so earlier state groups
            # can shrink the search window for subsequent ones.
            sg_positions = [p for p in positions if p <= best_pos]
            sg_hashes: list[bytes] = []
            for pos in sg_positions:
                state_hash = self.compute_mamba_align_state_hash(
                    sg, pos, group_block_ids
                )
                sg_hashes.append(state_hash if state_hash is not None else b"")
            try:
                idx = lookup_on_reverse(sg_hashes)
            except Exception as e:
                logger.error(
                    f"mamba-align state reverse lookup error for "
                    f"group={sg.group_id}. {type(e).__name__}: {e}"
                )
                if on_lookup_error is not None:
                    on_lookup_error()
                return 0, 0, []
            if idx < 0:
                # This state group has no state at any candidate position.
                return 0, 0, []
            sg_pos = sg_positions[idx]
            if sg_pos < best_pos:
                best_pos = sg_pos

        external_hit_tokens = best_pos - num_computed_tokens
        if external_hit_tokens <= 0:
            return 0, 0, []

        # Collect mamba state hashes for GC heat update.
        mamba_prefetch_hashes: list[bytes] = []
        for pos in range(
            self.lcm_block_size,
            best_pos + self.lcm_block_size,
            self.lcm_block_size,
        ):
            for sg in self.state_groups:
                state_hash = self.compute_mamba_align_state_hash(
                    sg, pos, group_block_ids
                )
                if state_hash is not None:
                    mamba_prefetch_hashes.append(state_hash)

        return (
            external_hit_tokens,
            external_hit_tokens // self.lcm_block_size,
            mamba_prefetch_hashes,
        )
