# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
"""Semantic KV-cache description: groups, cache kinds, block sizing.

Ported from connector v2 (``dev_connector`` branch,
``ucm/integration/vllm/v2/ucm_kv_cache.py`` L1-605) on 2026-09-26 so the
v1 connector shares v2's vocabulary and the future cutover stays small.
Excluded from the port: the proxy-facing batch orchestrator
(``UCMKVCacheLayout`` and the Transfer types it needs -- the proxy is not
part of this port) and v2's own hash chains. Adaptations are import paths
only; types, parsing and validation are unchanged.

Physical placement lives in ``.group``/``.view``; record templates in
``.record``. This module keeps the semantic layer: ``parse_kv_cache_config``
turns vLLM's ``KVCacheConfig`` into :class:`UCMKVCacheSpec` -- per-group
``UCMKVCacheGroupInfo`` with stamped ``tail_blocks`` and per-layer
``UCMLayerSpec`` with mirrored 0.29 descriptors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.kv_cache_interface import (
    KVCacheSpecKind,
    get_kv_cache_spec_kind,
)

from .group import TensorDescriptor
from .view import LAYOUT_DEBUG, layout_debug

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheSpec,
    )


_SLIDING_KINDS = frozenset(
    (KVCacheSpecKind.SLIDING_WINDOW, KVCacheSpecKind.SLIDING_WINDOW_MLA)
)


@dataclass(frozen=True)
class UCMLayerSpec:
    """One registered layer; storage_block_size counts stored states, not tokens."""

    layer_name: str
    layer_index: int
    kv_cache_spec: "KVCacheSpec"
    storage_block_size: int
    num_blocks: int
    # 0.29 placement: the declaration covering this layer and the
    # layer's position in it; None on 0.26 (per-tensor overlay).
    descriptor: "TensorDescriptor | None" = None
    descriptor_position: int = 0


@dataclass(frozen=True)
class UCMKVCacheGroupInfo:
    """One native KV group; token_block_size is tokens covered by one block ID."""

    group_id: int
    layers: tuple[UCMLayerSpec, ...]
    group_spec: "KVCacheSpec"
    token_block_size: int
    kinds: frozenset[KVCacheSpecKind]
    tail_tokens: int | None = None
    # vLLM blocks one ucm key's window spans (0 = the group stores
    # nothing); the parser stamps it once the cache block size is known.
    tail_blocks: int = 0
    is_eagle_group: bool = False

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def is_attention(self) -> bool:
        return KVCacheSpecKind.MAMBA not in self.kinds

    @property
    def is_sliding_window(self) -> bool:
        return not self.kinds.isdisjoint(_SLIDING_KINDS)

    @property
    def is_state_snapshot(self) -> bool:
        return KVCacheSpecKind.MAMBA in self.kinds


@dataclass(frozen=True)
class UCMKVCacheSpec:
    """UCM policy sizes, all in tokens.

    scheduler_block_size is what vLLM's
    ``resolve_kv_cache_block_sizes`` reports for this engine -- the
    token-alignment invariant of the resident KV pool (single group:
    ``cache_config.block_size``; multiple groups: LCM).  ucm_cache_block_size
    is both the record unit and the shared hash chain's key granularity:
    the smallest token_block_size across the full-attention groups
    (token_block_size already folds compression), or the scheduler block
    when the model has no full-attention group.
    """

    groups: tuple[UCMKVCacheGroupInfo, ...]
    scheduler_block_size: int
    ucm_cache_block_size: int
    device_type: str

    @property
    def alignment_block_size(self) -> int:
        """The multi-group boundary every external store aligns to.

        The model-check harness trims its source prompt to a multiple of
        this (lcm with the native group block sizes), so a dump always
        covers complete cache blocks of every group.  The cache_block_size
        is that boundary: FA chains hash at it and every group's window
        within it is whole or a measured fraction.
        """

        return self.ucm_cache_block_size

    @property
    def attn_groups(self) -> tuple[UCMKVCacheGroupInfo, ...]:
        return tuple(group for group in self.groups if group.is_attention)

    @property
    def state_groups(self) -> tuple[UCMKVCacheGroupInfo, ...]:
        return tuple(group for group in self.groups if group.is_state_snapshot)

    @property
    def sw_groups(self) -> tuple[UCMKVCacheGroupInfo, ...]:
        return tuple(group for group in self.groups if group.is_sliding_window)

    @property
    def fa_groups(self) -> tuple[UCMKVCacheGroupInfo, ...]:
        return tuple(
            group
            for group in self.groups
            if group.is_attention and not group.is_sliding_window
        )

    @property
    def wa_groups(self) -> tuple[UCMKVCacheGroupInfo, ...]:
        return tuple(group for group in self.groups if group.is_sliding_window)

    def dispatch_routes(
        self,
    ) -> tuple[
        tuple[Literal["FA", "WA", "State"], tuple["UCMKVCacheGroupInfo", ...]], ...
    ]:
        """The routing table every dump/load works over: key kind -> groups.

        FA holds the full-attention groups; WA the sliding groups that
        re-store a window tail (tail 0 groups store nothing); State the
        mamba snapshot groups. Empty kinds are absent, and
        ``group_ucm_block_ids`` / dispatch plans index these in order.
        """

        routes: list[
            tuple[Literal["FA", "WA", "State"], tuple["UCMKVCacheGroupInfo", ...]]
        ] = []
        if self.fa_groups:
            routes.append(("FA", self.fa_groups))
        wa_stored = tuple(
            group for group in self.wa_groups if (group.tail_tokens or 0) > 0
        )
        if wa_stored:
            routes.append(("WA", wa_stored))
        if self.state_groups:
            routes.append(("State", self.state_groups))
        return tuple(routes)

    @property
    def layer_to_group(self) -> Mapping[str, int]:
        return {
            layer.layer_name: group.group_id
            for group in self.groups
            for layer in group.layers
        }


def _group_tail_blocks(group: UCMKVCacheGroupInfo, ucm_block_size: int) -> int:
    """vLLM blocks one ucm key's window spans (0 = the group stores nothing).

    HMA's ``tail_blocks`` with the one generalization v2 needs: a tail
    that does not divide the block still keeps its partial head block
    (ceil, not HMA's floor).  A state snapshot is one indivisible
    checkpoint page; a full-attention key spans the whole ucm block
    (several blocks when the unit is the larger side, the containing
    block otherwise).
    """

    token_block = group.token_block_size
    if group.is_state_snapshot:
        return 1
    if group.is_sliding_window:
        tail = group.tail_tokens or 0
        return (tail + token_block - 1) // token_block if tail > 0 else 0
    return (ucm_block_size + token_block - 1) // token_block


def _concrete_specs(
    group: "KVCacheGroupSpec",
) -> tuple[tuple[str, "KVCacheSpec"], ...]:
    group_spec = group.kv_cache_spec
    nested = getattr(group_spec, "kv_cache_specs", None)
    names = tuple(getattr(group, "layer_names", ()))
    if nested:
        missing = [name for name in names if name not in nested]
        if missing:
            raise ValueError(f"KV cache group is missing specs for layers {missing}")
        return tuple((name, nested[name]) for name in names)
    return tuple((name, group_spec) for name in names)


def _spec_tokens_per_state(spec: "KVCacheSpec") -> int:
    """Compression ratio of one stored state (DSV4 C4A = 4).

    Ascend 0.26 names the field ``compress_ratio``; vLLM 0.29 renamed it to
    ``tokens_per_state`` (vLLM #51718) with identical semantics for DSV4
    (int > 1 compresses multiple tokens into one stored state). Mamba
    specs carry a -1 sentinel on 0.29 which callers treat as no
    compression; sliding and state groups carry 1.
    """
    value = getattr(spec, "compress_ratio", None)
    if value is None:
        value = getattr(spec, "tokens_per_state", 1)
    if value is None:
        return 1
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def _layer_index(name: str, device_type: str, num_hidden_layers: int | None) -> int:
    """Ascend MTP uses local IDs; official 0.29 already uses global IDs."""
    index = extract_layer_index(name)
    if device_type.lower() == "npu" and "mtp" in name.split("."):
        if num_hidden_layers is None or num_hidden_layers <= 0:
            raise ValueError("MTP layer IDs require model num_hidden_layers")
        if index < num_hidden_layers:
            index += num_hidden_layers
    return index


def _classify(
    group: "KVCacheGroupSpec",
    concrete: Sequence[tuple[str, "KVCacheSpec"]],
) -> frozenset[KVCacheSpecKind]:
    specs = (
        tuple(spec for _, spec in concrete)
        or tuple(getattr(group.kv_cache_spec, "kv_cache_specs", {}).values())
        or (group.kv_cache_spec,)
    )
    spec_kinds = tuple(get_kv_cache_spec_kind(spec) for spec in specs)
    unknown = tuple(
        type(spec).__qualname__
        for spec, kind in zip(specs, spec_kinds)
        if kind == KVCacheSpecKind.UNKNOWN
    )
    if unknown:
        raise TypeError(f"Unsupported KV cache spec types: {sorted(set(unknown))}")

    kinds = frozenset(spec_kinds)
    if KVCacheSpecKind.MAMBA in kinds and len(kinds) != 1:
        raise TypeError(f"Mamba and attention specs cannot share a KV group: {kinds}")
    return kinds


def _parse_descriptors(
    kv_cache_tensors: "Sequence[object]",
) -> dict[str, tuple["TensorDescriptor", int]]:
    """Mirror vLLM 0.29 kv_cache_tensors; undeclared configs yield {}."""

    declared_at: dict[str, tuple["TensorDescriptor", int]] = {}
    for entry in kv_cache_tensors:
        layers = tuple(str(name) for name in getattr(entry, "layers", ()) or ())
        if not layers:
            raise ValueError("A declared KV cache tensor covers no layers")
        descriptor = TensorDescriptor(
            layers=layers,
            offset=int(getattr(entry, "offset", 0)),
            layer_stride=int(getattr(entry, "layer_stride")),
            block_stride=int(getattr(entry, "block_stride")),
        )
        if (
            descriptor.offset < 0
            or descriptor.layer_stride < 0
            or descriptor.block_stride <= 0
        ):
            raise ValueError(f"Invalid declared placement: {descriptor}")
        for position, name in enumerate(layers):
            if name in declared_at:
                raise ValueError(
                    f"Layer {name} is covered by more than one declared tensor"
                )
            declared_at[name] = (descriptor, position)
    return declared_at


def parse_kv_cache_config(
    kv_cache_config: "KVCacheConfig",
    *,
    scheduler_block_size: int,
    ucm_cache_block_size: int | None = None,
    device_type: str = "npu",
    attention_tokens_per_state: Mapping[int, int] | None = None,
    num_hidden_layers: int | None = None,
) -> UCMKVCacheSpec:
    """Describe logical groups and per-layer storage from KVCacheConfig.

    vLLM 0.29's scheduler replaces a UniformTypeKVCacheSpecs map with one
    representative spec. For DSV4 that loses C4/C128 per-layer ratios.
    The connector supplies attention_tokens_per_state from the model config
    on both scheduler and worker; direct callers with full specs can omit it.
    This mapping applies to full attention, never compressor state tensors.
    """

    attention_tokens_per_state = attention_tokens_per_state or {}

    # 0.29 kv_cache_tensors carry a layers placement per tensor; 0.26
    # entries (size + shared_by) do not, so only the declared ones parse.
    kv_cache_tensors = tuple(getattr(kv_cache_config, "kv_cache_tensors", ()) or ())
    declared = tuple(
        tensor
        for tensor in kv_cache_tensors
        if getattr(tensor, "layers", None) is not None
    )
    declared_at = _parse_descriptors(declared)
    raw_groups = tuple(getattr(kv_cache_config, "kv_cache_groups", ()))
    if not raw_groups:
        raise ValueError("kv_cache_config.kv_cache_groups must not be empty")

    classified: list[
        tuple[
            "KVCacheGroupSpec",
            tuple[tuple[str, "KVCacheSpec"], ...],
            frozenset[KVCacheSpecKind],
        ]
    ] = []
    layer_indices: dict[str, int] = {}
    for raw_group in raw_groups:
        concrete = _concrete_specs(raw_group)
        layer_indices.update(
            (name, _layer_index(name, device_type, num_hidden_layers))
            for name, _ in concrete
        )
        kinds = _classify(raw_group, concrete)
        if KVCacheSpecKind.MAMBA in kinds:
            check_specs = (
                concrete
                or tuple(getattr(raw_group.kv_cache_spec, "kv_cache_specs", {}).items())
                or (("", raw_group.kv_cache_spec),)
            )
            modes = {
                str(getattr(spec, "mamba_cache_mode", None)) for _, spec in check_specs
            }
            if modes != {"align"}:
                raise ValueError(
                    "connector v2 supports Mamba state only with "
                    f"mamba_cache_mode='align', got {sorted(modes)}"
                )
            block_sizes = {int(getattr(spec, "block_size")) for _, spec in check_specs}
            if block_sizes != {scheduler_block_size}:
                raise ValueError(
                    "Mamba align block size must equal cache_config.block_size="
                    f"{scheduler_block_size}, got {sorted(block_sizes)}"
                )
        classified.append((raw_group, concrete, kinds))

    device_type = str(device_type).lower()
    if len(raw_groups) != 1 and ucm_cache_block_size is not None:
        raise ValueError(
            "custom ucm_cache_block_size is supported only for a single KV group"
        )

    # Compression detection drives only the per-layer ratio recovery in
    # the group loop below (0.29's representative spec loses per-layer
    # ratios); the cache block itself comes from the FA groups' token
    # spans, which already fold compression.
    has_compression = any(
        max((_spec_tokens_per_state(spec) for _, spec in concrete), default=1) > 1
        for _raw_group, concrete, kinds in classified
        if KVCacheSpecKind.MAMBA not in kinds and kinds.isdisjoint(_SLIDING_KINDS)
    )

    groups: list[UCMKVCacheGroupInfo] = []
    fa_token_blocks: list[int] = []
    attention_tokens_per_state_by_layer: dict[int, int] = {}
    if has_compression:
        for _, concrete, kinds in classified:
            if KVCacheSpecKind.MAMBA in kinds or not kinds.isdisjoint(_SLIDING_KINDS):
                continue
            for name, concrete_spec in concrete:
                attention_tokens_per_state_by_layer[layer_indices[name]] = (
                    attention_tokens_per_state.get(
                        layer_indices[name], _spec_tokens_per_state(concrete_spec)
                    )
                )
    num_blocks = int(getattr(kv_cache_config, "num_blocks", 0))
    for group_id, (raw_group, concrete, kinds) in enumerate(classified):
        representative = (
            concrete[0][1]
            if concrete
            else next(
                iter(getattr(raw_group.kv_cache_spec, "kv_cache_specs", {}).values()),
                raw_group.kv_cache_spec,
            )
        )
        physical_block_size = int(getattr(raw_group.kv_cache_spec, "block_size"))
        if KVCacheSpecKind.MAMBA in kinds:
            # Mamba state blocks follow the scheduler block (align check
            # above); they never fold tokens.
            token_block_size = physical_block_size
        else:
            # Ascend 0.26 attention specs report the storage span and
            # carry compress_ratio: the token span is the product.  vLLM
            # 0.29 reports token spans directly, so its spec block is
            # already the token span.
            ascend_ratio = getattr(representative, "compress_ratio", None)
            token_block_size = (
                physical_block_size * int(ascend_ratio)
                if ascend_ratio
                else physical_block_size
            )
            if kinds.isdisjoint(_SLIDING_KINDS):
                fa_token_blocks.append(token_block_size)
        layers: list[UCMLayerSpec] = []
        for index, (name, spec) in enumerate(concrete):
            # Normalize to the number of stored states one group block
            # spans.  Ascend 0.26 reports the C4 storage span as block_size
            # directly; vLLM 0.29 reports the logical span and the storage
            # axis is block_size // tokens_per_state (DSV4 C4A: 256/4=64
            # states; C128A: 256/128=2; uncompressed specs keep the block).
            logical = int(getattr(spec, "block_size"))
            ratio = _spec_tokens_per_state(spec)
            if has_compression and kinds.isdisjoint(_SLIDING_KINDS):
                ratio = attention_tokens_per_state_by_layer[layer_indices[name]]
            if device_type == "npu":
                # Ascend's replicated DCP indexer allocates consecutive kernel
                # rows per logical block, as declared by its native cache spec.
                storage_block_size = logical * int(
                    getattr(spec, "sfa_dcp_replicated_indexer_size", 1)
                )
            elif ratio > 1 and logical % ratio == 0:
                storage_block_size = logical // ratio
            else:
                storage_block_size = logical
            layers.append(
                UCMLayerSpec(
                    name,
                    layer_indices[name],
                    spec,
                    storage_block_size,
                    num_blocks,
                    *(declared_at.get(name, (None, 0))),
                )
            )
        tail_tokens: int | None = None
        if not kinds.isdisjoint(_SLIDING_KINDS):
            # What a sliding group re-stores at each hash boundary -- not
            # necessarily the whole window: swa_cache keeps the full
            # window, a compressor state cache keeps window minus its
            # layer's compression ratio, and window == ratio leaves
            # nothing to store (tail 0 groups join no chain).
            tails: set[int] = set()
            tail_specs = concrete or tuple(
                getattr(raw_group.kv_cache_spec, "kv_cache_specs", {}).items()
            )
            for name, concrete_spec in tail_specs:
                window = int(getattr(concrete_spec, "sliding_window"))
                if name.lower().endswith("swa_cache"):
                    tail = window
                else:
                    layer_index = layer_indices.get(name)
                    if layer_index is None:
                        layer_index = _layer_index(name, device_type, num_hidden_layers)
                    ratio = attention_tokens_per_state_by_layer.get(
                        layer_index, attention_tokens_per_state.get(layer_index)
                    )
                    if ratio is None:
                        raise ValueError(
                            "Cannot find matching full-attention compression ratio "
                            f"for sliding layer {layer_index}"
                        )
                    tail = window - ratio
                if tail < 0:
                    raise ValueError(f"Negative sliding tail for {name}: {tail}")
                tails.add(tail)
            if not tail_specs:
                raise NotImplementedError(
                    "An empty PP sliding group needs global per-layer tail semantics; "
                    "this engine projection does not provide enough information"
                )
            if len(tails) != 1:
                raise ValueError(
                    f"Group {group_id} has inconsistent tail sizes {sorted(tails)}"
                )
            tail_tokens = tails.pop()
        groups.append(
            UCMKVCacheGroupInfo(
                group_id=group_id,
                layers=tuple(layers),
                group_spec=raw_group.kv_cache_spec,
                token_block_size=token_block_size,
                kinds=kinds,
                tail_tokens=tail_tokens,
                is_eagle_group=any(
                    "eagle" in layer.layer_name.lower() for layer in layers
                ),
            )
        )

    state_groups = tuple(group for group in groups if group.is_state_snapshot)
    if state_groups:
        if not any(group.is_attention for group in groups):
            raise ValueError(
                "State-only KV cache groups are unsupported: mamba snapshots "
                "restore behind a full-attention prefix"
            )
        mismatched_groups = {
            group.group_id: group.token_block_size
            for group in groups
            if group.token_block_size != scheduler_block_size
        }
        if mismatched_groups:
            raise ValueError(
                "Mamba align requires every KV group block size to equal "
                f"cache_config.block_size={scheduler_block_size}, got "
                f"{mismatched_groups}"
            )
    if len(groups) == 1 and ucm_cache_block_size is not None:
        selected_block = ucm_cache_block_size
        if (
            selected_block < scheduler_block_size
            or selected_block % scheduler_block_size
        ):
            raise ValueError(
                "ucm_cache_block_size must be a positive multiple of "
                "scheduler_block_size"
            )
    elif fa_token_blocks:
        selected_block = min(fa_token_blocks)
    else:
        selected_block = scheduler_block_size

    # Flat plan windows need a uniform per-key block count on FA chains:
    # every FA group's token span must divide, or be divided by, the
    # cache block -- otherwise a key straddles blocks and the window
    # shape varies per key.
    for group in groups:
        if not (group.is_attention and not group.is_sliding_window):
            continue
        if selected_block % group.token_block_size and (
            group.token_block_size % selected_block
        ):
            raise ValueError(
                f"Full-attention group {group.group_id} token_block_size "
                f"{group.token_block_size} neither divides nor is a "
                f"multiple of ucm_cache_block_size {selected_block}"
            )

    # Sliding tails keep a static window shape: the boundary must land
    # on the group's block grid, so its token block must divide the
    # cache block.
    for group in groups:
        if not group.is_sliding_window or not (group.tail_tokens or 0):
            continue
        if selected_block % group.token_block_size:
            raise ValueError(
                f"Sliding-window group {group.group_id} token_block_size "
                f"{group.token_block_size} must divide "
                f"ucm_cache_block_size {selected_block}"
            )

    groups = [
        replace(group, tail_blocks=_group_tail_blocks(group, selected_block))
        for group in groups
    ]

    if LAYOUT_DEBUG:
        for group in groups:
            kind_names = ",".join(sorted(kind.value for kind in group.kinds))
            layout_debug(
                f"spec group={group.group_id} layers={group.num_layers} "
                f"kinds={{{kind_names}}} token_block={group.token_block_size} "
                f"tail={group.tail_tokens} tail_blocks={group.tail_blocks}"
            )
        layout_debug(
            f"spec scheduler_block={scheduler_block_size} "
            f"ucm_cache_block={selected_block} "
            f"device={device_type}"
        )

    return UCMKVCacheSpec(
        groups=tuple(groups),
        scheduler_block_size=scheduler_block_size,
        ucm_cache_block_size=selected_block,
        device_type=device_type,
    )
