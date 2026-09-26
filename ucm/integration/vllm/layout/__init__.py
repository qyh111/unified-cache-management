# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
"""Per-group KV-cache addressing layouts for the UCM connector.

Ported from connector v2 (``dev_connector`` branch,
``ucm/integration/vllm/v2/layout/`` + ``record_layout.py`` +
``ucm_kv_cache.py``'s semantic layer) so the v1 connector shares v2's
vocabulary and the future cutover stays small. Not ported: v2's hash
chains (the v1 side keeps vLLM-compatible hashes via ``request_hasher``),
the byte-range proxy and the Transfer batch orchestrator.

One entry point, :func:`parse_kv_cache_config`: turn vLLM's
``KVCacheConfig`` into :class:`UCMKVCacheSpec`; :func:`build_group_layouts`
then hands the runtime views to :class:`KVCacheGroupLayout`, which
flattens them into addressing columns, and :class:`GroupRecordLayout`
compiles a group's record template (window rows, partial head row,
record bytes). Group-internal alignment (ghost slots, e.g.
MiniMax/glm5.2 indexer) and cross-group padding for the one-store /
layerwise schema are layered above this package by the connector-side
aligned subclasses.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from .group import BlockAccess, BlockFirstView, KVCacheGroupLayout, TensorDescriptor
from .record import GroupRecordLayout
from .schema import LayerShardSchema, Slot, StoreSchema
from .spec import (
    UCMKVCacheGroupInfo,
    UCMKVCacheSpec,
    UCMLayerSpec,
    parse_kv_cache_config,
)
from .view import LAYOUT_DEBUG, MemorySegment, build_layer_view, layout_debug

if TYPE_CHECKING:
    import torch


def build_group_layouts(
    spec: UCMKVCacheSpec,
    kv_caches: Mapping[str, "torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]"],
) -> dict[int, KVCacheGroupLayout]:
    """One :class:`KVCacheGroupLayout` per non-empty KV group of the spec."""

    return {
        group.group_id: KVCacheGroupLayout(
            group, kv_caches, device_type=spec.device_type
        )
        for group in spec.groups
        if group.layers
    }


__all__ = [
    "BlockAccess",
    "BlockFirstView",
    "GroupRecordLayout",
    "KVCacheGroupLayout",
    "LAYOUT_DEBUG",
    "LayerShardSchema",
    "MemorySegment",
    "Slot",
    "StoreSchema",
    "TensorDescriptor",
    "UCMKVCacheGroupInfo",
    "UCMKVCacheSpec",
    "UCMLayerSpec",
    "build_group_layouts",
    "build_layer_view",
    "layout_debug",
    "parse_kv_cache_config",
]
