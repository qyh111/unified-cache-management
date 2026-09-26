# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
"""Compose per-group record templates into one store's slot schema.

With a uniform ``token_block_size`` across groups (the one-store
precondition, enforced by ``parse_kv_cache_config`` for mamba-align
models), every key kind's record can span the *same* fixed slot table:
the groups' record templates concatenated in ``dispatch_routes`` order.
A key of one kind fills its own groups' slots with real pointers and
leaves the other kinds' slots as ghosts (``ptr=0``, declared width) --
the same convention the v1 layouts already use for MiniMax/SharedIndexer
ghost segments (``ptr=0, copy_size=size, block_stride=0``; device-buffer
registration filters ``addr==0``).

Two geometries come out of one slot table:

* :class:`StoreSchema` -- the bulk store: ``tensor_size_list`` is the flat
  slot table, one shard per record (``shard_index = 0``).
* :class:`LayerShardSchema` -- the layerwise store: every model layer is
  one shard (``shard_index = layer_id``) and all shards must present the
  *identical* slot list; a layer fills the slots owned by its group and
  ghosts the others. Group-internal raggedness (dense layers missing an
  indexer slot) is auto-aligned by appending ghost columns to shorter
  layers, mirroring what the MiniMax/SharedIndexer layouts hand-write.

Layerwise further requires ``tail_blocks == 1`` per group (one window row
per key), which holds for FA/State under the uniform block size.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .record import GroupRecordLayout
from .view import LAYOUT_DEBUG, layout_debug

IO_ALIGNMENT = 4096


@dataclass(frozen=True)
class Slot:
    """One slot of a store's fixed record schema.

    ``route`` is the key kind owning real data for this slot ("FA", "WA"
    or "State"); records of the other kinds ghost it. ``layer_name``/
    ``layer_id`` identify the physical column (layerwise shards resolve by
    them); ``row`` is the window row inside the owning group's template.
    """

    route: str
    group_id: int
    row: int
    layer_name: str
    layer_id: int
    size_bytes: int


@dataclass(frozen=True)
class StoreSchema:
    """One store's fixed record schema over all groups.

    Slots are concatenated in ``dispatch_routes`` order; inside a group,
    row-major over its ``GroupRecordLayout`` template (row 0 carries the
    partial head row when a tail is not block aligned). ``group_spans``
    maps ``group_id -> (start, count)`` inside the slot table;
    ``group_bases`` maps ``group_id -> byte offset of the group's part``
    within one record.
    """

    slots: tuple[Slot, ...]
    kind_masks: dict[str, np.ndarray]
    group_spans: dict[int, tuple[int, int]]
    group_bases: dict[int, int]
    tensor_size_list: tuple[int, ...]
    shard_size: int
    record_bytes: int
    _records: dict[int, GroupRecordLayout]

    @classmethod
    def build(
        cls,
        routes: Sequence[tuple[str, Sequence[int]]],
        records: Mapping[int, GroupRecordLayout],
    ) -> "StoreSchema":
        """``routes`` is ``spec.dispatch_routes()`` reduced to
        ``(kind, group_ids)`` pairs, keeping this layer spec-free."""
        slots: list[Slot] = []
        kind_lists: dict[str, list[int]] = {}
        spans: dict[int, tuple[int, int]] = {}
        bases: dict[int, int] = {}
        cursor = 0
        for route, group_ids in routes:
            indices = kind_lists.setdefault(route, [])
            for group_id in group_ids:
                record = records.get(group_id)
                if record is None:
                    raise ValueError(f"group {group_id} has no record layout")
                layout = record.access.layout
                spans[group_id] = (
                    len(slots),
                    len(layout.layer_names) * record.blocks_per_key,
                )
                bases[group_id] = cursor
                for row in range(record.blocks_per_key):
                    for column, layer_name in enumerate(layout.layer_names):
                        slots.append(
                            Slot(
                                route=route,
                                group_id=group_id,
                                row=row,
                                layer_name=layer_name,
                                layer_id=int(layout.layer_ids[column]),
                                size_bytes=int(
                                    record.access.segment_bytes[row][column]
                                ),
                            )
                        )
                        indices.append(len(slots) - 1)
                cursor += record.record_bytes
        sizes = tuple(slot.size_bytes for slot in slots)
        schema = cls(
            slots=tuple(slots),
            kind_masks={
                kind: _mask_from_indices(len(slots), indices)
                for kind, indices in kind_lists.items()
            },
            group_spans=spans,
            group_bases=bases,
            tensor_size_list=sizes,
            shard_size=round_up(sum(sizes), IO_ALIGNMENT),
            record_bytes=cursor,
            _records=dict(records),
        )
        if LAYOUT_DEBUG:
            layout_debug(
                f"store-schema slots={len(schema.slots)} "
                f"record_bytes={schema.record_bytes} "
                f"shard_size={schema.shard_size} "
                f"kinds={sorted(schema.kind_masks)}"
            )
        return schema

    def resolve_group(
        self,
        group_id: int,
        blocks: Sequence[int] | np.ndarray,
        *,
        token_offsets: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One group's keys over the full slot table; the other groups ghost.

        Per-group physical addressing for independent block-id chains: the
        group's own slots resolve through its record template, every other
        group's slots keep the declared width with ``ptr=0``. ``blocks``
        carries ``key_count * tail_blocks`` vLLM block ids of this group's
        chain. Returns flattened ``(offsets, ptrs, sizes)`` of length
        ``key_count * len(slots)``, key-major.
        """
        start, count = self.group_spans[group_id]
        record = self._records[group_id]
        block_ids = np.asarray(blocks, dtype=np.uint64)
        rows = len(block_ids) // record.blocks_per_key
        if len(block_ids) % record.blocks_per_key:
            raise ValueError("blocks must contain complete access windows")
        total = len(self.slots)
        offsets = np.zeros((rows, total), dtype=np.uint64)
        ptrs = np.zeros((rows, total), dtype=np.uint64)
        sizes = np.tile(
            np.asarray(self.tensor_size_list, dtype=np.uint64), (rows, 1)
        )
        # Ghost groups keep their template offsets with null pointers.
        for gid, (gstart, gcount) in self.group_spans.items():
            gtemplate = self._records[gid].ucm_block_offsets.reshape(-1)
            offsets[:, gstart : gstart + gcount] = np.tile(
                np.asarray(self.group_bases[gid], dtype=np.uint64) + gtemplate,
                (rows, 1),
            )
        real_offsets, real_ptrs, _, _ = record.resolve(
            block_ids, rows, token_offsets=token_offsets
        )
        offsets[:, start : start + count] = (
            np.asarray(self.group_bases[group_id], dtype=np.uint64)
            + real_offsets.reshape(rows, count)
        )
        ptrs[:, start : start + count] = real_ptrs.reshape(rows, count)
        return offsets.reshape(-1), ptrs.reshape(-1), sizes.reshape(-1)

    def resolve(
        self,
        kind: str,
        key_count: int,
        group_blocks: Mapping[int, Sequence[int] | np.ndarray],
        *,
        token_offsets: Mapping[int, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Resolve one key kind's records over the full slot table.

        ``group_blocks[gid]`` carries ``key_count * tail_blocks`` vLLM block
        ids for each group of this kind. Real slots resolve through the
        group's record template; the other kinds' slots ghost with
        ``ptr=0`` at their declared widths. Returns flattened
        ``(offsets, ptrs, sizes)`` of length ``key_count * len(slots)``,
        key-major.
        """
        if kind not in self.kind_masks:
            raise ValueError(f"unknown key kind {kind!r}")
        total = len(self.slots)
        offsets = np.zeros((key_count, total), dtype=np.uint64)
        ptrs = np.zeros((key_count, total), dtype=np.uint64)
        sizes = np.tile(
            np.asarray(self.tensor_size_list, dtype=np.uint64), (key_count, 1)
        )
        for group_id, (start, count) in self.group_spans.items():
            record = self._records[group_id]
            template = record.ucm_block_offsets.reshape(-1)
            rows = key_count
            offsets[:, start : start + count] = np.tile(
                np.asarray(self.group_bases[group_id], dtype=np.uint64)
                + template,
                (rows, 1),
            )
            if not self.kind_masks[kind][start]:
                continue  # ghost group: ptr stays 0, sizes stay declared
            blocks = np.asarray(group_blocks[group_id], dtype=np.uint64)
            group_offsets = None
            if token_offsets is not None and group_id in token_offsets:
                group_offsets = token_offsets[group_id]
            real_offsets, real_ptrs, _, _ = record.resolve(
                blocks,
                key_count,
                token_offsets=group_offsets,
            )
            if len(real_ptrs) != rows * count:
                raise ValueError(
                    f"group {group_id} resolved {len(real_ptrs)} entries for "
                    f"{rows * count} slots"
                )
            offsets[:, start : start + count] = (
                np.asarray(self.group_bases[group_id], dtype=np.uint64)
                + real_offsets.reshape(rows, count)
            )
            ptrs[:, start : start + count] = real_ptrs.reshape(rows, count)
        return offsets.reshape(-1), ptrs.reshape(-1), sizes.reshape(-1)


@dataclass(frozen=True)
class LayerShardSchema:
    """The layerwise geometry: one identical slot list per model layer.

    Slots are the position-wise union of the groups' per-layer column
    schemas, concatenated in ``dispatch_routes`` order. A layer fills the
    slots owned by its group (real columns from its layout) and ghosts the
    others; group-internal raggedness is auto-aligned by appending ghost
    columns to shorter layers (the MiniMax indexer pattern). All layers
    share one ``tensor_size_list`` -- the store's per-shard slot list.
    """

    slots: tuple[Slot, ...]
    layer_ids: tuple[int, ...]
    tensor_size_list: tuple[int, ...]
    shard_size: int
    _records: dict[int, GroupRecordLayout]
    _positions: dict[int, tuple[tuple[int, int], ...]]
    """layer_id -> ((slot index, group_id), ...) of the layer's real slots."""

    @classmethod
    def build(
        cls,
        routes: Sequence[tuple[str, Sequence[int]]],
        records: Mapping[int, GroupRecordLayout],
    ) -> "LayerShardSchema":
        """``routes`` is ``spec.dispatch_routes()`` reduced to
        ``(kind, group_ids)`` pairs, keeping this layer spec-free."""
        slots: list[Slot] = []
        positions: dict[int, list[tuple[int, int]]] = {}
        layer_ids: set[int] = set()

        for route, group_ids in routes:
            for group_id in group_ids:
                record = records.get(group_id)
                if record is None:
                    raise ValueError(f"group {group_id} has no record layout")
                if record.blocks_per_key != 1:
                    raise ValueError(
                        f"layerwise shards require tail_blocks == 1 for group "
                        f"{group_id}, got {record.blocks_per_key}"
                    )
                layout = record.access.layout
                layer_columns = _per_layer_columns(layout, record)
                layer_ids.update(layer_columns)
                position_schema = _align_layer_columns(layer_columns)
                for column, (size_bytes, donor_name, donor_id) in enumerate(
                    position_schema
                ):
                    slots.append(
                        Slot(
                            route=route,
                            group_id=group_id,
                            row=0,
                            layer_name=donor_name,
                            layer_id=donor_id,
                            size_bytes=size_bytes,
                        )
                    )
                    # A layer owns a position iff its own column list extends
                    # there; shorter layers ghost the tail (MiniMax indexer).
                    for layer_id, columns in layer_columns.items():
                        if column < len(columns):
                            positions.setdefault(layer_id, []).append(
                                (len(slots) - 1, group_id)
                            )
        if not slots:
            raise ValueError("layer shard schema is empty")
        sizes = tuple(slot.size_bytes for slot in slots)
        schema = cls(
            slots=tuple(slots),
            layer_ids=tuple(sorted(layer_ids)),
            tensor_size_list=sizes,
            shard_size=round_up(sum(sizes), IO_ALIGNMENT),
            _records=dict(records),
            _positions={
                layer_id: tuple(entries) for layer_id, entries in positions.items()
            },
        )
        if LAYOUT_DEBUG:
            layout_debug(
                f"layer-shard-schema slots={len(schema.slots)} "
                f"layers={len(schema.layer_ids)} shard_size={schema.shard_size}"
            )
        return schema

    def resolve_layer(
        self,
        layer_id: int,
        key_count: int,
        group_blocks: Mapping[int, Sequence[int] | np.ndarray],
        *,
        token_offsets: Mapping[int, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resolve one layer's shard: ``(ptrs, sizes)`` over the shard slots.

        Real slots come from the owning group's record template restricted
        to this layer (``segment_mask(layer_ids=[layer_id])``); the other
        groups' slots ghost with ``ptr=0``. Ghost size is the declared
        width, so every shard presents the same slot list to the store.
        """
        total = len(self.slots)
        ptrs = np.zeros((key_count, total), dtype=np.uint64)
        sizes = np.tile(
            np.asarray(self.tensor_size_list, dtype=np.uint64), (key_count, 1)
        )
        entries = self._positions.get(layer_id, ())
        if not entries:
            return ptrs.reshape(-1), sizes.reshape(-1)
        group_id = entries[0][1]
        record = self._records[group_id]
        blocks = np.asarray(group_blocks[group_id], dtype=np.uint64)
        group_offsets = None
        if token_offsets is not None and group_id in token_offsets:
            group_offsets = token_offsets[group_id]
        _, real_ptrs, real_sizes, _ = record.resolve(
            blocks,
            key_count,
            token_offsets=group_offsets,
            segment_mask=record.access.layout.segment_mask(layer_ids=[layer_id]),
        )
        if len(real_ptrs) != key_count * len(entries):
            raise ValueError(
                f"layer {layer_id} resolved {len(real_ptrs)} entries for "
                f"{key_count * len(entries)} slots"
            )
        for index, (slot_index, _) in enumerate(entries):
            ptrs[:, slot_index] = real_ptrs.reshape(key_count, -1)[:, index]
            sizes[:, slot_index] = real_sizes.reshape(key_count, -1)[:, index]
        return ptrs.reshape(-1), sizes.reshape(-1)


def _per_layer_columns(
    layout, record: GroupRecordLayout
) -> dict[int, list[tuple[str, int, bool]]]:
    """``layer_id -> [(column name, size bytes, ghost=False), ...]``.

    Column order follows the layout's (layer_index, layer_name) order;
    sizes are the record template's first-row widths -- the partial head
    row of a sliding group, whole-block payload otherwise -- so a shard's
    slot list always matches what the record template resolves.
    """
    columns: dict[int, list[tuple[str, int, bool]]] = {}
    for index, layer_id in enumerate(layout.layer_ids.tolist()):
        columns.setdefault(int(layer_id), []).append(
            (
                layout.layer_names[index],
                int(record.access.segment_bytes[0][index]),
                False,
            )
        )
    return columns


def _align_layer_columns(
    layer_columns: Mapping[int, list[tuple[str, int, bool]]],
) -> list[tuple[int, str, int]]:
    """Uniform per-layer schema: ``(size, donor name, donor layer_id)``.

    The longest layer's column list defines the positions; every other
    layer must match it as a prefix (same sizes in the same order) and
    auto-ghosts the missing tail -- the append pattern the v1 layouts use
    for MiniMax/SharedIndexer. Exotic interleavings need an explicit
    alignment subclass and are rejected here. Whether a position is real
    or ghosted is decided *per layer* in :class:`LayerShardSchema`.
    """
    longest: list[tuple[str, int, bool]] = []
    longest_id = -1
    for layer_id, columns in layer_columns.items():
        if len(columns) > len(longest):
            longest = columns
            longest_id = layer_id
    if not longest:
        raise ValueError("group has no columns")
    for layer_id, columns in layer_columns.items():
        for position, (name, size, _) in enumerate(columns):
            expected = longest[position]
            if size != expected[1]:
                raise ValueError(
                    f"layer {layer_id} column {name!r} ({size}B) does not "
                    f"match the group schema position {position} "
                    f"({expected[0]!r}, {expected[1]}B); group-internal "
                    "columns must be prefix-aligned"
                )
    return [(size, name, longest_id) for name, size, _ in longest]


def _mask_from_indices(total: int, indices: Sequence[int]) -> np.ndarray:
    mask = np.zeros(total, dtype=np.bool_)
    if indices:
        mask[np.asarray(indices, dtype=np.int64)] = True
    return mask


def round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment
