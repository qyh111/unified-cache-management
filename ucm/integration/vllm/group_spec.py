# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
"""Declarative group schema for UCM connectors (connector refactor, phase 1).

This module is the *specification layer* of the planned connector refactor:
one :class:`GroupSpec` declares, per KV-cache group (full attention / sliding
window / mamba state), everything the rest of the system needs to derive
without reverse-engineering live tensors:

* the scheduling face — hash granularity relations and the dump/load block
  selection rule (dump and load share one rule, as proven in connector v2);
* the physical face — the per-record segment schema, including ghost padding
  slots, from which a store's fixed ``tensor_size_list`` is compiled.

It intentionally imports nothing beyond the standard library so that the
invariants can be unit-tested on any machine without vLLM/UCM installed.

Evidence map (develop branch, 2026-09-26):
* ``KVCacheSegment`` four-tuple is the pre-existing segment primitive:
  ucm_connector.py L378-382.
* per-group scheduling metadata today: hma_connector.py ``KVCacheGroupMeta``
  L40-47 / ``_init_group_metas`` L472-583; hla_connector.py ``GroupInfo``
  L135-149 / ``KVCacheGroupManager`` L152-412.
* ghost padding today (three hand-written variants of one idea):
  ucm_connector.py L623-629 (SharedIndexer), L1201-1205 (MiniMax M3);
  bulk paths drop ghosts (L1283, L818-819), registration filters
  ``addr==0 or size==0`` (L1646-1655).
* store fixed-length constraint: ``tensor_size_list`` frozen at store init,
  ``shard_size = block_size = round_up(sum, 4096)`` — base L1614-1623,
  hma_connector.py L724-727.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Sequence

IO_ALIGNMENT = 4096
"""Store record alignment: shard_size = block_size = round_up(sum, 4096)."""


class GroupKind(str, Enum):
    """Transfer semantics of a KV-cache group."""

    FULL_ATTENTION = "fa"
    WINDOW_ATTENTION = "wa"
    STATE = "state"


@dataclass(frozen=True)
class SegmentSchema:
    """One slot inside a group's per-row record schema.

    ``size_bytes`` is the payload width of the slot for one row. A *ghost*
    slot (``ghost=True``) occupies schema width so rows keep a fixed
    rectangular layout, but carries no backing memory: the materializer must
    emit ``ptr=0, block_stride=0, buffer_size=0`` and device-buffer
    registration must skip it.
    """

    name: str
    size_bytes: int
    ghost: bool = False

    def __post_init__(self) -> None:
        if self.size_bytes < 0:
            raise ValueError(f"segment {self.name!r}: negative size {self.size_bytes}")


@dataclass(frozen=True)
class BlockRef:
    """One (boundary, row) addressing result of the shared selection rule.

    ``block_index`` indexes into the group's own vLLM block-id chain;
    ``token_offset`` positions the boundary inside that physical block when
    the hash granularity ``unit`` is smaller than the group's
    ``token_block_size`` (e.g. the Ascend C128 group of DSV4,
    hma_connector.py L1253-1257) or when a short WA tail hides at the block
    tail (hma L1275-1283).
    """

    group_id: int
    boundary: int
    row: int
    block_index: int
    token_offset: int = 0


@dataclass(frozen=True)
class GroupSpec:
    """Declarative schema of one KV-cache group.

    Scheduling face
    ---------------
    ``token_block_size`` (tb): tokens covered by one physical block of this
    group. ``tail_tokens``: the window payload stored per hash boundary —
    0 for a whole-block FA chain, ``unit`` when the FA record is one hash
    block of content (FAWA's convention, hma L508), the window width for
    WA, or the state page for STATE. ``tail_blocks``: physical rows per
    record; parsers choose :func:`tail_blocks_ceil` (general form, keeps
    partial head blocks) or :func:`tail_blocks_floor` (FAWA closed form).

    Physical face
    -------------
    ``segments``: the slot schema of one row. Rows of one group share the
    same schema; a full record spans ``tail_blocks`` rows.
    """

    group_id: int
    name: str
    kind: GroupKind
    token_block_size: int
    tail_tokens: int = 0
    tail_blocks: int = 1
    layer_names: tuple[str, ...] = ()
    segments: tuple[SegmentSchema, ...] = ()

    def __post_init__(self) -> None:
        if self.token_block_size <= 0:
            raise ValueError(f"group {self.name!r}: token_block_size must be positive")
        if self.tail_tokens < 0:
            raise ValueError(f"group {self.name!r}: negative tail_tokens")
        if self.tail_blocks <= 0:
            raise ValueError(f"group {self.name!r}: tail_blocks must be positive")
        if self.kind is not GroupKind.FULL_ATTENTION and self.tail_tokens == 0:
            raise ValueError(f"group {self.name!r}: {self.kind.value} groups need a tail")
        if len(self.segments) != len({s.name for s in self.segments}):
            raise ValueError(f"group {self.name!r}: duplicate segment names")

    # -- derived scheduling quantities (v2 vocabulary) -------------------

    def window_span(self, unit: int) -> int:
        """Tokens alive in the head (oldest) block of a window; 0 = all full.

        WA: ``tail_tokens % tb``. FA 1:N (``unit > tb``): ``unit % tb``.
        """
        if self.kind is GroupKind.FULL_ATTENTION:
            return unit % self.token_block_size if unit > self.token_block_size else 0
        return self.tail_tokens % self.token_block_size

    # -- derived physical quantities -------------------------------------

    @property
    def row_size_bytes(self) -> int:
        return sum(s.size_bytes for s in self.segments)

    @property
    def real_row_size_bytes(self) -> int:
        """Bulk-path payload: ghost slots are dropped before flattening."""
        return sum(s.size_bytes for s in self.segments if not s.ghost)

    @property
    def record_bytes(self) -> int:
        """Fixed record width a store must reserve for one key (ghosts in)."""
        return self.row_size_bytes * self.tail_blocks

    # -- invariants -------------------------------------------------------

    def validate_hash_granularity(self, unit: int) -> None:
        """Check that hash boundaries land on this group's block grid.

        Mirrors the joint constraint enforced today at hma_connector.py
        L534-538 (``max_tb % unit == 0``) and the per-kind divisibility of
        connector v2 (FA ``tb`` and ``unit`` must divide each other).
        """
        tb = self.token_block_size
        if self.kind is GroupKind.FULL_ATTENTION:
            if unit % tb and tb % unit:
                raise ValueError(
                    f"group {self.name!r}: unit={unit} and tb={tb} must divide "
                    "each other for FA groups"
                )
        else:
            if tb % unit:
                raise ValueError(
                    f"group {self.name!r}: tb={tb} must be a multiple of "
                    f"unit={unit} so dump boundaries fall on the block grid"
                )


def tail_blocks_ceil(tail_tokens: int, token_block_size: int) -> int:
    """General form: keep partial head blocks (connector v2 semantics).

    A window of 96 tokens over tb=64 yields 2 rows, the head one holding
    32 tokens — the behaviour HMA's floor variant loses for general shapes.
    """
    if tail_tokens <= 0:
        return 0
    return math.ceil(tail_tokens / token_block_size)


def tail_blocks_floor(tail_tokens: int, token_block_size: int) -> int:
    """FAWA/DSV4 closed form (hma_connector.py L523: ``max(tail // tb, 1)``).

    The remainder is absorbed as a tail offset inside the last block
    (``_extract_wa_ptr`` L1275-1283). A zero tail still yields one row;
    such groups produce no records and the callers skip them.
    """
    return max(tail_tokens // token_block_size, 1)


# ---------------------------------------------------------------------------
# Shared dump/load selection rule
# ---------------------------------------------------------------------------


def boundary_range(token_start: int, token_end: int, unit: int) -> range:
    """Hash boundaries k whose token span ``[k*unit, (k+1)*unit)`` lies fully
    inside ``[token_start, token_end)``.

    This is the one rule both directions use: v1 re-derives it three times
    (base L2034-2049, hla L1171-1240, hma L1051-1092); connector v2 proved
    dump and load must select identical keys.
    """
    if token_end <= token_start or unit <= 0:
        return range(0)
    first = -(-token_start // unit)  # ceil: first boundary fully covered
    last = token_end // unit  # floor: exclusive end of fully covered boundaries
    return range(first, max(last, first))


def select_blocks(
    spec: GroupSpec,
    boundaries: Sequence[int],
    unit: int,
) -> Iterator[list[BlockRef]]:
    """Yield, per boundary, the group's rows as :class:`BlockRef` entries.

    FA: one row anchored at the block containing the boundary's last token
    (hma L1012-1015); ``token_offset`` positions the boundary inside the
    block when ``unit < tb``. WA: ``tail_blocks`` rows ending at that block,
    head row carrying the partial span when the tail is not block aligned
    (hma L992-1003). STATE: one page per boundary.

    Which boundaries to pass is *policy*, not part of the rule: FA/STATE
    dumps pass every boundary in the interval; WA dumps pass every boundary
    (block-wise) or only the newest one (chunk-wise); WA loads pass only the
    newest boundary ("always fetch the full WA tail on load", hma L1004-1011
    and L1346).
    """
    tb = spec.token_block_size
    for k in boundaries:
        last_token = (k + 1) * unit - 1
        anchor = last_token // tb
        if spec.kind is GroupKind.FULL_ATTENTION:
            offset = (k * unit) % tb if unit < tb else 0
            yield [BlockRef(spec.group_id, k, 0, anchor, offset)]
            continue
        if spec.kind is GroupKind.STATE:
            yield [BlockRef(spec.group_id, k, 0, anchor)]
            continue
        rows = []
        for row in range(spec.tail_blocks):
            block_index = anchor - (spec.tail_blocks - 1) + row
            offset = spec.tail_tokens % tb if row == 0 else 0
            rows.append(BlockRef(spec.group_id, k, row, block_index, offset))
        yield rows


# ---------------------------------------------------------------------------
# Store-boundary compilation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardSchema:
    """One store instance: a fixed ``tensor_size_list`` compiled from groups.

    Stores differ *only* in their schema class today (FA vs WA namespace,
    sizes and aggregation flags — hma L584-743); making that explicit lets
    the connector hold N stores derived mechanically instead of hardcoded
    pairs.

    ``uniform_rows=True`` declares the row-per-shard contract used by the
    layerwise path: every row becomes one ``shard_index`` and all rows must
    share the same width (hla L1599-1620). Bulk stores flatten all rows into
    a single shard, where per-row widths may differ across groups.
    """

    name: str
    rows: tuple[tuple[SegmentSchema, ...], ...]
    uniform_rows: bool = False
    io_alignment: int = IO_ALIGNMENT

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError(f"shard {self.name!r}: empty schema")
        if self.uniform_rows:
            widths = {sum(s.size_bytes for s in row) for row in self.rows}
            if len(widths) > 1:
                raise ValueError(
                    f"shard {self.name!r}: uniform_rows requires one fixed row "
                    f"width, got {sorted(widths)}"
                )

    @property
    def tensor_size_list(self) -> list[int]:
        """Flattened per-slot sizes; ghosts included (fixed record length)."""
        return [s.size_bytes for row in self.rows for s in row]

    @property
    def real_size_list(self) -> list[int]:
        """Bulk-path payload sizes; ghosts dropped (base L1283 / L818-819)."""
        return [s.size_bytes for row in self.rows for s in row if not s.ghost]

    @property
    def shard_size(self) -> int:
        return round_up(sum(self.tensor_size_list), self.io_alignment)

    @property
    def ghost_slots(self) -> tuple[tuple[int, SegmentSchema], ...]:
        """(flat index, segment) of ghost slots, for registration filtering."""
        return tuple((i, s) for i, s in self._iter_slots() if s.ghost)

    def _iter_slots(self) -> Iterator[tuple[int, SegmentSchema]]:
        index = 0
        for row in self.rows:
            for segment in row:
                yield index, segment
                index += 1


def round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def shard_from_groups(
    name: str,
    groups: Sequence[GroupSpec],
    *,
    drop_ghosts: bool = False,
    uniform_rows: bool = False,
) -> ShardSchema:
    """Compile groups into one store schema.

    Row order follows the store contract: per group, ``tail_blocks`` rows of
    that group's segment schema (hma ``_store_tensor_size_list`` L793-825).
    ``drop_ghosts`` materializes the bulk variant where schema padding is
    removed before flattening; the row/layerwise variant keeps ghosts so
    every row stays rectangular (base L498-545, L1261-1278).
    """
    compiled_rows: list[tuple[SegmentSchema, ...]] = []
    for group in groups:
        row = group.segments
        if drop_ghosts:
            row = tuple(s for s in row if not s.ghost)
        if not row:
            raise ValueError(
                f"shard {name!r}: group {group.name!r} has no real segments"
            )
        for _ in range(group.tail_blocks):
            compiled_rows.append(row)
    return ShardSchema(name, tuple(compiled_rows), uniform_rows=uniform_rows)
