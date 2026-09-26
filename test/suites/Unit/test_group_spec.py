# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
"""Unit tests for ucm.integration.vllm.group_spec (pure stdlib, no vLLM).

Run directly (no pytest required, mirrors the v2 test convention):

    PYTHONPATH=. python test/suites/Unit/test_group_spec.py
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


def _load_group_spec():
    """Import group_spec; fall back to direct file load when ucm's runtime
    dependencies (e.g. wrapt via ucm/__init__.py) are not installed."""
    try:
        from ucm.integration.vllm import group_spec

        return group_spec
    except ImportError:
        root = Path(__file__).resolve().parents[3]
        path = root / "ucm" / "integration" / "vllm" / "group_spec.py"
        spec = importlib.util.spec_from_file_location("group_spec_standalone", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


_gs = _load_group_spec()

BlockRef = _gs.BlockRef
GroupKind = _gs.GroupKind
GroupSpec = _gs.GroupSpec
IO_ALIGNMENT = _gs.IO_ALIGNMENT
SegmentSchema = _gs.SegmentSchema
ShardSchema = _gs.ShardSchema
boundary_range = _gs.boundary_range
round_up = _gs.round_up
select_blocks = _gs.select_blocks
shard_from_groups = _gs.shard_from_groups
tail_blocks_ceil = _gs.tail_blocks_ceil
tail_blocks_floor = _gs.tail_blocks_floor


def fa_spec(**overrides) -> GroupSpec:
    kwargs = dict(
        group_id=0,
        name="fa",
        kind=GroupKind.FULL_ATTENTION,
        token_block_size=256,
        tail_tokens=0,
        tail_blocks=1,
        segments=(SegmentSchema("k", 512), SegmentSchema("v", 512)),
    )
    kwargs.update(overrides)
    return GroupSpec(**kwargs)


def wa_spec(**overrides) -> GroupSpec:
    kwargs = dict(
        group_id=1,
        name="swa",
        kind=GroupKind.WINDOW_ATTENTION,
        token_block_size=64,
        tail_tokens=64,
        tail_blocks=1,
        segments=(SegmentSchema("k", 128),),
    )
    kwargs.update(overrides)
    return GroupSpec(**kwargs)


class TestGroupSpecInvariants(unittest.TestCase):
    def test_fa_allows_zero_tail(self):
        spec = fa_spec()
        self.assertEqual(spec.record_bytes, 1024)

    def test_fa_records_one_hash_block_of_content(self):
        """FAWA convention: FA group with tail=unit expresses a full block."""
        spec = fa_spec(token_block_size=256, tail_tokens=256, tail_blocks=1)
        self.assertEqual(spec.record_bytes, 1024)

    def test_wa_requires_tail(self):
        with self.assertRaises(ValueError):
            wa_spec(tail_tokens=0)

    def test_state_requires_tail(self):
        with self.assertRaises(ValueError):
            GroupSpec(
                group_id=2,
                name="mamba",
                kind=GroupKind.STATE,
                token_block_size=256,
                tail_tokens=0,
            )

    def test_duplicate_segment_names_rejected(self):
        with self.assertRaises(ValueError):
            fa_spec(
                segments=(SegmentSchema("k", 512), SegmentSchema("k", 256)),
            )

    def test_negative_and_zero_fields_rejected(self):
        with self.assertRaises(ValueError):
            fa_spec(token_block_size=0)
        with self.assertRaises(ValueError):
            wa_spec(tail_tokens=-1)
        with self.assertRaises(ValueError):
            wa_spec(tail_blocks=0)
        with self.assertRaises(ValueError):
            fa_spec(segments=(SegmentSchema("k", -1),))


class TestDerivedQuantities(unittest.TestCase):
    def test_tail_blocks_ceil_keeps_partial_head(self):
        """window=96 over tb=64 -> 2 rows, head row holds 32 tokens (v2)."""
        self.assertEqual(tail_blocks_ceil(96, 64), 2)
        self.assertEqual(tail_blocks_ceil(64, 64), 1)
        self.assertEqual(tail_blocks_ceil(0, 64), 0)

    def test_tail_blocks_floor_is_fawa_closed_form(self):
        """hma L523: max(tail // tb, 1); zero tail still yields one row."""
        self.assertEqual(tail_blocks_floor(96, 64), 1)
        self.assertEqual(tail_blocks_floor(128, 64), 2)
        self.assertEqual(tail_blocks_floor(0, 64), 1)

    def test_window_span(self):
        # WA: tail % tb
        self.assertEqual(wa_spec(tail_tokens=96, token_block_size=64).window_span(64), 32)
        self.assertEqual(wa_spec(tail_tokens=128, token_block_size=64).window_span(64), 0)
        # FA 1:N: unit % tb
        self.assertEqual(fa_spec(token_block_size=4096).window_span(6144), 2048)
        self.assertEqual(fa_spec(token_block_size=4096).window_span(8192), 0)
        self.assertEqual(fa_spec(token_block_size=256).window_span(256), 0)

    def test_ghost_aware_sizes(self):
        spec = fa_spec(
            segments=(
                SegmentSchema("k", 512),
                SegmentSchema("indexer", 128, ghost=True),
            )
        )
        self.assertEqual(spec.row_size_bytes, 640)
        self.assertEqual(spec.real_row_size_bytes, 512)
        self.assertEqual(spec.record_bytes, 640)


class TestHashGranularity(unittest.TestCase):
    def test_fa_divisible_pair_passes(self):
        fa_spec(token_block_size=4096).validate_hash_granularity(128)

    def test_fa_indivisible_pair_rejected(self):
        with self.assertRaises(ValueError):
            fa_spec(token_block_size=64).validate_hash_granularity(100)

    def test_wa_requires_tb_multiple_of_unit(self):
        wa_spec(token_block_size=768).validate_hash_granularity(256)
        with self.assertRaises(ValueError):
            wa_spec(token_block_size=100).validate_hash_granularity(256)


class TestBoundaryRange(unittest.TestCase):
    def test_aligned_interval_covers_four_boundaries(self):
        self.assertEqual(list(boundary_range(0, 1024, 256)), [0, 1, 2, 3])

    def test_partial_boundaries_excluded(self):
        # [10, 300): no boundary fully covered
        self.assertEqual(list(boundary_range(10, 300, 256)), [])
        # [256, 512): exactly boundary 1
        self.assertEqual(list(boundary_range(256, 512, 256)), [1])
        # [257, 768): boundary 1 covers [256,512) which is clipped at the
        # start; only boundary 2 [512,768) lies fully inside
        self.assertEqual(list(boundary_range(257, 768, 256)), [2])

    def test_degenerate_inputs(self):
        self.assertEqual(list(boundary_range(100, 100, 256)), [])
        self.assertEqual(list(boundary_range(200, 100, 256)), [])
        self.assertEqual(list(boundary_range(0, 256, 0)), [])


class TestSelectBlocks(unittest.TestCase):
    def test_fa_aligned(self):
        spec = fa_spec(token_block_size=256)
        (refs,) = list(select_blocks(spec, [5], unit=256))
        self.assertEqual(refs, [BlockRef(0, 5, 0, 5, 0)])

    def test_fa_offset_when_unit_below_tb(self):
        """Ascend C128 style: hash=128 inside tb=4096 blocks (hma L1253)."""
        spec = fa_spec(name="c128", token_block_size=4096)
        (refs,) = list(select_blocks(spec, [10], unit=128))
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].block_index, 0)  # last token 1407 -> block 0
        self.assertEqual(refs[0].token_offset, 1280)  # 10*128 % 4096

    def test_wa_block_wise_rows_end_at_anchor(self):
        """tail=96/tb=64: row0 is the partial head (32 tokens), row1 full."""
        spec = wa_spec(tail_tokens=96, token_block_size=64, tail_blocks=2)
        (refs,) = list(select_blocks(spec, [7], unit=64))
        self.assertEqual(
            refs,
            [
                BlockRef(1, 7, 0, 6, 32),
                BlockRef(1, 7, 1, 7, 0),
            ],
        )

    def test_state_single_page_per_boundary(self):
        spec = GroupSpec(
            group_id=2,
            name="mamba",
            kind=GroupKind.STATE,
            token_block_size=256,
            tail_tokens=256,
            tail_blocks=1,
            segments=(SegmentSchema("state", 4096),),
        )
        (refs,) = list(select_blocks(spec, [3], unit=256))
        self.assertEqual(refs, [BlockRef(2, 3, 0, 3, 0)])


class TestShardSchema(unittest.TestCase):
    def test_bulk_store_flattens_groups_with_uneven_rows(self):
        """FAWA-style store: FA group + WA group, row widths may differ."""
        fa = fa_spec()
        wa = wa_spec(tail_tokens=96, token_block_size=64, tail_blocks=2)
        shard = shard_from_groups("fa", [fa, wa])
        # 1 FA row (2 slots) + 2 WA rows (1 slot each)
        self.assertEqual(shard.tensor_size_list, [512, 512, 128, 128])
        self.assertFalse(shard.uniform_rows)
        self.assertEqual(shard.shard_size, round_up(1280, IO_ALIGNMENT))

    def test_ghosts_kept_in_tensor_size_list_but_dropped_in_real_list(self):
        spec = fa_spec(
            segments=(
                SegmentSchema("k", 512),
                SegmentSchema("indexer", 128, ghost=True),
            )
        )
        bulk = shard_from_groups("bulk", [spec], drop_ghosts=True)
        self.assertEqual(bulk.tensor_size_list, [512])
        self.assertEqual(bulk.real_size_list, [512])
        row_shard = shard_from_groups("rows", [spec], uniform_rows=True)
        self.assertEqual(row_shard.tensor_size_list, [512, 128])
        self.assertEqual(row_shard.real_size_list, [512])
        self.assertEqual(row_shard.ghost_slots, ((1, SegmentSchema("indexer", 128, True)),))

    def test_uniform_rows_violation_rejected(self):
        fa = fa_spec()
        wa = wa_spec(tail_tokens=96, token_block_size=64, tail_blocks=2)
        with self.assertRaises(ValueError):
            shard_from_groups("rows", [fa, wa], uniform_rows=True)

    def test_shard_size_alignment(self):
        spec = fa_spec(segments=(SegmentSchema("k", 100),))
        shard = shard_from_groups("fa", [spec])
        self.assertEqual(shard.shard_size, IO_ALIGNMENT)  # 100 -> 4096

    def test_empty_shard_rejected(self):
        with self.assertRaises(ValueError):
            ShardSchema(name="empty", rows=())

    def test_drop_ghosts_rejects_all_ghost_group(self):
        spec = fa_spec(segments=(SegmentSchema("indexer", 128, ghost=True),))
        with self.assertRaises(ValueError):
            shard_from_groups("fa", [spec], drop_ghosts=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
