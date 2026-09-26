# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
"""Unit tests for group_plan (the per-group dispatch planner).

Needs vllm installed (via group_hash's spec helpers); run on the remote
stacks or any env with vllm:

    PYTHONPATH=. python test/suites/Unit/test_group_plan.py
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from ucm.integration.vllm.group_plan import (
    fa_block_slice,
    plan_dump,
    plan_load,
    state_anchor_slice,
)

LCM = 16


def fa_group(group_id=0):
    return SimpleNamespace(group_id=group_id, block_size=LCM, is_mamba_align=False)


def state_group(group_id=1):
    return SimpleNamespace(
        group_id=group_id, block_size=LCM, is_mamba_align=True
    )


def fake_state_hash(group, seq_len, group_ucm_ids):
    """LCM-aligned boundaries hash; anything else is unresolvable."""
    if seq_len <= 0 or seq_len % LCM:
        return None
    return f"state:{group.group_id}:{seq_len}".encode()


def chains():
    group_ucm_ids = {
        0: [b"f0", b"f1", b"f2", b"f3"],
        1: [b"", b"", b"", b""],
    }
    group_vllm_ids = {0: [1, 2, 3, 4], 1: [1, 2, 3, 4]}
    return group_ucm_ids, group_vllm_ids


class TestFaBlockSlice(unittest.TestCase):
    def test_boundary_aligned_range(self):
        group = fa_group()
        group_ucm_ids, group_vllm_ids = chains()
        sl = fa_block_slice(group, group_ucm_ids[0], group_vllm_ids[0], 16, 48)
        self.assertEqual(sl.ucm_block_ids, [b"f1", b"f2"])
        self.assertEqual(sl.vllm_block_ids, [2, 3])
        self.assertFalse(sl.is_state)

    def test_null_vllm_ids_skipped(self):
        group = fa_group()
        ucm = [b"f0", b"f1", b"f2", b"f3"]
        vllm = [1, 0, 3, 4]  # block 1 is a mamba-align placeholder
        sl = fa_block_slice(group, ucm, vllm, 16, 64)
        self.assertEqual(sl.ucm_block_ids, [b"f1", b"f2", b"f3"])
        self.assertEqual(sl.vllm_block_ids, [3, 4])

    def test_empty_range_returns_empty_slice(self):
        group = fa_group()
        group_ucm_ids, group_vllm_ids = chains()
        sl = fa_block_slice(group, group_ucm_ids[0], group_vllm_ids[0], 44, 44)
        self.assertEqual(sl.size, 0)


class TestStateAnchor(unittest.TestCase):
    def test_dump_anchor_indexes_directly(self):
        group = state_group()
        group_ucm_ids, group_vllm_ids = chains()
        sl = state_anchor_slice(
            group, group_ucm_ids, group_vllm_ids, fake_state_hash, 64, "dump"
        )
        self.assertEqual(sl.vllm_block_ids, [4])  # (64-1)//16 = 3 -> chain[3]
        self.assertEqual(sl.ucm_block_ids, [b"state:1:64"])
        self.assertTrue(sl.is_state)

    def test_load_anchor_reverse_searches_non_null(self):
        group = state_group()
        group_ucm_ids, group_vllm_ids = chains()
        # anchor index 2 points at a null placeholder; reverse search finds 3
        sl = state_anchor_slice(
            group, group_ucm_ids, group_vllm_ids, fake_state_hash, 48, "load"
        )
        self.assertEqual(sl.vllm_block_ids, [3])

    def test_all_null_chain_skips_anchor(self):
        group = state_group()
        group_ucm_ids, _ = chains()
        sl = state_anchor_slice(
            group,
            group_ucm_ids,
            {1: [0, 0, 0, 0]},
            fake_state_hash,
            48,
            "load",
        )
        self.assertIsNone(sl)

    def test_unresolvable_hash_skips_anchor(self):
        group = state_group()
        _, group_vllm_ids = chains()
        sl = state_anchor_slice(
            group,
            {1: [b"", b"", b"", b""]},
            group_vllm_ids,
            lambda g, s, c: None,  # e.g. primary prefix hash missing
            48,
            "load",
        )
        self.assertIsNone(sl)

    def test_non_lcm_boundary_unresolvable(self):
        group = state_group()
        group_ucm_ids, group_vllm_ids = chains()
        sl = state_anchor_slice(
            group, group_ucm_ids, group_vllm_ids, fake_state_hash, 48 + 8, "dump"
        )
        self.assertIsNone(sl)


class TestPlanLoad(unittest.TestCase):
    def test_fa_first_then_state_anchor(self):
        groups = [fa_group(), state_group()]
        group_ucm_ids, group_vllm_ids = chains()
        slices = plan_load(
            groups,
            LCM,
            fake_state_hash,
            group_ucm_ids,
            group_vllm_ids,
            hbm_hit_block_num=1,
            total_hit_block_num=3,
        )
        self.assertEqual([s.group_id for s in slices], [0, 1])
        self.assertEqual([s.is_state for s in slices], [False, True])
        self.assertEqual(slices[0].ucm_block_ids, [b"f1", b"f2"])
        self.assertEqual(slices[1].ucm_block_ids, [b"state:1:48"])
        # MLA full-attn count semantics: state anchors excluded
        self.assertEqual(
            sum(len(s.ucm_block_ids) for s in slices if not s.is_state), 2
        )

    def test_no_state_groups(self):
        groups = [fa_group()]
        group_ucm_ids, group_vllm_ids = chains()
        slices = plan_load(
            groups,
            LCM,
            fake_state_hash,
            group_ucm_ids,
            group_vllm_ids,
            hbm_hit_block_num=0,
            total_hit_block_num=2,
        )
        self.assertEqual(len(slices), 1)
        self.assertEqual(slices[0].vllm_block_ids, [1, 2])


class TestPlanDump(unittest.TestCase):
    def test_aligned_dump_end_anchors_state(self):
        groups = [fa_group(), state_group()]
        group_ucm_ids, group_vllm_ids = chains()
        slices = plan_dump(
            groups,
            LCM,
            fake_state_hash,
            group_ucm_ids,
            group_vllm_ids,
            token_processed=32,
            dump_tok_end=64,
        )
        self.assertEqual([s.group_id for s in slices], [0, 1])
        self.assertEqual(slices[0].ucm_block_ids, [b"f2", b"f3"])
        self.assertEqual(slices[1].ucm_block_ids, [b"state:1:64"])

    def test_unaligned_dump_end_skips_state(self):
        groups = [fa_group(), state_group()]
        group_ucm_ids, group_vllm_ids = chains()
        slices = plan_dump(
            groups,
            LCM,
            fake_state_hash,
            group_ucm_ids,
            group_vllm_ids,
            token_processed=32,
            dump_tok_end=44,
        )
        # FA range [32,44) covers no full block; boundary 32 < first_lcm 48
        self.assertEqual(len(slices), 1)
        self.assertEqual(slices[0].size, 0)

    def test_partial_dump_range(self):
        groups = [fa_group(), state_group()]
        group_ucm_ids, group_vllm_ids = chains()
        slices = plan_dump(
            groups,
            LCM,
            fake_state_hash,
            group_ucm_ids,
            group_vllm_ids,
            token_processed=32,
            dump_tok_end=48,
        )
        # boundary 48 is the first LCM boundary of the dump window
        self.assertEqual(slices[0].ucm_block_ids, [b"f2"])
        self.assertEqual(slices[1].ucm_block_ids, [b"state:1:48"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
