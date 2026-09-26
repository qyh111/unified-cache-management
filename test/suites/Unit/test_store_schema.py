# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
"""Unit tests for the store/layer-shard schema composition layer.

Same fake-tensor approach as test_layout_base.py: numpy only, no torch or
vllm. Run directly:

    PYTHONPATH=. python test/suites/Unit/test_store_schema.py
"""

from __future__ import annotations

import types
import unittest

from test_layout_base import (
    DuckGroup,
    DuckLayer,
    FakeTensor,
    NB,
    TB,
)

from ucm.integration.vllm.layout.group import KVCacheGroupLayout
from ucm.integration.vllm.layout.record import GroupRecordLayout
from ucm.integration.vllm.layout.schema import LayerShardSchema, StoreSchema

PAGE = TB * 2 * 3 * 2  # 192: one dense attention block page
STATE_PAGE = 64


def attention_layer(name, index, base):
    layer = DuckLayer(name, index, NB, TB)
    tensor = FakeTensor(
        shape=(NB, TB, 2, 3),
        strides=(TB * 2 * 3, 2 * 3, 3, 1),
        element_size=2,
        base=base,
    )
    return layer, tensor


def state_layer(name, index, base):
    layer = DuckLayer(name, index, NB, STATE_PAGE)
    tensor = FakeTensor(
        shape=(NB, STATE_PAGE), strides=(STATE_PAGE, 1), element_size=1, base=base
    )
    return layer, tensor


def make_hybrid():
    """qwen-like hybrid: FA group + mamba State group, uniform tb=16."""
    fa_layers = [attention_layer("l0", 0, 1000), attention_layer("l1", 1, 5000)]
    st_layers = [state_layer("m0", 2, 7000), state_layer("m1", 3, 8000)]
    fa_group = DuckGroup(group_id=0, token_block_size=TB, layers=tuple(l for l, _ in fa_layers))
    st_group = DuckGroup(
        group_id=1,
        token_block_size=TB,
        layers=tuple(l for l, _ in st_layers),
        is_state_snapshot=True,
        tail_tokens=TB,
        tail_blocks=1,
    )
    fa_layout = KVCacheGroupLayout(
        fa_group,
        {"l0": fa_layers[0][1], "l1": fa_layers[1][1]},
        device_type="npu",
    )
    st_layout = KVCacheGroupLayout(
        st_group,
        {"m0": st_layers[0][1], "m1": st_layers[1][1]},
        device_type="npu",
    )
    records = {
        0: GroupRecordLayout.build(fa_group, fa_layout, TB),
        1: GroupRecordLayout.build(st_group, st_layout, TB),
    }
    routes = (("FA", (0,)), ("State", (1,)))
    return fa_group, st_group, fa_layout, st_layout, records, routes


class TestStoreSchema(unittest.TestCase):
    def test_slot_table_and_sizes(self):
        *_, records, routes = make_hybrid()
        schema = StoreSchema.build(routes, records)
        self.assertEqual([s.route for s in schema.slots], ["FA", "FA", "State", "State"])
        self.assertEqual(
            schema.tensor_size_list, (PAGE, PAGE, STATE_PAGE, STATE_PAGE)
        )
        self.assertEqual(schema.record_bytes, 2 * PAGE + 2 * STATE_PAGE)
        self.assertEqual(schema.shard_size, 4096)
        self.assertEqual(schema.group_spans[0], (0, 2))
        self.assertEqual(schema.group_spans[1], (2, 2))
        self.assertEqual(schema.group_bases[1], 2 * PAGE)
        self.assertEqual(schema.kind_masks["FA"].tolist(), [True, True, False, False])
        self.assertEqual(schema.kind_masks["State"].tolist(), [False, False, True, True])

    def test_resolve_fa_key_ghosts_state_slots(self):
        *_, records, routes = make_hybrid()
        schema = StoreSchema.build(routes, records)
        offsets, ptrs, sizes = schema.resolve("FA", 2, {0: [3, 5]})
        self.assertEqual(sizes.tolist(), [PAGE, PAGE, STATE_PAGE, STATE_PAGE] * 2)
        # key 0
        self.assertEqual(ptrs[:4].tolist(), [1000 + 3 * PAGE, 5000 + 3 * PAGE, 0, 0])
        self.assertEqual(offsets[:4].tolist(), [0, PAGE, 2 * PAGE, 2 * PAGE + STATE_PAGE])
        # key 1
        self.assertEqual(ptrs[4:6].tolist(), [1000 + 5 * PAGE, 5000 + 5 * PAGE])
        # every key's record spans the same fixed width
        self.assertEqual(offsets[-1] + sizes[-1], schema.record_bytes)

    def test_resolve_state_key_ghosts_fa_slots(self):
        *_, records, routes = make_hybrid()
        schema = StoreSchema.build(routes, records)
        offsets, ptrs, sizes = schema.resolve("State", 1, {1: [4]})
        self.assertEqual(ptrs.tolist(), [0, 0, 7000 + 4 * STATE_PAGE, 8000 + 4 * STATE_PAGE])
        self.assertEqual(offsets.tolist(), [0, PAGE, 2 * PAGE, 2 * PAGE + STATE_PAGE])
        self.assertEqual(offsets[-1] + sizes[-1], schema.record_bytes)

    def test_unknown_kind_rejected(self):
        *_, records, routes = make_hybrid()
        schema = StoreSchema.build(routes, records)
        with self.assertRaises(ValueError):
            schema.resolve("WA", 1, {})

    def test_wa_partial_row_width_in_schema(self):
        """A sliding group's slot width is its partial head row, not the
        whole page: tail=12 over tb=16 -> 144B, and its pointer skips to
        the window head (tb - tail = 4 tokens in)."""
        wa_layer, wa_tensor = attention_layer("w0", 4, 9000)
        wa_group = DuckGroup(
            group_id=2,
            token_block_size=TB,
            layers=(wa_layer,),
            is_sliding_window=True,
            tail_tokens=12,
            tail_blocks=1,
        )
        wa_layout = KVCacheGroupLayout(
            wa_group, {"w0": wa_tensor}, device_type="npu"
        )
        records = dict(make_hybrid()[4])
        records[2] = GroupRecordLayout.build(wa_group, wa_layout, TB)
        routes = (("FA", (0,)), ("WA", (2,)), ("State", (1,)))
        schema = StoreSchema.build(routes, records)
        self.assertEqual(
            schema.tensor_size_list, (PAGE, PAGE, 12 * 12, STATE_PAGE, STATE_PAGE)
        )
        self.assertEqual(schema.record_bytes, 2 * PAGE + 144 + 2 * STATE_PAGE)
        offsets, ptrs, sizes = schema.resolve("WA", 1, {2: [2]})
        self.assertEqual(ptrs.tolist(), [0, 0, 9000 + 2 * PAGE + 4 * 12, 0, 0])
        self.assertEqual(offsets.tolist(), [0, PAGE, 2 * PAGE, 2 * PAGE + 144, 2 * PAGE + 144 + STATE_PAGE])
        self.assertEqual(offsets[-1] + sizes[-1], schema.record_bytes)


class TestLayerShardSchema(unittest.TestCase):
    def test_shard_slots_uniform_across_layers(self):
        *_, records, routes = make_hybrid()
        schema = LayerShardSchema.build(routes, records)
        self.assertEqual(schema.tensor_size_list, (PAGE, STATE_PAGE))
        self.assertEqual(schema.shard_size, 4096)
        self.assertEqual(schema.layer_ids, (0, 1, 2, 3))
        # every layer resolves to the same slot size list
        for layer_id, blocks in ((0, {0: [3]}), (2, {1: [4]})):
            _, sizes = schema.resolve_layer(layer_id, 1, blocks)
            self.assertEqual(sizes.tolist(), [PAGE, STATE_PAGE])

    def test_fa_layer_real_and_state_ghost(self):
        *_, records, routes = make_hybrid()
        schema = LayerShardSchema.build(routes, records)
        ptrs, sizes = schema.resolve_layer(0, 1, {0: [3]})
        self.assertEqual(ptrs.tolist(), [1000 + 3 * PAGE, 0])
        self.assertEqual(sizes.tolist(), [PAGE, STATE_PAGE])

    def test_state_layer_real_and_fa_ghost(self):
        *_, records, routes = make_hybrid()
        schema = LayerShardSchema.build(routes, records)
        ptrs, sizes = schema.resolve_layer(2, 1, {1: [4]})
        self.assertEqual(ptrs.tolist(), [0, 7000 + 4 * STATE_PAGE])
        self.assertEqual(sizes.tolist(), [PAGE, STATE_PAGE])

    def test_group_internal_ghost_auto_align(self):
        """MiniMax pattern: sparse layer owns an extra indexer column;
        dense layers auto-ghost the tail so shards stay uniform."""
        fa_layers = [
            attention_layer("l0", 0, 1000),
            attention_layer("l1", 1, 5000),
        ]
        # give l0 a second column: a small indexer cache (4B per state row)
        indexer = FakeTensor(
            shape=(NB, TB, 1, 4), strides=(TB * 4, 4, 4, 1), element_size=1, base=3000
        )
        kv_caches = {
            "l0": (fa_layers[0][1], indexer),
            "l1": fa_layers[1][1],
        }
        fa_group = DuckGroup(
            group_id=0, token_block_size=TB, layers=(fa_layers[0][0], fa_layers[1][0])
        )
        fa_layout = KVCacheGroupLayout(fa_group, kv_caches, device_type="npu")
        self.assertEqual(len(fa_layout.layer_names), 3)  # k, indexer, k
        records = {0: GroupRecordLayout.build(fa_group, fa_layout, TB)}
        schema = LayerShardSchema.build((("FA", (0,)),), records)
        self.assertEqual(schema.tensor_size_list, (PAGE, TB * 4))
        # dense layer l1 owns only the first position; indexer slot ghosts
        ptrs, sizes = schema.resolve_layer(1, 1, {0: [5]})
        self.assertEqual(ptrs.tolist(), [5000 + 5 * PAGE, 0])
        self.assertEqual(sizes.tolist(), [PAGE, TB * 4])
        # sparse layer l0 owns both
        ptrs, sizes = schema.resolve_layer(0, 1, {0: [5]})
        self.assertEqual(ptrs.tolist(), [1000 + 5 * PAGE, 3000 + 5 * TB * 4])
        self.assertEqual(sizes.tolist(), [PAGE, TB * 4])

    def test_prefix_mismatch_rejected(self):
        layer0, tensor0 = attention_layer("l0", 0, 1000)
        layer1 = DuckLayer("l1", 1, NB, TB)
        # l1's page width differs from l0's with no prefix excuse
        broken = FakeTensor(
            shape=(NB, TB, 2, 4), strides=(TB * 2 * 4, 2 * 4, 4, 1),
            element_size=2, base=5000,
        )
        fa_group = DuckGroup(group_id=0, token_block_size=TB, layers=(layer0, layer1))
        fa_layout = KVCacheGroupLayout(
            fa_group,
            {"l0": tensor0, "l1": broken},
            device_type="npu",
        )
        records = {0: GroupRecordLayout.build(fa_group, fa_layout, TB)}
        with self.assertRaises(ValueError):
            LayerShardSchema.build((("FA", (0,)),), records)


if __name__ == "__main__":
    unittest.main(verbosity=2)
