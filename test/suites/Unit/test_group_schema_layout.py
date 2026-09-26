# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
"""Unit tests for GroupSchemaLayout (the v2-pipeline layout facade).

Needs vllm + torch (real spec classes drive parse_kv_cache_config); run
on the remote stacks:

    PYTHONPATH=. python test/suites/Unit/test_group_schema_layout.py
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    SlidingWindowSpec,
)

from ucm.integration.vllm.ucm_connector import GroupSchemaLayout

NB = 8
TB = 16
HEADS = 2
CHANNELS = 8
DTYPE = torch.bfloat16

DEVICE_TYPE = getattr(current_platform, "device_type", None) or "cpu"
LAYER_NAMES = ("model.layers.0.self_attn", "model.layers.1.self_attn")


def make_kv_tensor(base):
    """One layer's KV page: BNHC dense on NPU, BHNC permuted elsewhere."""
    if DEVICE_TYPE == "npu":
        return torch.zeros(
            (NB, TB, HEADS, CHANNELS), dtype=DTYPE
        ).as_strided((NB, TB, HEADS, CHANNELS), (TB * HEADS * CHANNELS, HEADS * CHANNELS, CHANNELS, 1)) + base * 0
    # 0.29-style logical [B, H, N, C] view over [B, N, H, C] memory
    return torch.zeros((NB, TB, HEADS, CHANNELS), dtype=DTYPE).as_strided(
        (NB, HEADS, TB, CHANNELS),
        (TB * HEADS * CHANNELS, CHANNELS, HEADS * CHANNELS, 1),
    )


def make_kv_cache_config(num_groups=1, sliding=False):
    spec = (
        SlidingWindowSpec(
            block_size=TB,
            num_kv_heads=HEADS,
            head_size=CHANNELS,
            dtype=DTYPE,
            sliding_window=TB,
        )
        if sliding
        else FullAttentionSpec(
            block_size=TB, num_kv_heads=HEADS, head_size=CHANNELS, dtype=DTYPE
        )
    )
    groups = [
        SimpleNamespace(layer_names=list(LAYER_NAMES), kv_cache_spec=spec)
        for _ in range(num_groups)
    ]
    return SimpleNamespace(
        num_blocks=NB, kv_cache_groups=groups, kv_cache_tensors=()
    )


def make_vllm_config():
    return SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(num_hidden_layers=2)
        ),
        cache_config=SimpleNamespace(block_size=TB),
    )


def make_kvcaches():
    return {
        name: make_kv_tensor(index * 1000)
        for index, name in enumerate(LAYER_NAMES)
    }


PAGE = TB * HEADS * CHANNELS * DTYPE.itemsize


class TestGroupSchemaLayoutBulk(unittest.TestCase):
    def setUp(self):
        self.caches = make_kvcaches()
        self.layout = GroupSchemaLayout(
            self.caches,
            {"use_layerwise": False},
            make_vllm_config(),
            make_kv_cache_config(),
        )

    def test_arrays_reflect_registered_tensors(self):
        self.assertFalse(self.layout.use_layerwise)
        self.assertEqual(self.layout.base_ptrs.shape, (2,))
        for index, name in enumerate(LAYER_NAMES):
            tensor = self.caches[name]
            self.assertEqual(
                int(self.layout.base_ptrs[index]), tensor.data_ptr()
            )
            self.assertEqual(
                int(self.layout.block_stride_lists[index]),
                tensor.stride(0) * tensor.element_size(),
            )
            self.assertEqual(int(self.layout.tensor_size_lists[index]), PAGE)

    def test_properties(self):
        self.assertEqual(self.layout.tensor_size_list, [PAGE, PAGE])
        self.assertEqual(self.layout.shard_size, 2 * PAGE)
        self.assertEqual(self.layout.block_size, 2 * PAGE)

    def test_extract_block_addrs(self):
        addrs = self.layout.extract_block_addrs([3, 5])
        self.assertEqual(addrs.shape, (2, 2))
        tensor = self.caches[LAYER_NAMES[0]]
        self.assertEqual(
            int(addrs[0][0]), tensor.data_ptr() + 3 * tensor.stride(0) * 2
        )
        self.assertEqual(
            int(addrs[1][0]), tensor.data_ptr() + 5 * tensor.stride(0) * 2
        )


class TestGroupSchemaLayoutLayerwise(unittest.TestCase):
    def setUp(self):
        self.caches = make_kvcaches()
        self.layout = GroupSchemaLayout(
            self.caches,
            {"use_layerwise": True},
            make_vllm_config(),
            make_kv_cache_config(),
        )

    def test_rows_uniform(self):
        self.assertTrue(self.layout.use_layerwise)
        self.assertEqual(self.layout.tensor_size_lists.shape, (2, 1))
        self.assertEqual(
            self.layout.tensor_size_lists.tolist(), [[PAGE], [PAGE]]
        )
        self.assertEqual(self.layout.tensor_size_list, [PAGE])
        self.assertEqual(self.layout.shard_size, PAGE)

    def test_extract_layer_first(self):
        addrs = self.layout.extract_block_addrs([2], layer_first=True)
        self.assertEqual(addrs.shape, (2, 1, 1))
        for index, name in enumerate(LAYER_NAMES):
            tensor = self.caches[name]
            self.assertEqual(
                int(addrs[index][0][0]),
                tensor.data_ptr() + 2 * tensor.stride(0) * 2,
            )


class TestMultiGroupHybrid(unittest.TestCase):
    """FA + mamba State groups through one store: per-group resolution."""

    STATE_PAGE = 12  # shapes ((4,), (8,)) int8

    def make_hybrid_config(self):
        fa_spec = FullAttentionSpec(
            block_size=TB, num_kv_heads=HEADS, head_size=CHANNELS, dtype=DTYPE
        )
        mamba_spec = MambaSpec(
            shapes=((4,), (8,)),
            dtypes=(torch.int8, torch.int8),
            block_size=TB,
            mamba_cache_mode="align",
        )
        groups = [
            SimpleNamespace(layer_names=list(LAYER_NAMES), kv_cache_spec=fa_spec),
            SimpleNamespace(
                layer_names=("model.layers.2.mamba", "model.layers.3.mamba"),
                kv_cache_spec=mamba_spec,
            ),
        ]
        return SimpleNamespace(
            num_blocks=NB, kv_cache_groups=groups, kv_cache_tensors=()
        )

    def make_hybrid_caches(self):
        caches = make_kvcaches()
        for index, name in enumerate(("model.layers.2.mamba", "model.layers.3.mamba")):
            caches[name] = torch.zeros((NB, self.STATE_PAGE), dtype=torch.int8)
        return caches

    def setUp(self):
        self.caches = self.make_hybrid_caches()
        self.layout = GroupSchemaLayout(
            self.caches,
            {"use_layerwise": False},
            make_vllm_config(),
            self.make_hybrid_config(),
        )

    def test_union_slot_table(self):
        schema = self.layout.store_schema
        self.assertEqual(len(schema.slots), 4)
        self.assertEqual(
            schema.tensor_size_list,
            (PAGE, PAGE, self.STATE_PAGE, self.STATE_PAGE),
        )
        self.assertEqual(schema.record_bytes, 2 * PAGE + 2 * self.STATE_PAGE)
        self.assertEqual([s.route for s in schema.slots], ["FA", "FA", "State", "State"])

    def test_resolve_group_fa_key_ghosts_state_slots(self):
        caches = self.caches
        offsets, ptrs, sizes = self.layout.resolve_group(0, [3])
        self.assertEqual(sizes.tolist(), [PAGE, PAGE, self.STATE_PAGE, self.STATE_PAGE])
        self.assertEqual(
            ptrs.tolist(),
            [
                caches[LAYER_NAMES[0]].data_ptr() + 3 * PAGE,
                caches[LAYER_NAMES[1]].data_ptr() + 3 * PAGE,
                0,
                0,
            ],
        )
        self.assertEqual(
            offsets.tolist(), [0, PAGE, 2 * PAGE, 2 * PAGE + self.STATE_PAGE]
        )
        self.assertEqual(offsets[-1] + sizes[-1], schema_record := self.layout.store_schema.record_bytes)
        self.assertEqual(schema_record, 2 * PAGE + 2 * self.STATE_PAGE)

    def test_resolve_group_state_key_ghosts_fa_slots(self):
        caches = self.caches
        m0 = caches["model.layers.2.mamba"]
        m1 = caches["model.layers.3.mamba"]
        offsets, ptrs, sizes = self.layout.resolve_group(1, [4])
        self.assertEqual(
            ptrs.tolist(),
            [0, 0, m0.data_ptr() + 4 * self.STATE_PAGE, m1.data_ptr() + 4 * self.STATE_PAGE],
        )
        self.assertEqual(
            offsets.tolist(), [0, PAGE, 2 * PAGE, 2 * PAGE + self.STATE_PAGE]
        )
        self.assertEqual(offsets[-1] + sizes[-1], self.layout.store_schema.record_bytes)

    def test_flat_extract_raises_for_multi_group(self):
        with self.assertRaises(ValueError):
            self.layout.extract_block_addrs([3])


class TestScopeGuards(unittest.TestCase):
    def test_multi_group_rejected(self):
        with self.assertRaises(NotImplementedError):
            GroupSchemaLayout(
                make_kvcaches(),
                {"use_layerwise": False},
                make_vllm_config(),
                make_kv_cache_config(num_groups=2),
            )

    def test_sliding_group_rejected(self):
        with self.assertRaises(NotImplementedError):
            GroupSchemaLayout(
                make_kvcaches(),
                {"use_layerwise": False},
                make_vllm_config(),
                make_kv_cache_config(sliding=True),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
