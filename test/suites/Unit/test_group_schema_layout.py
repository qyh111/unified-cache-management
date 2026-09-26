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
        self.layout = GroupSchemaLayout(
            make_kvcaches(),
            {"use_layerwise": False},
            make_vllm_config(),
            make_kv_cache_config(),
        )

    def test_arrays_reflect_registered_tensors(self):
        self.assertFalse(self.layout.use_layerwise)
        self.assertEqual(self.layout.base_ptrs.shape, (2,))
        for index, name in enumerate(LAYER_NAMES):
            tensor = make_kvcaches()[name]
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
        tensor = make_kvcaches()[LAYER_NAMES[0]]
        self.assertEqual(
            int(addrs[0][0]), tensor.data_ptr() + 3 * tensor.stride(0) * 2
        )
        self.assertEqual(
            int(addrs[1][0]), tensor.data_ptr() + 5 * tensor.stride(0) * 2
        )


class TestGroupSchemaLayoutLayerwise(unittest.TestCase):
    def setUp(self):
        self.layout = GroupSchemaLayout(
            make_kvcaches(),
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
            tensor = make_kvcaches()[name]
            self.assertEqual(
                int(addrs[index][0][0]),
                tensor.data_ptr() + 2 * tensor.stride(0) * 2,
            )


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
