# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
"""Unit tests for ucm.integration.vllm.layout (ported connector-v2 base).

The layout layer is duck-typed over runtime tensors (it only reads
shape/stride/element_size/data_ptr), so these tests use lightweight fakes
and need nothing beyond numpy. Run directly:

    PYTHONPATH=. python test/suites/Unit/test_layout_base.py

When ucm's runtime dependencies are absent (no wrapt/vllm), the loader
stubs the package chain and imports only the numpy-only submodules
(view/group/record); the spec parser (vllm-dependent) is exercised on the
remote stacks together with the model-check regression.
"""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path


def _load_layout():
    """Import the real package when ucm's deps exist; otherwise stub the
    package chain and load only the numpy-only submodules."""
    try:
        from ucm.integration.vllm import layout

        return layout
    except Exception:
        pass
    root = Path(__file__).resolve().parents[3]
    pkg_dir = root / "ucm" / "integration" / "vllm" / "layout"
    for name, path in (
        ("ucm", None),
        ("ucm.integration", None),
        ("ucm.integration.vllm", None),
        ("ucm.integration.vllm.layout", pkg_dir),
    ):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)] if path else []
            sys.modules[name] = module
    view = importlib.import_module("ucm.integration.vllm.layout.view")
    group = importlib.import_module("ucm.integration.vllm.layout.group")
    record = importlib.import_module("ucm.integration.vllm.layout.record")
    return types.SimpleNamespace(view=view, group=group, record=record)


L = _load_layout()


class FakeTensor:
    """The tensor surface build_layer_view actually reads."""

    def __init__(self, shape, strides, element_size=1, base=0):
        self.shape = tuple(shape)
        self._strides = tuple(strides)
        self._element_size = element_size
        self._base = base

    def stride(self, index):
        return self._strides[index]

    def element_size(self):
        return self._element_size

    def data_ptr(self):
        return self._base


@dataclass(frozen=True)
class DuckLayer:
    """UCMLayerSpec's field surface (spec.py needs vllm to import)."""

    layer_name: str
    layer_index: int
    num_blocks: int
    storage_block_size: int
    kv_cache_spec: object = None
    descriptor: object = None
    descriptor_position: int = 0


@dataclass(frozen=True)
class DuckGroup:
    """UCMKVCacheGroupInfo's field surface for the layout layer."""

    group_id: int
    token_block_size: int
    layers: tuple
    is_state_snapshot: bool = False
    is_sliding_window: bool = False
    tail_tokens: object = None
    tail_blocks: int = 1


NB = 8
TB = 16
HEADS = 2
CHANNELS = 3
ES = 2
PAGE = TB * HEADS * CHANNELS * ES  # 192: one dense block page


def dense_layer(name, index, base=0):
    """NPU BNHC dense attention: [nb, tb, H, C]."""
    layer = DuckLayer(
        layer_name=name,
        layer_index=index,
        num_blocks=NB,
        storage_block_size=TB,
    )
    tensor = FakeTensor(
        shape=(NB, TB, HEADS, CHANNELS),
        strides=(TB * HEADS * CHANNELS, HEADS * CHANNELS, CHANNELS, 1),
        element_size=ES,
        base=base,
    )
    return layer, tensor


def make_two_layer_group():
    layer0, tensor0 = dense_layer("l0", 0, base=1000)
    layer1, tensor1 = dense_layer("l1", 1, base=5000)
    group = DuckGroup(group_id=0, token_block_size=TB, layers=(layer0, layer1))
    layout = L.group.KVCacheGroupLayout(
        group, {"l0": tensor0, "l1": tensor1}, device_type="npu"
    )
    return group, layout


class TestLayerViews(unittest.TestCase):
    def test_npu_dense_attention_whole_block(self):
        layer, tensor = dense_layer("a", 0)
        view = L.view.build_layer_view(
            tensor, layer, state_snapshot=False, device_type="npu"
        )
        self.assertEqual(len(view.segments), 1)
        segment = view.segments[0]
        self.assertEqual(segment.block_stride_bytes, PAGE)
        self.assertEqual(segment.states_per_block, TB)
        self.assertEqual(segment.bytes_per_state, HEADS * CHANNELS * ES)
        self.assertEqual(segment.payload_bytes, PAGE)

    def test_cuda_bhnc_permuted_view(self):
        """0.29 [B,H,N,C] logical view over [B,N,H,C] memory."""
        layer = DuckLayer("a", 0, NB, TB)
        tensor = FakeTensor(
            shape=(NB, HEADS, TB, CHANNELS),
            strides=(TB * HEADS * CHANNELS, CHANNELS, HEADS * CHANNELS, 1),
            element_size=ES,
            base=0,
        )
        view = L.view.build_layer_view(
            tensor, layer, state_snapshot=False, device_type="cuda"
        )
        self.assertEqual(len(view.segments), 1)
        segment = view.segments[0]
        self.assertEqual(segment.block_stride_bytes, PAGE)
        self.assertEqual(segment.bytes_per_state, HEADS * CHANNELS * ES)
        self.assertEqual(segment.payload_bytes, PAGE)

    def test_hnc_head_split(self):
        """Layer-major heads: token stride == channels, heads far apart."""
        layer = DuckLayer("a", 0, NB, TB)
        tensor = FakeTensor(
            shape=(NB, TB, HEADS, CHANNELS),
            strides=(TB * HEADS * CHANNELS, CHANNELS, TB * CHANNELS, 1),
            element_size=ES,
            base=0,
        )
        view = L.view.build_layer_view(
            tensor, layer, state_snapshot=False, device_type="npu"
        )
        self.assertEqual(len(view.segments), HEADS)
        for head, segment in enumerate(view.segments):
            self.assertEqual(segment.base_ptr, head * TB * CHANNELS * ES)
            self.assertEqual(segment.block_stride_bytes, PAGE)
            self.assertEqual(segment.states_per_block, TB)
            self.assertEqual(segment.bytes_per_state, CHANNELS * ES)
            self.assertEqual(segment.payload_bytes, TB * CHANNELS * ES)

    def test_state_combined_page(self):
        """int8 padded page: payload == page_stride; content rides inside."""
        page = 64
        spec = types.SimpleNamespace(
            shapes=((4,), (8,)),
            dtypes=(
                types.SimpleNamespace(itemsize=1),
                types.SimpleNamespace(itemsize=2),
            ),
            page_size_bytes=page,
        )
        layer = DuckLayer("mamba.0", 0, NB, page, kv_cache_spec=spec)
        tensor = FakeTensor(
            shape=(NB, page), strides=(page, 1), element_size=1, base=0
        )
        view = L.view.build_layer_view(
            tensor, layer, state_snapshot=True, device_type="npu"
        )
        segment = view.segments[0]
        self.assertEqual(segment.block_stride_bytes, page)
        self.assertEqual(segment.states_per_block, 1)
        self.assertEqual(segment.bytes_per_state, 4 * 1 + 8 * 2)
        self.assertEqual(segment.payload_bytes, 4 * 1 + 8 * 2)


class TestGroupLayoutAccess(unittest.TestCase):
    def test_columns(self):
        _, layout = make_two_layer_group()
        self.assertEqual(layout.layer_names, ("l0", "l1"))
        self.assertEqual(list(layout.base_ptrs), [1000, 5000])
        self.assertEqual(list(layout.block_strides), [PAGE, PAGE])
        self.assertEqual(list(layout.states_per_block), [TB, TB])
        self.assertEqual(list(layout.state_strides), [12, 12])
        self.assertIsNone(layout.block_first)

    def test_whole_block_resolve(self):
        _, layout = make_two_layer_group()
        access = layout.compile_access()
        ptrs, sizes = access.resolve([3, 5])
        self.assertEqual(
            ptrs.tolist(),
            [
                [1000 + 3 * PAGE, 5000 + 3 * PAGE],
                [1000 + 5 * PAGE, 5000 + 5 * PAGE],
            ],
        )
        self.assertEqual(sizes.tolist(), [[PAGE, PAGE], [PAGE, PAGE]])

    def test_partial_token_resolve(self):
        _, layout = make_two_layer_group()
        access = layout.compile_access(token_offsets=4, token_counts=6)
        ptrs, sizes = access.resolve([7])
        self.assertEqual(
            ptrs.tolist(),
            [[1000 + 7 * PAGE + 4 * 12, 5000 + 7 * PAGE + 4 * 12]],
        )
        self.assertEqual(sizes.tolist(), [[6 * 12, 6 * 12]])

    def test_block_range_checked(self):
        _, layout = make_two_layer_group()
        with self.assertRaises(ValueError):
            layout.compile_access().resolve([NB])

    def test_extract_segments(self):
        _, layout = make_two_layer_group()
        ptrs, sizes = layout.extract_segments([2], [4], [10])
        self.assertEqual(
            ptrs.tolist(),
            [[1000 + 2 * PAGE + 4 * 12, 5000 + 2 * PAGE + 4 * 12]],
        )
        self.assertEqual(sizes.tolist(), [[6 * 12, 6 * 12]])


class TestBlockFirstSpan(unittest.TestCase):
    """Declared interleaved placement: one group == one IO span per block."""

    LAYER_STRIDE = PAGE
    BLOCK_STRIDE = 2 * PAGE  # declarations are byte quantities

    def make_layout(self):
        descriptor = L.group.TensorDescriptor(
            layers=("l0", "l1"),
            offset=0,
            layer_stride=self.LAYER_STRIDE,
            block_stride=self.BLOCK_STRIDE,
        )
        layer0 = DuckLayer(
            "l0", 0, NB, TB, descriptor=descriptor, descriptor_position=0
        )
        layer1 = DuckLayer(
            "l1", 1, NB, TB, descriptor=descriptor, descriptor_position=1
        )
        group = DuckGroup(group_id=0, token_block_size=TB, layers=(layer0, layer1))
        # row stride padded to the block stride (rows_per_block == 1, so the
        # dense-row check does not apply and the padding rides inside);
        # strides are in elements, descriptor quantities in bytes.
        strides = (self.BLOCK_STRIDE // ES, HEADS * CHANNELS, CHANNELS, 1)
        layout = L.group.KVCacheGroupLayout(
            group,
            {
                "l0": FakeTensor((NB, TB, HEADS, CHANNELS), strides, ES, base=1000),
                "l1": FakeTensor(
                    (NB, TB, HEADS, CHANNELS), strides, ES, base=1000 + PAGE
                ),
            },
            device_type="npu",
        )
        return group, layout

    def test_span_detected(self):
        _, layout = self.make_layout()
        self.assertIsNotNone(layout.block_first)
        self.assertEqual(layout.block_first.base_ptr, 1000)
        self.assertEqual(layout.block_first.block_stride, self.BLOCK_STRIDE)
        self.assertEqual(layout.block_first.block_size_bytes, 2 * PAGE)

    def test_record_merges_whole_blocks(self):
        group, layout = self.make_layout()
        record = L.record.GroupRecordLayout.build(group, layout, TB)
        self.assertTrue(record.merge_whole_blocks)
        self.assertEqual(record.record_bytes, 2 * PAGE)
        offsets, ptrs, sizes, entries = record.resolve([3], key_count=1)
        self.assertEqual(offsets.tolist(), [0])
        self.assertEqual(ptrs.tolist(), [1000 + 3 * self.BLOCK_STRIDE])
        self.assertEqual(sizes.tolist(), [2 * PAGE])
        self.assertEqual(entries, 1)


class TestRecordLayout(unittest.TestCase):
    def test_fa_whole_block_fragmented(self):
        group, layout = make_two_layer_group()
        record = L.record.GroupRecordLayout.build(group, layout, TB)
        self.assertFalse(record.merge_whole_blocks)
        self.assertEqual(record.record_bytes, 2 * PAGE)
        offsets, ptrs, sizes, entries = record.resolve([3], key_count=1)
        self.assertEqual(offsets.tolist(), [0, PAGE])
        self.assertEqual(ptrs.tolist(), [1000 + 3 * PAGE, 5000 + 3 * PAGE])
        self.assertEqual(sizes.tolist(), [PAGE, PAGE])
        self.assertEqual(entries, 2)

    def test_wa_partial_head_row(self):
        """tail=100 over tb=16: 7 rows, row 0 holds the 4-token head tail."""
        group, layout = make_two_layer_group()
        sliding = DuckGroup(
            group_id=group.group_id,
            token_block_size=TB,
            layers=group.layers,
            is_sliding_window=True,
            tail_tokens=100,
            tail_blocks=7,
        )
        record = L.record.GroupRecordLayout.build(sliding, layout, TB)
        self.assertEqual(record.blocks_per_key, 7)
        # row 0: the 4-token partial; rows 1..6 start after it
        self.assertEqual(record.ucm_block_offsets[0].tolist(), [0, 4 * 12])
        self.assertEqual(record.ucm_block_offsets[1].tolist(), [2 * 4 * 12, 2 * 4 * 12 + PAGE])
        self.assertEqual(record.record_bytes, 2 * 4 * 12 + 6 * 2 * PAGE)
        offsets, ptrs, sizes, entries = record.resolve([5] * 7, key_count=1)
        self.assertEqual(sizes.tolist(), [4 * 12, 4 * 12] + [PAGE] * 12)
        self.assertEqual(entries, 7 * 2)
        # row 0 pointers skip to the window head (tb - partial = 12 tokens in)
        self.assertEqual(ptrs[0], 1000 + 5 * PAGE + 12 * 12)
        self.assertEqual(ptrs[7], 1000 + 5 * PAGE)

    def test_state_whole_page_only(self):
        page = 64
        spec = types.SimpleNamespace(page_size_bytes=page)
        layer = DuckLayer("mamba.0", 0, NB, page, kv_cache_spec=spec)
        tensor = FakeTensor((NB, page), (page, 1), 1, base=7000)
        group = DuckGroup(
            group_id=0,
            token_block_size=TB,
            layers=(layer,),
            is_state_snapshot=True,
            tail_tokens=TB,
            tail_blocks=1,
        )
        layout = L.group.KVCacheGroupLayout(
            group, {"mamba.0": tensor}, device_type="npu"
        )
        record = L.record.GroupRecordLayout.build(group, layout, TB)
        self.assertFalse(record.dynamic_token_offsets)
        offsets, ptrs, sizes, entries = record.resolve([4], key_count=1)
        self.assertEqual(ptrs.tolist(), [7000 + 4 * page])
        self.assertEqual(sizes.tolist(), [page])
        self.assertEqual(entries, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
