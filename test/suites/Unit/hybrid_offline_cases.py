"""Offline byte-address and v1 schema tests; requires only NumPy.

Engine SPI stubs are confined to this standalone test process. Byte-copy
expectations below use NumPy indexing, independently of the layout resolver.
"""

import ctypes
import re
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
for name, path in (
    ("ucm", ROOT / "ucm"),
    ("ucm.integration", ROOT / "ucm/integration"),
    ("ucm.integration.vllm", ROOT / "ucm/integration/vllm"),
):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
for name in (
    "vllm",
    "vllm.model_executor",
    "vllm.model_executor.models",
    "vllm.distributed",
    "vllm.distributed.kv_transfer",
    "vllm.distributed.kv_transfer.kv_connector",
    "vllm.distributed.kv_transfer.kv_connector.v1",
    "vllm.v1",
    "vllm.v1.core",
):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
utils = types.ModuleType("vllm.model_executor.models.utils")
utils.extract_layer_index = lambda name: int(re.search(r"layers\.(\d+)", name)[1])
sys.modules[utils.__name__] = utils
base = types.ModuleType("vllm.distributed.kv_transfer.kv_connector.v1.base")
base.KVConnectorMetadata = type("KVConnectorMetadata", (), {})
sys.modules[base.__name__] = base

from ucm.integration.vllm.hybrid.spec import parse_kv_cache_config
from ucm.integration.vllm.hybrid.store_layout import (
    HybridStoreLayout,
    slot_sizes,
    validate_spec,
)
from ucm.integration.vllm.hybrid.scheduler import UCMDispatcher, UCMGroupDispatchPlan
from ucm.integration.vllm.request_hasher import RequestHasher


@dataclass
class FullAttentionSpec:
    block_size: int = 4


@dataclass
class MambaSpec:
    block_size: int = 4
    mamba_cache_mode: str = "align"
    shapes: tuple = ()


@dataclass
class SlidingWindowSpec:
    block_size: int = 4
    sliding_window: int = 8


class Tensor:
    """Real allocated bytes and strides exposed through the torch view ABI."""

    def __init__(self, array):
        self.array = array
        self.shape = array.shape

    def data_ptr(self):
        return self.array.ctypes.data

    def stride(self, axis):
        return self.array.strides[axis] // self.array.itemsize

    def element_size(self):
        return self.array.itemsize

    def untyped_storage(self):
        array = self.array
        while isinstance(array.base, np.ndarray):
            array = array.base
        return NS(data_ptr=lambda: array.ctypes.data, nbytes=lambda: array.nbytes)


def cache_spec(groups, device="npu", tensors=()):
    raw = NS(
        num_blocks=5,
        kv_cache_tensors=tensors,
        kv_cache_groups=[
            NS(layer_names=list(names), kv_cache_spec=spec) for names, spec in groups
        ],
    )
    return parse_kv_cache_config(raw, scheduler_block_size=4, device_type=device), raw


def view(width, offset=0):
    # Padding between blocks tests copy_size != block_stride.
    backing = np.arange(5 * (width + 7), dtype=np.uint8).reshape(5, width + 7)
    return Tensor(backing[:, offset : offset + width])


def plan(kind, ids, keys=None, group_id=0):
    keys = keys or tuple(bytes([i + 1]) * 16 for i in range(len(ids[0])))
    return UCMGroupDispatchPlan(
        kind, keys, 0, len(keys) * 4, tuple(np.asarray(row) for row in ids), group_id
    )


class ByteStore:
    """Only fixed sizes + shard indices; deliberately no sizes/offsets API."""

    def __init__(self, layout):
        self.layout = layout
        self.data = {}

    def dump(self, transfer):
        for key, ptrs in zip(transfer.keys, transfer.ptrs):
            record = self.data.setdefault(
                key,
                bytearray(
                    self.layout.row_count
                    * ((self.layout.shard_size + 4095) // 4096 * 4096)
                ),
            )
            offset = int(self.layout.ucm_block_offsets[transfer.shard_index, 0])
            for ptr, size in zip(ptrs, self.layout.sizes):
                size = int(size)
                if ptr:
                    record[offset : offset + size] = ctypes.string_at(int(ptr), size)
                offset += size

    def load(self, transfer):
        for key, ptrs in zip(transfer.keys, transfer.ptrs):
            offset = int(self.layout.ucm_block_offsets[transfer.shard_index, 0])
            record = self.data[key]
            for ptr, size in zip(ptrs, self.layout.sizes):
                size = int(size)
                if ptr:
                    ctypes.memmove(
                        int(ptr), bytes(record[offset : offset + size]), size
                    )
                offset += size


class LayoutTests(unittest.TestCase):
    def assert_padding_roundtrip(self, spec, views, layerwise):
        """Independent whole-allocation oracle, including untouched guard bytes."""
        layout = HybridStoreLayout(spec, views, layerwise=layerwise)
        tensors = [
            t
            for value in views.values()
            for t in (value if isinstance(value, tuple) else (value,))
        ]
        for index, tensor in enumerate(tensors):
            pattern = (np.arange(tensor.array.size, dtype=np.uint64) + 37 * index) % 251
            tensor.array[...] = pattern.reshape(tensor.array.shape).astype(
                tensor.array.dtype
            )
        store = ByteStore(layout)
        transfers = []
        for gid in layout.routes:
            keys = (bytes([gid, 1]) * 8, bytes([gid, 2]) * 8)
            src = plan(layout.kinds[gid], [[1, 2]], keys, group_id=gid)
            dst = plan(layout.kinds[gid], [[3, 4]], keys, group_id=gid)
            transfers.append(dst)
            for row in range(layout.row_count):
                store.dump(layout.resolve(src, row))
        expected = []
        for value in views.values():
            for tensor in value if isinstance(value, tuple) else (value,):
                backing = tensor.array
                while isinstance(backing.base, np.ndarray):
                    backing = backing.base
                tensor.array[3:5] = 0xEE
                reference = backing.copy()
                # Independent logical tensor indexing, not resolver addresses.
                target = np.ndarray(
                    tensor.array.shape,
                    dtype=tensor.array.dtype,
                    buffer=reference,
                    offset=tensor.array.ctypes.data - backing.ctypes.data,
                    strides=tensor.array.strides,
                )
                target[3:5] = tensor.array[1:3]
                expected.append((backing, reference))
        # Poison every persisted null slot, so an accidental load of padding
        # corrupts the destination and is detected by the allocation oracle.
        for dst in transfers:
            for row in range(layout.row_count):
                transfer = layout.resolve(dst, row)
                for key, ptrs in zip(transfer.keys, transfer.ptrs):
                    for col, ptr in enumerate(ptrs):
                        if not ptr:
                            start = int(layout.ucm_block_offsets[row, col])
                            count = int(layout.sizes[col])
                            store.data[key][start : start + count] = (
                                bytes([0xA5]) * count
                            )
                store.load(transfer)
        for actual, reference in expected:
            np.testing.assert_array_equal(actual, reference)
        report = layout.padding_report()
        for group in report["groups"].values():
            self.assertEqual(
                group["payload_bytes"]
                + group["padding_bytes"]
                + group["alignment_bytes"],
                group["stored_block_bytes"],
            )
        return layout

    def test_glm_documented_sfa_li_padding_matrix_roundtrip(self):
        # Exact per-block sizes from GLM52_ASCEND_SHARED_INDEXER_KVCACHE_LAYOUT_DESIGN.
        # These are offline byte-view fixtures, not an actual quantized engine run.
        for sfa in (False, True):
            attention_sizes = [83968] if sfa else [131072, 16384]
            for mode in ("bf16", "li_c8", "mixed"):
                for layerwise in (False, True):
                    with self.subTest(sfa=sfa, mode=mode, layerwise=layerwise):
                        views = {}
                        for i in range(3):
                            views[f"model.layers.{i}.self_attn.attn"] = tuple(
                                view(n) for n in attention_sizes
                            )
                        views["model.layers.0.self_attn.indexer.k_cache"] = (
                            view(32768) if mode == "bf16" else (view(16384), view(256))
                        )
                        if mode == "mixed":
                            views["model.layers.2.self_attn.indexer.k_cache"] = view(
                                32768
                            )
                        spec, _ = cache_spec([(list(views), FullAttentionSpec())])
                        layout = self.assert_padding_roundtrip(spec, views, layerwise)
                        if layerwise:
                            suffix = {
                                "bf16": [32768],
                                "li_c8": [16384, 256],
                                "mixed": [16384, 16384, 256],
                            }[mode]
                            self.assertEqual(
                                layout.tensor_size_list, attention_sizes + suffix
                            )
                            self.assertEqual(
                                layout.padding_report()["groups"][0]["rows"][1][
                                    "padding_bytes"
                                ],
                                sum(suffix),
                            )
                        else:
                            self.assertEqual(
                                layout.padding_report()["groups"][0]["padding_bytes"], 0
                            )

    def test_four_group_state_tuple_and_combined_padding_roundtrip(self):
        for combined in (False, True):
            for layerwise in (False, True):
                with self.subTest(combined=combined, layerwise=layerwise):
                    views, groups = {}, []
                    for gid in range(4):
                        names = [f"model.layers.{gid + i*4}.cache" for i in range(2)]
                        state_spec = MambaSpec(shapes=((3,), (16,)))
                        state_spec.dtypes = (NS(itemsize=1), NS(itemsize=1))
                        groups.append(
                            (names, FullAttentionSpec() if gid == 0 else state_spec)
                        )
                        for name in names:
                            views[name] = (
                                (view(32) if combined else (view(16), view(16)))
                                if gid == 0
                                else (view(19) if combined else (view(3), view(16)))
                            )
                    spec, _ = cache_spec(groups)
                    layout = self.assert_padding_roundtrip(spec, views, layerwise)
                    self.assertEqual(
                        layout.tensor_size_list, [3, 16, 16] * (1 if layerwise else 2)
                    )
                    report = layout.padding_report()["groups"]
                    self.assertEqual(report[0]["padding_bytes"], 6)
                    for gid in (1, 2, 3):
                        self.assertEqual(report[gid]["padding_bytes"], 32)

    def test_minimax_dense_sparse_padding_roundtrip(self):
        for separate in (False, True):
            for layerwise in (False, True):
                views = {
                    f"model.layers.{i}.attn": (
                        (view(16), view(16)) if separate else view(32)
                    )
                    for i in range(3)
                }
                views["model.layers.2.index_cache"] = view(8)
                spec, _ = cache_spec([(list(views), FullAttentionSpec())])
                layout = self.assert_padding_roundtrip(spec, views, layerwise)
                self.assertEqual(
                    layout.padding_report()["groups"][0]["padding_bytes"],
                    16 if layerwise else 0,
                )

    def test_cpu_cuda_native_pages_copy_existing_engine_tail_without_slots(self):
        for device in ("cpu", "cuda"):
            for layerwise in (False, True):
                for interleaved in (False, True):
                    with self.subTest(
                        device=device, layerwise=layerwise, interleaved=interleaved
                    ):
                        groups, views, descriptors, allocations = [], {}, [], []
                        for gid in range(4):
                            names = [
                                f"model.layers.{gid + 4*i}.cache" for i in range(2)
                            ]
                            raw = FullAttentionSpec() if gid == 0 else MambaSpec()
                            raw.page_size_bytes = 32
                            groups.append((names, raw))
                            if interleaved:
                                backing = (
                                    (np.arange(5 * 2 * 32, dtype=np.uint16) + gid * 17)
                                    .astype(np.uint8)
                                    .reshape(5, 2, 32)
                                )
                                descriptors.append(
                                    NS(
                                        layers=names,
                                        offset=0,
                                        layer_stride=32,
                                        block_stride=64,
                                    )
                                )
                                for i, name in enumerate(names):
                                    views[name] = Tensor(
                                        backing[:, i, : 32 if gid == 0 else 19]
                                    )
                                allocations.append((gid, backing))
                            else:
                                for i, name in enumerate(names):
                                    backing = (
                                        (
                                            np.arange(5 * 32, dtype=np.uint16)
                                            + gid * 17
                                            + i * 7
                                        )
                                        .astype(np.uint8)
                                        .reshape(5, 32)
                                    )
                                    views[name] = Tensor(
                                        backing[:, : 32 if gid == 0 else 19]
                                    )
                                    allocations.append((gid, backing))
                        spec, _ = cache_spec(groups, device=device, tensors=descriptors)
                        layout = HybridStoreLayout(spec, views, layerwise=layerwise)
                        self.assertEqual(layout.policy, "native_state_page")
                        self.assertEqual(
                            layout.tensor_size_list, [32] if layerwise else [32, 32]
                        )
                        self.assertTrue(
                            all(
                                g["padding_bytes"] == 0
                                for g in layout.padding_report()["groups"].values()
                            )
                        )
                        store = ByteStore(layout)
                        for gid in range(4):
                            src = plan(
                                "FA" if gid == 0 else "State",
                                [[1]],
                                keys=(bytes([gid]) * 16,),
                                group_id=gid,
                            )
                            for row in range(layout.row_count):
                                store.dump(layout.resolve(src, row))
                        expected = []
                        for _, backing in allocations:
                            reference = backing.copy()
                            reference[3] = reference[
                                1
                            ]  # Entire engine page, including the existing tail.
                            backing[3] = 0xEE
                            expected.append((backing, reference))
                        for gid in range(4):
                            dst = plan(
                                "FA" if gid == 0 else "State",
                                [[3]],
                                keys=(bytes([gid]) * 16,),
                                group_id=gid,
                            )
                            for row in range(layout.row_count):
                                store.load(layout.resolve(dst, row))
                        for actual, reference in expected:
                            np.testing.assert_array_equal(actual, reference)

    def test_native_page_rejects_unproven_or_out_of_bounds_expansion(self):
        fa, state = "model.layers.0.attn", "model.layers.1.state"
        a, b = FullAttentionSpec(), MambaSpec()
        a.page_size_bytes = b.page_size_bytes = 32
        spec, _ = cache_spec([([fa], a), ([state], b)], device="cpu")
        attention = Tensor(np.zeros((5, 32), dtype=np.uint8))
        # The view fits but the last full page does not.
        short = np.zeros(4 * 32 + 19, dtype=np.uint8)
        tensor = Tensor(
            np.ndarray((5, 19), dtype=np.uint8, buffer=short, strides=(32, 1))
        )
        with self.assertRaisesRegex(ValueError, "allocation"):
            HybridStoreLayout(spec, {fa: attention, state: tensor}, layerwise=True)
        backing = np.zeros((5, 33), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "descriptor"):
            HybridStoreLayout(
                spec, {fa: attention, state: Tensor(backing[:, 1:20])}, layerwise=True
            )
        with self.assertRaisesRegex(ValueError, "one registered"):
            HybridStoreLayout(
                spec, {fa: attention, state: (view(3), view(16))}, layerwise=True
            )
        b.page_size_bytes = 40
        state_tensor = Tensor(np.zeros((5, 40), dtype=np.uint8))
        # Unequal declared pages no longer reach the native-page builder:
        # they route to the general semantic policy (Qwen4Exp mixes state
        # page sizes), which this minimal fixture cannot satisfy.
        with self.assertRaisesRegex(ValueError, "State policy"):
            HybridStoreLayout(
                spec, {fa: attention, state: state_tensor}, layerwise=True
            )

    def test_model_check_with_real_hybrid_layout(self):
        from unittest.mock import patch

        sys.path.insert(0, str(ROOT / "toolkit"))
        from ucm_toolkit.tools.model_check import hybrid as check

        class Access:
            def __init__(self, views):
                pass

            def synchronize(self):
                pass

            def write(self, ptr, data):
                ctypes.memmove(ptr, data, len(data))

            def read(self, ptr, size):
                return ctypes.string_at(ptr, size)

        checker_connector = type(
            "UCMHybridConnector",
            (),
            {"__module__": "ucm.integration.vllm.hybrid_connector"},
        )
        for state in (False, True):
            for layerwise in (False, True):
                with self.subTest(state=state, layerwise=layerwise):
                    names = ["model.layers.0.attn", "model.layers.2.attn"]
                    groups = [(names, FullAttentionSpec())]
                    views = {name: (view(8), view(8)) for name in names}
                    if state:
                        groups.append(
                            (
                                ["model.layers.1.mamba", "model.layers.3.mamba"],
                                MambaSpec(),
                            )
                        )
                        for i in (1, 3):
                            views[f"model.layers.{i}.mamba"] = (view(3), view(8))
                    spec, _ = cache_spec(groups)
                    layout = HybridStoreLayout(spec, views, layerwise=layerwise)
                    impl = checker_connector()
                    impl.kv_cache_layout = layout
                    worker = NS(connector=impl)
                    source = [
                        plan(
                            layout.kinds[gid],
                            [[1]],
                            keys=(str(gid).encode(),),
                            group_id=gid,
                        )
                        for gid in layout.routes
                    ]
                    target = [
                        plan(
                            layout.kinds[gid],
                            [[3]],
                            keys=(str(gid).encode(),),
                            group_id=gid,
                        )
                        for gid in layout.routes
                    ]
                    metadata = NS(
                        requests={"r": NS(dump_plans=source, load_plans=target)}
                    )
                    fixture = NS(kv_caches=views)
                    store = ByteStore(layout)
                    with patch.object(check, "TensorByteAccess", Access):
                        check.fill_hybrid(fixture, worker, metadata)
                        for item in source:
                            for row in range(layout.row_count):
                                store.dump(layout.resolve(item, row))
                        check.poison_hybrid(fixture, worker, metadata)
                        with self.assertRaises(AssertionError):
                            check.compare_hybrid(fixture, worker, metadata)
                        for item in target:
                            for row in range(layout.row_count):
                                store.load(layout.resolve(item, row))
                        self.assertGreater(
                            check.compare_hybrid(fixture, worker, metadata), 0
                        )

    def test_partition_never_crosses_a_real_segment_boundary(self):
        self.assertEqual(slot_sizes([[12, 8], [8, 4, 4], [12]]).tolist(), [8, 4, 4, 4])

    def test_single_group_shared_indexer_bf16_c8_and_missing(self):
        names = [f"model.layers.{i}.attn" for i in range(3)]
        indexers = ["model.layers.0.indexer", "model.layers.2.indexer"]
        spec, _ = cache_spec([(names + indexers, FullAttentionSpec())])
        views = {name: view(16) for name in names}
        views[indexers[0]] = view(16)  # BF16 indexer
        views[indexers[1]] = (view(8), view(4))  # C8 indexer + scales
        for layerwise in (False, True):
            layout = HybridStoreLayout(spec, views, layerwise=layerwise)
            source = plan("FA", [[1]])
            store = ByteStore(layout)
            for row in range(layout.row_count):
                store.dump(layout.resolve(source, row))
            expected = {
                name: [
                    t.array[1].copy()
                    for t in (value if isinstance(value, tuple) else (value,))
                ]
                for name, value in views.items()
            }
            sentinels = []
            for value in views.values():
                for tensor in value if isinstance(value, tuple) else (value,):
                    tensor.array[3] = 0
                    sentinels.append((tensor.array.base, tensor.array.base.copy()))
            for row in range(layout.row_count):
                store.load(layout.resolve(plan("FA", [[3]], source.keys), row))
            for name, value in views.items():
                for tensor, reference in zip(
                    value if isinstance(value, tuple) else (value,), expected[name]
                ):
                    np.testing.assert_array_equal(tensor.array[3], reference)
                    # The seven bytes of allocation padding must stay intact.
                    np.testing.assert_array_equal(
                        tensor.array.base.reshape(5, -1)[3, tensor.shape[1] :],
                        next(
                            saved
                            for base_array, saved in sentinels
                            if base_array is tensor.array.base
                        ).reshape(5, -1)[3, tensor.shape[1] :],
                    )
            if layerwise:
                self.assertTrue((layout.resolve(source, 1).ptrs == 0).any())
            else:
                self.assertFalse((layout.resolve(source, 0).ptrs == 0).any())

    def test_independent_mamba_allocations_and_group_block_ids(self):
        fa = "model.layers.3.attn"
        states = [f"model.layers.{i}.mamba" for i in range(3)]
        spec, _ = cache_spec(
            [([fa], FullAttentionSpec())] + [([name], MambaSpec()) for name in states]
        )
        views = {
            fa: (view(16), view(16)),
            **{name: (view(3), view(16)) for name in states},
        }
        for layerwise in (False, True):
            layout = HybridStoreLayout(spec, views, layerwise=layerwise)
            self.assertEqual(layout.row_count, 1)
            self.assertEqual(layout.tensor_size_list, [3, 16, 16])
            fa_transfer = layout.resolve(plan("FA", [[2]]), 0)
            self.assertEqual(fa_transfer.ptrs[0, 0], 0)
            self.assertEqual(
                int(fa_transfer.ptrs[0, 1]), views[fa][0].array[2].ctypes.data
            )
            for gid, name in enumerate(states, 1):
                transfer = layout.resolve(plan("State", [[gid]], group_id=gid), 0)
                self.assertEqual(
                    transfer.ptrs.tolist(),
                    [
                        [
                            views[name][0].array[gid].ctypes.data,
                            views[name][1].array[gid].ctypes.data,
                            0,
                        ]
                    ],
                )
                self.assertEqual(
                    layout.resolve(plan("State", [[0]], group_id=gid), 0).ptrs.shape[0],
                    0,
                )

    def test_multiple_fa_groups_use_their_own_block_tables(self):
        a, b = "model.layers.0.attn", "model.layers.1.attn"
        spec, _ = cache_spec([([a], FullAttentionSpec()), ([b], FullAttentionSpec())])
        views = {a: view(8), b: view(8)}
        layout = HybridStoreLayout(spec, views, layerwise=False)
        transfer = layout.resolve(plan("FA", [[1, 2]], group_id=0), 0)
        other = layout.resolve(plan("FA", [[3, 4]], group_id=1), 0)
        self.assertEqual(int(transfer.ptrs[0, 0]), views[a].array[1].ctypes.data)
        self.assertEqual(int(other.ptrs[0, 0]), views[b].array[3].ctypes.data)
        self.assertEqual(transfer.ptrs.dtype, np.uint64)
        self.assertTrue(transfer.ptrs.flags.c_contiguous)

    def test_head_separated_strides_and_nonzero_view_offset(self):
        name = "model.layers.0.attn"
        spec, _ = cache_spec([([name], FullAttentionSpec())], device="cuda")
        backing = np.arange(5 * 2 * 4 * 5, dtype=np.uint8).reshape(5, 2, 4, 5)
        # Dense channels inside each head, with a gap between heads.
        tensor = Tensor(backing[:, :, :, :])
        layout = HybridStoreLayout(spec, {name: tensor}, layerwise=False)
        transfer = layout.resolve(plan("FA", [[2]]), 0)
        self.assertEqual(
            transfer.ptrs.tolist(), [[backing[2, h].ctypes.data for h in range(2)]]
        )

    def test_029_declared_interleaved_layers(self):
        names = ["model.layers.0.attn", "model.layers.1.attn"]
        backing = np.arange(5 * 2 * 8, dtype=np.uint8).reshape(5, 2, 8)
        descriptor = NS(layers=names, offset=0, layer_stride=8, block_stride=16)
        spec, _ = cache_spec([(names, FullAttentionSpec())], tensors=[descriptor])
        layout = HybridStoreLayout(
            spec,
            {name: Tensor(backing[:, i]) for i, name in enumerate(names)},
            layerwise=True,
        )
        self.assertEqual(
            int(layout.resolve(plan("FA", [[4]]), 1).ptrs[0, 0]),
            backing[4, 1].ctypes.data,
        )

    def test_semantic_indexer_slots_and_minimax_dense_layers(self):
        for role in ("indexer", "index_cache"):
            names = [f"model.layers.{i}.attn" for i in range(3)]
            indexes = [f"model.layers.{i}.{role}" for i in (0, 2)]
            spec, _ = cache_spec([(names + indexes, FullAttentionSpec())])
            views = {name: view(16) for name in names}
            views[indexes[0]] = view(16)
            views[indexes[1]] = (view(8), view(4)) if role == "indexer" else view(16)
            layout = HybridStoreLayout(spec, views, layerwise=True)
            if role == "indexer":
                self.assertEqual(layout.tensor_size_list, [16, 8, 8, 4])
                self.assertEqual(layout.rows[0][0].columns.tolist(), [0, 1, 2])
                self.assertEqual(layout.rows[0][2].columns.tolist(), [0, 1, 3])
                ptrs = layout.resolve(plan("FA", [[2]]), 0).ptrs[0]
                self.assertEqual(
                    int(ptrs[2]), views[indexes[0]].array[2].ctypes.data + 8
                )
                self.assertEqual(ptrs[3], 0)
            else:
                self.assertEqual(layout.policy, "minimax_m3")
                self.assertEqual(layout.tensor_size_list, [16, 16])
            self.assertEqual(layout.rows[0][1].columns.tolist(), [0])

    def test_full_c8_does_not_reserve_bf16_tail(self):
        names = ["model.layers.0.attn", "model.layers.1.attn", "model.layers.0.indexer"]
        spec, _ = cache_spec([(names, FullAttentionSpec())])
        layout = HybridStoreLayout(
            spec,
            {names[0]: view(16), names[1]: view(16), names[2]: (view(8), view(4))},
            layerwise=True,
        )
        self.assertEqual(layout.tensor_size_list, [16, 8, 4])
        self.assertEqual(layout.rows[0][1].columns.tolist(), [0])

    def test_state_combined_page_and_no_implicit_extra_rows(self):
        fa, state = "model.layers.1.attn", "model.layers.0.mamba"
        raw_state = MambaSpec(shapes=((3,), (8,)))
        raw_state.dtypes = (NS(itemsize=1), NS(itemsize=1))
        spec, _ = cache_spec([([fa], FullAttentionSpec()), ([state], raw_state)])
        views = {fa: (view(8), view(8)), state: view(11)}
        layout = HybridStoreLayout(spec, views, layerwise=True)
        transfer = layout.resolve(plan("State", [[2]], group_id=1), 0)
        self.assertEqual(layout.tensor_size_list, [3, 8, 8])
        self.assertEqual(
            transfer.ptrs.tolist(),
            [
                [
                    views[state].array[2].ctypes.data,
                    views[state].array[2].ctypes.data + 3,
                    0,
                ]
            ],
        )
        spec, _ = cache_spec(
            [
                ([fa], FullAttentionSpec()),
                ([state, "model.layers.2.mamba"], MambaSpec()),
            ]
        )
        views[state] = (view(3), view(8))
        views["model.layers.2.mamba"] = (view(3), view(8))
        with self.assertRaisesRegex(ValueError, "equal local layer"):
            HybridStoreLayout(spec, views, layerwise=True)

    def test_qwen_four_groups_capacity_and_real_slot_sizes(self):
        groups, views = [], {}
        fa_parts = (view(1572864), view(1572864))
        state_parts = (view(30720), view(1572864))
        for offset in (3, 0, 1, 2):
            names = [f"model.layers.{i}.cache" for i in range(offset, 64, 4)]
            groups.append((names, FullAttentionSpec() if offset == 3 else MambaSpec()))
            views.update(
                {name: fa_parts if offset == 3 else state_parts for name in names}
            )
        spec, _ = cache_spec(groups)
        for layerwise in (True, False):
            layout = HybridStoreLayout(spec, views, layerwise=layerwise)
            self.assertEqual(layout.row_count, 16 if layerwise else 1)
            self.assertEqual(
                layout.tensor_size_list,
                [30720, 1572864, 1572864] * (1 if layerwise else 16),
            )
            self.assertEqual(layout.block_size, 50823168)
            self.assertEqual(set(layout.rows), {0, 1, 2, 3})

    def test_bad_block_ids_and_bad_window_shape_fail_before_io(self):
        name = "model.layers.0.attn"
        spec, _ = cache_spec([([name], FullAttentionSpec())])
        layout = HybridStoreLayout(spec, {name: view(8)}, layerwise=False)
        for ids in ([-1], [5]):
            with self.assertRaises(ValueError):
                layout.resolve(plan("FA", [ids]), 0)
        with self.assertRaises(ValueError):
            layout.resolve(plan("FA", [[1]], (b"a" * 16, b"b" * 16)), 0)

    def test_swa_and_unequal_groups_are_rejected(self):
        fa = "model.layers.0.attn"
        spec, _ = cache_spec(
            [
                ([fa], FullAttentionSpec()),
                (["model.layers.1.attn"], FullAttentionSpec(8)),
            ]
        )
        with self.assertRaisesRegex(ValueError, "equal"):
            validate_spec(spec)
        with self.assertRaisesRegex(ValueError, "FAWA"):
            cache_spec(
                [
                    ([fa], FullAttentionSpec()),
                    (["model.layers.1.attn"], SlidingWindowSpec()),
                ]
            )

    def test_qwen4_exp_ring_group_is_transient_not_rejected(self):
        # vLLM 0.30 Qwen4Exp registers the indexer raw-key ring as
        # CircularBufferSpec: one block per request, prefix_cacheable=False.
        # The parser must keep the group out of persistent routing instead
        # of failing the whole spec.
        class CircularBufferSpec:
            block_size = 4

        spec, _ = cache_spec(
            [
                (["model.layers.0.linear_attn"], MambaSpec()),
                (
                    ["model.layers.3.self_attn.indexer.raw_key_cache"],
                    CircularBufferSpec(),
                ),
                (
                    ["model.layers.3.self_attn.indexer.compressed_key_cache"],
                    FullAttentionSpec(),
                ),
            ]
        )
        rings = [
            group
            for group in spec.groups
            if "raw_key_cache" in group.layers[0].layer_name
        ]
        self.assertEqual(len(rings), 1)
        self.assertTrue(rings[0].transient)
        self.assertEqual(rings[0].tail_tokens, 0)
        persistent_names = [
            layer.layer_name for group in spec.persistent_groups for layer in group.layers
        ]
        self.assertNotIn(
            "model.layers.3.self_attn.indexer.raw_key_cache", persistent_names
        )
        self.assertIn(
            "model.layers.3.self_attn.indexer.compressed_key_cache", persistent_names
        )
        self.assertIn("model.layers.0.linear_attn", persistent_names)


class NamespaceTests(unittest.TestCase):
    def test_compressed_scheduler_specs_share_worker_namespace(self):
        from dataclasses import replace
        from ucm.integration.vllm.hybrid.namespace import storage_namespace

        names = ["model.layers.0.attn", "model.layers.0.indexer", "model.layers.1.attn"]
        worker, _ = cache_spec([(names, FullAttentionSpec())])
        group = worker.groups[0]
        # Model the engine's per-layer spec map followed by representative compression.
        physical = tuple(
            replace(
                layer,
                kv_cache_spec=NS(
                    block_size=4,
                    head_size=128 if "indexer" in layer.layer_name else 576,
                ),
            )
            for layer in group.layers
        )
        worker = replace(worker, groups=(replace(group, layers=physical),))
        scheduler = replace(
            worker,
            groups=(
                replace(
                    group,
                    layers=tuple(
                        replace(
                            layer, kv_cache_spec=physical[0].kv_cache_spec, num_blocks=2
                        )
                        for layer in physical
                    ),
                ),
            ),
        )
        kwargs = dict(
            model={"model_type": "glm_moe_dsa"},
            device="npu",
            cache_dtype="auto",
            model_dtype="bf16",
            tp=2,
            layerwise=True,
            versions={"vllm": "test"},
            additional_config={"enable_sparse_li_c8": False},
        )
        self.assertNotEqual(
            str(worker.groups[0].layers[1].kv_cache_spec),
            str(scheduler.groups[0].layers[1].kv_cache_spec),
        )
        expected = storage_namespace(worker, **kwargs)
        self.assertEqual(storage_namespace(scheduler, **kwargs), expected)
        for key, value in (
            ("additional_config", {"enable_sparse_li_c8": True}),
            ("quantization", "ascend"),
            ("model_dtype", "fp16"),
            ("cache_dtype", "int8"),
            ("layerwise", False),
            ("tp", 4),
        ):
            self.assertNotEqual(
                storage_namespace(worker, **(kwargs | {key: value})), expected
            )
        changed = replace(worker, groups=(replace(worker.groups[0], group_id=1),))
        self.assertNotEqual(storage_namespace(changed, **kwargs), expected)


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.spec, _ = cache_spec(
            [
                (["model.layers.0.attn"], FullAttentionSpec()),
                (["model.layers.1.mamba"], MambaSpec()),
            ]
        )
        config = NS(
            model_config=NS(model="model", dtype="bf16"),
            parallel_config=NS(tensor_parallel_size=1),
        )
        self.hasher = RequestHasher(config, 0)
        self.lookup = NS(
            lookup_on_prefix=lambda keys: -1, lookup_on_reverse=lambda keys: -1
        )
        self.dispatcher = UCMDispatcher(self.spec, self.lookup, self.hasher, b"seed")
        self.request = NS(request_id="r", all_token_ids=list(range(16)))

    def test_group_key_isolation_and_common_state_boundary(self):
        spec, _ = cache_spec(
            [(["model.layers.3.attn"], FullAttentionSpec())]
            + [([f"model.layers.{i}.state"], MambaSpec()) for i in range(3)]
        )
        d = UCMDispatcher(spec, self.lookup, self.hasher, b"seed")
        chains = d.lookup(self.request, 0).group_ucm_block_ids
        self.assertEqual(len({keys[1] for keys in chains}), 4)
        # Latest hits are 3, 2, 3; index 2 is NOT a common boundary.
        # The true common boundary is index 0 (4 tokens).
        present = set(chains[1][i] for i in (0, 2))
        present.update(chains[2][i] for i in (0, 1))
        present.update(chains[3][i] for i in (0, 2))
        self.lookup.lookup_on_prefix = lambda keys: len(keys) - 1
        self.lookup.lookup_on_reverse = lambda keys: next(
            (i for i in range(len(keys) - 1, -1, -1) if keys[i] in present), -1
        )
        self.assertEqual(d.lookup(self.request, 0).external_hit_tokens, 4)
        present.remove(chains[2][0])
        self.assertEqual(d.lookup(self.request, 0).external_hit_tokens, 0)
        self.lookup.lookup_on_prefix = lambda keys: -1
        d.lookup(self.request, 0)
        d.update_blocks("r", [[1, 2, 3, 4]] * 4, append=False)
        plans = d.build_metadata({"r": 4}).requests["r"].dump_plans
        self.assertEqual([p.group_id for p in plans], [0, 1, 2, 3])
        self.assertTrue(all(len(p.windows) == 1 for p in plans))

    def test_group_tag_overflow_rejected_and_fa_prefix_intersected(self):
        from ucm.integration.vllm.hybrid.scheduler import _key_tag

        self.assertNotEqual(
            _key_tag("State", group_id=0), _key_tag("State", group_id=15)
        )
        with self.assertRaisesRegex(ValueError, "4-bit"):
            _key_tag("State", group_id=16)
        spec, _ = cache_spec(
            [([f"model.layers.{i}.attn"], FullAttentionSpec()) for i in range(2)]
        )
        dispatcher = UCMDispatcher(spec, self.lookup, self.hasher, b"seed")
        chains = dispatcher.lookup(self.request, 0).group_ucm_block_ids
        self.lookup.lookup_on_prefix = lambda keys: (
            0 if keys[0] == chains[1][0] else len(keys) - 1
        )
        self.assertEqual(dispatcher.lookup(self.request, 0).external_hit_tokens, 4)

    def test_state_not_published_at_a_stale_boundary(self):
        self.dispatcher.lookup(self.request, 0)
        self.dispatcher.update_blocks("r", [[1, 2, 3, 4], [0, 2, 3, 4]], append=False)
        plans = self.dispatcher.build_metadata({"r": 6}).requests["r"].dump_plans
        self.assertEqual([p.hash_group for p in plans], ["FA"])
        plans = self.dispatcher.build_metadata({"r": 2}).requests["r"].dump_plans
        self.assertEqual([p.hash_group for p in plans], ["FA", "State"])
        self.assertEqual(plans[-1].windows[0].tolist(), [2])

    def test_state_boundary_limits_full_attention_prefix(self):
        self.lookup.lookup_on_prefix = lambda keys: len(keys) - 1
        self.lookup.lookup_on_reverse = lambda keys: 1
        result = self.dispatcher.lookup(self.request, 0)
        self.assertEqual(result.external_hit_tokens, 8)
        self.assertNotEqual(
            result.group_ucm_block_ids[0][1], result.group_ucm_block_ids[1][1]
        )

    def test_shared_prefix_requests_keep_distinct_destinations(self):
        self.lookup.lookup_on_prefix = lambda keys: len(keys) - 1
        self.lookup.lookup_on_reverse = lambda keys: len(keys) - 1
        for request_id, blocks in (("r", [1, 2, 3, 4]), ("s", [4, 3, 2, 1])):
            self.dispatcher.lookup(
                NS(request_id=request_id, all_token_ids=list(range(16))), 0
            )
            self.dispatcher.update_blocks(request_id, [blocks, blocks], append=False)
        meta = self.dispatcher.build_metadata({"r": 1, "s": 1})
        a, b = meta.requests["r"].load_plans[0], meta.requests["s"].load_plans[0]
        self.assertEqual(a.keys, b.keys)
        self.assertNotEqual(a.windows[0].tolist(), b.windows[0].tolist())

    def test_request_extra_semantics_are_not_silently_dropped(self):
        # This environment lacks the native extra-key helper: fail closed.
        self.request.cache_salt = "tenant-A"
        with self.assertRaisesRegex(RuntimeError, "KV-affecting semantics"):
            self.dispatcher.lookup(self.request, 0)

    def test_scheduler_new_resume_delta_and_finished(self):
        self.dispatcher.lookup(self.request, 0)
        cached = NS(req_ids=[], resumed_req_ids=set(), new_block_ids=[])
        output = NS(
            scheduled_new_reqs=[NS(req_id="r", block_ids=([1], [2]))],
            scheduled_cached_reqs=cached,
            num_scheduled_tokens={"r": 4},
            preempted_req_ids=set(),
            finished_req_ids=set(),
        )
        self.dispatcher.build_from_scheduler_output(output)
        output.scheduled_new_reqs = []
        output.scheduled_cached_reqs = NS(
            req_ids=["r"], resumed_req_ids=set(), new_block_ids=[([3], [4])]
        )
        self.dispatcher.build_from_scheduler_output(output)
        self.assertEqual(
            self.dispatcher.requests["r"].group_vllm_block_ids, ([1, 3], [2, 4])
        )
        output.scheduled_cached_reqs = NS(
            req_ids=["r"], resumed_req_ids={"r"}, new_block_ids=[([4, 3, 2], [1, 2, 3])]
        )
        self.dispatcher.build_from_scheduler_output(output)
        self.assertEqual(
            self.dispatcher.requests["r"].group_vllm_block_ids, ([4, 3, 2], [1, 2, 3])
        )
        output.scheduled_cached_reqs = cached
        output.num_scheduled_tokens = {}
        output.finished_req_ids = {"r"}
        self.dispatcher.build_from_scheduler_output(output)
        self.assertNotIn("r", self.dispatcher.requests)


class LifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import the actual Hybrid hook implementation with the old connector
        # reduced to its lifecycle helpers. Native store calls are recorded.
        import enum
        from unittest.mock import Mock

        base.SupportsHMA = type("SupportsHMA", (), {})
        base.KVConnectorRole = enum.Enum("KVConnectorRole", "SCHEDULER WORKER")
        platform = types.ModuleType("vllm.platforms")
        platform.current_platform = NS(device_type="npu")
        sys.modules[platform.__name__] = platform
        legacy = types.ModuleType("ucm.integration.vllm.ucm_connector")

        class Direct:
            def has_connector_metadata(self):
                return self.metadata is not None

            def _get_connector_metadata(self):
                return self.metadata

            def _get_dump_event_handle(self):
                return 91

            def _flush_pending_dump_tasks(self):
                for task in self._pending_dump_tasks:
                    try:
                        self._rank_consistency.wait_dump(task.task)
                    finally:
                        self.device.destroy_event_handle(task.event_handle)
                self._pending_dump_tasks.clear()

        legacy.UCMDirectConnector = Direct
        legacy.PendingDumpTask = lambda task, reqs, event: NS(
            task=task, request_ids=reqs, event_handle=event
        )
        legacy._get_store_io_sizes = lambda shard, block: (
            ((shard + 4095) // 4096) * 4096,
            ((shard + 4095) // 4096) * 4096 * (block // shard),
        )
        legacy._scheduler_read_block_size = lambda: 4096
        legacy._use_ucm_connector_cpu_affinity = lambda: False
        sys.modules[legacy.__name__] = legacy
        device = types.ModuleType("ucm.integration.vllm.device")
        device.create_device = lambda: None
        sys.modules[device.__name__] = device
        logger = types.ModuleType("ucm.logger")
        logger.init_logger = lambda name: Mock()
        sys.modules[logger.__name__] = logger
        factory = types.ModuleType("ucm.store.factory_v1")
        factory.UcmConnectorFactoryV1 = Mock()
        sys.modules[factory.__name__] = factory
        from ucm.integration.vllm.hybrid_connector import UCMHybridConnector

        cls.connector_class = UCMHybridConnector

    def make_connector(self):
        from unittest.mock import Mock
        from ucm.integration.vllm.hybrid.scheduler import (
            UCMConnectorMetadata,
            RequestDispatchMeta,
        )

        names = ["model.layers.0.attn", "model.layers.0.indexer", "model.layers.1.attn"]
        spec, _ = cache_spec([(names, FullAttentionSpec())])
        layout = HybridStoreLayout(
            spec, {name: view(8) for name in names}, layerwise=True
        )
        c = self.connector_class.__new__(self.connector_class)
        c.spec, c.kv_cache_layout = spec, layout
        c.layer_name_to_id = layout.layer_name_to_id
        c.metadata = UCMConnectorMetadata(
            requests={
                "r": RequestDispatchMeta(
                    "r", (plan("FA", [[2]]),), (plan("FA", [[1]]),)
                )
            }
        )
        c.store = object()  # No commit method exists.
        c.tp_rank = 0
        c.use_layerwise = True
        c.device = Mock()
        c._rank_consistency = Mock()
        c._rank_consistency.submit_dump.side_effect = lambda *args: object()
        c._rank_consistency.submit_load.side_effect = lambda *args: object()
        c._connector_worker_meta = Mock()
        c._invalid_block_ids = set()
        c._load_tasks, c._pending_dump_tasks = {}, []
        c._seen_names, c._saved_rows, c._failed_load_reqs = set(), set(), set()
        c._dump_request_ids, c._save_complete = set(), False
        return c

    def test_same_layer_views_wait_for_all_hooks_and_last_row_falls_back(self):
        c = self.make_connector()
        c.start_load_kv(None)
        c.wait_for_layer_load("model.layers.0.attn")
        self.assertEqual(c._rank_consistency.wait_load.call_count, 1)
        c.save_kv_layer("model.layers.0.attn", None, None)
        self.assertEqual(c._rank_consistency.submit_dump.call_count, 0)
        c.save_kv_layer("model.layers.0.indexer", None, None)
        self.assertEqual(c._rank_consistency.submit_dump.call_count, 1)
        c.save_kv_layer("model.layers.0.indexer", None, None)
        c.wait_for_save()
        self.assertEqual(c._rank_consistency.submit_dump.call_count, 2)
        self.assertEqual(c._rank_consistency.wait_dump.call_count, 2)
        self.assertEqual(c._rank_consistency.wait_load.call_count, 2)
        self.assertEqual(c.device.destroy_event_handle.call_count, 2)
        c.wait_for_save()
        self.assertEqual(c._rank_consistency.submit_dump.call_count, 2)

    def test_load_failure_suppresses_dump_but_drains_other_tasks(self):
        c = self.make_connector()
        c._rank_consistency.wait_load.side_effect = [RuntimeError("IO"), None]
        c.start_load_kv(None)
        c.wait_for_layer_load("model.layers.0.attn")
        c.wait_for_save()
        c._rank_consistency.submit_dump.assert_not_called()
        self.assertEqual(c._rank_consistency.wait_load.call_count, 2)
        self.assertEqual(c._invalid_block_ids, {2})

    def test_submit_failure_destroys_event_and_drains_successful_tasks(self):
        c = self.make_connector()
        c._rank_consistency.submit_dump.side_effect = [object(), RuntimeError("IO")]
        c.start_load_kv(None)
        c.wait_for_save()
        self.assertEqual(c.device.destroy_event_handle.call_count, 2)
        self.assertEqual(c._rank_consistency.wait_dump.call_count, 1)

    def test_tp_keys_are_isolated(self):
        c = self.make_connector()
        keys = (b"k" * 16,)
        self.assertEqual(c._store_keys(keys), list(keys))
        c.tp_rank = 1
        c.request_hasher = lambda key: b"r" * 16
        self.assertNotEqual(c._store_keys(keys), list(keys))


class Glm53IntegrationTests(unittest.TestCase):
    def make_glm(self, backend):
        from test_glm53_layout import fixture, Tensor as CapturedTensor

        capture = fixture(backend)
        raw_groups, caches = {}, {}
        descriptors = {}
        for record in capture["tensors"]:
            gid, name = record["group"], record["name"]
            spec_name = (
                "KpoolTailSpec"
                if gid == 1
                else "MambaSpec" if gid >= 2 else "MLAAttentionSpec"
            )
            concrete = type(spec_name, (), {})()
            concrete.block_size = 4 if gid == 1 else 4352
            concrete.page_size_bytes = record["descriptor"]["block_stride"]
            concrete.tokens_per_state = 4 if name.endswith(".indexer.k_cache") else 1
            concrete.mamba_cache_mode = "align"
            raw_groups.setdefault(gid, {})[name] = concrete
            caches[name] = tuple(CapturedTensor(v) for v in record["views"])
            descriptors[tuple(record["descriptor"]["layers"])] = NS(
                **record["descriptor"]
            )
        groups = [
            NS(
                layer_names=list(items),
                kv_cache_spec=NS(
                    block_size=4 if gid == 1 else 4352, kv_cache_specs=items
                ),
            )
            for gid, items in sorted(raw_groups.items())
        ]
        config = NS(
            num_blocks=capture["num_blocks"],
            kv_cache_groups=groups,
            kv_cache_tensors=list(descriptors.values()),
        )
        spec = parse_kv_cache_config(
            config,
            scheduler_block_size=4352,
            device_type=backend,
            layout_policy="glm53",
        )
        return spec, caches

    def test_glm53_store_compilation(self):
        for backend, group_count, row_count, index_bytes in (
            ("cuda", 6, 11, 143616),
            ("npu", 5, 12, 278528),
        ):
            spec, caches = self.make_glm(backend)
            self.assertEqual(len(spec.groups), group_count)
            self.assertTrue(spec.groups[1].transient)
            self.assertEqual(
                [g.group_id for g in spec.persistent_groups],
                [0, *range(2, group_count)],
            )
            self.assertFalse(spec.sw_groups)
            for layerwise in (True, False):
                with self.subTest(backend=backend, layerwise=layerwise):
                    layout = HybridStoreLayout(spec, caches, layerwise=layerwise)
                    self.assertEqual(layout.row_count, row_count if layerwise else 1)
                    self.assertEqual(
                        layout.block_size, row_count * (4456448 + index_bytes)
                    )
                    self.assertNotIn(1, layout.routes)
                    for gid, rows in layout.rows.items():
                        kind = "FA" if gid == 0 else "State"
                        plan = UCMGroupDispatchPlan(
                            kind, (b"a", b"b"), 0, 8704, (np.array([1, 2]),), gid
                        )
                        for i, row in enumerate(rows):
                            transfer = layout.resolve(plan, i)
                            self.assertEqual(
                                transfer.ptrs.shape, (2, len(layout.sizes))
                            )
                            ghost = sorted(
                                set(range(len(layout.sizes))) - set(row.columns)
                            )
                            self.assertFalse(transfer.ptrs[:, ghost].any())
                            if len(row.columns):
                                np.testing.assert_array_equal(
                                    transfer.ptrs[1, row.columns]
                                    - transfer.ptrs[0, row.columns],
                                    row.strides,
                                )
                            if gid and layerwise:
                                self.assertFalse(
                                    transfer.ptrs[
                                        :,
                                        [
                                            j
                                            for j, r in enumerate(layout.slot_roles)
                                            if r == "index"
                                        ],
                                    ].any()
                                )

    def test_glm53_keeps_native_tables_and_old_guards(self):
        spec, caches = self.make_glm("cuda")
        from ucm.integration.vllm.hybrid.scheduler import RequestState

        dispatcher = UCMDispatcher(spec, None, lambda x: b"x", b"seed")
        dispatcher.requests["r"] = RequestState(
            group_vllm_block_ids=tuple([] for _ in spec.groups)
        )
        dispatcher.update_blocks("r", ([1], [2], [3], [4], [5], [6]), append=False)
        self.assertEqual(dispatcher.requests["r"].group_vllm_block_ids[2], [3])
        self.assertEqual(
            [groups[0].group_id for _, groups in spec.dispatch_routes()],
            [0, 2, 3, 4, 5],
        )




class PerRoleCompressionTests(unittest.TestCase):
    def test_compression_ratios_key_by_layer_name_not_index(self):
        # Qwen4Exp registers a ratio-4 indexer view next to a ratio-1
        # full-attention view of the SAME model layer; keying ratios by
        # layer index would let one role shadow the other, and the Mamba
        # groups must never consult the attention ratio map at all.
        class MLAAttentionSpec:  # named to classify as MLA
            block_size = 4
            tokens_per_state = 4

        fa = "model.layers.3.self_attn.attn"
        idx = "model.layers.3.self_attn.indexer.compressed_key_cache"
        attention_group = NS(
            layer_names=[idx, fa],
            kv_cache_spec=NS(
                block_size=4,
                kv_cache_specs={idx: MLAAttentionSpec(), fa: FullAttentionSpec()},
            ),
        )
        mamba_group = NS(
            layer_names=["model.layers.0.linear_attn"], kv_cache_spec=MambaSpec()
        )
        raw = NS(
            num_blocks=5,
            kv_cache_tensors=(),
            kv_cache_groups=[mamba_group, attention_group],
        )
        spec = parse_kv_cache_config(raw, scheduler_block_size=4, device_type="cpu")
        by_name = {
            layer.layer_name: layer
            for group in spec.groups
            for layer in group.layers
        }
        self.assertEqual(by_name[idx].storage_block_size, 1)
        self.assertEqual(by_name[fa].storage_block_size, 4)
        self.assertEqual(by_name["model.layers.0.linear_attn"].storage_block_size, 4)


class NativePagePolicyTests(unittest.TestCase):
    def test_native_state_pages_need_one_declared_size(self):
        from ucm.integration.vllm.hybrid.store_layout import _native_state_page_sizes

        def layer(name, page):
            return NS(layer_name=name, kv_cache_spec=NS(page_size_bytes=page))

        def spec(pages):
            return NS(
                persistent_groups=[
                    NS(layers=[layer(f"l{i}", page) for i, page in enumerate(pages)])
                ]
            )

        self.assertEqual(_native_state_page_sizes(spec([1835008, 1835008])), {1835008})
        # Qwen4Exp mixes 3207168B linear pages with a 184320B ple page.
        self.assertEqual(
            _native_state_page_sizes(spec([3207168, 184320])), {3207168, 184320}
        )
        self.assertIsNone(_native_state_page_sizes(spec([1835008, None])))


if __name__ == "__main__":
    unittest.main()
