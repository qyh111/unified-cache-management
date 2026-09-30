"""Portable schema/dispatch/address tests, using NumPy rather than an engine.

The FAWA classes are compiled verbatim from their AST to avoid importing
torch/vllm/native store libraries. Only device tensors and services are faked.
"""

import ast
import importlib.util
import math
import sys
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / "ucm/integration/vllm"
loader = importlib.util.spec_from_file_location(
    "dsv41_layout", PATH / "dsv41_layout.py"
)
schema = importlib.util.module_from_spec(loader)
sys.modules[loader.name] = schema
loader.loader.exec_module(schema)


class Tensor:
    def __init__(self, a):
        self.a = a
        self.shape = a.shape
        self.dtype = a.dtype

    def dim(self):
        return self.a.ndim

    def stride(self, axis=None):
        s = tuple(x // self.a.itemsize for x in self.a.strides)
        return s if axis is None else s[axis]

    def element_size(self):
        return self.a.itemsize

    def data_ptr(self):
        return self.a.ctypes.data

    def __getitem__(self, key):
        return Tensor(self.a[key])


source = ast.parse((PATH / "hma_connector.py").read_text(encoding="utf-8"))
classes = {
    "KVCacheGroupLayout",
    "KVCacheGroupMeta",
    "FAWARequestMeta",
    "FAWARequestDispatchMeta",
    "UCMFAWAConnector",
    "UCMFAWAConnectorMetadata",
}
module = ast.Module(
    body=[
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
    ]
    + [
        node
        for node in source.body
        if isinstance(node, ast.ClassDef) and node.name in classes
    ],
    type_ignores=[],
)
env = dict(
    np=np,
    math=math,
    dataclass=dataclass,
    field=field,
    Tuple=tuple,
    torch=NS(Tensor=Tensor),
    UCMDirectConnector=type("Base", (), {}),
    KVConnectorMetadata=type("Metadata", (), {}),
    SupportsHMA=type("SupportsHMA", (), {}),
    extract_layer_index=lambda name: int(name.split(".layers.")[1].split(".")[0]),
    logger=NS(info=lambda *a: None, info_once=lambda *a: None),
    compile_dsv41_layout=schema.compile_dsv41_layout,
)
exec(
    compile(ast.fix_missing_locations(module), str(PATH / "hma_connector.py"), "exec"),
    env,
)
Connector, Layout = env["UCMFAWAConnector"], env["KVCacheGroupLayout"]


def spec(cls, **kw):
    obj = type(cls, (), {})()
    obj.__dict__.update(kw)
    return obj


def fixture(ascend=False):
    prefix = "Ascend" if ascend else ""

    def cache(block=128, ratio=1, index=False, window=None):
        return spec(
            prefix + ("SlidingWindowMLASpec" if window else "MLAAttentionSpec"),
            block_size=block,
            num_kv_heads=1,
            head_size=128 if index else 512,
            dtype="int8" if index and ascend else "bfloat16" if ascend else "uint8",
            tokens_per_state=ratio,
            sliding_window=window,
            state_content_bytes=None if ascend else 132 if index else 584,
            storage_block_size=block // ratio,
            scale_dim=1 if ascend and index else 0,
            scale_dtype="float16",
        )

    def group(layers, block):
        return NS(
            layer_names=list(layers),
            kv_cache_spec=NS(block_size=block, kv_cache_specs=layers),
        )

    main, ring = {}, {}
    for layer, ratio in ((2, 2), (8, 2), (14, 2), (20, 1)):
        owner = f"m.layers.{layer}.attn"
        main[owner + (".long_kv_cache" if ascend else "")] = cache(ratio=ratio)
        main[owner + ".indexer.k_cache"] = cache(ratio=ratio, index=True)
        if ratio == 2:
            ring[owner + ".compressor.state_cache"] = spec(
                "CircularBufferSpec",
                block_size=32 if ascend else 8,
                head_size=1024,
                dtype="float32",
                tokens_per_state=1,
            )
    groups = [group(main, 128), group(ring, 32 if ascend else 8)]
    block = 128 if ascend else 32
    for start in range(0, 40, 4):
        groups.append(
            group(
                {
                    f"m.layers.{i}.attn.swa_cache": cache(block=block, window=128)
                    for i in range(start, start + 4)
                },
                block,
            )
        )
    config = NS(
        speculative_config=None,
        parallel_config=NS(),
        model_config=NS(hf_text_config=NS(model_type="deepseek_v41_text")),
    )
    return NS(kv_cache_groups=groups), config


class SchemaTests(unittest.TestCase):
    def test_cpu_payload_excludes_native_padding(self):
        kv, config = fixture()
        plan = schema.compile_dsv41_layout(kv, config, ascend=False)
        self.assertEqual(plan.file_size, {"FA": 229376, "WA": 2990080})
        self.assertEqual(plan.groups[1].tail_tokens, 0)
        self.assertEqual(plan.groups[2].tail_blocks, 4)

    def test_npu_scale_is_per_row_and_bf16_is_two_bytes(self):
        kv, config = fixture(True)
        plan = schema.compile_dsv41_layout(kv, config, ascend=True)
        self.assertEqual(
            plan.groups[0].layers["m.layers.2.attn.indexer.k_cache"], (64, (8192, 128))
        )
        self.assertEqual(
            plan.groups[0].layers["m.layers.20.attn.long_kv_cache"], (128, (131072,))
        )
        self.assertEqual(plan.file_size, {"FA": 372736, "WA": 5242880})

    def test_rejects_wrong_owner_and_speculative(self):
        kv, config = fixture()
        config.speculative_config = NS(num_speculative_tokens=1)
        with self.assertRaisesRegex(ValueError, "MTP"):
            schema.compile_dsv41_layout(kv, config, ascend=False)
        config.speculative_config = None
        main = kv.kv_cache_groups[0]
        main.layer_names.remove("m.layers.2.attn")
        with self.assertRaisesRegex(ValueError, "paired C2"):
            schema.compile_dsv41_layout(kv, config, ascend=False)

    def test_rejects_unsupported_parallel_and_blocks(self):
        for field in (
            "pipeline_parallel_size",
            "decode_context_parallel_size",
            "prefill_context_parallel_size",
        ):
            kv, config = fixture()
            setattr(config.parallel_config, field, 2)
            with self.assertRaisesRegex(ValueError, field):
                schema.compile_dsv41_layout(kv, config, ascend=False)
        kv, config = fixture()
        kv.kv_cache_groups[0].kv_cache_spec.block_size = 256
        with self.assertRaises(ValueError):
            schema.compile_dsv41_layout(kv, config, ascend=False)

    def test_namespace_changes_with_schema_not_dictionary_order(self):
        kv, config = fixture()
        a = schema.compile_dsv41_layout(kv, config, ascend=False)
        kv.kv_cache_groups[0].layer_names.reverse()
        b = schema.compile_dsv41_layout(kv, config, ascend=False)
        self.assertEqual(a.namespace, b.namespace)
        kv.kv_cache_groups[0].kv_cache_spec.kv_cache_specs[
            "m.layers.2.attn"
        ].state_content_bytes = 528
        c = schema.compile_dsv41_layout(kv, config, ascend=False)
        self.assertNotEqual(a.namespace, c.namespace)


class DispatchTests(unittest.TestCase):
    def connector(self):
        kv, config = fixture()
        c = Connector.__new__(Connector)
        c._kv_cache_config, c._vllm_config = kv, config
        c.group_metas, c.fa_group_ids, c.window_group_ids = {}, [], []
        c._init_group_metas()
        c.wa_dump_block_wise = False
        return c

    def test_live_tail_only_ring_ids_are_never_indexed(self):
        c = self.connector()
        # A private ring has one ID, not four token-prefix IDs.
        self.assertEqual(c._slice_group_block_ids(1, [9], np.array([511]), False), [])
        self.assertEqual(
            c._slice_group_block_ids(2, list(range(1, 17)), np.array([511]), False),
            [13, 14, 15, 16],
        )

    def test_exact_prompt_hit_queries_previous_stored_even_boundary(self):
        c = self.connector()
        c.persist_token_threshold = c.load_tokens_threshold = 0
        c.requests_meta = {}
        c.request_block_hasher = lambda request: [b"a", b"b", b"c", b"d"]
        seen = []

        def lookup(keys):
            seen.append(keys)
            return len(keys)

        c._lookup_external_hit_blocks = lookup
        c._prefetch_hit_key_hotness = lambda *args: None
        hit, _ = c.get_num_new_matched_tokens(NS(num_tokens=512, request_id="req"), 0)
        self.assertEqual(seen, [[b"a", b"b", b"c"]])
        self.assertEqual(hit, 384)
        self.assertEqual(c.requests_meta["req"].token_processed, 384)

    def test_dispatch_only_publishes_aligned_non_null_window(self):
        for tokens, null_tail, expected in (
            (512, False, True),
            (513, False, False),
            (512, True, False),
        ):
            c = self.connector()
            req = NS(
                vllm_block_ids=(),
                hbm_hit_block_num=0,
                total_hit_block_num=0,
                num_token_ids=tokens,
                token_processed=0,
                ucm_block_ids=[b"a", b"b", b"c", b"d"],
            )
            ids = ([1, 2, 3, 4, 5], [9]) + tuple(list(range(1, 18)) for _ in range(10))
            if null_tail:
                ids[2][12] = 0
            dispatch = c._generate_dispatch_meta(req, tokens, ids)
            self.assertEqual(dispatch.dump_wa, expected)
            self.assertEqual(dispatch.dump_vllm_block_ids[1], [])
            self.assertEqual(len(dispatch.dump_keys), 4)
            if not expected:
                self.assertTrue(
                    all(not dispatch.dump_vllm_block_ids[g] for g in c.window_group_ids)
                )

    def test_mixed_npu_rows_and_scale_addresses(self):
        # Separate physical strides, including a nonzero view offset, as in
        # native overlay allocations. Expected addresses come from NumPy.
        views, rows = {}, {}
        keep = []
        for layer, nrows in ((2, 64), (20, 128)):
            parts = []
            for width, dtype in ((128, np.int8), (1, np.float16)):
                backing = np.zeros(3 * 147712 + 128, np.uint8)
                arr = np.ndarray(
                    (3, nrows, 1, width),
                    dtype=dtype,
                    buffer=backing,
                    offset=64,
                    strides=(
                        147712,
                        width * np.dtype(dtype).itemsize,
                        width * np.dtype(dtype).itemsize,
                        np.dtype(dtype).itemsize,
                    ),
                )
                keep.append(arr)
                parts.append(Tensor(arr))
            name = f"m.layers.{layer}.attn.indexer.k_cache"
            views[name] = tuple(parts)
            rows[name] = nrows
        layout = Layout(
            views,
            is_ascend_layout=True,
            expected_block_size=128,
            expected_layer_rows=rows,
        )
        self.assertEqual(
            layout.segment_tensor_size_list(128, 128), [8192, 128, 16384, 256]
        )
        self.assertEqual(
            layout.extract_addrs(np.array([2]))[0].tolist(),
            [a[2].ctypes.data for a in keep],
        )

    def test_wait_for_save_drains_inflight_before_return(self):
        c = self.connector()
        events = []
        c.fa_store = c.wa_store = NS(check=lambda task: False)
        c.tp_size = 1
        c.tp_dump_tasks = {
            ("req",): [NS(store=c.wa_store, task="pending", event_handle=7)]
        }
        c._get_connector_metadata = lambda: env["UCMFAWAConnectorMetadata"]()
        c._rank_consistency = NS(wait_dump=lambda task: events.append(("wait", task)))
        c.device = NS(
            destroy_event_handle=lambda handle: events.append(("destroy", handle))
        )
        c.wait_for_save()
        self.assertEqual(events, [("wait", "pending"), ("destroy", 7)])
        self.assertEqual(c.tp_dump_tasks, {})

    def test_rejects_wrong_physical_rows_or_gapped_content(self):
        name = "m.layers.2.attn"
        data = Tensor(np.zeros((3, 128, 1, 512), dtype=np.uint16))
        with self.assertRaisesRegex(ValueError, "block size mismatch"):
            Layout({name: data}, is_ascend_layout=True, expected_layer_rows={name: 64})
        data = Tensor(np.zeros((3, 64, 1, 1024), dtype=np.uint16)[:, :, :, ::2])
        with self.assertRaisesRegex(ValueError, "dense"):
            Layout({name: data}, is_ascend_layout=True, expected_layer_rows={name: 64})

    def test_poisoned_ring_is_not_read_after_even_restore(self):
        # Independent dependency oracle for C2: pair [2k, 2k+1]. Chunk
        # splits may be odd; the first post-restore chunk starts even.
        for capacity in (8, 32):
            ring = [None] * capacity
            start = 128
            for length in (1, 3, 5, 2, 1):
                inputs = list(range(start, start + length))
                for pos in inputs:
                    if pos % 2:
                        previous = (
                            pos - 1 if pos - 1 >= start else ring[(pos - 1) % capacity]
                        )
                        self.assertEqual(previous, pos - 1)
                for pos in inputs[-capacity:]:
                    ring[pos % capacity] = pos
                start += length


if __name__ == "__main__":
    unittest.main()
