"""Captured-layout address oracle; no engine, accelerator or native store needed."""

import importlib.util
import json
import math
from pathlib import Path
import sys
import types
import unittest
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = "_glm53_layout_test"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "ucm/integration/vllm/hybrid/layout")]
sys.modules[PACKAGE] = package


def load(name):
    spec = importlib.util.spec_from_file_location(
        f"{PACKAGE}.{name}", Path(package.__path__[0]) / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


view = load("view")
load("policies")
glm = load("glm53")
ELEMENT = {"torch.uint8": 1, "torch.int8": 1, "torch.bfloat16": 2, "torch.float32": 4}


class Tensor:
    def __init__(self, data):
        self.data = dict(data)
        self.shape = tuple(data["shape"])

    def stride(self, axis):
        return self.data["stride"][axis]

    def element_size(self):
        return ELEMENT[self.data["dtype"]]

    def data_ptr(self):
        return 0x100000000 + self.data["offset"]

    def untyped_storage(self):
        return NS(
            data_ptr=lambda: 0x100000000, nbytes=lambda: self.data["storage_bytes"]
        )


def fixture(backend):
    return json.loads(
        (
            Path(__file__).with_name("fixtures") / f"glm53_{backend}_layout.json"
        ).read_text()
    )


def layer(record, blocks):
    return NS(
        layer_name=record["name"],
        layer_index=int(record["name"].split("layers.")[1].split(".")[0]),
        num_blocks=blocks,
        storage_block_size=1088 if "indexer" in record["name"] else 4352,
        descriptor=NS(
            **{
                k: record["descriptor"][k]
                for k in ("layers", "offset", "layer_stride", "block_stride")
            }
        ),
        descriptor_position=record["position"],
        kv_cache_spec=NS(page_size_bytes=record["descriptor"]["block_stride"]),
    )


class Glm53LayoutTests(unittest.TestCase):
    def test_captured_page_addresses(self):
        for backend in ("cuda", "npu"):
            capture = fixture(backend)
            blocks = capture["num_blocks"]
            for record in capture["tensors"]:
                if record["group"] == 1 or (backend == "npu" and record["group"] >= 2):
                    continue
                # NPU main's second component is zero-width rope, not payload.
                tensor = Tensor(record["views"][0])
                with self.subTest(backend=backend, name=record["name"]):
                    item = layer(record, blocks)
                    physical = (
                        glm.build_tiled_page_view(tensor, item)
                        if backend == "cuda"
                        else view.build_layer_view(
                            tensor, item, state_snapshot=False, device_type="npu"
                        )
                    )
                    segment = physical.segments[0]
                    rows = tensor.shape[0] // blocks
                    for block in (0, 1, blocks - 1):
                        # Independent oracle: locate the first and last kernel
                        # rows using the captured native tensor's element stride.
                        first = (
                            tensor.data_ptr()
                            + block * rows * tensor.stride(0) * tensor.element_size()
                        )
                        end = (
                            tensor.data_ptr()
                            + (block + 1)
                            * rows
                            * tensor.stride(0)
                            * tensor.element_size()
                        )
                        self.assertEqual(
                            segment.base_ptr + block * segment.block_stride_bytes, first
                        )
                        self.assertEqual(first + segment.payload_bytes, end)

    def test_policy_preserves_groups_and_missing_indexer(self):
        for backend, counts, index_bytes in (
            ("cuda", [11, 9, 9, 8, 8], 143616),
            ("npu", [11, 12, 11, 11], 278528),
        ):
            capture = fixture(backend)
            grouped = {}
            for record in capture["tensors"]:
                gid = record["group"]
                if gid == 1:
                    continue
                item = layer(record, capture["num_blocks"])
                if backend == "npu":
                    tensors = [
                        Tensor(raw)
                        for raw in record["views"]
                        if math.prod(raw["shape"]) > 0
                    ]
                    physical = view.build_layer_view(
                        tensors, item, state_snapshot=gid >= 2, device_type="npu"
                    )
                else:
                    physical = glm.build_tiled_page_view(
                        Tensor(record["views"][0]), item
                    )
                grouped.setdefault(gid, {}).setdefault(item.layer_index, []).append(
                    (item, physical)
                )
            rows = {
                gid: [
                    ("FA" if gid == 0 else "State", entries)
                    for _, entries in sorted(layers.items())
                ]
                for gid, layers in grouped.items()
            }
            policy, regions, mapped = glm.compile_glm53_policy(rows)
            self.assertEqual(policy, "glm53")
            self.assertEqual(regions, ("main", "index"))
            self.assertEqual(list(mapped), [0, *range(2, len(counts) + 1)])
            self.assertEqual([len(r) for r in mapped.values()], counts)
            for gid, mapped_rows in mapped.items():
                for row in mapped_rows:
                    payload = sum(s.payload_bytes for s in row["main"])
                    self.assertEqual(
                        payload, 4341760 if backend == "npu" and gid else 4456448
                    )
                    if gid == 0:
                        self.assertEqual(
                            sum(s.payload_bytes for s in row["index"]), index_bytes
                        )
                    else:
                        self.assertNotIn("index", row)

    def test_rejects_descriptor_mismatch_and_overrun(self):
        capture = fixture("cuda")
        record = capture["tensors"][0]
        for corruption in ("origin", "stride", "capacity", "position", "rows"):
            with self.subTest(corruption=corruption):
                t, item = Tensor(record["views"][0]), layer(
                    record, capture["num_blocks"]
                )
                if corruption == "origin":
                    item.descriptor.offset += 1
                elif corruption == "stride":
                    item.descriptor.block_stride += 1
                elif corruption == "capacity":
                    t.data["storage_bytes"] = (
                        t.data["offset"]
                        + item.num_blocks * item.kv_cache_spec.page_size_bytes
                        - 1
                    )
                elif corruption == "position":
                    item.descriptor_position = -1
                else:
                    t.shape = (t.shape[0] - 1, *t.shape[1:])
                with self.assertRaises(ValueError):
                    glm.build_tiled_page_view(t, item)

    def test_rejects_tail_page(self):
        capture = fixture("cuda")
        record = next(t for t in capture["tensors"] if t["group"] == 1)
        with self.assertRaisesRegex(ValueError, "tail"):
            glm.build_tiled_page_view(
                Tensor(record["views"][0]), layer(record, capture["num_blocks"])
            )

    def test_policy_rejects_incomplete_or_inconsistent_rows(self):
        def entry(name, payload):
            segment = view.MemorySegment(4096, 8192, 1, payload, payload)
            return (
                NS(layer_name=name),
                view.LayerView(
                    name, 0, (view.ComponentView((1, payload), (8192, 1), (segment,)),)
                ),
            )

        main = entry("layers.0.attn", 100)
        index = entry("layers.0.indexer.k_cache", 20)
        state = entry("layers.1.state", 90)
        cases = [
            {0: [("FA", [main])], 2: [("State", [state])]},
            {0: [("FA", [main, index, index])], 2: [("State", [state])]},
            {
                0: [("FA", [main, index])],
                2: [("State", [entry("layers.1.state", 101)])],
            },
            {0: [("FA", [main, index])], 1: [("WA", [state])]},
            {
                0: [("FA", [main, entry("layers.0.indexer.tail_cache", 20)])],
                2: [("State", [state])],
            },
        ]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                glm.compile_glm53_policy(rows)


if __name__ == "__main__":
    unittest.main()
