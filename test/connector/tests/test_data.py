"""Infrastructure regressions; no connector implementation/metadata is inspected."""

import hashlib
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    raise unittest.SkipTest("Data/store checks require CPU torch") from error
from connector.data_check.comparison import KVComparison, view_spans
from connector.data_check.memory import Memory, generate
from connector.runtime.environment import prepare_python_ucm

prepare_python_ucm()
from connector.data_check import mock_store


def native(state=False):
    spec = SimpleNamespace(
        block_size=4, tokens_per_state=-1 if state else 1, tp_replicated=False
    )
    return SimpleNamespace(
        num_blocks=4,
        prefix_cacheable_group_ids=[0],
        kv_cache_groups=[SimpleNamespace(layer_names=["a", "b"], kv_cache_spec=spec)],
    )


def fixture(state=False):
    views = {name: torch.zeros((4, 16), dtype=torch.uint8) for name in ("a", "b")}
    return KVComparison(native(state), views, views)


class KVComparisonTest(unittest.TestCase):
    def test_strided_typed_view_keeps_offset_and_holes(self):
        backing = torch.arange(64, dtype=torch.uint8)
        view = torch.as_strided(backing.view(torch.int16), (2, 2, 3), (12, 3, 1), 2)
        spans = view_spans(view)
        self.assertEqual(
            [s.tolist() for s in spans], [list(range(4, 16)), list(range(28, 40))]
        )
        self.assertEqual(view_spans(torch.empty((4, 0))), [])
        self.assertEqual(len(view_spans(backing.view(2, 4, 8).permute(1, 0, 2))), 1)

    def test_poison_rejects_noop_wrong_layer_and_wrong_destination(self):
        source, target = fixture(), fixture()
        expected = source.source([[1]], False, 1)
        self.assertNotEqual(expected[(0, 0, "a", 0, 0)], expected[(0, 0, "b", 0, 0)])
        target.target([[2]], 4, expected)
        with self.assertRaisesRegex(AssertionError, "Restored KV differs"):
            target.compare_layer("a")
        target.views["a"][2].copy_(source.views["b"][1])
        with self.assertRaises(AssertionError):
            target.compare_layer("a")
        target.views["a"][3].copy_(source.views["a"][1])
        with self.assertRaises(AssertionError):
            target.compare_layer("a")
        for name in source.views:
            target.views[name][2].copy_(source.views[name][1])
            target.compare_layer(name)
        self.assertEqual(target.finish(), (2, 32))

    def test_state_boundary_and_running_slots_both_checked(self):
        source, target = fixture(True), fixture(True)
        expected = source.source([[0, 1]], True, 1)
        target.target([[0, 2, 3]], 8, expected)
        for name in source.views:
            target.views[name][2].copy_(source.views[name][1])
        with self.assertRaises(AssertionError):
            target.compare_layer("a")
        for name in source.views:
            target.views[name][3].copy_(source.views[name][1])
            target.compare_layer(name)
        self.assertEqual(target.finish(), (4, 64))

    def test_rank_replication_comes_from_native_model_identity(self):
        a, b = fixture(), fixture()
        self.assertNotEqual(a.source([[1]], False, 0), b.source([[1]], False, 1))
        self.assertEqual(a.source([[1]], True, 0), b.source([[1]], True, 1))


class MockStoreTest(unittest.TestCase):
    def setUp(self):
        self.tensor = torch.empty(4 * 8192, dtype=torch.uint8)
        self.memory = Memory([self.tensor])
        self.memory.seed({self.tensor.untyped_storage()._cdata: "kv"})
        self.ledger, self.lock = {}, threading.RLock()
        mock_store.bind(self.memory, self.ledger, self.lock)
        self.ptr = self.tensor.data_ptr()
        self.config = dict(
            unique_id="engine",
            tensor_size_list=[8192],
            block_size=16384,
            shard_size=8192,
        )
        self.store = mock_store.HashStore(self.config)

    def dump(self, key=b"key", shard=0, offset=0):
        return self.store.dump_data([key], [shard], [[self.ptr + offset]])

    def test_opaque_shards_all_required_and_namespace_isolated(self):
        pending = self.dump()
        self.assertFalse(self.store.lookup([b"key"])[0])
        self.store.wait(pending)
        self.assertFalse(self.store.lookup([b"key"])[0])
        self.store.wait(self.dump(shard=1, offset=8192))
        self.assertTrue(self.store.lookup([b"key"])[0])
        other = mock_store.HashStore(dict(self.config, unique_id="other"))
        self.assertFalse(other.lookup([b"key"])[0])
        self.assertFalse(self.store.lookup([b"unknown"])[0])

    def test_cold_replay_uses_recorded_source_and_actual_destination(self):
        expected = hashlib.sha256(self.memory.read(self.ptr, 8192)).hexdigest()
        self.store.wait(self.dump())
        # Discard source allocation from the store's registry entirely.
        target = torch.full((8192,), 255, dtype=torch.uint8)
        target_memory = Memory([target])
        mock_store.bind(target_memory, self.ledger, self.lock)
        task = self.store.load_data([b"key"], [0], [[target.data_ptr()]])
        self.assertTrue(bool((target == 255).all()))
        self.store.wait(task)
        self.assertEqual(
            hashlib.sha256(memoryview(target.numpy())).hexdigest(), expected
        )
        # Ledger values contain only lengths, hex digests and replay recipes.
        record = self.ledger[("engine", b"key", 0)]
        size, digest, parts = record[0]
        self.assertIsInstance(digest, str)
        self.assertTrue(
            all(
                isinstance(seed, str) and isinstance(start, int) and isinstance(n, int)
                for seed, start, n in parts
            )
        )
        self.assertEqual(sum(p[2] for p in parts), size)
        self.assertNotEqual(
            self.memory.read(self.ptr, 4096), self.memory.read(self.ptr + 4096, 4096)
        )

    def test_wrong_key_shard_corruption_and_bounds_fail(self):
        self.store.wait(self.dump())
        for key, shard in ((b"missing", 0), (b"key", 1)):
            task = self.store.load_data([key], [shard], [[self.ptr + 8192]])
            with self.assertRaises(mock_store.StoreNotFoundError):
                self.store.wait(task)
        with self.assertRaises(ValueError):
            self.dump(offset=self.tensor.numel())
        with self.assertRaises(ValueError):
            self.dump(shard=2)
        self.tensor[0] ^= 1
        with self.assertRaisesRegex(AssertionError, "fixture recipe"):
            self.dump()
        size, digest, parts = self.ledger[("engine", b"key", 0)][0]
        self.ledger[("engine", b"key", 0)] = [(size, "corrupt", parts)]
        task = self.store.load_data([b"key"], [0], [[self.ptr + 8192]])
        for _ in range(2):
            with self.assertRaisesRegex(AssertionError, "Corrupt"):
                self.store.check(task)

    def test_null_padding_reserves_slot_without_dereference(self):
        task = self.store.dump_data([b"padding"], [0], [[0]])
        self.store.wait(task)
        self.store.wait(self.store.load_data([b"padding"], [0], [[0]]))
        with self.assertRaisesRegex(AssertionError, "absent source"):
            self.store.wait(self.store.load_data([b"padding"], [0], [[self.ptr]]))

    def test_interval_replay_preserves_fragments_and_aliases(self):
        alias = Memory([self.tensor, self.tensor[3:]])
        self.assertEqual(len(alias.buffers), 1)
        seed = hashlib.sha256(b"alternate").hexdigest()
        data = generate(seed, 77, 9000)
        self.memory.write(self.ptr + 3, data)
        self.memory.assign(self.ptr + 3, len(data), [(seed, 77, len(data))])
        parts = self.memory.recipe(self.ptr, 16384)
        self.assertEqual(self.memory.replay(parts), self.memory.read(self.ptr, 16384))
        self.assertGreater(len(parts), 1)
        self.store.wait(self.dump())


if __name__ == "__main__":
    unittest.main(verbosity=2)
