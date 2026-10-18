"""Checks that the independent oracle detects wrong bytes and store identities."""

import hashlib
import threading
from types import SimpleNamespace

import pytest


@pytest.fixture
def data():
    torch = pytest.importorskip("torch", reason="Byte helper checks require CPU torch")
    from connector.runtime.environment import prepare_python_ucm

    prepare_python_ucm()
    from connector.data_check import mock_store
    from connector.data_check.comparison import KVComparison, view_spans
    from connector.data_check.memory import Memory, generate

    return SimpleNamespace(
        torch=torch,
        store=mock_store,
        oracle=KVComparison,
        spans=view_spans,
        memory=Memory,
        generate=generate,
    )


def oracle(data):
    spec = SimpleNamespace(block_size=4, tokens_per_state=1, tp_replicated=False)
    native = SimpleNamespace(
        num_blocks=4,
        prefix_cacheable_group_ids=[0],
        kv_cache_groups=[SimpleNamespace(layer_names=["a", "b"], kv_cache_spec=spec)],
    )
    views = {n: data.torch.zeros((4, 16), dtype=data.torch.uint8) for n in ("a", "b")}
    return data.oracle(native, views, views)


def test_strides_and_storage_offsets(data):
    raw = data.torch.arange(64, dtype=data.torch.uint8)
    view = data.torch.as_strided(raw.view(data.torch.int16), (2, 2, 3), (12, 3, 1), 2)
    assert [s.tolist() for s in data.spans(view)] == [
        list(range(4, 16)),
        list(range(28, 40)),
    ]
    assert len(data.spans(raw.view(2, 4, 8).permute(1, 0, 2))) == 1


def test_independent_logical_identity_and_poison(data):
    source, target = oracle(data), oracle(data)
    expected = source.source([[1]], False, 1)
    assert expected == target.source([[3]], False, 1)
    target.target([[2]], 4, expected)
    with pytest.raises(AssertionError, match="Restored KV differs"):
        target.compare_layer("a")
    target.views["a"][2].copy_(source.views["b"][1])
    with pytest.raises(AssertionError, match="Restored KV differs"):
        target.compare_layer("a")
    for name in source.views:
        target.views[name][2].copy_(source.views[name][1])
        target.compare_layer(name)
    assert target.finish() == (2, 32)


def test_tp_shards_and_mla_replication(data):
    a, b = oracle(data), oracle(data)
    assert a.source([[1]], False, 0) != b.source([[1]], False, 1)
    assert a.source([[1]], True, 0) == b.source([[1]], True, 1)
    b.prefix_identity = "different-native-request-prefix"
    assert a.source([[1]], True, 0) != b.source([[1]], True, 0)


def store_fixture(data):
    tensor = data.torch.empty(8192, dtype=data.torch.uint8)
    memory = data.memory([tensor])
    memory.seed({tensor.untyped_storage()._cdata: "kv"})
    ledger, lock = {}, threading.RLock()
    data.store.bind(memory, ledger, lock)
    config = dict(
        unique_id="test", tensor_size_list=[4096], block_size=8192, shard_size=4096
    )
    return tensor, memory, ledger, lock, data.store.HashStore(config)


def test_cold_store_and_opaque_shards(data):
    tensor, source, ledger, lock, store = store_fixture(data)
    for shard in range(2):
        store.wait(
            store.dump_data([b"key"], [shard], [[tensor.data_ptr() + 4096 * shard]])
        )
    assert store.lookup([b"key"])[0]
    target = data.torch.full((8192,), 255, dtype=data.torch.uint8)
    target_memory = data.memory([target])
    data.store.bind(target_memory, ledger, lock)
    store.wait(store.load_data([b"key"], [1], [[target.data_ptr()]]))
    assert target[:4096].equal(tensor[4096:])
    with pytest.raises(data.store.StoreNotFoundError):
        store.wait(store.load_data([b"missing"], [0], [[target.data_ptr()]]))
    with pytest.raises(ValueError, match="outside registered"):
        store.wait(
            store.load_data([b"key"], [0], [[target.data_ptr() + target.numel()]])
        )
    assert all(isinstance(part[0], str) for part in ledger[("test", b"key", 0)][0][2])


def test_corruption_null_padding_and_recipe_fragments(data):
    tensor, memory, ledger, lock, store = store_fixture(data)
    seed = hashlib.sha256(b"alternate").hexdigest()
    replacement = data.generate(seed, 77, 4000)
    memory.write(tensor.data_ptr() + 3, replacement)
    memory.assign(
        tensor.data_ptr() + 3, len(replacement), [(seed, 77, len(replacement))]
    )
    assert memory.replay(memory.recipe(tensor.data_ptr(), 8192)) == memory.read(
        tensor.data_ptr(), 8192
    )
    store.wait(store.dump_data([b"key"], [0], [[tensor.data_ptr()]]))
    tensor[0] ^= 1
    with pytest.raises(AssertionError, match="fixture recipe"):
        store.dump_data([b"key"], [0], [[tensor.data_ptr()]])
    store.wait(store.dump_data([b"null"], [0], [[0]]))
    store.wait(store.load_data([b"null"], [0], [[0]]))
    with pytest.raises(AssertionError, match="absent source"):
        store.wait(store.load_data([b"null"], [0], [[tensor.data_ptr()]]))


@pytest.mark.parametrize("axis", [1, 2])
def test_partial_compute_preserves_history_and_future_poison(data, axis):
    spec = SimpleNamespace(block_size=4, tokens_per_state=1, tp_replicated=False)
    native = SimpleNamespace(
        num_blocks=4,
        prefix_cacheable_group_ids=[0],
        kv_cache_groups=[SimpleNamespace(layer_names=["a"], kv_cache_spec=spec)],
    )
    shape = (4, 4, 2, 3) if axis == 1 else (4, 2, 4, 3)
    source_view = data.torch.empty(shape, dtype=data.torch.uint8)
    target_view = data.torch.empty_like(source_view)
    source = data.oracle(native, {"a": source_view}, {"a": source_view}, {"a": axis})
    target = data.oracle(native, {"a": target_view}, {"a": target_view}, {"a": axis})
    source.source([[1]], False, 0)
    target.memory.poison()
    target.write_new_layer("a", [[1]], 1, 2, 0, False)
    index = [slice(None)] * 4
    index[0], index[axis] = 1, 1
    assert target_view[tuple(index)].equal(source_view[tuple(index)])
    index[axis] = 0
    assert bool((target_view[tuple(index)] == 255).all())
    index[axis] = slice(2, None)
    assert bool((target_view[tuple(index)] == 255).all())
    assert bool((target_view[2:] == 255).all())
