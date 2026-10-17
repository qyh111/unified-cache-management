"""Store dependency: keep digests/recipes only, replay into supplied CPU pointers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from ucm.store.pipeline.errors import StoreNotFoundError
from ucm.store.ucmstore_v1 import Task, UcmKVStoreBaseV1

MEMORY = LEDGER = LOCK = None


def bind(memory, ledger, lock):
    global MEMORY, LEDGER, LOCK
    MEMORY, LEDGER, LOCK = memory, ledger, lock


@dataclass
class Pending(Task):
    action: object
    complete: bool = False
    error: Exception | None = None

    def finish(self):
        if not self.complete:
            try:
                self.action()
            except Exception as error:
                self.error = error
            self.complete = True
        if self.error is not None:
            raise self.error


class HashStore(UcmKVStoreBaseV1):
    def __init__(self, config):
        super().__init__(config)
        self.namespace = config["unique_id"]
        self.sizes = tuple(int(x) for x in config.get("tensor_size_list", ()))
        schema = (self.namespace, "schema")
        if self.sizes:
            shards, remainder = divmod(
                int(config["block_size"]), int(config["shard_size"])
            )
            if remainder or shards < 1:
                raise ValueError("Invalid store shard geometry")
            with LOCK:
                old = LEDGER.get(schema)
                if old is not None and old != shards:
                    raise ValueError("Conflicting store schemas")
                LEDGER[schema] = shards

    def cc_store(self):
        raise AssertionError("Python mock store has no C++ pointer")

    def lookup(self, block_ids):
        count = LEDGER.get((self.namespace, "schema"), 0)
        return np.asarray(
            [
                count > 0
                and all(
                    (self.namespace, key, shard) in LEDGER for shard in range(count)
                )
                for key in block_ids
            ],
            dtype=bool,
        )

    def lookup_on_prefix(self, block_ids):
        for index, present in enumerate(self.lookup(block_ids)):
            if not present:
                return index - 1
        return len(block_ids) - 1

    def lookup_on_reverse(self, block_ids):
        hits = np.flatnonzero(self.lookup(block_ids))
        return int(hits[-1]) if len(hits) else -1

    def prefetch(self, block_ids):
        pass

    def _rows(self, keys, shards, addresses):
        addresses = np.asarray(addresses, dtype=np.uint64)
        if addresses.shape != (len(keys), len(self.sizes)) or len(keys) != len(shards):
            raise ValueError("Store key/shard/address dimensions differ")
        count = LEDGER[(self.namespace, "schema")]
        for key, shard, row in zip(keys, shards, addresses):
            shard = int(shard)
            if shard < 0 or shard >= count:
                raise ValueError("Invalid store shard index")
            yield (self.namespace, key, shard), row

    def dump_data(self, block_ids, shard_index, src_addr, prerequisite_handle=0):
        if prerequisite_handle:
            raise AssertionError("CPU mock must not receive a device event")
        records = []
        for identity, row in self._rows(block_ids, shard_index, src_addr):
            segments = []
            for address, size in zip(row, self.sizes):
                # Store SPI reserves the slot but skips nullptr/zero-sized
                # components, just like Trans::Stream's scatter/gather copies.
                if not int(address) or not size:
                    segments.append((size, None, None))
                    continue
                recipe = MEMORY.recipe(int(address), size)
                digest = hashlib.sha256(MEMORY.read(int(address), size)).hexdigest()
                if hashlib.sha256(MEMORY.replay(recipe)).hexdigest() != digest:
                    raise AssertionError("Dump source differs from its fixture recipe")
                segments.append((size, digest, recipe))
            records.append((identity, segments))

        def publish():
            with LOCK:
                for identity, segments in records:
                    previous = LEDGER.get(identity)
                    if previous is not None and previous != segments:
                        raise AssertionError(
                            "Conflicting payloads for the same store key/shard"
                        )
                    LEDGER[identity] = segments

        return Pending(publish)

    def load_data(self, block_ids, shard_index, dst_addr):
        rows = list(self._rows(block_ids, shard_index, dst_addr))

        def restore():
            for identity, row in rows:
                segments = LEDGER.get(identity)
                if segments is None:
                    raise StoreNotFoundError("Missing mock key/shard")
                if tuple(s[0] for s in segments) != self.sizes:
                    raise ValueError("Load segment sizes differ from dump")
                for address, (size, digest, recipe) in zip(row, segments):
                    if not int(address) or not size:
                        continue
                    if recipe is None:
                        raise AssertionError(
                            "Live destination maps to an absent source component"
                        )
                    data = MEMORY.replay(recipe)
                    if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
                        raise AssertionError("Corrupt replay record")
                    MEMORY.write(int(address), data)
                    MEMORY.assign(int(address), size, recipe)

        return Pending(restore)

    def dump(self, block_ids, shard_index, src_tensor):
        return self.dump_data(
            block_ids, shard_index, [[t.data_ptr() for t in row] for row in src_tensor]
        )

    def load(self, block_ids, shard_index, dst_tensor):
        return self.load_data(
            block_ids, shard_index, [[t.data_ptr() for t in row] for row in dst_tensor]
        )

    def wait(self, task):
        task.finish()

    def check(self, task):
        task.finish()
        return True


def summarize(ledger):
    """Observed store identities, for reporting only; never comparison input."""
    records = {}
    for identity in ledger.keys():
        entry = records.setdefault(identity[0], {"records": 0, "shards": set()})
        if len(identity) == 3:
            entry["records"] += 1
            entry["shards"].add(identity[2])
    for entry in records.values():
        entry["shards"] = sorted(entry["shards"])
    return records
