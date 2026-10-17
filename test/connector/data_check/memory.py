"""Deterministic CPU bytes and bounded address access, independent of UCM."""

from __future__ import annotations

import bisect
import hashlib

PAGE = 4096


def generate(seed: str, offset: int, size: int) -> bytes:
    """Random-access SHAKE stream; neighbouring pages have distinct contents."""
    result = bytearray()
    while size:
        block, start = divmod(offset, PAGE)
        count = min(size, PAGE - start)
        page = hashlib.shake_256(
            bytes.fromhex(seed) + block.to_bytes(8, "little")
        ).digest(PAGE)
        result.extend(page[start : start + count])
        offset += count
        size -= count
    return bytes(result)


class Memory:
    """Only registered CPU allocations may be read/written by the mock store.

    Recipes describe bytes, not tensors or addresses. They survive source
    worker destruction; no source pointer is dereferenced during load.
    """

    def __init__(self, tensors):
        from .comparison import storage_bytes

        storages = {}
        for tensor in tensors:
            if tensor.device.type != "cpu":
                raise ValueError("Mock memory requires CPU tensors")
            storages[tensor.untyped_storage()._cdata] = storage_bytes(tensor)
        self.buffers = sorted(storages.values(), key=lambda x: x.data_ptr())
        self.starts = [x.data_ptr() for x in self.buffers]
        self.recipes = []
        self.seeded = False

    def locate(self, address, size):
        address, size = int(address), int(size)
        if size < 0:
            raise ValueError("Negative transfer length")
        index = bisect.bisect_right(self.starts, address) - 1
        if (
            index < 0
            or address + size > self.starts[index] + self.buffers[index].numel()
        ):
            raise ValueError(
                f"Store pointer/size is outside registered KV memory: address={address}, size={size}"
            )
        return index, address - self.starts[index]

    def read(self, address, size):
        if not size:
            return b""
        index, offset = self.locate(address, size)
        return memoryview(self.buffers[index].numpy())[offset : offset + int(size)]

    def write(self, address, data):
        if not data:
            return
        index, offset = self.locate(address, len(data))
        memoryview(self.buffers[index].numpy())[offset : offset + len(data)] = data

    def assign(self, address, size, parts):
        """Replace a byte interval's replay recipe, splitting existing intervals."""
        if not size:
            return
        self.locate(address, size)
        end = address + size
        kept = []
        for start, stop, seed, stream in self.recipes:
            if stop <= address or start >= end:
                kept.append((start, stop, seed, stream))
            else:
                if start < address:
                    kept.append((start, address, seed, stream))
                if stop > end:
                    kept.append((end, stop, seed, stream + end - start))
        cursor = address
        for seed, stream, length in parts:
            kept.append((cursor, cursor + length, seed, stream))
            cursor += length
        if cursor != end:
            raise ValueError("Replay recipe length mismatch")
        self.recipes = sorted(kept)

    def recipe(self, address, size):
        if not size:
            return []
        self.locate(address, size)
        cursor, end, parts = int(address), int(address) + int(size), []
        for start, stop, seed, stream in self.recipes:
            if stop <= cursor:
                continue
            if start > cursor:
                break
            count = min(stop, end) - cursor
            parts.append((seed, stream + cursor - start, count))
            cursor += count
            if cursor == end:
                return parts
        raise ValueError("Dump contains bytes without a source/replay recipe")

    @staticmethod
    def replay(parts):
        return b"".join(generate(seed, offset, size) for seed, offset, size in parts)

    def seed(self, names, rank=0):
        """Source data is chosen by the fixture before connector calls."""
        for index, tensor in enumerate(self.buffers):
            name = names[tensor.untyped_storage()._cdata]
            seed = hashlib.sha256(repr((name, rank)).encode()).hexdigest()
            for offset in range(0, tensor.numel(), 1 << 20):
                data = generate(seed, offset, min(1 << 20, tensor.numel() - offset))
                self.write(tensor.data_ptr() + offset, data)
            self.assign(tensor.data_ptr(), tensor.numel(), [(seed, 0, tensor.numel())])
        self.seeded = True

    def poison(self):
        for tensor in self.buffers:
            tensor.fill_(255)
        self.recipes.clear()
