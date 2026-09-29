"""Model-check byte oracle for the Hybrid connector, independent of v2 Proxy.

Transfer ranges use the connector's registered layout. This checks dispatch and
roundtrip consistency, not an independent proof of its physical address parser.
"""

from __future__ import annotations

import hashlib
import os


def is_hybrid_worker(worker):
    connector = getattr(worker, "connector", worker)
    return any(
        cls.__module__ == "ucm.integration.vllm.hybrid_connector"
        and cls.__name__ == "UCMHybridConnector"
        for cls in type(connector).__mro__
    )


def segments(worker, metadata, phase):
    """Yield logical key, fixed slot offset, actual pointer and payload length."""
    if not is_hybrid_worker(worker):
        raise TypeError("Hybrid checker requires UCMHybridConnector")
    layout = getattr(worker, "connector", worker).kv_cache_layout
    if layout is None:
        raise ValueError("Hybrid KV caches are not registered")
    offsets = layout.ucm_block_offsets
    for request in metadata.requests.values():
        for plan in getattr(request, f"{phase}_plans"):
            for row in range(layout.row_count):
                transfer = layout.resolve(plan, row)
                for key, ptrs in zip(transfer.keys, transfer.ptrs, strict=True):
                    for col, ptr in enumerate(ptrs):
                        if ptr:  # Ghost slots have no device allocation.
                            yield (
                                bytes(key),
                                int(offsets[row, col]),
                                int(ptr),
                                int(layout.sizes[col]),
                            )


def payload(key, offset, size):
    # Include TP rank; a mistaken cross-rank load must not pass.
    seed = hashlib.sha256(
        bytes(key)
        + int(offset).to_bytes(8, "little")
        + int(os.getenv("RANK", "0")).to_bytes(4, "little")
    ).digest()
    pattern = bytes((seed[i % len(seed)] + i) % 251 for i in range(251))
    return (pattern * ((size + 250) // 251))[:size]


class TensorByteAccess:
    """Bound raw pointer operations to actual registered Torch allocations."""

    def __init__(self, kv_caches):
        import torch

        self.buffers = []
        self.devices = set()
        seen = set()
        for value in kv_caches.values():
            for tensor in value if isinstance(value, (tuple, list)) else (value,):
                storage = tensor.untyped_storage()
                base, size = int(storage.data_ptr()), int(storage.nbytes())
                if not base or size <= 0 or (base, size) in seen:
                    continue
                seen.add((base, size))
                view = torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
                    storage, 0, (size,), (1,)
                )
                self.buffers.append((base, base + size, view))
                self.devices.add(tensor.device)
        if not self.buffers:
            raise ValueError("No registered KV allocations")

    def synchronize(self):
        import torch

        for device in self.devices:
            if device.type in ("cuda", "npu"):
                getattr(torch, device.type).synchronize(device)

    def view(self, ptr, size):
        for base, end, tensor in self.buffers:
            if base <= ptr and ptr + size <= end:
                return tensor[ptr - base : ptr - base + size]
        raise ValueError(f"Hybrid pointer range outside allocations: {ptr}, {size}")

    def read(self, ptr, size):
        return self.view(ptr, size).cpu().numpy().tobytes()

    def write(self, ptr, data):
        import torch

        target = self.view(ptr, len(data))
        source = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        target.copy_(source.to(target.device))


def _check(fixture, worker, metadata, mode):
    access = TensorByteAccess(fixture.kv_caches)
    access.synchronize()
    count = 0
    for key, offset, ptr, size in segments(
        worker, metadata, "dump" if mode == "fill" else "load"
    ):
        expected = payload(key, offset, size)
        if mode == "compare":
            if access.read(ptr, size) != expected:
                raise AssertionError(
                    f"Hybrid byte mismatch: key={key.hex()}, offset={offset}, "
                    f"ptr={ptr}, size={size}"
                )
        else:
            # Every destination byte differs from the expected value, so a
            # skipped or partial load cannot pass merely because of allocator contents.
            access.write(
                ptr,
                (
                    expected
                    if mode == "fill"
                    else bytes(value ^ 255 for value in expected)
                ),
            )
        count += 1
    access.synchronize()
    if not count:
        raise AssertionError(f"Hybrid {mode} selected no real payload segments")
    return count


def fill_hybrid(fixture, worker, metadata):
    return _check(fixture, worker, metadata, "fill")


def poison_hybrid(fixture, worker, metadata):
    return _check(fixture, worker, metadata, "poison")


def compare_hybrid(fixture, worker, metadata):
    return _check(fixture, worker, metadata, "compare")
