"""The storage-neutral synchronous Proxy boundary used by connector v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable


class UCMProxyError(RuntimeError):
    """Normalized error raised by the v2 Proxy adapter."""


@runtime_checkable
class UCMProxy(Protocol):
    def lookup(self, block_ids: Sequence[bytes]) -> Sequence[bool]: ...

    def load(
        self,
        block_ids: Sequence[bytes],
        offsets: Sequence[int],
        ptrs: Sequence[int],
        sizes: Sequence[int],
    ) -> Any: ...

    def dump(
        self,
        block_ids: Sequence[bytes],
        offsets: Sequence[int],
        ptrs: Sequence[int],
        sizes: Sequence[int],
    ) -> Any: ...

    def wait(self, task: Any) -> None: ...


@dataclass(frozen=True)
class UCMProxyBatch:
    block_ids: tuple[bytes, ...]
    offsets: tuple[int, ...]
    ptrs: tuple[int, ...]
    sizes: tuple[int, ...]

    @property
    def total_bytes(self) -> int:
        return sum(self.sizes)


class UCMProxyAdapter:
    """Validate and normalize calls without knowing the backing Store."""

    def __init__(
        self,
        proxy: UCMProxy,
        record_sizes: dict[bytes, int] | None = None,
    ) -> None:
        self._proxy = proxy
        self._record_sizes = record_sizes or {}

    @staticmethod
    def _keys(block_ids: Sequence[bytes]) -> tuple[bytes, ...]:
        keys = tuple(bytes(key) for key in block_ids)
        invalid = [index for index, key in enumerate(keys) if len(key) != 16]
        if invalid:
            raise ValueError(f"UCM block IDs must be 16 bytes; invalid indexes={invalid}")
        return keys

    def lookup(self, block_ids: Sequence[bytes]) -> tuple[bool, ...]:
        keys = self._keys(block_ids)
        try:
            result = tuple(bool(value) for value in self._proxy.lookup(keys))
        except Exception as exc:
            raise UCMProxyError("Proxy lookup failed") from exc
        if len(result) != len(keys):
            raise UCMProxyError(
                f"Proxy lookup returned {len(result)} results for {len(keys)} keys"
            )
        return result

    def _wait(self, operation: str, task: Any) -> None:
        if task is None:
            return
        wait = getattr(self._proxy, "wait", None)
        if not callable(wait):
            raise UCMProxyError(
                f"Proxy {operation} returned an asynchronous task but does not "
                "provide wait(task)"
            )
        wait(task)

    def _batch(
        self,
        block_ids: Sequence[bytes],
        offsets: Sequence[int],
        ptrs: Sequence[int],
        sizes: Sequence[int],
    ) -> UCMProxyBatch:
        keys = self._keys(block_ids)
        normalized = (
            tuple(int(value) for value in offsets),
            tuple(int(value) for value in ptrs),
            tuple(int(value) for value in sizes),
        )
        lengths = {len(keys), *(len(values) for values in normalized)}
        if len(lengths) != 1:
            raise ValueError(
                "block_ids, offsets, ptrs and sizes must have identical lengths"
            )
        normalized_offsets, normalized_ptrs, normalized_sizes = normalized
        for index, (key, offset, ptr, size) in enumerate(
            zip(keys, normalized_offsets, normalized_ptrs, normalized_sizes)
        ):
            if offset < 0 or ptr <= 0 or size <= 0:
                raise ValueError(
                    f"Invalid Proxy segment at index {index}: "
                    f"offset={offset}, ptr={ptr}, size={size}"
                )
            record_size = self._record_sizes.get(key)
            if record_size is not None and offset + size > record_size:
                raise ValueError(
                    f"Proxy segment {index} exceeds record: "
                    f"offset={offset}, size={size}, record_size={record_size}"
                )
        return UCMProxyBatch(
            keys, normalized_offsets, normalized_ptrs, normalized_sizes
        )

    def load(
        self,
        block_ids: Sequence[bytes],
        offsets: Sequence[int],
        ptrs: Sequence[int],
        sizes: Sequence[int],
    ) -> None:
        batch = self._batch(block_ids, offsets, ptrs, sizes)
        if not batch.block_ids:
            return
        try:
            task = self._proxy.load(
                batch.block_ids, batch.offsets, batch.ptrs, batch.sizes
            )
            self._wait("load", task)
        except Exception as exc:
            if isinstance(exc, UCMProxyError):
                raise
            raise UCMProxyError("Proxy load failed") from exc

    def dump(
        self,
        block_ids: Sequence[bytes],
        offsets: Sequence[int],
        ptrs: Sequence[int],
        sizes: Sequence[int],
    ) -> None:
        batch = self._batch(block_ids, offsets, ptrs, sizes)
        if not batch.block_ids:
            return
        try:
            task = self._proxy.dump(
                batch.block_ids, batch.offsets, batch.ptrs, batch.sizes
            )
            self._wait("dump", task)
        except Exception as exc:
            if isinstance(exc, UCMProxyError):
                raise
            raise UCMProxyError("Proxy dump failed") from exc
