# SPDX-License-Identifier: MIT
"""Shared cross-process byte store with thin v1 and v2 mock adapters.

Both v1 and v2 connectors write real bytes from CPU KV pointers into the same
file-backed repository, while recording per-rank write audits so rank 0 can
detect out-of-bounds pointers, missing shards, and duplicate replica writes.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Callable
from unittest.mock import patch

import numpy as np


class BufferRegistry:
    """Tracks valid host pointer ranges belonging to allocated native KV views."""

    def __init__(self) -> None:
        self._ranges: list[tuple[int, int]] = []
        self._dynamic_providers: list[Any] = []

    def register(self, base_ptr: int, num_bytes: int) -> None:
        if base_ptr <= 0 or num_bytes <= 0:
            return
        self._ranges.append((int(base_ptr), int(base_ptr) + int(num_bytes)))
        self._ranges.sort()

    def register_dynamic_provider(self, provider) -> None:
        """Register a callback returning live (base_ptr, num_bytes) staging buffers."""
        self._dynamic_providers.append(provider)

    def assert_valid(self, ptr: int, num_bytes: int, context: str) -> None:
        if ptr < 0 or num_bytes < 0:
            raise AssertionError(
                f"Invalid KV range during {context}: {ptr=}, {num_bytes=}"
            )
        if ptr == 0 or num_bytes == 0:
            return
        end = ptr + num_bytes
        for lo, hi in self._ranges:
            if lo <= ptr and end <= hi:
                return
        for provider in self._dynamic_providers:
            for base_ptr, size in provider():
                if base_ptr > 0 and size > 0 and base_ptr <= ptr and end <= base_ptr + size:
                    return
        raise AssertionError(
            f"Out-of-bounds KV pointer during {context}: "
            f"ptr={ptr:#x}, bytes={num_bytes}, registered={len(self._ranges)} buffers"
        )


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _append_audit(audit_path: Path, record: dict[str, Any]) -> None:
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


class SharedFileByteStore:
    """Lock-free cross-process byte repository shared by all torchrun ranks."""

    def __init__(
        self,
        root: Path,
        audit_dir: Path,
        world_rank: int,
        *,
        dump_delay_polls: int = 0,
    ) -> None:
        self.root = Path(root)
        self.audit_dir = Path(audit_dir)
        self.world_rank = int(world_rank)
        self.dump_delay_polls = dump_delay_polls
        self.io_errors: list[AssertionError] = []
        self.pending_dumps = 0
        self.audit_file = self.audit_dir / f"rank_{self.world_rank}.jsonl"
        self.prefix_lookups: list[tuple[Path, tuple[bytes, ...]]] | None = None
        self.root.mkdir(parents=True, exist_ok=True)
        self.audit_dir.mkdir(parents=True, exist_ok=True)

    def _namespace_label(self, namespace_dir: Path) -> str:
        try:
            return str(namespace_dir.resolve().relative_to(self.root.resolve()))
        except ValueError:
            return str(namespace_dir.resolve())

    # ---- v1 shard-based storage --------------------------------------------

    def write_v1_shard(
        self,
        namespace_dir: Path,
        key: bytes,
        shard_index: int,
        num_shards: int,
        payload: bytes,
    ) -> None:
        key_hex = key.hex()
        shard_path = namespace_dir / f"{key_hex}.s{shard_index}.bin"
        _atomic_write_bytes(shard_path, payload)
        _append_audit(
            self.audit_file,
            {
                "version": "v1",
                "namespace": self._namespace_label(namespace_dir),
                "key": key_hex,
                "shard": int(shard_index),
                "num_shards": num_shards,
                "offset": 0,
                "size": len(payload),
                "rank": self.world_rank,
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
        )
        if all(
            (namespace_dir / f"{key_hex}.s{idx}.bin").is_file()
            for idx in range(num_shards)
        ):
            _atomic_write_bytes(namespace_dir / f"{key_hex}.committed", b"1")

    def read_v1_shard(
        self, namespace_dir: Path, key: bytes, shard_index: int
    ) -> bytes | None:
        shard_path = namespace_dir / f"{key.hex()}.s{shard_index}.bin"
        if not shard_path.is_file():
            return None
        return shard_path.read_bytes()

    def is_v1_committed(self, namespace_dir: Path, key: bytes) -> bool:
        return (namespace_dir / f"{key.hex()}.committed").is_file()

    # ---- v2 offset-span storage --------------------------------------------

    def write_v2_span(
        self, namespace_dir: Path, key: bytes, offset: int, payload: bytes
    ) -> None:
        key_hex = key.hex()
        span_dir = namespace_dir / f"{key_hex}.spans"
        span_path = span_dir / f"r{self.world_rank}_off{offset}_len{len(payload)}.bin"
        _atomic_write_bytes(span_path, payload)
        _append_audit(
            self.audit_file,
            {
                "version": "v2",
                "namespace": self._namespace_label(namespace_dir),
                "key": key_hex,
                "shard": 0,
                "offset": int(offset),
                "size": len(payload),
                "rank": self.world_rank,
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
        )

    def commit_v2_key(self, namespace_dir: Path, key: bytes) -> None:
        _atomic_write_bytes(namespace_dir / f"{key.hex()}.committed", b"1")

    def is_v2_committed(self, namespace_dir: Path, key: bytes) -> bool:
        return (namespace_dir / f"{key.hex()}.committed").is_file()

    def read_v2_span(
        self, namespace_dir: Path, key: bytes, offset: int, size: int
    ) -> bytes | None:
        if size == 0:
            return b""
        span_dir = namespace_dir / f"{key.hex()}.spans"
        if not span_dir.is_dir():
            return None
        out = bytearray(size)
        covered = bytearray(size)
        req_start = offset
        req_end = offset + size
        for span_file in sorted(span_dir.glob("*.bin")):
            parts = span_file.stem.split("_")
            span_offset = int(parts[1][3:])
            span_len = int(parts[2][3:])
            span_end = span_offset + span_len
            lo = max(req_start, span_offset)
            hi = min(req_end, span_end)
            if lo >= hi:
                continue
            data = span_file.read_bytes()
            src_lo = lo - span_offset
            dst_lo = lo - req_start
            length = hi - lo
            out[dst_lo : dst_lo + length] = data[src_lo : src_lo + length]
            covered[dst_lo : dst_lo + length] = b"\x01" * length
        if not all(covered):
            return None
        return bytes(out)


@dataclass
class _StoreTask:
    success: bool = True
    error: Exception | None = None
    pending_dump: Callable[[], None] | None = None
    remaining_polls: int = 0
    lock: Any = field(default_factory=Lock, repr=False)

    def complete(self) -> None:
        if self.pending_dump is not None:
            try:
                self.pending_dump()
            except Exception as exc:
                self.error = exc
            finally:
                self.pending_dump = None
        if self.error is not None:
            raise self.error


class MockStoreV1:
    """Drop-in UcmKVStoreBaseV1 implementation backed by SharedFileByteStore."""

    supports_distributed_backend_load = False

    def __init__(
        self,
        config: dict[str, Any],
        backing: SharedFileByteStore,
        buffers: BufferRegistry,
    ) -> None:
        self.config = dict(config)
        self._backing = backing
        self._buffers = buffers

        backends = config.get("storage_backends")
        if isinstance(backends, str):
            backends = [path for path in backends.split(":") if path]
        if backends:
            self._namespace_dir = Path(backends[0])
        else:
            unique_id = str(config.get("unique_id", "default"))
            self._namespace_dir = backing.root / unique_id
        self._namespace_dir.mkdir(parents=True, exist_ok=True)

        raw_sizes = config.get("tensor_size_list") or []
        self._tensor_sizes = [int(size) for size in raw_sizes]
        self._shard_size = int(config.get("shard_size") or sum(self._tensor_sizes) or 1)
        self._block_size = int(config.get("block_size") or self._shard_size)
        if self._shard_size <= 0 or any(size < 0 for size in self._tensor_sizes):
            raise AssertionError(f"Invalid v1 Store geometry: {config}")
        self._num_shards, remainder = divmod(self._block_size, self._shard_size)
        if (
            self._num_shards <= 0
            or remainder
            or sum(self._tensor_sizes) > self._shard_size
        ):
            raise AssertionError(f"Invalid v1 Store geometry: {config}")

    def cc_store(self) -> int:
        return 0

    def lookup(self, block_ids: list[bytes]) -> list[bool]:
        return [
            self._backing.is_v1_committed(self._namespace_dir, key)
            for key in block_ids
        ]

    def lookup_on_prefix(self, block_ids: list[bytes]) -> int:
        if self._backing.prefix_lookups is not None:
            self._backing.prefix_lookups.append((self._namespace_dir, tuple(block_ids)))
        for index, key in enumerate(block_ids):
            if not self._backing.is_v1_committed(self._namespace_dir, key):
                return index - 1
        return len(block_ids) - 1

    def lookup_on_reverse(self, block_ids: list[bytes]) -> int:
        for index in range(len(block_ids) - 1, -1, -1):
            if self._backing.is_v1_committed(self._namespace_dir, block_ids[index]):
                return index
        return -1

    def prefetch(self, block_ids: list[bytes]) -> None:
        return None

    def _normalize_ptr_rows(self, ptrs: Any) -> list[list[int]]:
        if isinstance(ptrs, np.ndarray):
            arr = ptrs
            if arr.ndim == 1:
                if not arr.size:
                    return []
                if not self._tensor_sizes:
                    raise AssertionError("v1 Store received pointers without tensor sizes")
                arr = arr.reshape(-1, len(self._tensor_sizes))
            if arr.ndim != 2:
                raise AssertionError(f"Invalid v1 pointer matrix shape: {arr.shape}")
            return [[int(value) for value in row] for row in arr.tolist()]
        rows = list(ptrs)
        if rows and not isinstance(rows[0], (list, tuple, np.ndarray)):
            return [[int(value)] for value in rows]
        return [[int(value) for value in row] for row in rows]

    def _validate_io(self, block_ids, shard_indices, ptrs, operation):
        rows = self._normalize_ptr_rows(ptrs)
        if len(block_ids) != len(shard_indices) or len(block_ids) != len(rows):
            raise AssertionError(
                f"v1 {operation} row count mismatch: keys={len(block_ids)}, "
                f"shards={len(shard_indices)}, rows={len(rows)}"
            )
        for key, shard, row in zip(block_ids, shard_indices, rows, strict=True):
            if not 0 <= shard < self._num_shards:
                raise AssertionError(
                    f"v1 {operation} shard {shard} outside [0, {self._num_shards})"
                )
            if len(row) != len(self._tensor_sizes):
                raise AssertionError(
                    f"v1 {operation} column count mismatch: ptrs={len(row)}, "
                    f"tensor_size_list={len(self._tensor_sizes)}"
                )
            for ptr, size in zip(row, self._tensor_sizes, strict=True):
                self._buffers.assert_valid(ptr, size, f"v1 {operation} {key.hex()[:8]}")
        return rows

    def dump_data(
        self,
        block_ids: list[bytes],
        shard_index: list[int],
        src_addr: Any,
        prerequisite_handle: int = 0,
    ) -> _StoreTask:
        try:
            rows = self._validate_io(block_ids, shard_index, src_addr, "dump_data")
        except (AssertionError, ValueError) as exc:
            self._backing.io_errors.append(AssertionError(str(exc)))
            raise
        keys = tuple(block_ids)
        shards = tuple(shard_index)

        def read_rows():
            for row in rows:
                yield b"".join(
                    ctypes.string_at(ptr, size) if ptr else bytes(size)
                    for ptr, size in zip(row, self._tensor_sizes, strict=True)
                )

        def persist(payloads):
            for key, shard, payload in zip(keys, shards, payloads, strict=True):
                self._backing.write_v1_shard(
                    self._namespace_dir,
                    key,
                    int(shard),
                    self._num_shards,
                    payload,
                )

        if self._backing.dump_delay_polls and keys:
            original = [hashlib.sha256(payload).digest() for payload in read_rows()]

            def complete_dump():
                try:
                    payloads = list(read_rows())
                    if [
                        hashlib.sha256(payload).digest() for payload in payloads
                    ] != original:
                        raise AssertionError(
                            "v1 Dump source changed before Store completion"
                        )
                    persist(payloads)
                except Exception as exc:
                    self._backing.io_errors.append(AssertionError(str(exc)))
                    raise
                finally:
                    self._backing.pending_dumps -= 1

            self._backing.pending_dumps += 1
            return _StoreTask(
                pending_dump=complete_dump,
                remaining_polls=self._backing.dump_delay_polls,
            )
        persist(read_rows())
        return _StoreTask()

    def load_data(
        self,
        block_ids: list[bytes],
        shard_index: list[int],
        dst_addr: Any,
        prerequisite_handle: int = 0,
        **kwargs,
    ) -> _StoreTask:
        from ucm.store.pipeline.errors import StoreNotFoundError

        try:
            rows = self._validate_io(block_ids, shard_index, dst_addr, "load_data")
        except (AssertionError, ValueError) as exc:
            self._backing.io_errors.append(AssertionError(str(exc)))
            raise
        expected_len = sum(self._tensor_sizes)
        for key, shard_id, ptr_row in zip(block_ids, shard_index, rows, strict=True):
            payload = self._backing.read_v1_shard(
                self._namespace_dir, key, int(shard_id)
            )
            if payload is None:
                return _StoreTask(
                    success=False,
                    error=StoreNotFoundError(
                        f"Missing v1 shard {shard_id} for key {key.hex()} "
                        f"in {self._namespace_dir}"
                    ),
                )
            if len(payload) != expected_len:
                raise AssertionError(
                    f"v1 shard byte length mismatch for {key.hex()} s{shard_id}: "
                    f"stored={len(payload)}, expected={expected_len}"
                )
            cursor = 0
            for ptr, size in zip(ptr_row, self._tensor_sizes):
                if ptr != 0 and size > 0:
                    ctypes.memmove(ptr, payload[cursor : cursor + size], size)
                cursor += size
        return _StoreTask()

    def wait(self, task: _StoreTask) -> None:
        with task.lock:
            task.complete()

    def check(self, task: _StoreTask) -> bool:
        with task.lock:
            if task.pending_dump is not None and task.remaining_polls:
                task.remaining_polls -= 1
                return False
            task.complete()
            return True


class MockSchedulerClientV2:
    """Drop-in v2 StoreSchedulerClient backed by SharedFileByteStore."""

    def __init__(
        self,
        launch_config: dict[str, Any],
        runtime: dict[str, Any],
        backing: SharedFileByteStore,
    ) -> None:
        self._backing = backing
        self._namespace_dir = backing.root / "v2_shared"
        self._namespace_dir.mkdir(parents=True, exist_ok=True)
        self._next_token = 1

    def on_new_request(self, request_id: str) -> None:
        return None

    def on_request_finished(self, request_id: str) -> None:
        return None

    def lookup_on_prefix(
        self, request_id: str, keys: tuple[bytes, ...]
    ) -> tuple[int, bool]:
        for index, key in enumerate(keys):
            if not self._backing.is_v2_committed(self._namespace_dir, key):
                return index - 1, False
        return len(keys) - 1, False

    def lookup_on_reverse(
        self, request_id: str, keys: tuple[bytes, ...]
    ) -> tuple[int, bool]:
        for index in range(len(keys) - 1, -1, -1):
            if self._backing.is_v2_committed(self._namespace_dir, keys[index]):
                return index, False
        return -1, False

    def _make_token(self, keys: tuple[bytes, ...]):
        from ucm.integration.vllm.v2.store_client import AddressedToken

        token_id = self._next_token
        self._next_token += 1
        return AddressedToken(
            keys=tuple(keys),
            payload=token_id.to_bytes(8, "little"),
        )

    def prepare_get(self, request_id: str, keys: tuple[bytes, ...]):
        return self._make_token(keys)

    def complete_get(self, request_id: str, token: Any) -> None:
        return None

    def prepare_put(self, request_id: str, keys: tuple[bytes, ...]):
        return self._make_token(keys)

    def complete_put(
        self, request_id: str, token: Any, success: bool = True
    ) -> None:
        if not success:
            return
        for key in getattr(token, "keys", ()):
            self._backing.commit_v2_key(self._namespace_dir, key)

    def prepare_prefetch(self, request_id: str, keys: tuple[bytes, ...]):
        return self._make_token(keys)

    def complete_prefetch(self, request_id: str, success: bool) -> None:
        return None

    def prepare_offload(self) -> tuple:
        return ()

    def complete_offload(self, token: Any) -> None:
        return None

    def close(self) -> None:
        return None


class MockWorkerClientV2:
    """Drop-in v2 StoreWorkerClient backed by SharedFileByteStore."""

    def __init__(
        self,
        launch_config: dict[str, Any],
        runtime: dict[str, Any],
        backing: SharedFileByteStore,
        buffers: BufferRegistry,
    ) -> None:
        self._backing = backing
        self._buffers = buffers
        self._namespace_dir = backing.root / "v2_shared"
        self._namespace_dir.mkdir(parents=True, exist_ok=True)
        self._next_sub_job = 2

    def _make_task(
        self,
        job: Any,
        *,
        is_layerwise: bool = False,
        prerequisite_handle: int = 0,
        success: bool = True,
        error: Exception | None = None,
    ):
        from ucm.integration.vllm.v2.store_client import TransferTask

        sub_job_id = None
        if is_layerwise:
            sub_job_id = self._next_sub_job
            self._next_sub_job += 2
        task = TransferTask(
            job_id=int(getattr(job, "job_id", 1)),
            sub_job_id=sub_job_id,
            prerequisite_handle=int(prerequisite_handle),
            success=success,
        )
        task._mock_error = error
        return task

    def submit_put(
        self,
        job: Any,
        transfer_batches: tuple,
        *,
        is_layerwise: bool = False,
        prerequisite_handle: int = 0,
    ) -> Any:
        for batch in transfer_batches:
            for key, offsets, ptrs, sizes in zip(
                batch.ucm_keys,
                batch.store_offsets,
                batch.device_ptrs,
                batch.byte_sizes,
            ):
                for offset, ptr, size in zip(offsets, ptrs, sizes):
                    offset_i, ptr_i, size_i = int(offset), int(ptr), int(size)
                    if size_i == 0:
                        continue
                    self._buffers.assert_valid(ptr_i, size_i, f"v2 put {key.hex()[:8]}")
                    payload = ctypes.string_at(ptr_i, size_i)
                    self._backing.write_v2_span(
                        self._namespace_dir, key, offset_i, payload
                    )
        return self._make_task(
            job,
            is_layerwise=is_layerwise,
            prerequisite_handle=prerequisite_handle,
            success=True,
        )

    def submit_get(
        self,
        job: Any,
        transfer_batches: tuple,
        *,
        is_layerwise: bool = False,
        prerequisite_handle: int = 0,
    ) -> Any:
        from ucm.integration.vllm.v2.store_client import StoreIOError

        for batch in transfer_batches:
            for key, offsets, ptrs, sizes in zip(
                batch.ucm_keys,
                batch.store_offsets,
                batch.device_ptrs,
                batch.byte_sizes,
            ):
                for offset, ptr, size in zip(offsets, ptrs, sizes):
                    offset_i, ptr_i, size_i = int(offset), int(ptr), int(size)
                    if size_i == 0:
                        continue
                    payload = self._backing.read_v2_span(
                        self._namespace_dir, key, offset_i, size_i
                    )
                    if payload is None:
                        return self._make_task(
                            job,
                            is_layerwise=is_layerwise,
                            prerequisite_handle=prerequisite_handle,
                            success=False,
                            error=StoreIOError(
                                f"Missing v2 span [{offset_i}, {offset_i + size_i}) "
                                f"for key {key.hex()}"
                            ),
                        )
                    self._buffers.assert_valid(ptr_i, size_i, f"v2 get {key.hex()[:8]}")
                    ctypes.memmove(ptr_i, payload, size_i)
        return self._make_task(
            job,
            is_layerwise=is_layerwise,
            prerequisite_handle=prerequisite_handle,
            success=True,
        )

    def submit_prefetch(self, job: Any) -> Any:
        return self._make_task(job, success=True)

    def submit_offload(self, job: Any) -> Any:
        return self._make_task(job, success=True)

    def poll_finished(self) -> None:
        return None

    def wait(self, task: Any) -> None:
        error = getattr(task, "_mock_error", None)
        if error is not None:
            raise error

    def close(self) -> None:
        return None


@contextmanager
def install_mock_store(
    connector_version: str,
    backing: SharedFileByteStore,
    buffers: BufferRegistry,
    *,
    use_host_device: bool,
):
    """Patch v1 or v2 Store creation and (on CPU) v1 accelerator device helpers."""
    from contextlib import ExitStack

    with ExitStack() as stack:
        if connector_version == "v1":
            if not use_host_device:
                raise ValueError("v1 mock Store requires CPU KV buffers")
            from ucm.store.factory_v1 import UcmConnectorFactoryV1

            def create_mock_v1(name, config, module_path=None):
                return MockStoreV1(config, backing, buffers)

            stack.enter_context(
                patch.object(
                    UcmConnectorFactoryV1,
                    "create_connector",
                    staticmethod(create_mock_v1),
                )
            )
            if use_host_device:
                from host_runtime import HostDeviceStub
                from ucm.integration.vllm import (
                    device as ucm_device,
                    hla_connector,
                    hma_connector,
                    ucm_connector,
                )

                stub = HostDeviceStub()
                for mod in (ucm_device, ucm_connector, hla_connector, hma_connector):
                    if hasattr(mod, "create_device"):
                        stack.enter_context(
                            patch.object(mod, "create_device", lambda _s=stub: _s)
                        )
        elif connector_version == "v2":
            from ucm.integration.vllm.v2 import connector_scheduler, connector_worker

            stack.enter_context(
                patch.object(
                    connector_scheduler,
                    "StoreSchedulerClient",
                    lambda launch_config, runtime: MockSchedulerClientV2(
                        launch_config, runtime, backing
                    ),
                )
            )
            stack.enter_context(
                patch.object(
                    connector_worker,
                    "StoreWorkerClient",
                    lambda launch_config, runtime: MockWorkerClientV2(
                        launch_config, runtime, backing, buffers
                    ),
                )
            )
        else:
            raise ValueError(f"Unsupported connector version: {connector_version}")
        yield
