# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import os
import secrets
import tempfile
import time

import numpy as np

from ucm.store.pipeline.connector import UcmPipelineStore


def make_host_buffers(block_number: int, tensor_sizes: list[int]):
    buffers = []
    for block_idx in range(block_number):
        block_buffers = []
        for tensor_idx, tensor_size in enumerate(tensor_sizes):
            seed = block_idx * len(tensor_sizes) + tensor_idx
            data = (np.arange(tensor_size, dtype=np.uint32) + seed).astype(np.uint8)
            block_buffers.append(data)
        buffers.append(block_buffers)
    return buffers


def make_cache_store(
    storage_backend: str,
    tensor_sizes: list[int],
    device_id: int,
) -> UcmPipelineStore:
    shard_size = sum(tensor_sizes)
    config = {
        "store_pipeline": "Cache|Posix",
        "storage_backends": [storage_backend],
        "unique_id": secrets.token_hex(8),
        "device_id": device_id,
        "tensor_size_list": tensor_sizes,
        "shard_size": shard_size,
        "block_size": shard_size,
        "cache_dump_from_host": True,
        "cache_buffer_capacity_gb": 1,
        "share_buffer_enable": True,
        "io_direct": False,
        "waiting_queue_depth": 16,
        "running_queue_depth": 1024,
        "timeout_ms": 10000,
        "posix_data_trans_concurrency": 4,
        "posix_lookup_concurrency": 4,
    }
    return UcmPipelineStore(config)


def make_posix_reader(
    storage_backend: str,
    shard_size: int,
    device_id: int,
) -> UcmPipelineStore:
    config = {
        "store_pipeline": "Posix",
        "storage_backends": [storage_backend],
        "device_id": device_id,
        "tensor_size": shard_size,
        "shard_size": shard_size,
        "block_size": shard_size,
        "io_direct": False,
        "timeout_ms": 10000,
        "posix_data_trans_concurrency": 4,
        "posix_lookup_concurrency": 4,
    }
    return UcmPipelineStore(config)


def wait_until_committed(
    store: UcmPipelineStore,
    block_ids: list[bytes],
    timeout_seconds: float = 10.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if all(store.lookup(block_ids)):
            return
        time.sleep(0.01)
    raise TimeoutError("Timed out waiting for PosixStore to commit dumped blocks")


def main():
    device_id = int(os.environ.get("UCM_TEST_DEVICE_ID", "0"))
    tensor_sizes = [4096, 8192, 16384]
    shard_size = sum(tensor_sizes)
    block_number = 8
    block_ids = [secrets.token_bytes(16) for _ in range(block_number)]
    shard_indexes = [0] * block_number
    source_buffers = make_host_buffers(block_number, tensor_sizes)
    source_addrs = [
        [buffer.ctypes.data for buffer in block_buffers]
        for block_buffers in source_buffers
    ]

    with tempfile.TemporaryDirectory(prefix="ucm_cache_host_dump_") as storage_backend:
        cache_store = make_cache_store(storage_backend, tensor_sizes, device_id)
        dump_task = cache_store.dump_data(block_ids, shard_indexes, source_addrs)
        cache_store.wait(dump_task)

        posix_reader = make_posix_reader(storage_backend, shard_size, device_id)
        wait_until_committed(posix_reader, block_ids)

        destination_buffers = [
            np.zeros(shard_size, dtype=np.uint8) for _ in range(block_number)
        ]
        destination_addrs = [[buffer.ctypes.data] for buffer in destination_buffers]
        load_task = posix_reader.load_data(
            block_ids,
            shard_indexes,
            destination_addrs,
        )
        posix_reader.wait(load_task)

        for block_idx, block_buffers in enumerate(source_buffers):
            expected = np.concatenate(block_buffers)
            np.testing.assert_array_equal(expected, destination_buffers[block_idx])

    print(
        "Cache|Posix host dump test passed: "
        f"blocks={block_number}, shard_idx=0, tensor_sizes={tensor_sizes}"
    )


if __name__ == "__main__":
    os.environ.setdefault("UC_LOGGER_LEVEL", "debug")
    main()
