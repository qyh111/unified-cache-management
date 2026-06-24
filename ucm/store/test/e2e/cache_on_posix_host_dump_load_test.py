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
from typing import List

import numpy as np

from ucm.store.pipeline.connector import UcmPipelineStore


def make_store(
    storage_backends: List[str],
    unique_id: str,
    tensor_sizes: List[int],
    block_number: int,
) -> UcmPipelineStore:
    shard_size = sum(tensor_sizes)
    return UcmPipelineStore(
        {
            "store_pipeline": "Cache|Posix",
            "storage_backends": storage_backends,
            "unique_id": unique_id,
            "device_id": int(os.environ.get("UCM_TEST_DEVICE_ID", "0")),
            "tensor_size_list": tensor_sizes,
            "shard_size": shard_size,
            "block_size": shard_size,
            "cache_buffer_capacity_gb": 1,
            "share_buffer_enable": True,
            "waiting_queue_depth": max(16, block_number),
            "running_queue_depth": max(1024, block_number * 4),
            "io_direct": os.environ.get("UCM_TEST_IO_DIRECT", "1") != "0",
            "cache_use_host_buffer": True,
            "posix_data_trans_concurrency": 8,
            "posix_lookup_concurrency": 8,
            "timeout_ms": 10000,
        }
    )


def make_host_buffers(block_number: int, tensor_sizes: List[int]) -> List[List[np.ndarray]]:
    buffers: List[List[np.ndarray]] = []
    for block_idx in range(block_number):
        row = []
        for tensor_idx, size in enumerate(tensor_sizes):
            data = np.arange(size, dtype=np.uint32)
            data = (data + block_idx * 31 + tensor_idx * 17) & 0xFF
            row.append(data.astype(np.uint8))
        buffers.append(row)
    return buffers


def to_addr_array(buffers: List[List[np.ndarray]]) -> np.ndarray:
    return np.array(
        [[tensor.ctypes.data for tensor in row] for row in buffers], dtype=np.uint64
    )


def wait_until_committed(store: UcmPipelineStore, block_ids: List[bytes]) -> None:
    deadline = time.perf_counter() + 10.0
    while time.perf_counter() < deadline:
        if all(store.lookup(block_ids)):
            return
        time.sleep(0.01)
    raise TimeoutError("blocks were not committed to PosixStore in time")


def main() -> None:
    tensor_sizes = [4096, 8192, 16384]
    shard_size = sum(tensor_sizes)
    assert shard_size % 4096 == 0
    block_number = 8
    shard_indexes = [0] * block_number
    block_ids = [secrets.token_bytes(16) for _ in range(block_number)]

    with tempfile.TemporaryDirectory(
        prefix="ucm-cache-posix-host-",
        dir=os.environ.get("UCM_TEST_STORAGE_BACKEND_PARENT"),
    ) as storage_dir:
        storage_backends = [storage_dir]
        source = make_host_buffers(block_number, tensor_sizes)
        writer = make_store(storage_backends, secrets.token_hex(8), tensor_sizes, block_number)
        task = writer.dump_data(block_ids, shard_indexes, to_addr_array(source))
        writer.wait(task)

        reader = make_store(storage_backends, secrets.token_hex(8), tensor_sizes, block_number)
        wait_until_committed(reader, block_ids)

        destination = [
            [np.zeros(size, dtype=np.uint8) for size in tensor_sizes]
            for _ in range(block_number)
        ]
        task = reader.load_data(block_ids, shard_indexes, to_addr_array(destination))
        reader.wait(task)

        for block_idx, (src_row, dst_row) in enumerate(zip(source, destination)):
            for tensor_idx, (src, dst) in enumerate(zip(src_row, dst_row)):
                if not np.array_equal(src, dst):
                    diff = np.flatnonzero(src != dst)
                    raise AssertionError(
                        f"mismatch at block={block_idx}, tensor={tensor_idx}, "
                        f"first_diff={diff[:10]}"
                    )


if __name__ == "__main__":
    os.environ.setdefault("UC_LOGGER_LEVEL", "info")
    main()
