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
import ctypes
import os
import secrets
import time
from typing import List


def split_storage_backends(storage_backends: str) -> List[str]:
    backends = [path for path in storage_backends.split(":") if path]
    if not backends:
        raise ValueError("storage_backends must not be empty.")
    return backends


def setup_store(storage_backends, block_size, stream_number, timeout_ms):
    from ucm.store.nfsstore import ucmnfsstore

    backends = split_storage_backends(storage_backends)
    for path in backends:
        os.makedirs(path, exist_ok=True)

    store = ucmnfsstore.NFSStore()
    param = ucmnfsstore.NFSStore.Config(backends, block_size, True)
    param.transferDeviceId = -1
    param.transferStreamNumber = stream_number
    param.transferTimeoutMs = timeout_ms

    ret = store.Setup(param)
    if ret != 0:
        raise RuntimeError(f"Failed to initialize ucmnfsstore, errcode: {ret}.")
    return store


def make_host_buffers(count, layer_size, block_layer):
    return [
        [
            ctypes.create_string_buffer(os.urandom(layer_size), layer_size)
            for _ in range(block_layer)
        ]
        for _ in range(count)
    ]


def make_empty_host_buffers(src_buffers):
    return [
        [ctypes.create_string_buffer(len(buf.raw)) for buf in block]
        for block in src_buffers
    ]


def flatten_buffers(hashes, buffers):
    block_ids = []
    offsets = []
    addrs = []
    lengths = []
    for hash_id, block in zip(hashes, buffers):
        offset = 0
        for buf in block:
            length = len(buf.raw)
            block_ids.append(hash_id)
            offsets.append(offset)
            addrs.append(ctypes.addressof(buf))
            lengths.append(length)
            offset += length
    return block_ids, offsets, addrs, lengths


def dump_from_host(store, hashes, buffers):
    results = [store.Alloc(hash_id) for hash_id in hashes]
    if sum(results) != 0:
        raise RuntimeError(f"Alloc failed, results={results}.")

    block_ids, offsets, addrs, lengths = flatten_buffers(hashes, buffers)
    task_id = store.DumpFromHost(block_ids, offsets, addrs, lengths)
    if task_id <= 0:
        raise RuntimeError(f"DumpFromHost failed, task_id={task_id}.")

    ret = store.Wait(task_id)
    if ret != 0:
        raise RuntimeError(f"Wait dump task failed, task_id={task_id}, ret={ret}.")

    for hash_id in hashes:
        store.Commit(hash_id, True)

    return sum(lengths)


def load_to_host(store, hashes, buffers):
    for hash_id in hashes:
        if not store.Lookup(hash_id):
            raise RuntimeError(f"Lookup failed, block_id={hash_id}.")

    block_ids, offsets, addrs, lengths = flatten_buffers(hashes, buffers)
    task_id = store.LoadToHost(block_ids, offsets, addrs, lengths)
    if task_id <= 0:
        raise RuntimeError(f"LoadToHost failed, task_id={task_id}.")

    ret = store.Wait(task_id)
    if ret != 0:
        raise RuntimeError(f"Wait load task failed, task_id={task_id}, ret={ret}.")

    return sum(lengths)


def compare_buffers(expected, actual):
    for block_idx, (expected_block, actual_block) in enumerate(zip(expected, actual)):
        for layer_idx, (expected_buf, actual_buf) in enumerate(
            zip(expected_block, actual_block)
        ):
            if expected_buf.raw != actual_buf.raw:
                raise AssertionError(
                    f"Data mismatch at block={block_idx}, layer={layer_idx}."
                )


def store_all_hashes(hashes, hash_file):
    file_path = os.path.join(os.path.dirname(__file__), hash_file)
    with open(file_path, "w", encoding="utf-8") as file:
        for hash_id in hashes:
            file.write(hash_id + "\n")


def positive_check(params):
    for name, value in params.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")


def main():
    os.environ.setdefault("UC_LOGGER_LEVEL", "debug")

    storage_backends = "."
    block_number = 16
    batch_size = 4
    block_layer = 8
    host_layer_size = 4096
    stream_number = 8
    timeout_ms = 30000
    hash_file = "kvcache_host_block_hashes.txt"

    positive_check(
        {
            "block_number": block_number,
            "batch_size": batch_size,
            "block_layer": block_layer,
            "host_layer_size": host_layer_size,
            "stream_number": stream_number,
            "timeout_ms": timeout_ms,
        }
    )

    block_size = host_layer_size * block_layer
    store = setup_store(
        storage_backends,
        block_size,
        stream_number,
        timeout_ms,
    )

    hashes = [secrets.token_hex(16) for _ in range(block_number)]
    total_written = 0
    total_read = 0
    start_time = time.perf_counter()

    total_batches = (block_number + batch_size - 1) // batch_size
    for batch in range(total_batches):
        start = batch * batch_size
        end = min(start + batch_size, block_number)
        current_hashes = hashes[start:end]
        expected = make_host_buffers(
            len(current_hashes), host_layer_size, block_layer
        )
        actual = make_empty_host_buffers(expected)

        total_written += dump_from_host(store, current_hashes, expected)
        total_read += load_to_host(store, current_hashes, actual)
        compare_buffers(expected, actual)
        print(f"batch {batch + 1}/{total_batches} passed, blocks={len(current_hashes)}")

    elapsed = time.perf_counter() - start_time
    store_all_hashes(hashes, hash_file)
    print(
        "Host->SSD test passed: "
        f"blocks={block_number}, "
        f"block_size={block_size}, "
        f"written={total_written} bytes, "
        f"read={total_read} bytes, "
        f"elapsed={elapsed:.3f}s"
    )


if __name__ == "__main__":
    main()
