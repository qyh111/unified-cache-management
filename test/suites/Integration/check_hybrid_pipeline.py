"""Hardware byte-roundtrip for Hybrid + the existing Cache|Posix pipeline.

Run in a Linux UCM environment with compiled native libraries and torch/vLLM:
  python test/suites/Integration/check_hybrid_pipeline.py --device npu:0 \
      --storage /tmp/ucm-hybrid-check --output hybrid-result.json

Uses synthetic KV allocations, not model weights or attention inference. Data
is copied to different physical blocks and compared using independent torch
indexing. Each run has a unique directory; existing caches are never removed.
"""

import argparse
import importlib.metadata
import json
import secrets
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", required=True, help="npu:0 or cuda:0 (use an idle device)"
    )
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("hybrid-result.json"))
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    import numpy as np
    import torch

    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401

        torch.npu.set_device(args.device)
    elif args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)
    else:
        parser.error("Only npu and cuda native transfer devices are supported")

    from ucm.integration.vllm.device import create_device
    from ucm.integration.vllm.hybrid.spec import (
        UCMLayerSpec,
        UCMKVCacheGroupInfo,
        UCMKVCacheSpec,
    )
    from ucm.integration.vllm.hybrid.spec_kind import KVCacheSpecKind
    from ucm.integration.vllm.hybrid.store_layout import HybridStoreLayout
    from ucm.integration.vllm.hybrid.scheduler import UCMGroupDispatchPlan
    from ucm.store.pipeline.connector import UcmPipelineStore

    run_root = args.storage / ("hybrid-" + secrets.token_hex(8))
    run_root.mkdir(parents=True)
    report = {
        "device": args.device,
        "storage": str(run_root),
        "python": sys.version,
        "packages": {},
        "cases": [],
        "success": False,
    }
    for package in ("torch", "torch-npu", "vllm", "vllm-ascend", "ucm"):
        try:
            report["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][package] = None
    device = create_device()

    def make_case(state):
        views, groups, backings = {}, [], []

        def add(name, widths):
            values = []
            for width in widths:
                # Leave 16 canary bytes between blocks; payload copies must
                # not overwrite them. Allocations do not share tensor storage.
                host = (
                    (torch.arange(5 * (width + 16)) % 251)
                    .to(torch.uint8)
                    .reshape(5, width + 16)
                )
                backing = host.to(args.device)
                values.append(backing[:, :width])
                backings.append((backing, width))
            views[name] = values[0] if len(values) == 1 else tuple(values)

        fa_names = (
            ["model.layers.3.attn"]
            if state
            else [f"model.layers.{i}.attn" for i in range(3)]
        )
        for name in fa_names:
            add(name, [64])
        if not state:
            add("model.layers.0.indexer", [32])
            add("model.layers.2.indexer", [16, 4])
            fa_names += ["model.layers.0.indexer", "model.layers.2.indexer"]
        layers = tuple(
            UCMLayerSpec(name, int(name.split(".")[2]), NS(block_size=4), 4, 5)
            for name in fa_names
        )
        groups.append(
            UCMKVCacheGroupInfo(
                0,
                layers,
                NS(block_size=4),
                4,
                frozenset({KVCacheSpecKind.FULL_ATTENTION}),
                tail_blocks=1,
            )
        )
        if state:
            state_names = [f"model.layers.{i}.mamba" for i in range(3)]
            for name in state_names:
                add(name, [12, 84])
            layers = tuple(
                UCMLayerSpec(
                    name, int(name.split(".")[2]), NS(block_size=4, shapes=()), 4, 5
                )
                for name in state_names
            )
            groups.append(
                UCMKVCacheGroupInfo(
                    1,
                    layers,
                    NS(block_size=4),
                    4,
                    frozenset({KVCacheSpecKind.MAMBA}),
                    tail_blocks=1,
                )
            )
        return (
            UCMKVCacheSpec(tuple(groups), 4, 4, args.device.split(":")[0]),
            views,
            backings,
        )

    try:
        for state in (False, True):
            for layerwise in (False, True):
                label = f"{'fa-mamba' if state else 'shared-indexer'}-{'layerwise' if layerwise else 'bulk'}"
                spec, views, backings = make_case(state)
                layout = HybridStoreLayout(spec, views, layerwise=layerwise)
                path = run_root / label
                path.mkdir()
                shard = (layout.shard_size + 4095) // 4096 * 4096
                config = dict(
                    store_pipeline="Cache|Posix",
                    unique_id=secrets.token_hex(8),
                    storage_backends=[str(path)],
                    device_id=int(args.device.split(":")[1]),
                    tensor_size_list=layout.tensor_size_list,
                    shard_size=shard,
                    block_size=shard * layout.row_count,
                    share_buffer_enable=False,
                    cache_buffer_capacity_gb=16,
                    cache_load_exclusive_buffer_number=8,
                    cache_load_backend_only=True,
                    cache_sdma_direct=False,
                    cache_io_aggregation=False,
                    use_gdr=False,
                    posix_gc_enable=False,
                    gpu_kv_buffer_addrs=layout.base_ptrs.tolist(),
                    gpu_kv_buffer_sizes=layout.buffer_sizes.tolist(),
                )
                worker = UcmPipelineStore(config)
                scheduler = UcmPipelineStore(config | {"device_id": -1})
                plans = []
                for kind, groups in spec.dispatch_routes():
                    plans.append(
                        UCMGroupDispatchPlan(
                            kind,
                            (secrets.token_bytes(16),),
                            0,
                            4,
                            tuple(np.array([1], dtype=np.uint64) for _ in groups),
                        )
                    )
                tasks = []
                event = device.get_event_handle()
                if not event:
                    device.synchronize()
                try:
                    for plan in plans:
                        for row in range(layout.row_count):
                            transfer = layout.resolve(plan, row)
                            tasks.append(
                                worker.dump_data(
                                    list(transfer.keys), [row], transfer.ptrs, event
                                )
                            )
                    for task in tasks:
                        worker.wait(task)
                finally:
                    if event:
                        device.destroy_event_handle(event)
                keys = [key for plan in plans for key in plan.keys]
                deadline = time.monotonic() + args.timeout
                while not all(scheduler.lookup(keys)):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"{label}: files not published within {args.timeout}s"
                        )
                    time.sleep(0.1)
                expected = []
                for backing, width in backings:
                    backing[3, :width].zero_()
                    reference = backing.cpu().clone()
                    reference[3, :width] = reference[1, :width]
                    expected.append(reference)
                device.synchronize()
                tasks = []
                for plan in plans:
                    target = UCMGroupDispatchPlan(
                        plan.hash_group,
                        plan.keys,
                        0,
                        4,
                        tuple(np.array([3], dtype=np.uint64) for _ in plan.windows),
                    )
                    for row in range(layout.row_count):
                        transfer = layout.resolve(target, row)
                        tasks.append(
                            worker.load_data(list(transfer.keys), [row], transfer.ptrs)
                        )
                for task in tasks:
                    worker.wait(task)
                device.synchronize()
                for (backing, _), reference in zip(backings, expected):
                    if not torch.equal(backing.cpu(), reference):
                        raise AssertionError(
                            f"{label}: payload or padding canary mismatch"
                        )
                report["cases"].append(
                    dict(
                        name=label,
                        success=True,
                        rows=layout.row_count,
                        tensor_size_list=layout.tensor_size_list,
                        shard_size=shard,
                        keys=[key.hex() for key in keys],
                    )
                )
                print(f"PASS {label}", flush=True)
                del worker, scheduler
        report["success"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
