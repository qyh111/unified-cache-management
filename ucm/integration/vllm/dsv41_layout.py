"""Explicit FAWA schema for the captured V4.1 128-token, no-draft layout.

No engine or device imports: scheduler sizes and worker layout checks share
the same spec-derived schema. Engine allocation padding is not store payload.
"""

import hashlib
import json
from dataclasses import dataclass


def layer_specs(group):
    spec = group.kv_cache_spec
    nested = getattr(spec, "kv_cache_specs", None)
    return {
        name: nested[name] if nested is not None else spec for name in group.layer_names
    }


def _itemsize(dtype):
    name = str(dtype).removeprefix("torch.")
    sizes = {
        "uint8": 1,
        "int8": 1,
        "float8_e4m3fn": 1,
        "float16": 2,
        "bfloat16": 2,
        "float32": 4,
    }
    if name not in sizes:
        raise ValueError(f"Unsupported DSV4.1 cache dtype: {dtype}")
    return sizes[name]


@dataclass(frozen=True)
class Dsv41Group:
    kind: str
    block_size: int
    tail_tokens: int
    tail_blocks: int
    # Per registered layer: physical rows and per-component full-page bytes.
    layers: dict


@dataclass(frozen=True)
class Dsv41Layout:
    groups: dict
    file_size: dict
    namespace: str
    hash_block_size: int = 128


def compile_dsv41_layout(kv_config, vllm_config, *, ascend):
    """Validate supported inputs before any store is constructed.

    A C2 ring is transient only for whole even-prefix restores without draft
    tokens. CPU fused_save_compress_norm reads predecessor history only for
    odd start positions; Ascend _pool_kernel reads history only for residual
    start_pos % 2. The next chunk starting at 128*k has neither dependency.
    Capacity divisibility is not the proof. Odd later chunk starts read rows
    produced by this request after restore. MTP/CP/PP need separate validation.
    """
    if getattr(vllm_config, "speculative_config", None) is not None:
        raise ValueError("DSV4.1 FAWA speculative/MTP restore is not supported yet")
    parallel = vllm_config.parallel_config
    for field in (
        "pipeline_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ):
        if getattr(parallel, field, 1) != 1:
            raise ValueError(f"DSV4.1 FAWA requires {field}=1")
    groups, all_specs, rings = {}, {}, []
    totals = {"FA": 0, "WA": 0}
    identity = []
    for gid, group in enumerate(kv_config.kv_cache_groups):
        specs = layer_specs(group)
        if not specs or set(specs) & set(all_specs):
            raise ValueError("DSV4.1 requires unique registered cache owners")
        all_specs.update(specs)
        kinds, windows, layers = set(), set(), {}
        block = int(group.kv_cache_spec.block_size)
        for name, spec in specs.items():
            cls = type(spec).__name__
            if int(spec.block_size) != block:
                raise ValueError("DSV4.1 group mixes logical block sizes")
            if cls == "CircularBufferSpec":
                if (
                    not name.endswith(".compressor.state_cache")
                    or block != (32 if ascend else 8)
                    or int(spec.head_size) != 1024
                    or str(spec.dtype).removeprefix("torch.") != "float32"
                ):
                    raise ValueError(f"Unverified DSV4.1 ring: {name}")
                rings.append(name)
                kinds.add("ring")
                identity.append((gid, name, cls, block, str(spec.dtype)))
                continue
            expected = (
                ("AscendMLAAttentionSpec", "AscendSlidingWindowMLASpec")
                if ascend
                else ("MLAAttentionSpec", "SlidingWindowMLASpec")
            )
            if cls not in expected or int(spec.num_kv_heads) != 1:
                raise ValueError(f"Unsupported DSV4.1 cache spec: {name}: {cls}")
            window = getattr(spec, "sliding_window", None)
            kind = "WA" if window is not None else "FA"
            kinds.add(kind)
            ratio = int(getattr(spec, "tokens_per_state", 1))
            if ratio not in (1, 2) or block % ratio:
                raise ValueError(f"Unsupported DSV4.1 compression: {name}: {ratio}")
            if kind == "FA" and block != 128:
                raise ValueError("DSV4.1 FAWA currently requires FA block_size=128")
            if kind == "WA":
                if not name.endswith(".swa_cache") or ratio != 1 or int(window) != 128:
                    raise ValueError(f"Unsupported DSV4.1 sliding window: {name}")
                windows.add(int(window))
                if block <= 0 or 128 % block:
                    raise ValueError("DSV4.1 SWA blocks must tile the 128-token tail")
            rows = block // ratio
            storage_rows = getattr(spec, "storage_block_size", rows)
            if storage_rows is not None and int(storage_rows) != rows:
                raise ValueError(
                    f"DSV4.1 storage rows disagree with compression: {name}"
                )
            content = getattr(spec, "state_content_bytes", None)
            row_bytes = (
                int(content)
                if content is not None
                else int(spec.head_size) * _itemsize(spec.dtype)
            )
            components = [rows * row_bytes]
            scale_dim = int(getattr(spec, "scale_dim", 0))
            if scale_dim:
                components.append(rows * scale_dim * _itemsize(spec.scale_dtype))
            if row_bytes <= 0 or scale_dim < 0:
                raise ValueError(f"Invalid DSV4.1 payload size: {name}")
            layers[name] = (rows, tuple(components))
            identity.append(
                (
                    gid,
                    name,
                    cls,
                    block,
                    ratio,
                    str(spec.dtype),
                    rows,
                    components,
                    window,
                    str(getattr(spec, "scale_dtype", None)),
                )
            )
        if len(kinds) != 1:
            raise ValueError("DSV4.1 cannot mix ring/FA/SWA roles within one group")
        kind = kinds.pop()
        tail = 0 if kind == "ring" else 128
        count = 128 // block if kind == "WA" else 1
        groups[gid] = Dsv41Group(kind, block, tail, count, layers)
        if kind != "ring":
            totals[kind] += sum(sum(sizes) for _, sizes in layers.values()) * count
    for name in rings:
        owner = name.removesuffix(".compressor.state_cache")
        if ascend:
            owner += ".long_kv_cache"
        spec = all_specs.get(owner)
        if (
            spec is None
            or int(getattr(spec, "tokens_per_state", 1)) != 2
            or getattr(spec, "sliding_window", None) is not None
            or int(spec.block_size) != 128
        ):
            raise ValueError(f"DSV4.1 ring has no paired C2 owner: {name}")
    # Every C2 main-cache owner needs a ring; indexer rows share that producer.
    for name, spec in all_specs.items():
        if (
            getattr(spec, "sliding_window", None) is None
            and int(getattr(spec, "tokens_per_state", 1)) == 2
            and ".indexer." not in name
        ):
            owner = name.removesuffix(".long_kv_cache") if ascend else name
            if owner + ".compressor.state_cache" not in rings:
                raise ValueError(f"DSV4.1 C2 owner has no ring: {name}")
    if not all(totals.values()):
        raise ValueError("DSV4.1 FAWA requires both FA and SWA payloads")
    digest = hashlib.sha256(
        json.dumps([ascend, sorted(identity)], sort_keys=True).encode()
    ).hexdigest()[:16]
    return Dsv41Layout(
        groups,
        {k: (v + 4095) // 4096 * 4096 for k, v in totals.items()},
        "dsv41-r1-" + digest,
    )
