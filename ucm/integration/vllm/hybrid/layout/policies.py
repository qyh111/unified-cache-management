"""Semantic slot policies; physical segments and block strides remain shared.

A policy maps a local model layer to named slot regions. The store compiler
partitions each region separately, preserving holes between semantic roles.
"""

from dataclasses import replace
import math


def span(segments, start, length):
    """Slice a logical concatenation without assuming physical adjacency."""
    result, cursor = [], 0
    end = start + length
    for segment in segments:
        lo, hi = max(start, cursor), min(end, cursor + segment.payload_bytes)
        if lo < hi:
            result.append(
                replace(
                    segment,
                    base_ptr=segment.base_ptr + lo - cursor,
                    payload_bytes=hi - lo,
                )
            )
        cursor += segment.payload_bytes
    if sum(s.payload_bytes for s in result) != length:
        raise ValueError("Semantic slot exceeds real segment payload")
    return tuple(result)


def size(segments):
    return sum(s.payload_bytes for s in segments)


def components(entries):
    return [
        component.segments for layer, view in entries for component in view.components
    ]


def state_components(entries):
    if len(entries) != 1:
        raise ValueError("State policy requires one registered state entry per layer")
    layer, view = entries[0]
    parts = components(entries)
    if len(parts) == 2:
        return parts
    spec = layer.kv_cache_spec
    shapes, dtypes = getattr(spec, "shapes", ()), getattr(spec, "dtypes", ())
    if len(parts) != 1 or len(shapes) != 2 or len(dtypes) != 2:
        raise ValueError(
            "State policy requires conv/state components or an explicit two-component byte page"
        )
    sizes = [math.prod(shape) * dtype.itemsize for shape, dtype in zip(shapes, dtypes)]
    if sum(sizes) != size(parts[0]):
        raise ValueError("Combined state page payload does not match conv/state spec")
    return [span(parts[0], 0, sizes[0]), span(parts[0], sizes[0], sizes[1])]


def compile_policy(layer_rows, has_state):
    """Return policy name, semantic region order, and mappings per group/layer."""
    names = [
        layer.layer_name.lower().split(".")
        for rows in layer_rows.values()
        for _, entries in rows
        for layer, _ in entries
    ]
    minimax = any("index_cache" in parts for parts in names)
    shared = any("indexer" in parts for parts in names)
    if has_state and (minimax or shared):
        raise ValueError("Combined State and Indexer layout needs an explicit policy")
    policy = (
        "state"
        if has_state
        else "minimax_m3" if minimax else "shared_indexer" if shared else "ordinary"
    )
    mapped = {}
    index_sizes, scale_sizes, attention_shapes = set(), set(), set()
    for group_id, rows in layer_rows.items():
        mapped[group_id] = []
        for kind, entries in rows:
            roles = {}
            if policy == "state":
                if kind == "State":
                    conv, state = state_components(entries)
                    roles = {"conv": conv, "data0": state}
                else:
                    parts = components(entries)
                    if len(parts) == 2:
                        roles = {"data0": parts[0], "data1": parts[1]}
                    elif len(parts) == 1 and size(parts[0]) % 2 == 0:
                        # Packed KV ABIs may interleave K/V; these are two byte
                        # ranges, not a claim that each range is exactly K or V.
                        half = size(parts[0]) // 2
                        roles = {
                            "data0": span(parts[0], 0, half),
                            "data1": span(parts[0], half, half),
                        }
                    else:
                        raise ValueError(
                            "State policy requires separate or packed two-part attention"
                        )
            else:
                attention, indexer = [], []
                for entry in entries:
                    parts = entry[0].layer_name.lower().split(".")
                    (
                        indexer
                        if ("index_cache" in parts or "indexer" in parts)
                        else attention
                    ).append(entry)
                parts = components(attention)
                if not parts:
                    raise ValueError(
                        "Layout requires real attention in each local layer"
                    )
                attention_shapes.add(tuple(size(part) for part in parts))
                roles = {f"attention{i}": part for i, part in enumerate(parts)}
                iparts = components(indexer)
                if policy == "minimax_m3" and len(iparts) > 1:
                    raise ValueError("MiniMax-M3 requires one Indexer component")
                if len(iparts) > 2:
                    raise ValueError(
                        "Shared Indexer requires indexer and optional scale"
                    )
                if iparts:
                    roles["index"] = iparts[0]
                    index_sizes.add(size(iparts[0]))
                if len(iparts) == 2:
                    roles["scale"] = iparts[1]
                    scale_sizes.add(size(iparts[1]))
            mapped[group_id].append(roles)
    if policy == "state":
        return policy, ("conv", "data0", "data1"), mapped
    if len(attention_shapes) != 1:
        raise ValueError(
            "Attention component sizes differ; select a dedicated layout policy"
        )
    order = [f"attention{i}" for i in range(len(next(iter(attention_shapes))))]
    if policy == "ordinary":
        return policy, tuple(order), mapped
    if len(scale_sizes) > 1:
        raise ValueError("Indexer scales must have uniform size")
    rows = [r for group in mapped.values() for r in group]
    if scale_sizes:
        c8_sizes = {size(r["index"]) for r in rows if "scale" in r}
        bf16_sizes = {
            size(r["index"]) for r in rows if "index" in r and "scale" not in r
        }
        if len(c8_sizes) != 1:
            raise ValueError("LI C8 Indexer sizes must be uniform")
        chunk = next(iter(c8_sizes))
        if bf16_sizes and bf16_sizes != {2 * chunk}:
            raise ValueError("Mixed BF16 Indexer must be twice the LI C8 size")
        if bf16_sizes:
            for row in rows:
                if "index" in row and "scale" not in row:
                    original = row["index"]
                    row["index"] = span(original, 0, chunk)
                    row["index_tail"] = span(original, chunk, chunk)
            order += ["index", "index_tail", "scale"]
        else:
            order += ["index", "scale"]
    else:
        if len(index_sizes) != 1:
            raise ValueError("BF16/MiniMax Indexer sizes must be uniform")
        order += ["index"]
    return policy, tuple(order), mapped
