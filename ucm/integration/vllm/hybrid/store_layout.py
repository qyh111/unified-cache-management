"""Compile physical segments into one fixed v1 Pipeline Store schema.

Dedicated policies choose semantic regions; boundaries are partitioned only
within each region. Missing roles use null pointers without device overreads.
Addresses are expanded over block IDs in NumPy, without a Python loop per key.
"""

from dataclasses import dataclass

import numpy as np

from .layout import build_group_layouts
from .layout.policies import compile_policy


def validate_spec(spec):
    if spec.sw_groups:
        raise ValueError("FA/SWA belongs to UCMFAWAConnector, not UCMHybridConnector")
    sizes = {group.token_block_size for group in spec.groups}
    if len(sizes) != 1 or spec.ucm_cache_block_size not in sizes:
        raise ValueError(
            "Hybrid requires equal group token_block_size and UCM block size"
        )
    if not spec.fa_groups:
        raise ValueError("Hybrid requires a full-attention group")
    if any(not group.layers for group in spec.groups):
        raise ValueError(
            "Hybrid does not yet support empty local groups / PP projections"
        )


def slot_sizes(rows):
    """Smallest common partition of densely packed, possibly unequal rows."""
    boundaries = {0}
    for sizes in rows:
        cursor = 0
        for size in sizes:
            if int(size) <= 0:
                raise ValueError("Real segments must have positive size")
            cursor += int(size)
            boundaries.add(cursor)
    if len(boundaries) < 2:
        raise ValueError("Hybrid has no physical payload")
    return np.diff(np.asarray(sorted(boundaries), dtype=np.uint64))


@dataclass(frozen=True)
class Row:
    layer_id: int | None
    names: frozenset[str]
    group_positions: np.ndarray
    bases: np.ndarray
    strides: np.ndarray
    columns: np.ndarray
    limits: np.ndarray


@dataclass(frozen=True)
class Transfer:
    keys: tuple[bytes, ...]
    shard_index: int
    ptrs: np.ndarray


class HybridStoreLayout:
    def __init__(self, spec, kv_caches, *, layerwise):
        validate_spec(spec)
        self.spec = spec
        self.layerwise = layerwise
        self.group_layouts = build_group_layouts(spec, kv_caches)
        self.routes = {
            groups[0].group_id: groups for _, groups in spec.dispatch_routes()
        }
        self.kinds = {
            groups[0].group_id: kind for kind, groups in spec.dispatch_routes()
        }
        self.layer_name_to_id = {
            layer.layer_name: layer.layer_index
            for group in spec.groups
            for layer in group.layers
        }
        layer_rows, layer_ids, names_by_group = {}, {}, {}
        for group_id, (group,) in self.routes.items():
            layout = self.group_layouts[group_id]
            ids = sorted({layer.layer_index for layer in group.layers})
            layer_ids[group_id] = ids
            layer_rows[group_id], names_by_group[group_id] = [], []
            for layer_id in ids:
                entries = [
                    (layer, layout.layer_views[layer.layer_name])
                    for layer in sorted(group.layers, key=lambda x: x.layer_name)
                    if layer.layer_index == layer_id
                ]
                layer_rows[group_id].append((self.kinds[group_id], entries))
                names_by_group[group_id].append(
                    frozenset(layer.layer_name for layer, _ in entries)
                )
        counts = {len(rows) for rows in layer_rows.values()}
        if len(counts) != 1:
            raise ValueError(
                "Single-store groups require equal local layer counts; no implicit ghost rows"
            )
        # Single-group bulk retains compact real segments, without per-layer padding.
        if not layerwise and len(self.routes) == 1:
            self.policy, regions = "compact", ("payload",)
            mapped = {
                gid: [
                    {
                        "payload": tuple(
                            s
                            for _, entries in rows
                            for _, view in entries
                            for s in view.segments
                        )
                    }
                ]
                for gid, rows in layer_rows.items()
            }
            layer_ids = {gid: [None] for gid in self.routes}
            names_by_group = {
                gid: [frozenset().union(*names)]
                for gid, names in names_by_group.items()
            }
        else:
            self.policy, regions, mapped = compile_policy(
                layer_rows, bool(spec.state_groups)
            )
        # Compile a common partition WITHIN each semantic region. Padding may
        # occur between conv, state/K, V and scale, never by repacking roles.
        region_sizes, region_columns = {}, {}
        sizes, slot_roles = [], []
        for region in regions:
            parts = slot_sizes(
                [s.payload_bytes for s in row.get(region, ())]
                for rows in mapped.values()
                for row in rows
            )
            region_sizes[region] = parts
            region_columns[region] = len(sizes)
            sizes.extend(parts.tolist())
            slot_roles.extend([region] * len(parts))
        self.sizes = np.asarray(sizes, dtype=np.uint64)
        self.slot_roles = tuple(slot_roles)
        self.rows = {}
        for group_id, mapped_rows in mapped.items():
            group = self.routes[group_id][0]
            # Derive allocation bounds from the owning source segment even
            # when a policy has split it into multiple semantic slots.
            sources = [
                (s, layer.num_blocks)
                for layer in group.layers
                for s in self.group_layouts[group_id]
                .layer_views[layer.layer_name]
                .segments
            ]
            compiled = []
            for index, roles in enumerate(mapped_rows):
                bases, strides, columns, limits = [], [], [], []
                for region in regions:
                    starts = np.cumsum(region_sizes[region]) - region_sizes[region]
                    cursor = 0
                    for segment in roles.get(region, ()):
                        end = cursor + segment.payload_bytes
                        cols = np.flatnonzero((starts >= cursor) & (starts < end))
                        candidates = [
                            limit
                            for src, limit in sources
                            if src.block_stride_bytes == segment.block_stride_bytes
                            and src.base_ptr <= segment.base_ptr
                            and segment.base_ptr + segment.payload_bytes
                            <= src.base_ptr + src.payload_bytes
                        ]
                        if not candidates:
                            raise ValueError(
                                "Policy segment is outside its physical source"
                            )
                        bases.extend(
                            (segment.base_ptr + starts[cols] - cursor).tolist()
                        )
                        strides.extend([segment.block_stride_bytes] * len(cols))
                        columns.extend((cols + region_columns[region]).tolist())
                        limits.extend([min(candidates)] * len(cols))
                        cursor = end
                compiled.append(
                    Row(
                        layer_ids[group_id][index],
                        names_by_group[group_id][index],
                        np.zeros(len(columns), dtype=np.intp),
                        np.asarray(bases, dtype=np.uint64),
                        np.asarray(strides, dtype=np.uint64),
                        np.asarray(columns, dtype=np.intp),
                        np.asarray(limits, dtype=np.uint64),
                    )
                )
            self.rows[group_id] = tuple(compiled)
        if not layerwise and self.policy != "compact":
            # Bulk packs the per-layer semantic template of ONE native group.
            width = len(self.sizes)
            self.sizes = np.tile(self.sizes, next(iter(counts)))
            self.slot_roles = self.slot_roles * next(iter(counts))
            for gid, rows in self.rows.items():
                self.rows[gid] = (
                    Row(
                        None,
                        frozenset().union(*(r.names for r in rows)),
                        np.concatenate([r.group_positions for r in rows]),
                        np.concatenate([r.bases for r in rows]),
                        np.concatenate([r.strides for r in rows]),
                        np.concatenate(
                            [r.columns + i * width for i, r in enumerate(rows)]
                        ),
                        np.concatenate([r.limits for r in rows]),
                    ),
                )
        self.row_count = len(next(iter(self.rows.values())))
        # Register the actual allocations, never ghost pointers or padded sizes.
        allocations = {}
        for value in kv_caches.values():
            tensors = value if isinstance(value, (tuple, list)) else (value,)
            for tensor in tensors:
                storage = tensor.untyped_storage()
                allocations[int(storage.data_ptr())] = int(storage.nbytes())
        self.base_ptrs = np.asarray(list(allocations), dtype=np.uint64)
        self.buffer_sizes = np.asarray(list(allocations.values()), dtype=np.uint64)

    @property
    def tensor_size_list(self):
        return self.sizes.tolist()

    @property
    def shard_size(self):
        return int(self.sizes.sum())

    @property
    def block_size(self):
        return self.shard_size * self.row_count

    @property
    def ucm_block_offsets(self):
        shard_bytes = (self.shard_size + 4095) // 4096 * 4096
        return (
            np.arange(self.row_count, dtype=np.uint64)[:, None] * shard_bytes
            + (np.cumsum(self.sizes) - self.sizes)[None, :]
        )

    def padding_report(self):
        """Explain semantic padding separately from v1 shard alignment.

        Diagnostic only: never scan keys or materialize pointer matrices here.
        A byte count describes one stored key for the selected native group.
        """
        aligned_shard = (self.shard_size + 4095) // 4096 * 4096
        groups = {}
        for group_id, rows in self.rows.items():
            details = []
            for index, row in enumerate(rows):
                payload = int(self.sizes[row.columns].sum())
                real = set(row.columns.tolist())
                details.append(
                    {
                        "shard_index": index,
                        "layer_id": row.layer_id,
                        "payload_bytes": payload,
                        "padding_bytes": self.shard_size - payload,
                        "padding_columns": [
                            i for i in range(len(self.sizes)) if i not in real
                        ],
                    }
                )
            groups[group_id] = {
                "kind": self.kinds[group_id],
                "payload_bytes": sum(r["payload_bytes"] for r in details),
                "padding_bytes": sum(r["padding_bytes"] for r in details),
                "alignment_bytes": (aligned_shard - self.shard_size) * len(rows),
                "stored_block_bytes": aligned_shard * len(rows),
                "rows": details,
            }
        return {"slot_roles": self.slot_roles, "groups": groups}

    def resolve(self, plan, row_index):
        if self.kinds[plan.group_id] != plan.hash_group:
            raise ValueError("Plan kind does not match its native group")
        row = self.rows[plan.group_id][row_index]
        count = len(plan.keys)
        windows = tuple(np.asarray(ids, dtype=np.int64) for ids in plan.windows)
        if len(windows) != len(self.routes[plan.group_id]) or any(
            ids.shape != (count,) for ids in windows
        ):
            raise ValueError(
                "Hybrid expects exactly one physical block per key and group"
            )
        # Mamba align tables use block zero as a null placeholder. Such a
        # boundary has no snapshot and must never produce a visible state key.
        valid = np.ones(count, dtype=bool)
        if plan.hash_group == "State":
            for ids in windows:
                valid &= ids != 0
        keys = (
            plan.keys
            if valid.all()
            else tuple(key for key, keep in zip(plan.keys, valid) if keep)
        )
        ptrs = np.zeros((len(keys), len(self.sizes)), dtype=np.uint64)
        if len(row.columns):
            blocks = np.stack(windows, axis=1)[valid][:, row.group_positions]
            if (blocks < 0).any() or (blocks >= row.limits).any():
                raise ValueError("Physical block ID is outside its allocation")
            ptrs[:, row.columns] = blocks.astype(np.uint64) * row.strides + row.bases
        return Transfer(keys, row_index, ptrs)
