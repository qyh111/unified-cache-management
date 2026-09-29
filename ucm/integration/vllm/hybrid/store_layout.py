"""Compile physical segments into one fixed v1 Pipeline Store schema.

The union of segment boundaries splits real segments; it never rounds a device
read up to a larger size. Short rows have null pointer slots at their tail.
Addresses are expanded over block IDs in NumPy, without a Python loop per key.
"""

from dataclasses import dataclass

import numpy as np

from .layout import build_group_layouts


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
        self.routes = dict(spec.dispatch_routes())
        self.layer_name_to_id = {
            layer.layer_name: layer.layer_index
            for group in spec.groups
            for layer in group.layers
        }
        raw_rows = {}
        for kind, groups in self.routes.items():
            layers = sorted(
                {layer.layer_index for group in groups for layer in group.layers}
            )
            rows = []
            for layer_id in layers if layerwise else [None]:
                segments, names = [], set()
                for position, group in enumerate(groups):
                    layout = self.group_layouts[group.group_id]
                    for layer in sorted(
                        group.layers,
                        key=lambda item: (item.layer_index, item.layer_name),
                    ):
                        if layer_id is not None and layer.layer_index != layer_id:
                            continue
                        names.add(layer.layer_name)
                        for segment in layout.layer_views[layer.layer_name].segments:
                            segments.append((position, segment, layer.num_blocks))
                rows.append((layer_id, frozenset(names), segments))
            raw_rows[kind] = rows
        self.sizes = slot_sizes(
            [segment.payload_bytes for _, segment, _ in segments]
            for rows in raw_rows.values()
            for _, _, segments in rows
        )
        starts = np.cumsum(self.sizes) - self.sizes
        self.row_count = max(len(rows) for rows in raw_rows.values())
        self.rows = {}
        for kind, rows in raw_rows.items():
            compiled = []
            for layer_id, names, segments in rows:
                cursor = 0
                positions, bases, strides, columns, limits = [], [], [], [], []
                for position, segment, limit in segments:
                    end = cursor + segment.payload_bytes
                    cols = np.flatnonzero((starts >= cursor) & (starts < end))
                    positions.extend([position] * len(cols))
                    bases.extend((segment.base_ptr + starts[cols] - cursor).tolist())
                    strides.extend([segment.block_stride_bytes] * len(cols))
                    columns.extend(cols.tolist())
                    limits.extend([limit] * len(cols))
                    cursor = end
                compiled.append(
                    Row(
                        layer_id,
                        names,
                        np.asarray(positions, dtype=np.intp),
                        np.asarray(bases, dtype=np.uint64),
                        np.asarray(strides, dtype=np.uint64),
                        np.asarray(columns, dtype=np.intp),
                        np.asarray(limits, dtype=np.uint64),
                    )
                )
            # Every key uses all fixed shards, including tail ghosts. This
            # preserves the existing last-shard publication convention.
            for _ in range(self.row_count - len(compiled)):
                compiled.append(
                    Row(
                        None,
                        frozenset(),
                        np.empty(0, dtype=np.intp),
                        np.empty(0, dtype=np.uint64),
                        np.empty(0, dtype=np.uint64),
                        np.empty(0, dtype=np.intp),
                        np.empty(0, dtype=np.uint64),
                    )
                )
            self.rows[kind] = tuple(compiled)
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

    def resolve(self, plan, row_index):
        row = self.rows[plan.hash_group][row_index]
        count = len(plan.keys)
        windows = tuple(np.asarray(ids, dtype=np.int64) for ids in plan.windows)
        if len(windows) != len(self.routes[plan.hash_group]) or any(
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
