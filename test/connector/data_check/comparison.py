"""Byte oracle over native CPU views/descriptors, independent of UCM addresses."""

from __future__ import annotations

import itertools


def components(value):
    return value if isinstance(value, (list, tuple)) else (value,)


def storage_bytes(tensor):
    import torch

    return torch.empty(0, dtype=torch.uint8).set_(
        tensor.untyped_storage(), 0, (tensor.untyped_storage().nbytes(),), (1,)
    )


def view_spans(tensor):
    """Coalesce the contiguous suffix, preserving physical offsets and holes."""
    if tensor.numel() == 0:
        return []
    # Canonicalize axes by physical stride before coalescing. BHNC views may
    # be backed by contiguous BNHC pages; enumerating logical H*N slices would
    # create hundreds of thousands of tiny snapshots for a plain GQA cache.
    order = sorted(range(tensor.ndim), key=tensor.stride, reverse=True)
    tensor = tensor.permute(order)
    size = tensor.element_size()
    split = tensor.ndim
    while split and (
        tensor.shape[split - 1] == 1
        or tensor.stride(split - 1) * tensor.element_size() == size
    ):
        split -= 1
        size *= tensor.shape[split]
    backing = storage_bytes(tensor)
    spans = []
    for indices in itertools.product(*(range(n) for n in tensor.shape[:split])):
        offset = tensor.storage_offset() * tensor.element_size() + sum(
            index * stride * tensor.element_size()
            for index, stride in zip(indices, tensor.stride())
        )
        if offset < 0 or offset + size > backing.numel():
            raise AssertionError("Native view exceeds its backing allocation")
        spans.append(backing.narrow(0, offset, size))
    return spans


class KVComparison:
    """Expectations come only from native views and engine block tables.

    Connector metadata remains opaque. Digests, never payload tensors, cross
    the source/reader process boundary.
    """

    def __init__(self, native, views, raw):
        import hashlib

        self.hash = hashlib.sha256
        self.native, self.views = native, views
        tensors = [t for value in raw.values() for t in components(value)]
        from .memory import Memory

        self.memory = Memory(tensors)
        self.names = {}
        for name, value in sorted(raw.items()):
            for tensor in components(value):
                self.names.setdefault(tensor.untyped_storage()._cdata, name)
        self.expected, self.targets, self.checked = {}, {}, set()

    def pages(self, group_id, blocks):
        group = self.native.kv_cache_groups[group_id]
        for index, block in enumerate(blocks):
            if not block:
                continue
            for name in group.layer_names:
                for component, tensor in enumerate(components(self.views[name])):
                    factor, remainder = divmod(tensor.shape[0], self.native.num_blocks)
                    if remainder or factor < 1:
                        raise ValueError("Native view rows do not match block count")
                    page = tensor[int(block) * factor : (int(block) + 1) * factor]
                    for segment, span in enumerate(view_spans(page)):
                        yield (group_id, index, name, component, segment), span

    def source(self, blocks, replicated, rank):
        self.memory.seed(self.names, 0 if replicated else rank)
        # State shards in an MLA hybrid remain rank-specific. Seed only their
        # actually allocated pages: native group views may share physical banks.
        from .memory import generate

        for group_id in self.native.prefix_cacheable_group_ids:
            group = self.native.kv_cache_groups[group_id]
            for identity, span in self.pages(group_id, blocks[group_id]):
                name = identity[2]
                spec = getattr(group.kv_cache_spec, "kv_cache_specs", {}).get(
                    name, group.kv_cache_spec
                )
                if spec.tokens_per_state < 0 and not spec.tp_replicated:
                    seed = self.hash(repr((identity, rank)).encode()).hexdigest()
                    data = generate(seed, 0, span.numel())
                    self.memory.write(span.data_ptr(), data)
                    self.memory.assign(
                        span.data_ptr(), span.numel(), [(seed, 0, span.numel())]
                    )
        expected = {}
        for group_id in self.native.prefix_cacheable_group_ids:
            for identity, span in self.pages(group_id, blocks[group_id]):
                expected[identity] = (
                    span.numel(),
                    self.hash(memoryview(span.numpy())).hexdigest(),
                )
        if not expected:
            raise AssertionError("Source has no persistent KV bytes")
        return expected

    def target(self, blocks, hit_tokens, expected):
        self.memory.poison()
        self.expected = expected
        self.targets, self.checked = {}, set()
        for group_id in self.native.prefix_cacheable_group_ids:
            group = self.native.kv_cache_groups[group_id]
            nested = getattr(group.kv_cache_spec, "kv_cache_specs", {})
            spec = next(iter(nested.values())) if nested else group.kv_cache_spec
            state = spec.tokens_per_state < 0
            limit = (hit_tokens + spec.block_size - 1) // spec.block_size
            selected = (
                list(blocks[group_id]) if state else list(blocks[group_id][:limit])
            )
            source_state_slots = [k[1] for k in expected if k[0] == group_id]
            for identity, span in self.pages(group_id, selected):
                source = identity
                if state:
                    # MRV1 align mode restores the imported checkpoint and
                    # running state slots from the last reusable native state.
                    source = (identity[0], max(source_state_slots), *identity[2:])
                if source not in expected:
                    raise AssertionError(
                        "Target selects KV absent from the source block table"
                    )
                self.targets[identity] = (source, span)
        if not self.targets:
            raise AssertionError("Target has no expected native KV bytes")
        covered = {key[0] for key in self.targets}
        if covered != set(self.native.prefix_cacheable_group_ids):
            raise AssertionError("Target does not cover every native persistent group")

    def compare_layer(self, name):
        for identity, (source, span) in self.targets.items():
            if identity[2] != name:
                continue
            size, digest = self.expected[source]
            if (
                span.numel() != size
                or self.hash(memoryview(span.numpy())).hexdigest() != digest
            ):
                raise AssertionError(f"Restored KV differs: {identity}")
            self.checked.add(identity)

    def finish(self):
        if self.checked != set(self.targets):
            raise AssertionError("Not every native layer was compared")
        return len(self.checked), sum(span.numel() for _, span in self.targets.values())
