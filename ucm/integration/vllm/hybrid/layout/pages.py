"""CPU/CUDA native page addressing for equal-page FA/State groups.

Copy the engine-owned page, including its existing unused tail. Never extend
an arbitrary view merely because its allocation has spare bytes.
"""

from dataclasses import dataclass

from .view import ComponentView, LayerView, MemorySegment, row_payload_bytes


@dataclass(frozen=True)
class PageGroupLayout:
    layer_views: dict[str, LayerView]


def build_page_group_layouts(spec, kv_caches):
    groups, sizes = {}, set()
    for group in spec.groups:
        layers = {}
        for layer in group.layers:
            value = kv_caches[layer.layer_name]
            tensors = tuple(value) if isinstance(value, (tuple, list)) else (value,)
            if len(tensors) != 1:
                raise ValueError(
                    "Native State page layout requires one registered page view per layer"
                )
            tensor = tensors[0]
            shape = tuple(int(x) for x in tensor.shape)
            strides = tuple(int(tensor.stride(i)) for i in range(len(shape)))
            element = int(tensor.element_size())
            page = getattr(layer.kv_cache_spec, "page_size_bytes", None)
            if page is None or int(page) <= 0:
                raise ValueError("Native page layout requires explicit page_size_bytes")
            page = int(page)
            if (
                len(shape) < 2
                or shape[0] != layer.num_blocks
                or any(s <= 0 for s in strides)
            ):
                raise ValueError(
                    "Native page view must expose one row per physical block"
                )
            payload = row_payload_bytes(shape, strides, element)
            stride = strides[0] * element
            if payload > page or stride < page:
                raise ValueError("Native page size conflicts with view payload/stride")
            storage = tensor.untyped_storage()
            allocation = int(storage.data_ptr())
            end = allocation + int(storage.nbytes())
            base = int(tensor.data_ptr())
            descriptor = layer.descriptor
            if descriptor is None:
                if stride != page or base != allocation:
                    raise ValueError(
                        "Native page without descriptor requires allocation-aligned page stride"
                    )
            else:
                if (
                    stride != descriptor.block_stride
                    or base
                    != allocation
                    + descriptor.offset
                    + layer.descriptor_position * descriptor.layer_stride
                ):
                    raise ValueError(
                        "Native page view disagrees with descriptor origin/stride"
                    )
                if descriptor.layer_stride < stride:
                    if (
                        descriptor.layer_stride < page
                        or len(descriptor.layers) * descriptor.layer_stride > stride
                    ):
                        raise ValueError("Native page would overlap another layer slot")
                elif (
                    len(descriptor.layers) > 1
                    and descriptor.layer_stride < layer.num_blocks * stride
                ):
                    raise ValueError("Native page layer ranges overlap")
            if (
                layer.num_blocks <= 0
                or base < allocation
                or base + (layer.num_blocks - 1) * stride + page > end
            ):
                raise ValueError("Native page range exceeds backing allocation")
            segment = MemorySegment(base, stride, 1, page, page)
            layers[layer.layer_name] = LayerView(
                layer.layer_name,
                layer.layer_index,
                (ComponentView(shape, strides, (segment,)),),
            )
            sizes.add(page)
        groups[group.group_id] = PageGroupLayout(layers)
    if len(sizes) != 1:
        raise ValueError(
            "Single-store native State pages must have equal page_size_bytes"
        )
    return groups
