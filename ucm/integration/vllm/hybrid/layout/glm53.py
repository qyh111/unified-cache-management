"""GLM5.3 layout primitives, explicitly selected by the future model adapter.

These do not select persistent groups or change scheduler/store semantics.
In particular, a KPool tail must never be passed as a persistent layer.
"""

from .view import ComponentView, LayerView, MemorySegment, row_payload_bytes
from .policies import size


def build_tiled_page_view(tensor, layer):
    """Preserve a CUDA engine-declared page, including packed indexer scales.

    CUDA main/indexer views tile a logical page with kernel rows. State
    exposes one byte row with an engine-owned unused page tail. Require
    descriptor evidence before copying that tail; allocation slack is not
    evidence of page ownership. NPU overlay descriptors use a different ABI;
    its real component views use build_layer_view instead.
    """
    shape = tuple(int(x) for x in tensor.shape)
    strides = tuple(int(tensor.stride(i)) for i in range(len(shape)))
    element = int(tensor.element_size())
    descriptor = layer.descriptor
    page = int(layer.kv_cache_spec.page_size_bytes)
    blocks = int(layer.num_blocks)
    if "tail_cache" in layer.layer_name.split("."):
        raise ValueError("KPool tail is not a persistent GLM5.3 page")
    if descriptor is None or blocks <= 0 or page <= 0:
        raise ValueError("GLM5.3 pages require a descriptor and positive sizes")
    if len(shape) < 2 or shape[0] <= 0 or shape[0] % blocks:
        raise ValueError("Kernel rows must tile logical blocks")
    if any(s <= 0 for s in strides) or any(d <= 0 for d in shape):
        raise ValueError("Page views require positive shape and strides")
    rows = shape[0] // blocks
    row_bytes = row_payload_bytes(shape, strides, element)
    stride = strides[0] * element * rows
    # Multi-row pages must be completely dense; only single-row State pages
    # may expose less payload than their declared page capacity.
    if stride != page or (rows > 1 and row_bytes != strides[0] * element):
        raise ValueError("Kernel rows do not form one contiguous declared page")
    if row_bytes > strides[0] * element:
        raise ValueError("Page payload overlaps the next row")
    position = layer.descriptor_position
    if not 0 <= position < len(descriptor.layers):
        raise ValueError("Invalid descriptor layer position")
    if descriptor.layers[position] != layer.layer_name:
        raise ValueError("Descriptor belongs to a different layer")
    storage = tensor.untyped_storage()
    allocation, capacity = int(storage.data_ptr()), int(storage.nbytes())
    base = int(tensor.data_ptr())
    if descriptor.block_stride != stride or base != (
        allocation + descriptor.offset + position * descriptor.layer_stride
    ):
        raise ValueError("Page origin/stride disagrees with descriptor")
    if len(descriptor.layers) > 1 and descriptor.layer_stride < blocks * page:
        raise ValueError("GLM5.3 layer-major page ranges overlap")
    if base < allocation or base + blocks * page > allocation + capacity:
        raise ValueError("Declared page range exceeds backing allocation")
    return LayerView(
        layer.layer_name,
        layer.layer_index,
        (ComponentView(shape, strides, (MemorySegment(base, stride, 1, page, page),)),),
    )


def compile_glm53_policy(layer_rows):
    """Map persistent native groups to main/index regions without regrouping.

    NPU State's conv/state segments remain distinct. The common region
    partition will pad their sum to the attention page capacity. CUDA State
    already includes the engine page tail. Row-count padding is the store
    compiler's responsibility, not a reason to combine State groups.
    """
    mapped, main_sizes, index_sizes, state_sizes = {}, set(), set(), []
    for group_id, rows in layer_rows.items():
        mapped[group_id] = []
        if not rows:
            raise ValueError("Empty local GLM5.3 group is unsupported")
        for kind, entries in rows:
            roles = {}
            if kind == "State":
                if len(entries) != 1:
                    raise ValueError("State row requires one registered layer")
                roles["main"] = entries[0][1].segments
                state_sizes.append(size(roles["main"]))
            elif kind == "FA":
                for layer, view in entries:
                    parts = layer.layer_name.split(".")
                    if "tail_cache" in parts:
                        raise ValueError("KPool tail must remain request-private")
                    role = "index" if "indexer" in parts else "main"
                    if role in roles:
                        raise ValueError("Duplicate GLM5.3 role in model layer")
                    roles[role] = view.segments
                if set(roles) != {"main", "index"}:
                    raise ValueError("GLM5.3 FA row requires main and indexer")
                main_sizes.add(size(roles["main"]))
                index_sizes.add(size(roles["index"]))
            else:
                raise ValueError("Only FA and State are persistent GLM5.3 groups")
            mapped[group_id].append(roles)
    if len(main_sizes) != 1 or len(index_sizes) != 1 or not state_sizes:
        raise ValueError("GLM5.3 requires uniform FA roles and State groups")
    if (
        min(index_sizes) <= 0
        or min(main_sizes) <= 0
        or any(s <= 0 or s > next(iter(main_sizes)) for s in state_sizes)
    ):
        raise ValueError("State payload must fit the declared main region")
    return "glm53", ("main", "index"), mapped
