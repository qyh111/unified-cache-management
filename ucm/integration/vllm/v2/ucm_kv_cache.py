"""Read-only KV-cache description and ragged runtime layout for connector v2."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Literal, Mapping, Sequence

from .ucm_proxy import UCMProxyBatch


class UCMGroupTag(Enum):
    ATTENTION = "attention"
    STATE = "state"
    SLIDING_WINDOW = "sliding_window"
    C4A = "c4a"


@dataclass(frozen=True)
class UCMLayerSpec:
    layer_name: str
    layer_index: int
    kv_cache_spec: Any


@dataclass(frozen=True)
class UCMKVCacheGroupInfo:
    group_id: int
    layers: tuple[UCMLayerSpec, ...]
    group_spec: Any
    token_block_size: int
    hash_block_size: int
    tags: frozenset[UCMGroupTag]
    tail_tokens: int | None = None
    is_eagle_group: bool = False

    @property
    def num_layers(self) -> int:
        return len(self.layers)


@dataclass(frozen=True)
class UCMKVCacheSpec:
    groups: tuple[UCMKVCacheGroupInfo, ...]
    scheduler_block_size: int
    alignment_block_size: int
    mode: Literal["direct", "hybrid", "dsv4"]
    chunk_size: int

    @property
    def attn_groups(self) -> tuple[UCMKVCacheGroupInfo, ...]:
        return tuple(g for g in self.groups if UCMGroupTag.ATTENTION in g.tags)

    @property
    def state_groups(self) -> tuple[UCMKVCacheGroupInfo, ...]:
        return tuple(g for g in self.groups if UCMGroupTag.STATE in g.tags)

    @property
    def sw_groups(self) -> tuple[UCMKVCacheGroupInfo, ...]:
        return tuple(g for g in self.groups if UCMGroupTag.SLIDING_WINDOW in g.tags)

    @property
    def c4a_group(self) -> UCMKVCacheGroupInfo | None:
        matches = tuple(g for g in self.groups if UCMGroupTag.C4A in g.tags)
        if len(matches) > 1:
            raise ValueError("More than one C4A group was classified")
        return matches[0] if matches else None

    @property
    def layer_to_group(self) -> Mapping[str, int]:
        return {
            layer.layer_name: group.group_id
            for group in self.groups
            for layer in group.layers
        }


def _layer_index(layer_name: str, fallback: int) -> int:
    match = re.search(r"(?:layers|layer)\.(\d+)", layer_name)
    return int(match.group(1)) if match else fallback


def _concrete_specs(group: Any) -> tuple[tuple[str, Any], ...]:
    group_spec = group.kv_cache_spec
    nested = getattr(group_spec, "kv_cache_specs", None)
    names = tuple(getattr(group, "layer_names", ()))
    if nested:
        missing = [name for name in names if name not in nested]
        if missing:
            raise ValueError(f"KV cache group is missing specs for layers {missing}")
        return tuple((name, nested[name]) for name in names)
    return tuple((name, group_spec) for name in names)


def _classify(group: Any, concrete: Sequence[tuple[str, Any]]) -> frozenset[UCMGroupTag]:
    names = tuple(name.lower() for name, _ in concrete)
    specs = tuple(spec for _, spec in concrete) or (group.kv_cache_spec,)
    type_names = tuple(type(spec).__name__.lower() for spec in specs)
    has_window = any(getattr(spec, "sliding_window", None) is not None for spec in specs)
    is_mamba = any("mamba" in name for name in type_names)
    is_compressor = any("compressor" in name or "state_cache" in name for name in names)
    is_c4 = any(
        getattr(spec, "compress_ratio", 1) == 4
        and getattr(spec, "sliding_window", None) is None
        for spec in specs
    )
    tags: set[UCMGroupTag] = set()
    if has_window:
        tags.add(UCMGroupTag.SLIDING_WINDOW)
    if is_mamba or is_compressor:
        tags.add(UCMGroupTag.STATE)
    else:
        tags.add(UCMGroupTag.ATTENTION)
    if is_c4:
        tags.update((UCMGroupTag.ATTENTION, UCMGroupTag.C4A))
    return frozenset(tags)


def parse_kv_cache_config(
    kv_cache_config: Any,
    *,
    scheduler_block_size: int,
    chunk_size: int | None = None,
) -> UCMKVCacheSpec:
    """Parse the vLLM 0.26 KVCacheConfig using only its stable fields."""

    raw_groups = tuple(getattr(kv_cache_config, "kv_cache_groups", ()))
    if not raw_groups:
        raise ValueError("kv_cache_config.kv_cache_groups must not be empty")
    if scheduler_block_size <= 0:
        raise ValueError("scheduler_block_size must be positive")

    classified: list[tuple[Any, tuple[tuple[str, Any], ...], frozenset[UCMGroupTag]]] = []
    dsv4 = False
    for raw_group in raw_groups:
        concrete = _concrete_specs(raw_group)
        tags = _classify(raw_group, concrete)
        classified.append((raw_group, concrete, tags))
        if any(
            "slidingwindowmla" in type(spec).__name__.replace("_", "").lower()
            for _, spec in concrete
        ):
            dsv4 = True

    hybrid = len(raw_groups) > 1 and any(
        UCMGroupTag.STATE in tags or UCMGroupTag.SLIDING_WINDOW in tags
        for _, _, tags in classified
    )
    mode: Literal["direct", "hybrid", "dsv4"] = (
        "dsv4" if dsv4 else "hybrid" if hybrid else "direct"
    )
    if mode != "direct" and chunk_size is not None:
        raise ValueError(f"custom chunk_size is not supported for {mode}")

    c4_sizes: set[int] = set()
    if mode == "dsv4":
        for raw_group, concrete, tags in classified:
            if UCMGroupTag.C4A in tags:
                c4_sizes.add(int(getattr(raw_group.kv_cache_spec, "block_size")))
        if len(c4_sizes) != 1:
            raise ValueError(
                f"DeepSeek V4 requires exactly one C4A block size, got {sorted(c4_sizes)}"
            )
        canonical_size = c4_sizes.pop() * 4
    else:
        canonical_size = scheduler_block_size

    groups: list[UCMKVCacheGroupInfo] = []
    attention_compress_ratio_by_layer: dict[int, int] = {}
    if mode == "dsv4":
        for _, concrete, tags in classified:
            if UCMGroupTag.ATTENTION not in tags or UCMGroupTag.SLIDING_WINDOW in tags:
                continue
            for fallback, (name, concrete_spec) in enumerate(concrete):
                attention_compress_ratio_by_layer[_layer_index(name, fallback)] = int(
                    getattr(concrete_spec, "compress_ratio", 1)
                )
    for group_id, (raw_group, concrete, tags) in enumerate(classified):
        representative = concrete[0][1] if concrete else raw_group.kv_cache_spec
        physical_block_size = int(getattr(raw_group.kv_cache_spec, "block_size"))
        compress_ratio = int(getattr(representative, "compress_ratio", 1))
        token_block_size = (
            physical_block_size * compress_ratio
            if mode == "dsv4"
            else physical_block_size
        )
        hash_block_size = canonical_size if mode == "dsv4" else physical_block_size
        layers = tuple(
            UCMLayerSpec(name, _layer_index(name, index), spec)
            for index, (name, spec) in enumerate(concrete)
        )
        tail_tokens: int | None = None
        if mode == "dsv4" and UCMGroupTag.SLIDING_WINDOW in tags:
            tails: set[int] = set()
            for fallback, (name, concrete_spec) in enumerate(concrete):
                window = int(getattr(concrete_spec, "sliding_window"))
                if name.lower().endswith("swa_cache"):
                    tail = window
                else:
                    layer_index = _layer_index(name, fallback)
                    if layer_index not in attention_compress_ratio_by_layer:
                        raise ValueError(
                            "Cannot find matching full-attention compression ratio "
                            f"for DSV4 layer {layer_index}"
                        )
                    tail = window - attention_compress_ratio_by_layer[layer_index]
                if tail < 0:
                    raise ValueError(f"Negative DSV4 tail for {name}: {tail}")
                tails.add(tail)
            if len(tails) != 1:
                raise ValueError(
                    f"DSV4 group {group_id} has inconsistent tail sizes {sorted(tails)}"
                )
            tail_tokens = tails.pop()
        groups.append(
            UCMKVCacheGroupInfo(
                group_id=group_id,
                layers=layers,
                group_spec=raw_group.kv_cache_spec,
                token_block_size=token_block_size,
                hash_block_size=hash_block_size,
                tags=tags,
                tail_tokens=tail_tokens,
                is_eagle_group=any("eagle" in layer.layer_name.lower() for layer in layers),
            )
        )

    if mode == "direct":
        selected_chunk = chunk_size or scheduler_block_size
        if selected_chunk < scheduler_block_size or selected_chunk % scheduler_block_size:
            raise ValueError(
                "chunk_size must be a positive multiple of scheduler_block_size"
            )
        alignment = scheduler_block_size
    elif mode == "hybrid":
        selected_chunk = scheduler_block_size
        alignment = math.lcm(*(group.token_block_size for group in groups))
    else:
        selected_chunk = canonical_size
        alignment = canonical_size

    return UCMKVCacheSpec(
        groups=tuple(groups),
        scheduler_block_size=scheduler_block_size,
        alignment_block_size=alignment,
        mode=mode,
        chunk_size=selected_chunk,
    )


@dataclass(frozen=True)
class UCMTensorViewLayout:
    base_ptr: int
    row_stride_bytes: int
    token_stride_bytes: int
    tokens_per_row: int
    rows_per_vllm_block: int
    bytes_per_token: int
    row_payload_bytes: int
    buffer_size_bytes: int


@dataclass(frozen=True)
class UCMLayerKVCacheLayout:
    layer_name: str
    layer_index: int
    group_id: int
    views: tuple[UCMTensorViewLayout, ...]


@dataclass(frozen=True)
class UCMGroupKVCacheLayout:
    group_id: int
    layers: tuple[UCMLayerKVCacheLayout, ...]


def _tensor_views(tensor: Any) -> tuple[Any, ...]:
    if isinstance(tensor, (tuple, list)):
        if not tensor:
            raise ValueError("KV cache component tuple must not be empty")
        return tuple(tensor)
    shape = tuple(int(value) for value in tensor.shape)
    if len(shape) == 5:
        raise ValueError(
            "Standalone 5-D combined KV tensors are not part of the verified "
            "Ascend vLLM 0.26 layouts; pass explicit component views or add a "
            "platform fixture before enabling this shape"
        )
    return (tensor,)


def _row_payload_bytes(
    shape: tuple[int, ...], strides: tuple[int, ...], element_size: int
) -> int:
    """Return one row's payload and reject non-dense trailing dimensions.

    The verified Ascend layouts may pad between rows, but each component payload
    after dimension 0 is dense.  Copying ``stride(0)`` bytes would incorrectly
    include another component (notably Kimi's shared Attention/Mamba page).
    """

    expected_stride = 1
    for size, stride in zip(reversed(shape[1:]), reversed(strides[1:])):
        if stride != expected_stride:
            raise ValueError(
                "KV tensor trailing dimensions must be dense; "
                f"shape={shape}, strides={strides}"
            )
        expected_stride *= size
    return expected_stride * element_size


def _view_layout(
    tensor: Any,
    expected_block_size: int,
    *,
    num_blocks: int,
    state_snapshot: bool = False,
) -> UCMTensorViewLayout:
    shape = tuple(int(value) for value in tensor.shape)
    if len(shape) < 2 or len(shape) > 4:
        raise ValueError(
            "Verified Ascend component views must be 2-D, 3-D, or 4-D, "
            f"got shape={shape}"
        )
    if any(value <= 0 for value in shape):
        raise ValueError(f"KV tensor dimensions must be positive, got shape={shape}")
    if shape[0] % num_blocks:
        raise ValueError(
            f"KV tensor first dimension {shape[0]} is not divisible by "
            f"num_blocks={num_blocks}"
        )
    element_size = int(tensor.element_size())
    strides = tuple(int(tensor.stride(index)) for index in range(len(shape)))
    row_stride = strides[0] * element_size
    row_payload = _row_payload_bytes(shape, strides, element_size)
    if row_stride < row_payload:
        raise ValueError(
            f"KV tensor row stride {row_stride} is smaller than payload {row_payload}"
        )
    rows_per_block = shape[0] // num_blocks
    if state_snapshot:
        return UCMTensorViewLayout(
            base_ptr=int(tensor.data_ptr()),
            row_stride_bytes=row_stride,
            token_stride_bytes=row_payload,
            tokens_per_row=1,
            rows_per_vllm_block=rows_per_block,
            bytes_per_token=row_payload,
            row_payload_bytes=row_payload,
            buffer_size_bytes=(shape[0] - 1) * row_stride + row_payload,
        )

    # All verified Ascend 0.26 attention/cache views use dimension 1 as the
    # storage-token axis.  Kimi MLA is the important non-trivial case:
    # (num_blocks * 6, 128, 1, width) represents one logical 768-token block.
    tokens_per_row = shape[1]
    if rows_per_block * tokens_per_row != expected_block_size:
        raise ValueError(
            "KV tensor does not match the concrete cache spec: "
            f"rows_per_vllm_block={rows_per_block}, tokens_per_row={tokens_per_row}, "
            f"expected physical block_size={expected_block_size}"
        )
    token_stride = strides[1] * element_size
    bytes_per_token = token_stride
    if tokens_per_row * bytes_per_token != row_payload:
        raise ValueError(
            "KV tensor token axis does not cover a dense row payload: "
            f"shape={shape}, strides={strides}"
        )
    return UCMTensorViewLayout(
        base_ptr=int(tensor.data_ptr()),
        row_stride_bytes=row_stride,
        token_stride_bytes=token_stride,
        tokens_per_row=tokens_per_row,
        rows_per_vllm_block=rows_per_block,
        bytes_per_token=bytes_per_token,
        row_payload_bytes=row_payload,
        buffer_size_bytes=(shape[0] - 1) * row_stride + row_payload,
    )


class UCMKVCacheLayout:
    """Ragged, per-layer physical layout with deterministic record offsets."""

    def __init__(
        self,
        spec: UCMKVCacheSpec,
        kv_caches: Mapping[str, Any],
        *,
        num_blocks: int,
    ) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.spec = spec
        self.num_blocks = num_blocks
        groups: dict[int, UCMGroupKVCacheLayout] = {}
        layers: dict[str, UCMLayerKVCacheLayout] = {}
        for group in spec.groups:
            group_layers: list[UCMLayerKVCacheLayout] = []
            for layer in sorted(
                group.layers,
                key=lambda item: (item.layer_index, item.layer_name),
            ):
                if layer.layer_name not in kv_caches:
                    raise ValueError(f"Missing KV cache tensor for {layer.layer_name}")
                physical_block_size = int(
                    getattr(layer.kv_cache_spec, "block_size", group.token_block_size)
                )
                raw_views = _tensor_views(kv_caches[layer.layer_name])
                is_hybrid_state = (
                    spec.mode == "hybrid" and UCMGroupTag.STATE in group.tags
                )
                if is_hybrid_state:
                    expected_shapes = getattr(layer.kv_cache_spec, "shapes", None)
                    if expected_shapes is not None:
                        expected_shapes = tuple(
                            tuple(int(value) for value in shape)
                            for shape in expected_shapes
                        )
                        actual_shapes = tuple(
                            tuple(int(value) for value in view.shape[1:])
                            for view in raw_views
                        )
                        if actual_shapes != expected_shapes:
                            raise ValueError(
                                f"State components for {layer.layer_name} do not "
                                f"match spec shapes: {actual_shapes} != {expected_shapes}"
                            )
                parsed_views = tuple(
                    _view_layout(
                        view,
                        physical_block_size,
                        num_blocks=num_blocks,
                        state_snapshot=is_hybrid_state,
                    )
                    for view in raw_views
                )
                if is_hybrid_state and any(
                    view.rows_per_vllm_block != 1 for view in parsed_views
                ):
                    raise ValueError(
                        "Verified Ascend state components require exactly one "
                        f"physical row per vLLM block for {layer.layer_name}"
                    )
                item = UCMLayerKVCacheLayout(
                    layer.layer_name, layer.layer_index, group.group_id, parsed_views
                )
                group_layers.append(item)
                layers[layer.layer_name] = item
            groups[group.group_id] = UCMGroupKVCacheLayout(
                group.group_id, tuple(group_layers)
            )
        self.groups = groups
        self.layers = layers

    def build_load_batches(self, metadata: Any, layer_name: str | None = None) -> UCMProxyBatch:
        return self._build_batches(metadata, "load_plans", layer_name)

    def build_dump_batches(self, metadata: Any, layer_name: str | None = None) -> UCMProxyBatch:
        return self._build_batches(metadata, "dump_plans", layer_name)

    def _build_batches(
        self, metadata: Any, plan_attribute: str, layer_name: str | None
    ) -> UCMProxyBatch:
        keys: list[bytes] = []
        offsets: list[int] = []
        ptrs: list[int] = []
        sizes: list[int] = []
        request_metas: Iterable[Any] = getattr(metadata, "requests", {}).values()
        for request_meta in request_metas:
            for plan in getattr(request_meta, plan_attribute):
                self._append_plan(plan, layer_name, keys, offsets, ptrs, sizes)
        return UCMProxyBatch(tuple(keys), tuple(offsets), tuple(ptrs), tuple(sizes))

    def _append_plan(
        self,
        plan: Any,
        layer_name: str | None,
        keys: list[bytes],
        offsets: list[int],
        ptrs: list[int],
        sizes: list[int],
    ) -> None:
        if not plan.keys:
            return
        token_count = plan.token_end - plan.token_start
        if token_count <= 0 or token_count % len(plan.keys):
            raise ValueError("Dispatch token range must divide evenly across keys")
        key_tokens = token_count // len(plan.keys)
        block_maps = {
            selection.group_id: {
                selection.start_block_index + index: block_id
                for index, block_id in enumerate(selection.block_ids)
            }
            for selection in plan.vllm_blocks
        }
        for key_index, key in enumerate(plan.keys):
            record_offset = 0
            key_start = plan.token_start + key_index * key_tokens
            key_end = key_start + key_tokens
            for group_id in sorted(block_maps):
                group_info = self.spec.groups[group_id]
                group_layout = self.groups[group_id]
                group_key_start = key_start
                if plan.hash_group == "WA":
                    if not group_info.tail_tokens:
                        continue
                    group_key_start = max(key_end - group_info.tail_tokens, 0)
                for layer in group_layout.layers:
                    for view in layer.views:
                        segments = self._segments_for_view(
                            group_info,
                            view,
                            block_maps[group_id],
                            group_key_start,
                            key_end,
                        )
                        for ptr, size in segments:
                            if layer_name is None or layer.layer_name == layer_name:
                                keys.append(key)
                                offsets.append(record_offset)
                                ptrs.append(ptr)
                                sizes.append(size)
                            record_offset += size

    def _segments_for_view(
        self,
        group: UCMKVCacheGroupInfo,
        view: UCMTensorViewLayout,
        block_map: Mapping[int, int],
        token_start: int,
        token_end: int,
    ) -> tuple[tuple[int, int], ...]:
        if self.spec.mode == "hybrid" and UCMGroupTag.STATE in group.tags:
            logical_block = max((token_end - 1) // group.token_block_size, 0)
            if logical_block not in block_map:
                raise ValueError(
                    f"Missing state checkpoint block {logical_block} "
                    f"for group {group.group_id}"
                )
            block_id = block_map[logical_block]
            if block_id < 0 or block_id >= self.num_blocks:
                raise ValueError(
                    f"vLLM block ID {block_id} is outside [0, {self.num_blocks})"
                )
            result: list[tuple[int, int]] = []
            first_row = block_id * view.rows_per_vllm_block
            for row in range(view.rows_per_vllm_block):
                ptr = view.base_ptr + (first_row + row) * view.row_stride_bytes
                size = view.row_payload_bytes
                if ptr + size > view.base_ptr + view.buffer_size_bytes:
                    raise ValueError("KV cache state segment exceeds registered tensor buffer")
                result.append((ptr, size))
            return tuple(result)
        result: list[tuple[int, int]] = []

        def append_segment(ptr: int, size: int) -> None:
            if not size:
                return
            if result and result[-1][0] + result[-1][1] == ptr:
                previous_ptr, previous_size = result[-1]
                result[-1] = (previous_ptr, previous_size + size)
            else:
                result.append((ptr, size))

        first = token_start // group.token_block_size
        last = (token_end - 1) // group.token_block_size
        for logical_block in range(first, last + 1):
            if logical_block not in block_map:
                raise ValueError(
                    f"Missing vLLM block {logical_block} for group {group.group_id}"
                )
            block_id = block_map[logical_block]
            if block_id < 0 or block_id >= self.num_blocks:
                raise ValueError(
                    f"vLLM block ID {block_id} is outside [0, {self.num_blocks})"
                )
            logical_begin = max(token_start, logical_block * group.token_block_size)
            logical_end = min(token_end, (logical_block + 1) * group.token_block_size)
            storage_tokens = view.rows_per_vllm_block * view.tokens_per_row
            numerator_begin = (
                logical_begin - logical_block * group.token_block_size
            ) * storage_tokens
            numerator_end = (
                logical_end - logical_block * group.token_block_size
            ) * storage_tokens
            if numerator_begin % group.token_block_size or numerator_end % group.token_block_size:
                raise ValueError(
                    "Logical token range cannot be represented exactly by tensor layout"
                )
            physical_begin = numerator_begin // group.token_block_size
            physical_end = numerator_end // group.token_block_size
            while physical_begin < physical_end:
                row_in_block, token_in_row = divmod(
                    physical_begin, view.tokens_per_row
                )
                row_end = min(
                    physical_end,
                    (row_in_block + 1) * view.tokens_per_row,
                )
                row_index = block_id * view.rows_per_vllm_block + row_in_block
                ptr = (
                    view.base_ptr
                    + row_index * view.row_stride_bytes
                    + token_in_row * view.token_stride_bytes
                )
                size = (row_end - physical_begin) * view.bytes_per_token
                if ptr + size > view.base_ptr + view.buffer_size_bytes:
                    raise ValueError("KV cache segment exceeds registered tensor buffer")
                append_segment(ptr, size)
                physical_begin = row_end
        return tuple(result)
