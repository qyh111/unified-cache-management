"""Explicit small FA fixture; model-config cases do not use this geometry."""

import copy
from pathlib import Path

from ..runtime.environment import require

BLOCK_SIZE = 128
NUM_BLOCKS = 32
TOKENS = 512
LAYERS = 2


def make_config(args, rank=0):
    from vllm.config import (
        CacheConfig,
        CompilationConfig,
        DeviceConfig,
        KVTransferConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
    )

    from .config import connector_settings

    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        enable_prefix_caching=True,
        num_gpu_blocks_override=NUM_BLOCKS,
    )
    cache_config.kv_cache_layout = "LBNHC"
    config = VllmConfig(
        model_config=ModelConfig(
            model=str(Path(args.output) / "fixture"),
            dtype="float16",
            seed=17,
            max_model_len=TOKENS * 2,
            skip_tokenizer_init=True,
            enforce_eager=True,
        ),
        cache_config=cache_config,
        parallel_config=ParallelConfig(
            tensor_parallel_size=args.tp,
            rank=rank,
            distributed_executor_backend="external_launcher",
        ),
        scheduler_config=SchedulerConfig(
            max_model_len=TOKENS * 2,
            max_num_batched_tokens=TOKENS * 2,
            max_num_seqs=2,
            async_scheduling=False,
            enable_chunked_prefill=True,
            is_encoder_decoder=False,
        ),
        device_config=DeviceConfig(args.backend),
        compilation_config=CompilationConfig(mode=0),
        kv_transfer_config=KVTransferConfig(
            kv_connector="UCMConnector",
            kv_connector_module_path="ucm.integration.vllm.ucm_connector",
            kv_role="kv_both",
            engine_id=Path(args.output).name,
            kv_load_failure_policy="fail",
            kv_connector_extra_config=connector_settings(args),
        ),
    )
    config.cache_config.num_gpu_blocks = NUM_BLOCKS
    # Independently constructed test processes represent one engine. Ascend's
    # platform initialization otherwise assigns a fresh suffix in each one.
    config.kv_transfer_config.engine_id = Path(args.output).name
    require(
        config.cache_config.block_size == BLOCK_SIZE,
        "Platform changed fixture block size",
    )
    return config


def make_native_cache_configs(config, tp):
    import torch
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    register_all_kvcache_specs(config)
    spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=config.model_config.get_num_kv_heads(config.parallel_config),
        head_size=config.model_config.get_head_size(),
        dtype=torch.float16,
    )
    specs = {f"model.layers.{i}.self_attn": spec for i in range(LAYERS)}
    configs = get_kv_cache_configs(
        config,
        [copy.deepcopy(specs) for _ in range(tp)],
        [NUM_BLOCKS * spec.page_size_bytes * LAYERS for _ in range(tp)],
    )
    for native in configs:
        native.kv_cache_layout = "LBNHC"
        require(native.num_blocks == NUM_BLOCKS, "Native block override was ignored")
    return configs, specs


def allocate_fixture(native, specs, device):
    """Reify the declared FA fixture using native allocation descriptors.

    This is intentionally a block-first synthetic view, not an emulation of
    any platform's attention backend. Keep backing allocations alive.
    """
    import torch

    allocations, views = [], {}
    for descriptor in native.kv_cache_tensors:
        backing = torch.zeros(descriptor.size, dtype=torch.uint8, device=device)
        allocations.append(backing)
        for layer_index, name in enumerate(descriptor.layers):
            spec = specs[name]
            itemsize = spec.dtype.itemsize
            shape = (
                native.num_blocks,
                2,
                spec.block_size,
                spec.num_kv_heads,
                spec.head_size,
            )
            strides = (
                descriptor.block_stride // itemsize,
                spec.block_size * spec.num_kv_heads * spec.head_size,
                spec.num_kv_heads * spec.head_size,
                spec.head_size,
                1,
            )
            offset = descriptor.offset + layer_index * descriptor.layer_stride
            views[name] = torch.as_strided(
                backing.view(spec.dtype),
                shape,
                strides,
                offset // itemsize,
            )
            require(
                views[name].stride(0) * itemsize == spec.page_size_bytes,
                "Fixture requires layer-compact full FA pages",
            )
    require(set(views) == set(specs), "Native descriptors did not cover fixture layers")
    return allocations, views
