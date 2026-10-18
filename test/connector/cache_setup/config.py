"""Shared engine/request configuration and public store-factory settings."""

from __future__ import annotations

import json
from pathlib import Path


def connector_settings(args):
    """The store is injected via the production factory's module-path contract."""
    config = {
        "storage_backends": str(Path(args.output) / "store"),
        "share_buffer_enable": False,
        "cache_buffer_capacity_gb": 32,
        "store_pipeline": "Cache|Posix",
        "io_direct": False,
        "posix_gc_enable": False,
        "data_dir_shard_bytes": 0,
        "waiting_queue_depth": 1024,
        "running_queue_depth": 4,
        "cache_stream_number": 1,
        "cache_load_exclusive_buffer_number": 2,
    }
    store = {
        "ucm_connector_name": "RealStore",
        "ucm_connector_module_path": "connector.data_check.real_store",
        "ucm_connector_config": config,
    }
    if args.store == "mock":
        store.update(
            ucm_connector_name="HashStore",
            ucm_connector_module_path="connector.data_check.mock_store",
        )
    return {
        "use_layerwise": args.mode == "layerwise",
        "use_request_async_load": args.async_load,
        "enable_event_sync": False,
        "enable_metrics": False,
        "load_tokens_threshold": 0,
        "ucm_connectors": [store],
    }


def register_model(model_path, backend):
    from vllm.model_executor.models import ModelRegistry

    architecture = json.loads((Path(model_path) / "config.json").read_text())[
        "architectures"
    ][0]
    if backend == "cuda":
        cls = ModelRegistry.models[architecture].load_model_cls()
    elif architecture == "Glm5NextForConditionalGeneration":
        from vllm_ascend.models.glm5next.model import Glm5NextForConditionalGeneration

        cls = Glm5NextForConditionalGeneration
    elif architecture == "DeepseekV41ForCausalLM":
        from vllm_ascend.models.deepseek_v41.model import (
            AscendDeepseekV41LLMForCausalLM,
        )

        cls = AscendDeepseekV41LLMForCausalLM
    else:
        raise ValueError(f"Unsupported native model architecture: {architecture}")
    # Inspect the actual source class in this process. A subprocess would lose
    # the explicit source-platform registration and generated version metadata.
    ModelRegistry.register_model(architecture, cls)
    return architecture


def make_native_config(args, rank=0):
    from vllm.config import KVTransferConfig
    from vllm.engine.arg_utils import EngineArgs

    register_model(args.model_config, args.backend)
    # Reuse only the store settings; geometry is entirely native below.
    # Do not instantiate the synthetic fixture's VllmConfig on this path.
    extra = connector_settings(args)
    options = {}
    if args.backend == "cuda":
        from vllm.config import AttentionConfig, KernelConfig

        options["attention_config"] = AttentionConfig(backend=args.attention_backend)
        options["kernel_config"] = KernelConfig(enable_jit_warmup=False)
        options["additional_config"] = {"kda_prefill_backend": "triton"}
    config = EngineArgs(
        **options,
        model=args.model_config,
        dtype="bfloat16",
        max_model_len=max(args.tokens or 16384, 16384),
        max_num_batched_tokens=max(args.tokens or 16384, 16384),
        max_num_seqs=2,
        tensor_parallel_size=args.tp,
        distributed_executor_backend="external_launcher",
        skip_tokenizer_init=True,
        load_format="dummy",
        limit_mm_per_prompt={"image": 0, "video": 0},
        enforce_eager=True,
        enable_prefix_caching=True,
        block_size=128,
        trust_remote_code=True,
        kv_transfer_config=KVTransferConfig(
            kv_connector="UCMConnector",
            kv_connector_module_path="ucm.integration.vllm.ucm_connector",
            kv_role="kv_both",
            engine_id=Path(args.output).name,
            kv_load_failure_policy="fail",
            kv_connector_extra_config=extra,
        ),
    ).create_engine_config()
    config.parallel_config.rank = rank
    if hasattr(args, "resolved_engine_id"):
        config.kv_transfer_config.engine_id = args.resolved_engine_id
    else:
        args.resolved_engine_id = config.kv_transfer_config.engine_id
    if config.model_config.multimodal_config is not None:
        # Requests contain token IDs only. KV consumer defaults otherwise turn
        # on precomputed MM embeddings and require a tokenizer/processor.
        config.model_config.multimodal_config.enable_mm_embeds = False
    resolve_native_length(args, config)
    return config


def resolve_native_length(args, config):
    if not args.tokens_explicit:
        args.tokens = max(1024, 3 * config.cache_config.block_size)
    maximum = args.tokens + config.cache_config.block_size
    config.model_config.max_model_len = maximum
    config.scheduler_config.max_model_len = maximum
    config.scheduler_config.max_num_batched_tokens = maximum
