# SPDX-License-Identifier: MIT
"""Construct MRV2 models on meta and allocate real KV views across TP x PCP x PP."""

from __future__ import annotations

import importlib.util
import json
import math
import os
from contextlib import ExitStack
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch


def build_kv_transfer_config(args, store_root: Path):
    from vllm.config import KVTransferConfig

    if args.connector == "v1":
        module_path = "ucm.integration.vllm.ucm_connector"
        if args.ucm_config:
            extra = {"UCM_CONFIG_FILE": str(args.ucm_config.resolve())}
        else:
            extra = {
                "use_layerwise": args.use_layerwise,
                "enable_event_sync": False,
                "load_tokens_threshold": 0,
                "ucm_connectors": [
                    {
                        "ucm_connector_name": "UcmMockPosixStore",
                        "ucm_connector_config": {
                            "storage_backends": str(store_root.resolve()),
                            "store_pipeline": "Posix",
                            "cache_buffer_capacity_gb": 16,
                        },
                    }
                ],
            }
    elif args.connector == "v2":
        module_path = "ucm.integration.vllm.v2.ucm_connector"
        if args.ucm_config:
            extra = {"UCM_CONFIG_FILE": str(args.ucm_config.resolve())}
        else:
            extra = {"use_layerwise": args.use_layerwise}
            if args.chunk_size:
                extra["chunk_size"] = args.chunk_size
    else:
        raise ValueError(f"Unsupported connector version: {args.connector}")

    ucm_child = dict(
        kv_connector="UCMConnector",
        kv_connector_module_path=module_path,
        kv_role="kv_both",
        kv_load_failure_policy="fail",
        kv_connector_extra_config=extra,
    )
    multi_mode = getattr(args, "multi_connector", "none")
    if multi_mode == "none":
        return KVTransferConfig(**ucm_child)

    pd_child = dict(
        kv_connector="MockPDHandshakeConnector",
        kv_connector_module_path="mock_pd_connector",
        kv_role="kv_both",
        kv_load_failure_policy="fail",
        kv_connector_extra_config={},
    )
    if multi_mode == "pd-first":
        children = [pd_child, ucm_child]
    elif multi_mode == "ucm-first":
        children = [ucm_child, pd_child]
    else:
        raise ValueError(f"Unsupported multi_connector mode: {multi_mode!r}")

    return KVTransferConfig(
        kv_connector="MultiConnector",
        kv_role="kv_both",
        kv_load_failure_policy="fail",
        kv_connector_extra_config={"connectors": children},
    )


def make_engine(args, store_root: Path):
    if args.platform in ("ascend-sim", "ascend"):
        from vllm_ascend.utils import adapt_patch

        adapt_patch(is_global_patch=True)

    from vllm.config import AttentionConfig
    from vllm.engine.arg_utils import EngineArgs

    model_config = json.loads((args.model / "config.json").read_text())
    architectures = model_config.get("architectures", ())
    needs_tokenizer = bool(model_config.get("vision_config")) or any(
        name.startswith(
            ("DeepseekV41", "Glm5Next", "MiniMaxM3", "Qwen3_5", "Qwen4Exp")
        )
        for name in architectures
    )
    world_size = args.tp * args.pcp * args.pp
    if getattr(args, "auto_align_chunks", 0) > 0:
        model_max_pos = (
            model_config.get("max_position_embeddings")
            or model_config.get("text_config", {}).get("max_position_embeddings")
            or 24576
        )
        # Reserve enough max_model_len headroom for padded Mamba/KDA or FAWA
        # alignments (e.g. 4 x 4096 = 16384 or 4 x 4224 = 16896) without
        # exceeding the model's position limit.
        max_model_len = min(int(model_max_pos), max(args.tokens + 16, 24576 + 16))
    else:
        max_model_len = args.tokens + 16
    max_num_batched_tokens = min(args.max_num_batched_tokens, max_model_len)
    engine_kwargs = dict(
        model=str(args.model.resolve()),
        skip_tokenizer_init=not needs_tokenizer,
        trust_remote_code=args.trust_remote_code,
        load_format="dummy",
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_chunked_prefill=True,
        max_num_seqs=2,
        tensor_parallel_size=args.tp,
        pipeline_parallel_size=args.pp,
        prefill_context_parallel_size=args.pcp,
        decode_context_parallel_size=args.dcp,
        distributed_executor_backend="mp" if world_size > 1 else None,
        dtype=args.dtype,
        quantization=args.quantization,
        speculative_config=args.speculative_config,
        kv_cache_dtype=args.kv_cache_dtype,
        mamba_ssm_cache_dtype=args.mamba_ssm_cache_dtype,
        attention_config=AttentionConfig(indexer_kv_dtype=args.indexer_kv_dtype),
        enforce_eager=True,
        enable_prefix_caching=True,
        async_scheduling=False,
        disable_hybrid_kv_cache_manager=False,
        block_size=args.block_size,
        kv_transfer_config=build_kv_transfer_config(args, store_root),
    )
    if args.cp_kv_cache_interleave_size == "auto":
        if args.dcp > 1:
            # Probe the platform-resolved block_size without kv_transfer_config
            # so Ascend's DCP interleave equality check sees the model's real
            # padded block_size (e.g. Mamba/KDA or DeepSeek-V4) instead of 1.
            probe_kwargs = dict(engine_kwargs, kv_transfer_config=None)
            probe_engine = EngineArgs(**probe_kwargs).create_engine_config()
            engine_kwargs["cp_kv_cache_interleave_size"] = int(
                probe_engine.cache_config.block_size
            )
    elif args.cp_kv_cache_interleave_size is not None:
        engine_kwargs["cp_kv_cache_interleave_size"] = args.cp_kv_cache_interleave_size

    engine = EngineArgs(**engine_kwargs).create_engine_config()
    if not engine.use_v2_model_runner:
        raise ValueError("This model-check requires the platform's MRV2 path")
    return engine


def install_ascend_worker(engine) -> None:
    from vllm_ascend.ascend_config import init_ascend_config
    from vllm_ascend.ascend_forward_context import sync_v2_extra_kwargs
    from vllm_ascend.mrv2_utils import apply_v2_model_runner_config_patch
    from vllm_ascend.utils import (
        adapt_patch,
        register_ascend_customop,
        set_potential_max_tokens,
    )

    set_potential_max_tokens(engine)
    apply_v2_model_runner_config_patch()
    sync_v2_extra_kwargs(engine)
    adapt_patch()
    from vllm_ascend import ops
    from vllm_ascend.patch.worker.patch_kda_recoverssm import patch_kda_recoverssm

    patch_kda_recoverssm(engine)
    ops.register_dummy_fusion_op()
    register_ascend_customop(engine)
    init_ascend_config(engine)


def make_model(engine, *, allow_kpool_host_stub: bool = False):
    import torch
    from vllm.model_executor.model_loader.utils import initialize_model
    from vllm.utils.torch_utils import set_default_torch_dtype

    class MetaPlacement(torch.overrides.TorchFunctionMode):
        """Keep weights and scales on meta even when constructors force a device."""

        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = dict(kwargs or {})
            device = kwargs.get("device")
            if "device" in kwargs and (
                isinstance(device, int) or (func is torch.empty and device is None)
            ):
                kwargs["device"] = torch.device("meta")
            if isinstance(device, (str, torch.device)) and torch.device(
                device
            ).type in ("cuda", "npu"):
                kwargs["device"] = torch.device("meta")
            if (
                func is torch.Tensor.to
                and len(args) > 1
                and isinstance(args[1], (str, torch.device))
                and torch.device(args[1]).type in ("cuda", "npu")
            ):
                args = (args[0], torch.device("meta"), *args[2:])
            return func(*args, **kwargs)

    with (
        set_default_torch_dtype(engine.model_config.dtype),
        torch.device("meta"),
        MetaPlacement(),
        ExitStack() as model_patches,
    ):
        architectures = engine.model_config.architectures or ()
        if any(name.startswith("DeepseekV41") for name in architectures):
            if engine.device_config.device_type == "npu":
                if (
                    importlib.util.find_spec(
                        "vllm_ascend.models.deepseek_v41.engram.embedding"
                    )
                    is not None
                ):
                    from vllm_ascend.models.deepseek_v41.engram.embedding import (
                        AscendParallelEngramEmbedding as EngramEmbedding,
                    )
                else:
                    from vllm_ascend.models.deepseek_v41 import (
                        ParallelEngramEmbedding as EngramEmbedding,
                    )
            else:
                from vllm.models.deepseek_v41.common import engram

                EngramEmbedding = engram.ParallelEngramEmbedding
                model_patches.enter_context(
                    patch.object(engram, "is_uva_available", lambda: True)
                )

            def meta_engram_weights(embedding):
                dtype = getattr(embedding, "storage_dtype", torch.float8_e4m3fn)
                codes = torch.empty(
                    (embedding.part_num_embeddings, embedding.dim),
                    dtype=dtype,
                    device="meta",
                )
                scales = None
                if dtype != torch.bfloat16:
                    scales = torch.empty(
                        (
                            embedding.part_num_embeddings,
                            embedding.dim // embedding.block_size,
                        ),
                        dtype=getattr(embedding, "scale_dtype", torch.uint8),
                        device="meta",
                    )
                return codes, scales

            model_patches.enter_context(
                patch.object(EngramEmbedding, "_allocate_weights", meta_engram_weights)
            )

        if engine.device_config.device_type == "cuda" and any(
            name.startswith("Qwen4Exp") for name in architectures
        ):
            from vllm.models.qwen4_exp.nvidia import ngram_embedding

            model_patches.enter_context(
                patch.object(
                    ngram_embedding,
                    "Qwen4ExpPLEPinnedHostEmbedding",
                    ngram_embedding.Qwen4ExpPLEDeviceEmbedding,
                )
            )

        if engine.device_config.device_type == "npu":
            from vllm_ascend.distributed import parallel_state as ascend_ps

            if ascend_ps._MC2 is not None:
                # ModelSlim W8A8/W4A8 MoE reads get_mc2_group().device_group to
                # query HCCL comm names and falls back on AttributeError. Host
                # Gloo groups have no NPU backend, so expose None during meta
                # model construction only and restore immediately afterward.
                model_patches.enter_context(
                    patch.object(ascend_ps._MC2, "device_group", None)
                )

            if (
                allow_kpool_host_stub
                and any(name.startswith("Glm5Next") for name in architectures)
                and importlib.util.find_spec("cann_ops_transformer") is None
            ):
                from vllm_ascend.attention.indexer_kpool import (
                    Glm5NextKPoolIndexerBackend,
                )

                # Meta model construction never runs forward; skip forward-only
                # cann_ops_transformer op handle lookup when the operator wheel
                # is absent in a CPU-only ascend-sim container.
                if int(os.environ.get("RANK", "0")) == 0:
                    print(
                        "[rank 0] Note: stubbing Glm5NextKPoolIndexerBackend._init_cann_ops "
                        "for host meta build (cann_ops_transformer absent)",
                        flush=True,
                    )
                model_patches.enter_context(
                    patch.object(
                        Glm5NextKPoolIndexerBackend,
                        "_init_cann_ops",
                        lambda self: None,
                    )
                )

        model = initialize_model(engine)
        if engine.speculative_config is not None:
            if engine.device_config.device_type == "npu":
                from vllm_ascend.worker.v2.spec_decode import init_speculator
            else:
                from vllm.v1.worker.gpu.spec_decode import init_speculator

            speculator = init_speculator(engine, torch.device("cpu"), None)
            speculator.load_model(model)
            draft_layers = speculator.draft_attn_layer_names
            if not draft_layers:
                raise ValueError("Speculative draft registered no attention layers")
            if any(not param.is_meta for param in speculator.model.parameters()):
                raise RuntimeError("Draft model materialized weights outside meta")
            model._ucm_test_speculator = speculator

    if any(not param.is_meta for param in model.parameters()):
        raise RuntimeError("Model construction materialized weights outside meta")
    return model


def native_kv_head_partitions(engine, model, specs) -> dict[str, int]:
    """Read GQA partition counts from the cache layer's owning QKV module."""
    from vllm.v1.kv_cache_interface import (
        AttentionSpec,
        MLAAttentionSpec,
        SlidingWindowMLASpec,
    )

    layers = engine.compilation_config.static_forward_context
    modules = dict(model.named_modules())
    module_names = {id(module): name for name, module in modules.items()}
    tp_size = engine.parallel_config.tensor_parallel_size
    if tp_size == 1:
        return {}
    partitions = {}
    for layer_name, spec in specs.items():
        if not isinstance(spec, AttentionSpec) or isinstance(
            spec, (MLAAttentionSpec, SlidingWindowMLASpec)
        ):
            continue
        max_tp_shards = getattr(spec, "max_tp_shards", None)
        if max_tp_shards is not None:
            count = min(tp_size, max_tp_shards)
            if count <= 0 or tp_size % count:
                raise ValueError(f"Invalid native GQA partition for {layer_name}")
            partitions[layer_name] = count
            continue

        module_name = module_names.get(id(layers.get(layer_name)))
        while module_name is not None:
            owner = modules[module_name]
            projection = getattr(owner, "qkv_proj", owner)
            replicas = getattr(projection, "num_kv_head_replicas", None)
            total_heads = getattr(owner, "total_num_kv_heads", None)
            if replicas is not None:
                if replicas <= 0 or tp_size % replicas:
                    raise ValueError(f"Invalid native GQA replica count for {layer_name}")
                partitions[layer_name] = tp_size // replicas
                break
            if total_heads is not None:
                partitions[layer_name] = min(tp_size, total_heads)
                break
            module_name = module_name.rpartition(".")[0] if module_name else None

        if layer_name not in partitions:
            # In ordinary GQA only the single-local-head case can have head
            # replicas. That case needs the owning projection's geometry.
            if spec.num_kv_heads <= 1:
                raise ValueError(
                    f"Cannot resolve native GQA partition for {layer_name}"
                )
            partitions[layer_name] = tp_size
        if partitions[layer_name] <= 0 or tp_size % partitions[layer_name]:
            raise ValueError(f"Invalid native GQA partition for {layer_name}")
    return partitions


def allocate(
    engine,
    kv_mib,
    scratch_dir,
    cleanup,
    *,
    kv_device,
    connector_version: str,
    auto_dcp_interleave: bool = False,
    schedule_resolver=None,
    chunk_size: int | None = None,
):
    from parallel import gather
    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
        parallel_state,
    )
    from vllm.distributed.kv_transfer.kv_connector.utils import (
        get_current_attn_backends,
    )
    from vllm.v1.attention.backends.utils import (
        get_supported_kv_cache_layouts,
        resolve_kv_cache_layout,
    )
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
    from vllm.v1.worker.gpu import attn_utils, model_runner
    from vllm.v1.worker.utils import AttentionGroup

    is_ascend = engine.device_config.device_type == "npu"
    cleanup.enter_context(set_current_vllm_config(engine))
    if is_ascend:
        install_ascend_worker(engine)
        live = (
            model_runner.get_kv_cache_spec,
            model_runner.init_kv_cache.__globals__["allocate_kv_cache"],
            model_runner.init_kv_cache.__globals__["bind_kv_cache_to_layers"],
            attn_utils.prepare_kernel_block_sizes,
            kv_cache_utils.get_kv_cache_configs.__globals__[
                "get_kv_cache_config_from_groups"
            ],
            kv_cache_utils.resolve_kv_cache_block_sizes,
        )
        for fn in live:
            if not fn.__module__.startswith("vllm_ascend."):
                raise RuntimeError(f"Ascend MRV2 patch missing at {fn.__name__}")

    parallel = engine.parallel_config
    rank = int(os.environ.get("RANK", "0"))
    world = parallel.world_size
    parallel.rank = rank

    rendezvous = "env://" if world > 1 else (Path(scratch_dir) / "rendezvous").as_uri()
    coordinator = parallel_state.GroupCoordinator

    def host_group(**kwargs):
        kwargs.update(
            use_device_communicator=False, use_message_queue_broadcaster=False
        )
        return coordinator(**kwargs)

    with ExitStack() as group_options:
        group_options.enter_context(
            patch.object(parallel_state, "GroupCoordinator", host_group)
        )
        if is_ascend:
            from vllm_ascend.patch.worker import patch_distributed

            group_options.enter_context(
                patch.object(
                    patch_distributed, "create_hccl_pg_options", lambda name: None
                )
            )
        init_distributed_environment(
            world, rank, rendezvous, rank, "gloo", timeout=timedelta(minutes=5)
        )
        cleanup.callback(destroy_distributed_environment)
        initialize_model_parallel(
            tensor_model_parallel_size=parallel.tensor_parallel_size,
            pipeline_model_parallel_size=parallel.pipeline_parallel_size,
            prefill_context_model_parallel_size=parallel.prefill_context_parallel_size,
            decode_context_model_parallel_size=parallel.decode_context_parallel_size,
            backend="gloo",
        )
        cleanup.callback(destroy_model_parallel)
        if is_ascend:
            from vllm_ascend.distributed.parallel_state import (
                destroy_ascend_model_parallel,
                init_ascend_model_parallel,
            )

            engram_cfg = getattr(engine, "engram_config", None)
            if (
                hasattr(parallel_state, "get_engram_dp_group")
                and parallel_state.get_engram_dp_group() is not None
            ):
                engram_cfg = None
            init_ascend_model_parallel(parallel, engram_cfg)
            cleanup.callback(destroy_ascend_model_parallel)

    register_all_kvcache_specs(engine)
    model = make_model(engine, allow_kpool_host_stub=(kv_device.type == "cpu"))
    specs = model_runner.get_kv_cache_spec(engine)
    if not specs:
        raise ValueError(f"{type(model).__name__} registered no KV cache specs")
    if any(getattr(spec, "non_causal", False) for spec in specs.values()):
        raise ValueError("Non-causal prefill disables native prefix caching")
    head_partitions = (
        native_kv_head_partitions(engine, model, specs)
        if connector_version == "v1"
        else {}
    )

    supported = get_supported_kv_cache_layouts(get_current_attn_backends(engine))
    packing = resolve_kv_cache_layout(
        engine, gather([layout.name for layout in supported]), specs.values()
    )

    def _plan_kv_configs():
        if kv_mib is None:
            groups = kv_cache_utils.get_kv_cache_groups(engine, dict(specs))
            # Reserve enough blocks for two requests (cold write + hot reload).
            per_req = kv_cache_utils._max_memory_usage_bytes_from_groups(engine, groups)
            one_block = kv_cache_utils._pool_bytes_per_block(engine, groups)
            kv_bytes = per_req * 2 + one_block * 4
        else:
            kv_bytes = kv_mib << 20

        all_inputs = gather((specs, kv_bytes))
        all_specs, budgets = zip(*all_inputs)
        configs = kv_cache_utils.get_kv_cache_configs(
            engine, list(all_specs), list(budgets)
        )
        for cfg in configs:
            cfg.kv_cache_layout = packing.name
        w_cfg = configs[rank]
        s_cfg = kv_cache_utils.generate_scheduler_kv_cache_config(configs)

        engine.cache_config.num_gpu_blocks = s_cfg.num_blocks
        active_groups = (
            s_cfg.prefix_cacheable_groups or s_cfg.kv_cache_groups
        )
        engine.cache_config.block_size = min(
            group.kv_cache_spec.block_size for group in active_groups
        )
        kv_cache_utils.update_kv_cache_capacity(engine, s_cfg)
        engine.adjust_dcp_kv_cache_interleave_size(w_cfg)
        if (
            auto_dcp_interleave
            and engine.parallel_config.decode_context_parallel_size > 1
        ):
            final_block_size = int(engine.cache_config.block_size)
            interleave = int(engine.parallel_config.cp_kv_cache_interleave_size)
            if interleave > final_block_size or final_block_size % interleave != 0:
                engine.parallel_config.cp_kv_cache_interleave_size = final_block_size
        engine.validate_block_size()

        b_size, h_size = kv_cache_utils.resolve_kv_cache_block_sizes(s_cfg, engine)
        align = kv_cache_utils.resolve_cache_hit_alignment_tokens(
            s_cfg, engine, b_size, h_size
        )
        engine.cache_config.hash_block_size = h_size
        engine.cache_config.cache_hit_alignment_tokens = align
        for cfg in (w_cfg, s_cfg):
            cfg.hash_block_size = h_size
            cfg.cache_hit_alignment_tokens = align
        return w_cfg, s_cfg, b_size, h_size, align

    worker_config, scheduler_config, block_size, hash_size, alignment = (
        _plan_kv_configs()
    )
    resolved_plan = None
    if schedule_resolver is not None:
        planned_alignment = math.lcm(
            int(block_size), int(alignment or 1), int(chunk_size or 1)
        )
        effective_tokens, effective_batch, prefix_len, expected_hit = schedule_resolver(
            planned_alignment, int(engine.model_config.max_model_len)
        )
        resolved_plan = (
            effective_tokens,
            effective_batch,
            prefix_len,
            expected_hit,
            planned_alignment,
        )
        final_max_len = effective_tokens + 16
        if (
            final_max_len != int(engine.model_config.max_model_len)
            or effective_batch != int(engine.scheduler_config.max_num_batched_tokens)
        ):
            # Re-plan KV capacity tightly to the final schedule BEFORE allocating
            # CPU tensors or constructing Worker/Connector/Scheduler so all
            # derived configs (max_in_flight_tokens, KVCacheManager, num_blocks)
            # match the actual run without post-hoc mutation.
            engine.model_config.max_model_len = final_max_len
            engine.scheduler_config.max_num_batched_tokens = effective_batch
            if getattr(engine.scheduler_config, "max_num_scheduled_tokens", None) is not None:
                engine.scheduler_config.max_num_scheduled_tokens = effective_batch
            engine.cache_config.num_gpu_blocks = None
            worker_config, scheduler_config, block_size, hash_size, alignment = (
                _plan_kv_configs()
            )
            rechecked_alignment = math.lcm(
                int(block_size), int(alignment or 1), int(chunk_size or 1)
            )
            if rechecked_alignment != planned_alignment:
                raise AssertionError(
                    f"Alignment changed after tightening max_model_len: "
                    f"before={planned_alignment}, after={rechecked_alignment}"
                )

    layers = engine.compilation_config.static_forward_context
    physical = deepcopy(worker_config)
    if is_ascend:
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise.layerwise_cache_layout import (
            apply_layerwise_kv_cache_plan,
        )

        apply_layerwise_kv_cache_plan(physical, engine)
    attn_utils.add_kv_sharing_layers_to_kv_cache_groups(
        attn_utils.get_shared_kv_cache_layers(engine), physical.kv_cache_groups
    )
    attn_groups = []
    for group_id, group in enumerate(physical.kv_cache_groups):
        spec = group.kv_cache_spec
        attn_groups.append(
            [
                AttentionGroup(
                    backend=layers[name].get_attn_backend(),
                    layer_names=[name],
                    kv_cache_spec=(
                        spec.kv_cache_specs[name]
                        if isinstance(spec, UniformTypeKVCacheSpecs)
                        else spec
                    ),
                    kv_cache_group_id=group_id,
                )
                for name in group.layer_names
                if spec.has_layer_views
            ]
        )
    kernel_sizes = attn_utils.prepare_kernel_block_sizes(physical, attn_groups)
    views = model_runner.init_kv_cache(
        layers, physical, kv_device, kernel_sizes, engine
    )
    for name, view in views.items():
        tensors = view if isinstance(view, (tuple, list)) else (view,)
        if any(tensor.device != kv_device for tensor in tensors):
            raise RuntimeError(f"Native allocator placed {name} outside {kv_device}")
    for name, target in attn_utils.get_shared_kv_cache_layers(engine).items():
        if target in head_partitions:
            head_partitions[name] = head_partitions[target]

    print(
        f"[rank {rank}] {type(model).__name__}: layout={packing.name}, "
        f"blocks={worker_config.num_blocks}, views={len(views)}, "
        f"block_size={block_size}, hash_size={hash_size}, alignment={alignment}, "
        f"max_model_len={engine.model_config.max_model_len}, "
        f"max_num_batched_tokens={engine.scheduler_config.max_num_batched_tokens}",
        flush=True,
    )
    return (
        worker_config,
        scheduler_config,
        block_size,
        hash_size,
        views,
        resolved_plan,
        head_partitions,
    )
