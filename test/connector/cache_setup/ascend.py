"""Native Ascend MRV1 KV construction with CPU allocations."""

import copy
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch


@contextmanager
def meta_construction(config):
    """Keep unexecuted compute parameters on meta; retain all cache owners."""
    import torch
    import vllm_ascend.models.deepseek_v41.model as dsv
    import vllm_ascend.models.glm5next.model as glm

    class UnexecutedMoE(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.is_internal_router = False

        def forward(self, *args, **kwargs):
            raise AssertionError("The cache test must not execute MoE")

    class UnexecutedEngramHash(torch.nn.Module):
        def __init__(self, vllm_config, text_config, swa_layer):
            super().__init__()
            self.lookback_depth = text_config.engram_max_ngram_size - 1
            self.use_slot_cache = not vllm_config.use_v2_model_runner

        def forward(self, *args, **kwargs):
            raise AssertionError("The cache test must not execute Engram hashing")

    original_to = torch.Tensor.to
    cpu_control = False

    def meta_to(tensor, *args, **kwargs):
        device = kwargs.get("device")
        if args and isinstance(args[0], (str, torch.device)):
            device = args[0]
        if device is not None and torch.device(device).type not in ("cpu", "meta"):
            if args and isinstance(args[0], (str, torch.device)):
                args = (torch.device("meta"), *args[1:])
            else:
                kwargs["device"] = "meta"
        elif tensor.is_meta and device is not None:
            if args and isinstance(args[0], (str, torch.device)):
                args = (torch.device("meta"), *args[1:])
            else:
                kwargs["device"] = "meta"
        return original_to(tensor, *args, **kwargs)

    def factory(original):
        def call(*args, **kwargs):
            if not cpu_control:
                kwargs["device"] = "meta"
            return original(*args, **kwargs)

        return call

    def cpu_function(original):
        def call(*args, **kwargs):
            nonlocal cpu_control
            cpu_control = True
            try:
                with torch.device("cpu"):
                    return original(*args, **kwargs)
            finally:
                cpu_control = False

        return call

    previous_dtype = torch.get_default_dtype()
    with ExitStack() as context:
        context.enter_context(patch.object(glm, "FusedMoEFactory", UnexecutedMoE))
        context.enter_context(patch.object(dsv, "FusedMoEFactory", UnexecutedMoE))
        context.enter_context(
            patch.object(dsv, "create_engram_hash_state", UnexecutedEngramHash)
        )
        context.enter_context(patch.object(torch.Tensor, "to", meta_to))
        for name in (
            "arange",
            "empty",
            "empty_like",
            "empty_strided",
            "eye",
            "full",
            "full_like",
            "linspace",
            "ones",
            "ones_like",
            "rand",
            "rand_like",
            "randn",
            "randn_like",
            "tensor",
            "zeros",
            "zeros_like",
        ):
            context.enter_context(
                patch.object(torch, name, factory(getattr(torch, name)))
            )
        from vllm_ascend.models.deepseek_v41.engram import embedding

        def meta_engram_weights(table):
            codes = torch.empty(
                (table.part_num_embeddings, table.dim),
                dtype=table.storage_dtype,
                device="meta",
            )
            scales = None
            if table.storage_dtype != torch.bfloat16:
                scales = torch.empty(
                    (table.part_num_embeddings, table.dim // table.block_size),
                    dtype=table.scale_dtype,
                    device="meta",
                )
            return codes, scales

        context.enter_context(
            patch.object(
                embedding.AscendParallelEngramEmbedding,
                "_allocate_weights",
                meta_engram_weights,
            )
        )

        context.enter_context(
            patch.object(
                embedding,
                "in_the_same_node_as",
                cpu_function(embedding.in_the_same_node_as),
            )
        )
        torch.set_default_dtype(config.model_config.dtype)
        try:
            with torch.device("meta"):
                yield
        finally:
            torch.set_default_dtype(previous_dtype)


def construct_cache(args, config, rank):
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.model_executor.model_loader.utils import initialize_model
    from vllm.platforms import current_platform
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
    from vllm.v1.worker.utils import AttentionGroup, prepare_kernel_block_sizes
    from vllm_ascend.ascend_config import get_ascend_config
    from vllm_ascend.distributed.parallel_state import init_ascend_model_parallel
    from vllm_ascend.utils import adapt_patch, enable_sfa, register_ascend_customop
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    adapt_patch()
    register_ascend_customop(config)
    from vllm.distributed.device_communicators.cpu_communicator import CpuCommunicator
    from vllm_ascend.patch.worker.patch_distributed import GroupCoordinatorPatch

    def initialize_cpu_communicator(group):
        group.device = torch.device("cpu")
        group.use_custom_op_call = False
        if group.use_device_communicator and group.world_size > 1:
            group.device_communicator = CpuCommunicator(
                cpu_group=group.cpu_group,
                device=group.device,
                device_group=group.device_group,
                unique_name=group.unique_name,
            )

    with (
        set_current_vllm_config(config),
        patch.object(
            GroupCoordinatorPatch,
            "_init_device_communicator",
            initialize_cpu_communicator,
        ),
        patch.object(
            type(current_platform),
            "get_device_communicator_cls",
            return_value="vllm.distributed.device_communicators.cpu_communicator.CpuCommunicator",
        ),
    ):
        init_distributed_environment(
            world_size=args.tp,
            rank=rank,
            local_rank=rank,
            distributed_init_method="file://"
            + str(Path(args.output) / ("gloo-" + args.phase)),
            backend="gloo",
        )
        initialize_model_parallel(tensor_model_parallel_size=args.tp, backend="gloo")
        init_ascend_model_parallel(config.parallel_config, config.engram_config)
        with meta_construction(config):
            model = initialize_model(config, prefix="")
        current_platform.update_block_size_for_backend(config)
        runner = NPUModelRunner.__new__(NPUModelRunner)
        for name in (
            "model_config",
            "cache_config",
            "compilation_config",
            "scheduler_config",
            "parallel_config",
        ):
            setattr(runner, name, getattr(config, name))
        runner.vllm_config = config
        runner.ascend_config = get_ascend_config()
        runner.model = model
        args.native_model_class = type(model).__module__ + "." + type(model).__name__
        args.native_runner_class = type(runner).__module__ + "." + type(runner).__name__
        runner.device = torch.device(args.device)
        runner.dtype = config.model_config.dtype
        runner.kv_cache_dtype = config.model_config.dtype
        runner.block_size = config.cache_config.block_size
        runner.tp_rank = rank
        from vllm.distributed import get_dcp_group, get_pcp_group

        runner.dcp_size = get_dcp_group().world_size
        runner.pcp_size = get_pcp_group().world_size
        runner.use_compress = hasattr(config.model_config.hf_config, "compress_ratios")
        runner.use_sparse = enable_sfa(config)
        runner.sparse_kv_offload_enabled = False
        runner.shared_kv_cache_layers = {}
        runner.runner_only_attn_layers = set()
        runner.sfa_dcp_replicated_indexer_size = 1
        runner.attn_groups = []
        runner.kv_caches = []
        specs = runner.get_kv_cache_spec()
        if not specs:
            raise AssertionError("Native model produced no cache specs")
        from vllm.distributed.kv_transfer.kv_connector.utils import (
            get_current_attn_backends,
        )
        from vllm.v1.attention.backends.utils import (
            get_supported_kv_cache_layouts,
            resolve_kv_cache_layout,
        )

        backends = get_current_attn_backends(config)
        layouts = [[layout.name for layout in get_supported_kv_cache_layouts(backends)]]
        resolve_kv_cache_layout(config, layouts, list(specs.values()))
        register_all_kvcache_specs(config)
        gathered = [None] * args.tp
        torch.distributed.all_gather_object(gathered, specs)
        minimum = 1
        while True:
            config.cache_config.num_gpu_blocks_override = minimum
            try:
                configs = kv_cache_utils.get_kv_cache_configs(
                    config, copy.deepcopy(gathered), [1 << 50] * args.tp
                )
                break
            except ValueError as error:
                if "KV cache is needed" not in str(
                    error
                ) and "No available memory for the cache blocks" not in str(error):
                    raise
                minimum *= 2
                if minimum > 8192:
                    raise
        config.cache_config.num_gpu_blocks_override = minimum * 3
        configs = kv_cache_utils.get_kv_cache_configs(
            config, copy.deepcopy(gathered), [1 << 50] * args.tp
        )
        native = configs[rank]
        native.kv_cache_layout = config.cache_config.kv_cache_layout
        runner.kv_cache_config = native
        runner.use_hybrid_blocks = len(native.kv_cache_groups) > 1
        owners = config.compilation_config.static_forward_context
        for group_id, group in enumerate(native.kv_cache_groups):
            entries = []
            for name in group.layer_names:
                spec = getattr(group.kv_cache_spec, "kv_cache_specs", {}).get(
                    name, group.kv_cache_spec
                )
                entries.append(
                    AttentionGroup(
                        owners[name].get_attn_backend(), [name], spec, group_id, []
                    )
                )
            runner.attn_groups.append(entries)
        kernel_block_sizes = prepare_kernel_block_sizes(native, runner.attn_groups)
        raw = {}
        allocate = runner._allocate_kv_cache_tensors

        def capture_allocations(native_config):
            raw.update(allocate(native_config))
            return raw

        with patch.object(runner, "_allocate_kv_cache_tensors", capture_allocations):
            views = runner.initialize_kv_cache_tensors(native, kernel_block_sizes)
        return native, views, raw, config, kernel_block_sizes
