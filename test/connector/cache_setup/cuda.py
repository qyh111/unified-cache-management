"""CUDA platform bootstrap and native GPUModelRunner KV construction on CPU."""

import copy
import sys
import types
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch


def prepare_cuda_imports(args):
    import torch

    def reject_cuda_initialization(*args, **kwargs):
        raise AssertionError("KV test attempted to initialize CUDA")

    torch.cuda._lazy_init = reject_cuda_initialization
    if not args.vllm_source:
        # Installed builds use their actual extensions. Do not conceal missing
        # libraries or ABI errors with the source-only import sentinels.
        return
    # CudaPlatform imports this registration-only extension unconditionally.
    # No kernels/schemas are invented: any attempted op invocation must fail.
    for name in ("vllm._C_stable_libtorch", "vllm.vllm_flash_attn._vllm_fa2_C"):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__ucm_unexecuted_extension__ = True
            sys.modules[name] = module


def select_cuda_platform(platform_module, args):
    from vllm.platforms.cuda import CudaPlatform
    from vllm.platforms.interface import DeviceCapability

    capability = DeviceCapability(*map(int, args.cuda_capability.split(".")))

    class SourceCudaPlatform(CudaPlatform):
        @classmethod
        def get_device_capability(cls, device_id=0):
            return capability

        @classmethod
        def get_device_name(cls, device_id=0):
            return "explicit-no-device-sm" + args.cuda_capability

        @classmethod
        def get_device_total_memory(cls, device_id=0):
            raise AssertionError("KV tests must use bounded native capacity planning")

        @classmethod
        def get_device_communicator_cls(cls):
            return (
                "vllm.distributed.device_communicators.cpu_communicator.CpuCommunicator"
            )

        @classmethod
        def is_pin_memory_available(cls):
            return False

    # Keep all CUDA config/backend/layout policies inherited from the source.
    # Only hardware probes and control communication are adapted.
    platform_module._current_platform = SourceCudaPlatform()


def prepare_cuda_compute_imports():
    from vllm import _custom_ops

    # Optional GEMM features are unexecuted; missing registration is not a KV
    # constraint. Do not advertise a compute kernel that was not loaded.
    _custom_ops.cutlass_scaled_mm_supports_block_fp8 = lambda *a: False
    _custom_ops.cutlass_scaled_mm_supports_fp8 = lambda *a: False
    _custom_ops.cutlass_scaled_mm_supports_fp4 = lambda *a: False
    _custom_ops.cutlass_group_gemm_supported = lambda *a: False


@contextmanager
def cuda_meta_construction(config):
    import torch

    class Unexecuted(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.is_internal_router = False

        def forward(self, *args, **kwargs):
            raise AssertionError("KV test attempted an unexecuted compute substitute")

    original_to = torch.Tensor.to

    def meta_to(tensor, *args, **kwargs):
        device = (
            args[0]
            if args and isinstance(args[0], (str, torch.device))
            else kwargs.get("device")
        )
        if device is not None and (
            tensor.is_meta or torch.device(device).type not in ("cpu", "meta")
        ):
            if args and isinstance(args[0], (str, torch.device)):
                args = (torch.device("meta"), *args[1:])
            else:
                kwargs["device"] = "meta"
        return original_to(tensor, *args, **kwargs)

    def factory(original):
        def call(*args, **kwargs):
            kwargs["device"] = "meta"
            return original(*args, **kwargs)

        return call

    def reject_compute(*args, **kwargs):
        raise AssertionError("KV-only test attempted a CUDA activation kernel")

    class UnexecutedStream:
        def __init__(self, *args, **kwargs):
            pass

    def skip_moe_weight_metadata(model):
        model.moe_layers = []
        model.moe_mlp_layers = []
        model.num_moe_layers = 0

    previous = torch.get_default_dtype()
    with ExitStack() as context:
        context.enter_context(patch.object(torch.cuda, "Stream", UnexecutedStream))
        context.enter_context(patch.object(torch.cuda, "Event", UnexecutedStream))
        for name in ("silu_and_mul", "silu_and_mul_with_clamp"):
            context.enter_context(
                patch.object(torch.ops._C, name, reject_compute, create=True)
            )
        architecture = config.model_config.architectures[0]
        if architecture == "Glm5NextForConditionalGeneration":
            import vllm.models.glm5next.nvidia.attention as glm_attention
            import vllm.models.glm5next.nvidia.model as glm

            context.enter_context(patch.object(glm, "FusedMoEFactory", Unexecuted))
            context.enter_context(
                patch.object(glm_attention, "SparseAttnIndexerKpool", Unexecuted)
            )
        elif architecture == "DeepseekV41ForCausalLM":
            import vllm.models.deepseek_v41.attention as dsv_attention
            import vllm.models.deepseek_v41.nvidia.model as dsv

            context.enter_context(patch.object(dsv, "DeepseekV4MoE", Unexecuted))
            context.enter_context(
                patch.object(
                    dsv.DeepseekV41LLMForCausalLM,
                    "set_moe_parameters",
                    skip_moe_weight_metadata,
                )
            )
            context.enter_context(patch.object(dsv, "NgramHashState", Unexecuted))
            context.enter_context(patch.object(dsv, "Engram", Unexecuted))
            context.enter_context(
                patch.object(dsv_attention, "SparseAttnIndexer", Unexecuted)
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
        torch.set_default_dtype(config.model_config.dtype)
        try:
            with torch.device("meta"):
                yield
        finally:
            torch.set_default_dtype(previous)


def construct_cuda_cache(args, config, rank):
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.distributed.kv_transfer.kv_connector.utils import (
        get_current_attn_backends,
    )
    from vllm.model_executor.model_loader.utils import initialize_model
    from vllm.v1.attention.backends.utils import (
        get_supported_kv_cache_layouts,
        resolve_kv_cache_layout,
    )
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
    from vllm.v1.worker import gpu_model_runner
    from vllm.v1.worker.utils import AttentionGroup, prepare_kernel_block_sizes

    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=args.tp,
            rank=rank,
            local_rank=rank,
            distributed_init_method="file://"
            + str(Path(args.output) / ("gloo-" + args.phase)),
            backend="gloo",
        )
        initialize_model_parallel(tensor_model_parallel_size=args.tp, backend="gloo")
        with cuda_meta_construction(config):
            model = initialize_model(config, prefix="")
        from vllm.platforms import current_platform

        from .config import resolve_native_length

        current_platform.update_block_size_for_backend(config)
        resolve_native_length(args, config)
        runner = gpu_model_runner.GPUModelRunner.__new__(
            gpu_model_runner.GPUModelRunner
        )
        runner.vllm_config = config
        runner.model_config = config.model_config
        runner.cache_config = config.cache_config
        runner.compilation_config = config.compilation_config
        runner.parallel_config = config.parallel_config
        runner.model = model
        args.native_model_class = type(model).__module__ + "." + type(model).__name__
        args.native_runner_class = type(runner).__module__ + "." + type(runner).__name__
        runner.device = torch.device(args.device)
        runner.shared_kv_cache_layers = {}
        runner.runner_only_attn_layers = set()
        runner.kv_caches = []
        specs = runner.get_kv_cache_spec()
        if not specs:
            raise AssertionError("Native CUDA model produced no KV specs")
        backends = get_current_attn_backends(config)
        resolve_kv_cache_layout(
            config,
            [[x.name for x in get_supported_kv_cache_layouts(backends)]],
            list(specs.values()),
        )
        register_all_kvcache_specs(config)
        gathered = [None] * args.tp
        torch.distributed.all_gather_object(gathered, specs)
        minimum = 1
        while True:
            config.cache_config.num_gpu_blocks_override = minimum
            try:
                kv_cache_utils.get_kv_cache_configs(
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
        native = kv_cache_utils.get_kv_cache_configs(
            config, copy.deepcopy(gathered), [1 << 50] * args.tp
        )[rank]
        native.kv_cache_layout = config.cache_config.kv_cache_layout
        owners = config.compilation_config.static_forward_context
        groups = []
        for group_id, group in enumerate(native.kv_cache_groups):
            groups.append(
                [
                    AttentionGroup(
                        owners[name].get_attn_backend(),
                        [name],
                        getattr(group.kv_cache_spec, "kv_cache_specs", {}).get(
                            name, group.kv_cache_spec
                        ),
                        group_id,
                        [],
                    )
                    for name in group.layer_names
                ]
            )
        kernels = prepare_kernel_block_sizes(native, groups)
        raw = {}
        allocate = gpu_model_runner.allocate_kv_cache

        def capture(*values, **kwargs):
            result = allocate(*values, **kwargs)
            raw.update(result)
            return result

        with patch.object(gpu_model_runner, "allocate_kv_cache", capture):
            views = runner.initialize_kv_cache_tensors(native, kernels)
        return native, views, raw, config, kernels
