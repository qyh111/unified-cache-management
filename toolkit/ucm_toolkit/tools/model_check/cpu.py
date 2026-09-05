# SPDX-License-Identifier: MIT
"""CPU（官方 vLLM CPU 构建）UCM KV-cache compatibility check without weights.

与 ascend.py/cuda.py 同一套 common 流程，只是：

1. 目标平台是官方 vLLM 的 CPU 构建（vllm-0.26.0+cpu 等）：平台按
   ``vllm_version_matches_substr("cpu")`` 自动激活 CpuPlatform，无需任何
   设备/驱动；
2. 模型在 meta device 上构造（不加载权重），runner 用官方
   GPUModelRunner（device=cpu），分布式 backend 用 gloo；
3. UCM 核心需以 **simu** 运行时编译（``export PLATFORM=simu`` 或留空执行
   ``pip install -e .``，setup.py 的 fallback 分支即
   ``-DRUNTIME_ENVIRONMENT=simu``），不依赖 NPU/CUDA 设备库。

运行（在装了官方 vllm CPU 构建 + simu 版 UCM + ucm_toolkit 的 venv 里）：

    ucm-toolkit run model-check --model /path/to/model --block-size 128 \
        --storage-backends /path/to/ucm_storage

与 ascend/cuda 的差异：

- ``make_model`` 不做任何 vllm-ascend patch（官方 vllm 没有平台插件）；
- ``patch_groups`` 在 CPU 上为空操作（官方分组/manager 工厂直接用）；
  多 group（如 DeepSeekV4 的 SWA 混合）在官方 vllm 的
  ``_get_kv_cache_groups_uniform_groups`` 存在页大小断言限制（SWA 子组页 >
  full-MLA 组页），非 GPU 平台会在此失败——这是上游代码边界，单组模型
  （如 GLM 系列）不受影响。
"""

from __future__ import annotations

import gc
import os
import time
import traceback
from typing import Any

import torch

from .config import load_config
from .common import (
    CacheFixture,
    UnsupportedEnvironment,
    current_vllm_config_context,
    log,
)
from .common import make_cache as make_common_cache
from .common import make_config as make_common_config
from .common import (
    no_real_device_move_from_meta_context,
    make_worker,
    schedule,
    verify,
)

# =========================== User configuration ===========================
config = load_config()
model = config.model
tokens = config.tokens
block_size = config.block_size
use_layerwise = config.use_layerwise
additional_config = config.additional_config
store_pipeline = config.store_pipeline
storage_backends = config.storage_backends
dtype = config.dtype
kv_cache_dtype = config.kv_cache_dtype
trust_remote_code = True
request_token_salt = time.time_ns() ^ os.getpid()


def _factory_kwargs_redirect_to_meta(kwargs: dict[str, Any]) -> dict[str, Any]:
    rewritten_kwargs = dict(kwargs)
    device_arg = rewritten_kwargs.get("device")
    if device_arg is None:
        rewritten_kwargs["device"] = torch.device("meta")
        return rewritten_kwargs
    try:
        target_device = torch.device(device_arg)
    except Exception:
        return rewritten_kwargs
    if target_device.type != "meta":
        rewritten_kwargs["device"] = torch.device("meta")
    return rewritten_kwargs


# ---------------------------------------------------------------------------
# 1. Construct VllmConfig (device="cpu")
# ---------------------------------------------------------------------------


def make_config() -> Any:
    return make_common_config(
        model,
        tokens,
        block_size,
        dtype,
        kv_cache_dtype,
        trust_remote_code,
        additional_config,
        store_pipeline,
        storage_backends,
        use_layerwise,
        "cpu",
    )


# ---------------------------------------------------------------------------
# 2. No-weight model + production ModelRunner
# ---------------------------------------------------------------------------


def make_model(vllm_config: Any) -> tuple[Any, Any]:
    """Build the production model structure on the meta device (official vLLM)."""

    from vllm.model_executor.model_loader.utils import initialize_model

    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(vllm_config.model_config.dtype)
        vllm_config.compilation_config.static_forward_context.clear()
        with (
            current_vllm_config_context(vllm_config),
            no_real_device_move_from_meta_context(_factory_kwargs_redirect_to_meta),
            torch.device("meta"),
        ):
            model = initialize_model(vllm_config, prefix="")
    except Exception as exc:
        raise UnsupportedEnvironment(
            "This model/vLLM combination cannot construct its model structure on "
            "the meta device with the official (CPU-build) vLLM. The checker will "
            "not fall back to loading checkpoint weights. "
            f"Original error: {type(exc).__name__}: {exc}\n"
            f"Original traceback:\n{traceback.format_exc()}"
        ) from exc
    finally:
        torch.set_default_dtype(previous_dtype)
    return model, vllm_config


def select_model_runner_cls() -> type[Any]:
    """Return the production runner class of the installed vLLM."""
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except Exception as exc:
        raise UnsupportedEnvironment(
            "CPU checks require vllm.v1.worker.gpu_model_runner.GPUModelRunner "
            "so KV tensors are initialized through the real runner path. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    return GPUModelRunner


def make_runner(
    vllm_config: Any,
    model: Any,
    device: torch.device,
) -> Any:
    """Instantiate the production ModelRunner without loading checkpoint weights."""

    runner_cls = select_model_runner_cls()
    try:
        with current_vllm_config_context(vllm_config):
            runner = runner_cls(vllm_config, device)
    except Exception as exc:
        raise UnsupportedEnvironment(
            "Production ModelRunner construction failed before any checkpoint "
            f"loader was called. Original error: {type(exc).__name__}: {exc}\n"
            f"Original traceback:\n{traceback.format_exc()}"
        ) from exc

    # load_model() normally installs this attribute.  The checker has already
    # constructed the same module hierarchy on meta, so attach it explicitly.
    runner.model = model
    return runner


def make_cache(vllm_config: Any, active_device: torch.device) -> CacheFixture:
    return make_common_cache(
        vllm_config,
        active_device,
        tokens,
        block_size,
        "gloo",
        torch.cpu.synchronize,
        make_model,
        make_runner,
    )


# ---------------------------------------------------------------------------
# Group patch (no-op on official vLLM / CPU)
# ---------------------------------------------------------------------------


def patch_groups(fixture: CacheFixture) -> None:
    """Official vLLM needs no Ascend-style manager patch on CPU.

    Single-group configs go through the upstream manager factory directly.
    DeepseekV4-style multi-group (SWA mixed) layouts hit the upstream
    ``_get_kv_cache_groups_uniform_groups`` page-size assertion on non-GPU
    platforms (see module docstring) and fail before this point.
    """
    return


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def check_config() -> None:
    if tokens <= 0 or block_size <= 0:
        raise ValueError("tokens and block_size must be positive")


def main() -> int:
    check_config()
    active_device = torch.device("cpu")
    log(f"device={active_device}")
    vllm_config = None
    fixture = None
    worker = None
    dispatch = None
    try:
        vllm_config = make_config()
        fixture = make_cache(vllm_config, active_device)
        # UCM's MLA shared buffer is created by the worker.  The scheduler reads
        # the worker-published id, so initialize the worker before Scheduler.
        worker = make_worker(fixture)
        dispatch = schedule(fixture, tokens, request_token_salt, patch_groups)
        verify(fixture, dispatch, worker, torch.cpu.synchronize)
        return 0
    finally:
        del dispatch, worker, fixture, vllm_config
        gc.collect()
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    raise SystemExit(main())