"""Explicit, scoped compute-only mocks for no-forward layout validation."""

from contextlib import contextmanager
import importlib
import os
from types import SimpleNamespace
from unittest.mock import patch


class NoCompute:
    def __call__(self, *args, **kwargs):
        raise AssertionError("FP8 compute was executed in a layout-only check")

    apply_weights = __call__


@contextmanager
def glm53_fp8_layout_mock(vllm_config):
    if os.environ.get("UCM_MODEL_CHECK_GLM53_FP8_MOCK") != "1":
        yield
        return
    model_type = getattr(vllm_config.model_config.hf_text_config, "model_type", None)
    if model_type not in ("glm5_next", "glm5_next_text"):
        raise ValueError("GLM5.3 FP8 layout mock requires a GLM5.3 config")
    fp8 = importlib.import_module("vllm.model_executor.layers.quantization.fp8")
    print("[ucm-kv-check] MOCK: FP8 linear compute factory only; forward forbidden; config/spec/dtype unchanged", flush=True)
    with patch.object(fp8, "init_fp8_linear_kernel", lambda **kwargs: NoCompute()):
        yield


@contextmanager
def qwen_next_qsa_layout_mock(vllm_config):
    if os.environ.get("UCM_MODEL_CHECK_QWEN_NEXT_QSA_MOCK") != "1":
        yield
        return
    model_type = getattr(vllm_config.model_config.hf_text_config, "model_type", None)
    if model_type not in ("qwen4_exp", "qwen4_exp_text"):
        raise ValueError("Qwen-Next QSA layout mock requires a Qwen4Exp config")
    qsa = importlib.import_module("vllm.models.qwen4_exp.nvidia.qsa")
    print("[ucm-kv-check] MOCK: QSA flash-attn availability only; forward forbidden; config/spec/dtype unchanged", flush=True)
    with patch.object(qsa, "is_flash_attn_varlen_func_available", lambda: True):
        yield


@contextmanager
def dsv41_fp8_layout_mock(vllm_config):
    """Unblock DeepSeek V4.1 construction on CPU without touching layout.

    Three construction-time gates are CPU-hostile and none influences a KV
    spec, shape, group, dtype or page: (a) the FP8 block-scaled linear kernel
    chooser requires AMX tiles, (b) ``VllmConfig._resolve_and_verify_engram_config``
    only builds an EngramConfig on CUDA although the Engram n-gram embedding
    registers no KV cache, and (c) the Engram embedding sizes its launch grid
    from the accelerator SM count. Forward stays forbidden via NoCompute.
    """

    if os.environ.get("UCM_MODEL_CHECK_DSV41_FP8_MOCK") != "1":
        yield
        return
    model_type = getattr(vllm_config.model_config.hf_text_config, "model_type", None)
    if model_type not in ("deepseek_v41", "deepseek_v41_text"):
        raise ValueError("DSV4.1 FP8 layout mock requires a deepseek_v41 config")
    fp8 = importlib.import_module("vllm.model_executor.layers.quantization.fp8")
    print(
        "[ucm-kv-check] MOCK: DSV4.1 FP8 linear kernel selection, Engram "
        "feature config and accelerator probes only; forward forbidden; "
        "config/spec/dtype unchanged",
        flush=True,
    )
    if getattr(vllm_config, "engram_config", None) is None:
        engram = importlib.import_module("vllm.config.engram")
        # Stays for the engine's later stages; the Engram lookup (forward)
        # is the only consumer of the CUDA offload settings.
        vllm_config.engram_config = engram.EngramConfig(
            cpu_offload=False, dp_shared_memory=False
        )
    torch = importlib.import_module("torch")
    fake_gpu = SimpleNamespace(multi_processor_count=132)
    with (
        patch.object(fp8, "init_fp8_linear_kernel", lambda **kwargs: NoCompute()),
        patch.object(torch.accelerator, "current_device_index", lambda: 0),
        patch.object(
            torch.cuda, "get_device_properties", lambda *args, **kwargs: fake_gpu
        ),
    ):
        yield
