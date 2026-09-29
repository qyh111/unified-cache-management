"""Explicit, scoped compute-only mocks for no-forward layout validation."""

from contextlib import contextmanager
import importlib
import os
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
