import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ucm_toolkit.tools.model_check.compute_mocks import glm53_fp8_layout_mock


class ComputeMockTests(unittest.TestCase):
    def test_scoped_factory_and_no_forward(self):
        original = object()
        module = NS(init_fp8_linear_kernel=original)
        config = NS(model_config=NS(hf_text_config=NS(model_type="glm5_next_text")))
        with patch.dict(os.environ, UCM_MODEL_CHECK_GLM53_FP8_MOCK="1"), patch(
            "ucm_toolkit.tools.model_check.compute_mocks.importlib.import_module", return_value=module
        ):
            with self.assertRaises(AssertionError):
                with glm53_fp8_layout_mock(config):
                    module.init_fp8_linear_kernel().apply_weights()
            self.assertIs(module.init_fp8_linear_kernel, original)

    def test_default_does_not_import_or_patch(self):
        with patch.dict(os.environ, UCM_MODEL_CHECK_GLM53_FP8_MOCK="0"), patch(
            "ucm_toolkit.tools.model_check.compute_mocks.importlib.import_module", side_effect=AssertionError
        ):
            with glm53_fp8_layout_mock(None):
                pass

    def test_rejects_other_models(self):
        with patch.dict(os.environ, UCM_MODEL_CHECK_GLM53_FP8_MOCK="1"), self.assertRaises(ValueError):
            with glm53_fp8_layout_mock(NS(model_config=NS(hf_text_config=NS(model_type="other")))):
                pass


class QwenNextQsaMockTests(unittest.TestCase):
    def test_scoped_availability_and_restore(self):
        original = object()
        module = NS(is_flash_attn_varlen_func_available=original)
        config = NS(model_config=NS(hf_text_config=NS(model_type="qwen4_exp_text")))
        with patch.dict(os.environ, UCM_MODEL_CHECK_QWEN_NEXT_QSA_MOCK="1"), patch(
            "ucm_toolkit.tools.model_check.compute_mocks.importlib.import_module", return_value=module
        ):
            from ucm_toolkit.tools.model_check.compute_mocks import (
                qwen_next_qsa_layout_mock,
            )

            with qwen_next_qsa_layout_mock(config):
                self.assertTrue(module.is_flash_attn_varlen_func_available())
            self.assertIs(module.is_flash_attn_varlen_func_available, original)

    def test_default_does_not_import_or_patch(self):
        from ucm_toolkit.tools.model_check.compute_mocks import (
            qwen_next_qsa_layout_mock,
        )

        with patch.dict(os.environ, UCM_MODEL_CHECK_QWEN_NEXT_QSA_MOCK="0"), patch(
            "ucm_toolkit.tools.model_check.compute_mocks.importlib.import_module", side_effect=AssertionError
        ):
            with qwen_next_qsa_layout_mock(None):
                pass

    def test_rejects_other_models(self):
        from ucm_toolkit.tools.model_check.compute_mocks import (
            qwen_next_qsa_layout_mock,
        )

        with patch.dict(os.environ, UCM_MODEL_CHECK_QWEN_NEXT_QSA_MOCK="1"), self.assertRaises(ValueError):
            with qwen_next_qsa_layout_mock(NS(model_config=NS(hf_text_config=NS(model_type="other")))):
                pass


class Dsv41Fp8MockTests(unittest.TestCase):
    def _modules(self):
        fp8_original = object()
        fp8_mod = NS(init_fp8_linear_kernel=fp8_original)
        engram_calls = []

        def engram_config_factory(**kwargs):
            engram_calls.append(kwargs)
            return NS(**kwargs)

        engram_mod = NS(EngramConfig=engram_config_factory)
        acc_original, props_original = object(), object()
        torch_mod = NS(
            accelerator=NS(current_device_index=acc_original),
            cuda=NS(get_device_properties=props_original),
        )

        def fake_import(name):
            return {
                "vllm.model_executor.layers.quantization.fp8": fp8_mod,
                "vllm.config.engram": engram_mod,
                "torch": torch_mod,
            }[name]

        return fp8_mod, fp8_original, engram_calls, torch_mod, acc_original, props_original, fake_import

    def test_scoped_patches_engram_config_and_no_forward(self):
        from ucm_toolkit.tools.model_check.compute_mocks import dsv41_fp8_layout_mock

        (
            fp8_mod,
            fp8_original,
            engram_calls,
            torch_mod,
            acc_original,
            props_original,
            fake_import,
        ) = self._modules()
        config = NS(
            model_config=NS(hf_text_config=NS(model_type="deepseek_v41_text")),
            engram_config=None,
        )
        with patch.dict(os.environ, UCM_MODEL_CHECK_DSV41_FP8_MOCK="1"), patch(
            "ucm_toolkit.tools.model_check.compute_mocks.importlib.import_module",
            side_effect=fake_import,
        ):
            with dsv41_fp8_layout_mock(config):
                self.assertEqual(engram_calls, [{"cpu_offload": False, "dp_shared_memory": False}])
                self.assertFalse(config.engram_config.cpu_offload)
                self.assertEqual(torch_mod.accelerator.current_device_index(), 0)
                self.assertEqual(
                    torch_mod.cuda.get_device_properties(0).multi_processor_count, 132
                )
                with self.assertRaises(AssertionError):
                    fp8_mod.init_fp8_linear_kernel().apply_weights()
            self.assertIs(fp8_mod.init_fp8_linear_kernel, fp8_original)
            self.assertIs(torch_mod.accelerator.current_device_index, acc_original)
            self.assertIs(torch_mod.cuda.get_device_properties, props_original)

    def test_keeps_existing_engram_config(self):
        from ucm_toolkit.tools.model_check.compute_mocks import dsv41_fp8_layout_mock

        (_, _, engram_calls, _, _, _, fake_import) = self._modules()
        existing = NS(cpu_offload=True)
        config = NS(
            model_config=NS(hf_text_config=NS(model_type="deepseek_v41")),
            engram_config=existing,
        )
        with patch.dict(os.environ, UCM_MODEL_CHECK_DSV41_FP8_MOCK="1"), patch(
            "ucm_toolkit.tools.model_check.compute_mocks.importlib.import_module",
            side_effect=fake_import,
        ):
            with dsv41_fp8_layout_mock(config):
                self.assertIs(config.engram_config, existing)
        self.assertEqual(engram_calls, [])

    def test_default_does_not_import_or_patch(self):
        from ucm_toolkit.tools.model_check.compute_mocks import dsv41_fp8_layout_mock

        with patch.dict(os.environ, UCM_MODEL_CHECK_DSV41_FP8_MOCK="0"), patch(
            "ucm_toolkit.tools.model_check.compute_mocks.importlib.import_module", side_effect=AssertionError
        ):
            with dsv41_fp8_layout_mock(None):
                pass

    def test_rejects_other_models(self):
        from ucm_toolkit.tools.model_check.compute_mocks import dsv41_fp8_layout_mock

        with patch.dict(os.environ, UCM_MODEL_CHECK_DSV41_FP8_MOCK="1"), self.assertRaises(ValueError):
            with dsv41_fp8_layout_mock(NS(model_config=NS(hf_text_config=NS(model_type="deepseek_v4")))):
                pass
