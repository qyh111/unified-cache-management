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
