"""CPU transfer simulation contract, without importing torch or vLLM."""

import ast
import os
import time
import unittest
from abc import ABC, abstractmethod
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import patch


class CpuSimulationTest(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[2] / "ucm/integration/vllm/device.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {
            "Device",
            "CpuDevice",
            "cpu_simulation_enabled",
            "create_device",
            "get_current_device_id",
        }
        nodes = [
            n
            for n in tree.body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
        ]
        self.platform = NS(device_type="cpu", is_cuda_alike=lambda: False)
        self.ns = dict(
            os=os,
            time=time,
            ABC=ABC,
            abstractmethod=abstractmethod,
            Any=Any,
            Optional=Optional,
            Tuple=Tuple,
            List=List,
            Dict=Dict,
            current_platform=self.platform,
            torch=NS(
                cuda=NS(current_device=lambda: 2), npu=NS(current_device=lambda: 3)
            ),
            CudaDevice=lambda: "cuda",
            NpuDevice=lambda: "npu",
        )
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), self.ns
        )

    def test_cpu_requires_explicit_opt_in(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(self.ns["create_device"]())
            with self.assertRaises(RuntimeError):
                self.ns["get_current_device_id"]()

    def test_cpu_worker_id_and_sync_contract(self):
        with patch.dict(os.environ, {"UCM_CPU_SIMULATION": "1", "LOCAL_RANK": "2"}):
            device = self.ns["create_device"]()
            self.assertIsInstance(device, self.ns["CpuDevice"])
            self.assertEqual(self.ns["get_current_device_id"](), 2)
            self.assertEqual(device.get_event_handle(), 0)
            device.synchronize()
            device.destroy_event_handle(0)
            device.destroy_event_handles()
            self.assertEqual(device.split_cores(2), ([], []))
            self.assertAlmostEqual(device.elapsed_time_ms(1.0, 1.25), 250)
            self.assertIsInstance(device.record_timing_event(), float)

    def test_negative_cpu_rank_rejected(self):
        with patch.dict(os.environ, {"UCM_CPU_SIMULATION": "1", "LOCAL_RANK": "-1"}):
            with self.assertRaises(ValueError):
                self.ns["get_current_device_id"]()

    def test_accelerator_selection_unchanged_even_with_cpu_flag(self):
        with patch.dict(os.environ, {"UCM_CPU_SIMULATION": "1"}):
            self.platform.device_type = "npu"
            self.assertEqual(self.ns["create_device"](), "npu")
            self.assertEqual(self.ns["get_current_device_id"](), 3)
            self.platform.device_type = "cuda"
            self.platform.is_cuda_alike = lambda: True
            self.assertEqual(self.ns["create_device"](), "cuda")
            self.assertEqual(self.ns["get_current_device_id"](), 2)

    def test_cpu_package_dispatch_without_accelerator_visibility(self):
        from ucm_toolkit.tools.model_check.adapter import ModelCheckTool

        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "ucm_toolkit.tools.model_check.adapter.importlib.util.find_spec",
                side_effect=lambda name: object() if name == "vllm" else None,
            ),
            patch("importlib.metadata.version", return_value="0.29.0+cpu"),
            patch(
                "ucm_toolkit.tools.model_check.adapter.run_command", return_value=0
            ) as run,
        ):
            self.assertEqual(ModelCheckTool().run(["--hybrid"]), 0)
            self.assertEqual(
                run.call_args.args[0][-1], "ucm_toolkit.tools.model_check.cpu"
            )
            env = run.call_args.kwargs["env"]
            self.assertNotIn("CUDA_VISIBLE_DEVICES", env)
            self.assertNotIn("ASCEND_RT_VISIBLE_DEVICES", env)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    unittest.main()
