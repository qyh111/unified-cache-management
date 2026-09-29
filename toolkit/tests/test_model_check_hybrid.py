"""Hybrid model-check dispatch and byte-oracle tests without engine imports."""

import ast
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ucm_toolkit.tools.model_check import hybrid
from ucm_toolkit.tools.model_check.adapter import ModelCheckTool
from ucm_toolkit.tools.model_check.config import (
    HYBRID_ENV,
    BUFFER_GB_ENV,
    EXCLUSIVE_ENV,
    _bool_env,
    _int_env,
)
from ucm_toolkit.errors import ToolkitError


def helper(name):
    # Load only the function under test: common.py normally requires torch.
    tree = ast.parse((ROOT / "ucm_toolkit/tools/model_check/common.py").read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    ns = dict(
        Any=object,
        CacheFixture=object,
        SimpleNamespace=NS,
        is_hybrid_worker=hybrid.is_hybrid_worker,
        HYBRID_ENV=HYBRID_ENV,
        BUFFER_GB_ENV=BUFFER_GB_ENV,
        EXCLUSIVE_ENV=EXCLUSIVE_ENV,
        _bool_env=_bool_env,
        _int_env=_int_env,
    )
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), "<common helper>", "exec"), ns
    )
    return ns[name]


class Offsets:
    def __getitem__(self, pair):
        row, col = pair
        return row * 4096 + (0, 4, 7)[col]


class LayoutDiagnosticsTest(unittest.TestCase):
    def test_ascend_multi_group_does_not_import_single_group_shim(self):
        tree = ast.parse((ROOT / "ucm_toolkit/tools/model_check/ascend.py").read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "patch_groups")
        ns = {"CacheFixture": object}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<ascend shim>", "exec"), ns)
        with patch("builtins.__import__", side_effect=AssertionError("unexpected legacy import")):
            ns["patch_groups"](NS(kv_cache_config=NS(kv_cache_groups=[object()] * 5)))

    def test_new_routes_do_not_probe_legacy_shared_tensor_abi(self):
        path = ROOT.parent / "ucm/integration/vllm/ucm_connector.py"
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        assignment = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "use_hybrid_linear_attention"
                    for t in n.targets)
        )
        probe_calls = []
        def probe(config):
            probe_calls.append(config)
            return True
        for fawa, hybrid_selected, expected_calls in (
            (True, True, 0), (True, False, 0), (False, True, 0), (False, False, 1)
        ):
            probe_calls.clear()
            ns = dict(use_fawa=fawa, use_hybrid=hybrid_selected,
                      kv_cache_config=object(),
                      UCMHybridLinearAttentionConnector=NS(supports_kv_cache_layout=probe))
            exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(path), "exec"), ns)
            self.assertEqual(len(probe_calls), expected_calls)
            self.assertEqual(ns["use_hybrid_linear_attention"], bool(expected_calls))

    def test_optional_storage_block_size_does_not_stop_model_check(self):
        fn = helper("log_cache_layout")
        messages = []
        fn.__globals__.update(
            log=messages.append,
            _runtime_tensor_signature=lambda value: ("cpu", "uint8"),
        )
        for physical_size in (None, 64):
            spec = NS(block_size=256, storage_block_size=physical_size)
            fixture = NS(
                kv_cache_config=NS(
                    num_blocks=3,
                    kv_cache_tensors=[NS(layers=["layer"], block_stride=1024, size=3072)],
                    kv_cache_groups=[NS(kv_cache_spec=spec, layer_names=["layer"])],
                ),
                kv_caches={"layer": object()},
            )
            fn(fixture)
            self.assertIn(f"256, {physical_size},", messages[-1])


class Layout:
    row_count = 2
    sizes = (4, 3, 2)
    ucm_block_offsets = Offsets()

    def resolve(self, plan, row):
        # Key 2 belongs to a distinct request/destination, even if keys match.
        ptrs = [
            [plan.base + i * 100 + row * 20, 0, plan.base + i * 100 + row * 20 + 10]
            for i in range(len(plan.keys))
        ]
        return NS(keys=plan.keys, ptrs=ptrs)


class UCMHybridConnector:
    __module__ = "ucm.integration.vllm.hybrid_connector"

    def __init__(self):
        self.kv_cache_layout = Layout()


class Memory:
    def __init__(self, data):
        self.data = data

    def synchronize(self):
        pass

    def write(self, ptr, data):
        self.data[ptr] = data

    def read(self, ptr, size):
        return self.data.get(ptr, b"")[:size]


class HybridModelCheckTests(unittest.TestCase):
    def setUp(self):
        self.worker = NS(connector=UCMHybridConnector())
        self.meta = NS(
            requests={
                "one": NS(
                    dump_plans=(NS(keys=(b"a", b"b"), base=1000),),
                    load_plans=(NS(keys=(b"a", b"b"), base=2000),),
                ),
                "two": NS(dump_plans=(), load_plans=(NS(keys=(b"a",), base=3000),)),
            }
        )

    def test_slots_offsets_ghost_and_duplicate_destinations(self):
        items = list(hybrid.segments(self.worker, self.meta, "load"))
        self.assertEqual(len(items), 12)
        self.assertEqual(items[0], (b"a", 0, 2000, 4))
        self.assertEqual(items[1], (b"a", 7, 2010, 2))
        self.assertEqual(items[4], (b"a", 4096, 2020, 4))
        self.assertEqual(items[8], (b"a", 0, 3000, 4))
        self.assertTrue(all(ptr for _, _, ptr, _ in items))

    def test_poison_detects_skipped_and_corrupt_load(self):
        fixture = NS(kv_caches={})
        with patch.object(hybrid, "TensorByteAccess", Memory):
            self.assertEqual(hybrid.fill_hybrid(fixture, self.worker, self.meta), 8)
            expected = {
                (k, off): fixture.kv_caches[ptr]
                for k, off, ptr, _ in hybrid.segments(self.worker, self.meta, "dump")
            }
            self.assertEqual(hybrid.poison_hybrid(fixture, self.worker, self.meta), 12)
            with self.assertRaises(AssertionError):
                hybrid.compare_hybrid(fixture, self.worker, self.meta)
            for key, offset, ptr, _ in hybrid.segments(self.worker, self.meta, "load"):
                fixture.kv_caches[ptr] = expected[key, offset]
            self.assertEqual(hybrid.compare_hybrid(fixture, self.worker, self.meta), 12)
            fixture.kv_caches[3000] = b"bad!"
            with self.assertRaisesRegex(AssertionError, "Hybrid byte mismatch"):
                hybrid.compare_hybrid(fixture, self.worker, self.meta)

    def test_empty_selection_fails(self):
        with patch.object(hybrid, "TensorByteAccess", Memory):
            with self.assertRaisesRegex(AssertionError, "no real payload"):
                hybrid.fill_hybrid(NS(kv_caches={}), self.worker, NS(requests={}))

    def test_rank_and_offset_change_pattern(self):
        with patch.dict(os.environ, {"RANK": "0"}):
            first = hybrid.payload(b"a", 0, 64)
            self.assertNotEqual(first, hybrid.payload(b"a", 4096, 64))
        with patch.dict(os.environ, {"RANK": "1"}):
            self.assertNotEqual(first, hybrid.payload(b"a", 0, 64))

    def test_hybrid_not_misclassified_as_v2_and_legacy_still_skips(self):
        batch = helper("_v2_batch")
        self.assertIsNone(batch(self.worker, self.meta, "dump"))
        self.assertIsNone(batch(NS(), NS(request_meta={}), "dump"))
        expected = object()
        v2 = NS(
            layout=NS(build_dump_batches=lambda meta: expected), use_layerwise=False
        )
        self.assertIs(batch(v2, self.meta, "dump"), expected)
        with self.assertRaises(ValueError):
            batch(NS(), self.meta, "dump")

    def test_config_default_legacy_16_and_hybrid_override(self):
        config = helper("make_ucm_config")
        with patch.dict(os.environ, {}, clear=True):
            result = config("Cache|Posix", "/cache", True)
            self.assertFalse(result["use_hybrid_connector"])
            self.assertEqual(
                result["ucm_connectors"][0]["ucm_connector_config"][
                    "cache_buffer_capacity_gb"
                ],
                16,
            )
        with patch.dict(os.environ, {HYBRID_ENV: "true", BUFFER_GB_ENV: "24"}):
            result = config("Cache|Posix", "/cache", False)
            self.assertTrue(result["use_hybrid_connector"])
            self.assertFalse(result["use_layerwise"])
            self.assertEqual(
                result["ucm_connectors"][0]["ucm_connector_config"][
                    "cache_buffer_capacity_gb"
                ],
                24,
            )
        with patch.dict(os.environ, {BUFFER_GB_ENV: "0"}):
            with self.assertRaises(ValueError):
                config("Cache|Posix", "/cache", True)

    def test_cli_hybrid_capacity_propagated(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "ucm_toolkit.tools.model_check.adapter._detect_platform",
                return_value="ascend",
            ),
            patch(
                "ucm_toolkit.tools.model_check.adapter.run_command", return_value=0
            ) as run,
        ):
            self.assertEqual(
                ModelCheckTool().run(["--hybrid", "--cache-buffer-capacity-gb", "16"]),
                0,
            )
            env = run.call_args.kwargs["env"]
            self.assertEqual(env[HYBRID_ENV], "true")
            self.assertEqual(env[BUFFER_GB_ENV], "16")
            self.assertEqual(
                env["UCM_MODEL_CHECK_CONNECTOR_MODULE_PATH"],
                "ucm.integration.vllm.ucm_connector",
            )

    def test_exclusive_config_and_cli_override(self):
        config = helper("make_ucm_config")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                config("Cache|Posix", "/cache", True)["ucm_connectors"][0][
                    "ucm_connector_config"
                ]["cache_load_exclusive_buffer_number"],
                512,
            )
        with (
            patch.dict(os.environ, {EXCLUSIVE_ENV: "256"}, clear=True),
            patch(
                "ucm_toolkit.tools.model_check.adapter._detect_platform",
                return_value="cpu",
            ),
            patch(
                "ucm_toolkit.tools.model_check.adapter.run_command", return_value=0
            ) as run,
        ):
            self.assertEqual(
                ModelCheckTool().run(["--cache-load-exclusive-buffer-number", "128"]), 0
            )
            self.assertEqual(run.call_args.kwargs["env"][EXCLUSIVE_ENV], "128")
            self.assertEqual(
                config("Cache|Posix", "/cache", True)["ucm_connectors"][0][
                    "ucm_connector_config"
                ]["cache_load_exclusive_buffer_number"],
                256,
            )
        with patch.dict(os.environ, {EXCLUSIVE_ENV: "0"}, clear=True):
            with self.assertRaises(ValueError):
                config("Cache|Posix", "/cache", True)
            with self.assertRaises(ToolkitError):
                ModelCheckTool().run([])

    def test_rank_configuration_matches_process(self):
        from ucm_toolkit.tools.model_check.config import configure_worker_rank

        config = NS(parallel_config=NS(rank=0))
        with patch.dict(os.environ, {"RANK": "3"}):
            configure_worker_rank(config)
        self.assertEqual(config.parallel_config.rank, 3)
        with patch.dict(os.environ, {"RANK": "-1"}):
            with self.assertRaises(ValueError):
                configure_worker_rank(config)

    def test_cpu_constraint_uses_native_type_and_fails_if_missing(self):
        from types import ModuleType
        from ucm_toolkit.tools.model_check.config import cpu_gqa_block_sizes

        module = ModuleType("vllm.v1.attention.backend")
        module.MultipleOf = lambda n: ("multiple", n)
        with patch.dict(sys.modules, {module.__name__: module}):
            self.assertEqual(cpu_gqa_block_sizes(), [("multiple", 16)])
        del module.MultipleOf
        with patch.dict(sys.modules, {module.__name__: module}):
            with self.assertRaisesRegex(RuntimeError, "native MultipleOf"):
                cpu_gqa_block_sizes()

    def test_hybrid_invalid_combination_never_launches(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("ucm_toolkit.tools.model_check.adapter.run_command") as run,
        ):
            for args in (
                ["--hybrid", "--pp", "2"],
                ["--hybrid", "--pcp", "2"],
                ["--hybrid", "--tp", "2", "--dcp", "2"],
                [
                    "--hybrid",
                    "--connector-module-path",
                    "ucm.integration.vllm.v2.ucm_connector",
                ],
                ["--hybrid", "--cache-buffer-capacity-gb", "0"],
            ):
                with self.assertRaises(ToolkitError):
                    ModelCheckTool().run(args)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
