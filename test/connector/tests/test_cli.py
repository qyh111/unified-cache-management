"""The standalone test launcher must work without importing vLLM."""

import contextlib
import io
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from connector.run import main, parse_args


class ConnectorCLITest(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.source = Path(self.directory.name)
        for package in ("vllm", "vllm_ascend"):
            (self.source / package).mkdir()
            (self.source / package / "__init__.py").touch()
        self.source_args = ["--backend", "npu", "--vllm-source", str(self.source)]

    def test_help_does_not_load_vllm(self):
        before = set(sys.modules)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as raised:
                main(["--help"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("--tp", output.getvalue())
        self.assertIn("--backend", output.getvalue())
        self.assertIn("--device", output.getvalue())
        self.assertFalse(
            any(name.startswith("vllm") for name in set(sys.modules) - before)
        )

    def test_rejects_unsupported_topologies_and_invalid_timeout(self):
        for arguments in (
            ["--tp", "4"],
            ["--timeout", "0"],
        ):
            with self.subTest(arguments=arguments):
                with (
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit),
                ):
                    parse_args(
                        [*self.source_args, "--output", "/tmp/example", *arguments]
                    )

    def test_cpu_allocation_keeps_npu_engine_backend(self):
        with (
            patch.dict("os.environ", {"VLLM_TARGET_DEVICE": "cpu"}),
            patch("subprocess.call", return_value=0) as launch,
        ):
            result = main(
                [*self.source_args, "--device", "cpu", "--output", "/tmp/example"]
            )
        self.assertEqual(result, 0)
        self.assertEqual(
            launch.call_args.args[0][:3], [sys.executable, "-m", "connector.scenario"]
        )
        env = launch.call_args.kwargs["env"]
        self.assertEqual(env["VLLM_PLUGINS"], "")
        self.assertEqual(env["ENABLE_UCM_PATCH"], "0")
        self.assertNotIn("VLLM_TARGET_DEVICE", env)
        command = launch.call_args.args[0]
        self.assertEqual(command[command.index("--backend") + 1], "npu")
        self.assertEqual(command[command.index("--device") + 1], "cpu")

    def test_ascend_keeps_normal_patch_path(self):
        with (
            patch.dict("os.environ", {"VLLM_PLUGINS": ""}),
            patch("subprocess.call", return_value=0) as launch,
        ):
            main(
                [
                    *self.source_args,
                    "--ascend-source",
                    str(self.source),
                    "--backend",
                    "npu",
                    "--output",
                    "/tmp/example",
                ]
            )
        self.assertEqual(launch.call_args.kwargs["env"]["ENABLE_UCM_PATCH"], "0")
        self.assertEqual(launch.call_args.kwargs["env"]["VLLM_PLUGINS"], "")
        self.assertIn(
            str(Path(__file__).resolve().parents[3]),
            launch.call_args.kwargs["env"]["PYTHONPATH"].split(":"),
        )

    def test_store_selection_controls_checkout_import(self):
        for store in ("mock", "real"):
            with (
                patch.dict("os.environ", {"PYTHONPATH": ""}),
                patch("subprocess.call", return_value=0) as launch,
            ):
                main(["--backend", "npu", "--store", store, "--output", "/tmp/example"])
            command = launch.call_args.args[0]
            self.assertEqual(command[command.index("--store") + 1], store)
            paths = launch.call_args.kwargs["env"]["PYTHONPATH"].split(":")
            self.assertEqual(
                str(Path(__file__).resolve().parents[3]) in paths, store == "mock"
            )
        self.assertEqual(
            parse_args(["--backend", "npu", "--output", "/tmp/a"]).store, "mock"
        )

    def test_installed_packages_need_no_source_arguments(self):
        for backend in ("npu",):
            with patch("subprocess.call", return_value=0) as launch:
                main(["--backend", backend, "--output", "/tmp/example"])
            command = launch.call_args.args[0]
            self.assertNotIn("--vllm-source", command)
            self.assertNotIn("--ascend-source", command)
            self.assertNotIn(None, command)
            self.assertEqual(parse_args(command[3:]).backend, backend)

    def test_source_selection_is_independent_for_each_package(self):
        for source_args in (
            self.source_args,
            ["--ascend-source", str(self.source)],
        ):
            with patch("subprocess.call", return_value=0) as launch:
                main([*source_args, "--backend", "npu", "--output", "/tmp/example"])
            command = launch.call_args.args[0]
            self.assertEqual("--vllm-source" in command, "--vllm-source" in source_args)
            self.assertEqual(
                "--ascend-source" in command, "--ascend-source" in source_args
            )

    def test_rejects_invalid_explicit_sources(self):
        for arguments in (
            ["--vllm-source", str(self.source / "missing")],
            ["--ascend-source", str(self.source)],
        ):
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                parse_args(
                    ["--backend", "cuda", *arguments, "--output", "/tmp/example"]
                )

    def test_model_config_file_is_normalized_and_forwarded(self):
        model = self.source / "config.json"
        model.write_text('{"architectures": ["example"]}')
        with patch("subprocess.call", return_value=0) as launch:
            main(
                [
                    *self.source_args,
                    "--ascend-source",
                    str(self.source),
                    "--backend",
                    "npu",
                    "--model-config",
                    str(model),
                    "--tokens",
                    "1024",
                    "--output",
                    "/tmp/example",
                ]
            )
        command = launch.call_args.args[0]
        self.assertEqual(
            command[command.index("--model-config") + 1], str(self.source.resolve())
        )
        self.assertEqual(command[command.index("--tokens") + 1], "1024")

    def test_cuda_requires_explicit_target_and_forwards_it(self):
        (self.source / "config.json").write_text('{"architectures": ["example"]}')
        base = [
            *self.source_args[2:],
            "--backend",
            "cuda",
            "--model-config",
            str(self.source),
            "--output",
            "/tmp/example",
        ]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(base)
        with patch("subprocess.call", return_value=0) as launch:
            main(
                [
                    *base,
                    "--cuda-capability",
                    "10.0",
                    "--attention-backend",
                    "FLASHMLA_SPARSE",
                ]
            )
        command = launch.call_args.args[0]
        self.assertEqual(command[command.index("--cuda-capability") + 1], "10.0")
        self.assertEqual(
            command[command.index("--attention-backend") + 1], "FLASHMLA_SPARSE"
        )
        env = launch.call_args.kwargs["env"]
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        self.assertNotIn("VLLM_TARGET_DEVICE", env)

    def test_cuda_rejects_ambiguous_profiles(self):
        (self.source / "config.json").write_text("{}")
        for capability in ("100", "sm100", "10.x", "7.0", "10.10"):
            with self.subTest(capability=capability):
                with (
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit),
                ):
                    parse_args(
                        [
                            *self.source_args[2:],
                            "--backend",
                            "cuda",
                            "--model-config",
                            str(self.source),
                            "--cuda-capability",
                            capability,
                            "--attention-backend",
                            "FLASHMLA_SPARSE",
                            "--output",
                            "/tmp/example",
                        ]
                    )

    def test_rejects_cpu_backend_and_unimplemented_device_io(self):
        for arguments in (
            ["--backend", "cpu", "--device", "cpu"],
            ["--backend", "npu", "--device", "npu"],
            ["--backend", "npu", "--device", "cuda"],
        ):
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                parse_args([*arguments, "--output", "/tmp/example"])

    def test_rejects_model_path_without_config_and_tokens_without_model(self):
        for arguments in (["--model-config", str(self.source)], ["--tokens", "512"]):
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                parse_args([*self.source_args, "--output", "/tmp/example", *arguments])


if __name__ == "__main__":
    unittest.main()
