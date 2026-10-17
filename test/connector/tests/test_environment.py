"""Source identity bootstrap must not invent a release or modify checkouts."""

import subprocess
import sys
import types
import unittest
from importlib import metadata
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from connector.cache_setup.config import resolve_native_length
from connector.runtime.environment import (
    capture_engine_modules,
    engine_version,
    source_version,
    validate_engine,
)


class SourceIdentityTest(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "vllm").mkdir()
        (self.root / "vllm/__init__.py").write_text("# fixture\n")
        self.git("init", "-q")
        self.git("add", ".")
        self.git(
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
        )
        self.git("tag", "v0.30.0")

    def git(self, *args):
        return subprocess.run(
            ["git", "-C", str(self.root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    def test_release_metadata_is_derived_without_writing_source(self):
        with patch.dict(sys.modules):
            self.assertEqual(source_version(self.root, "vllm", "0.30.0"), "0.30.0")
            self.assertEqual(sys.modules["vllm._version"].__version__, "0.30.0")
        self.assertFalse((self.root / "vllm/_version.py").exists())
        self.assertEqual(self.git("status", "--porcelain"), "")

    def test_different_release_is_rejected(self):
        with self.assertRaises(RuntimeError):
            source_version(self.root, "vllm", "0.29.0")

    def test_module_capture_does_not_trigger_lazy_imports(self):
        lazy = types.ModuleType("vllm._test_lazy")

        def lazy_getattr(name):
            sys.modules["_unexpected_lazy_import"] = types.ModuleType(
                "_unexpected_lazy_import"
            )
            raise AssertionError("Module metadata must not invoke lazy imports")

        lazy.__getattr__ = lazy_getattr
        sentinel = types.ModuleType("vllm._test_sentinel")
        sentinel.__ucm_unexecuted_extension__ = True
        with patch.dict(
            sys.modules, {"vllm._test_lazy": lazy, "vllm._test_sentinel": sentinel}
        ):
            captured = capture_engine_modules()
            self.assertIn(
                "vllm._test_sentinel", captured["unexecuted_extension_imports"]
            )
            self.assertNotIn("_unexpected_lazy_import", sys.modules)

    def test_dirty_source_is_rejected(self):
        (self.root / "vllm/__init__.py").write_text("# modified\n")
        with self.assertRaises(RuntimeError):
            source_version(self.root, "vllm", "0.30.0")


class InstalledEngineTest(unittest.TestCase):
    def test_installed_release_uses_real_metadata_without_git_or_version_stub(self):
        with (
            patch(
                "connector.runtime.environment.util.find_spec", return_value=object()
            ),
            patch(
                "connector.runtime.environment.metadata.version",
                return_value="0.30.0+empty",
            ),
            patch("subprocess.check_output") as git,
            patch.dict(sys.modules),
        ):
            before = sys.modules.get("vllm._version")
            self.assertEqual(engine_version("vllm", None, "0.30.0"), "0.30.0+empty")
            git.assert_not_called()
            self.assertIs(sys.modules.get("vllm._version"), before)

    def test_missing_installed_package_reports_source_option(self):
        for package, option in (
            ("vllm", "--vllm-source"),
            ("vllm_ascend", "--ascend-source"),
        ):
            with patch(
                "connector.runtime.environment.util.find_spec", return_value=None
            ):
                with self.assertRaisesRegex(RuntimeError, option):
                    engine_version(package, None, "0.30.0")

    def test_installed_wrong_or_missing_release_is_rejected(self):
        for version in ("0.29.0", "0.30.0rc1", "0.30.01"):
            with (
                patch(
                    "connector.runtime.environment.util.find_spec",
                    return_value=object(),
                ),
                patch(
                    "connector.runtime.environment.metadata.version",
                    return_value=version,
                ),
                self.assertRaisesRegex(RuntimeError, "installed"),
            ):
                engine_version("vllm", None, "0.30.0")
        with (
            patch(
                "connector.runtime.environment.util.find_spec", return_value=object()
            ),
            patch(
                "connector.runtime.environment.metadata.version",
                side_effect=metadata.PackageNotFoundError,
            ),
            self.assertRaisesRegex(RuntimeError, "no installed version metadata"),
        ):
            engine_version("vllm", None, "0.30.0")

    def test_explicit_source_does_not_fall_back_to_installed_package(self):
        with (
            patch(
                "connector.runtime.environment.source_version",
                side_effect=RuntimeError("bad source"),
            ),
            patch("connector.runtime.environment.metadata.version") as installed,
            self.assertRaisesRegex(RuntimeError, "bad source"),
        ):
            engine_version("vllm", "/explicit/source", "0.30.0")
        installed.assert_not_called()

    def test_import_identity_keeps_installed_paths_and_checks_explicit_sources(self):
        module = types.ModuleType("vllm")
        module.__file__ = "/installed/vllm/__init__.py"
        module.__version__ = "0.30.0+empty"
        validate_engine(module, None, "0.30.0+empty")
        with self.assertRaisesRegex(RuntimeError, "outside"):
            validate_engine(module, "/explicit/source", "0.30.0")
        with self.assertRaisesRegex(RuntimeError, "mismatch"):
            validate_engine(module, None, "0.29.0")

    def test_installed_cuda_does_not_substitute_compiled_extensions(self):
        from connector.cache_setup.cuda import prepare_cuda_imports

        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(_lazy_init=lambda: None)
        with patch.dict(sys.modules, {"torch": torch}):
            before = set(sys.modules)
            prepare_cuda_imports(types.SimpleNamespace(vllm_source=None))
            self.assertEqual(set(sys.modules), before)
            with self.assertRaisesRegex(AssertionError, "initialize CUDA"):
                torch.cuda._lazy_init()


class ModelConfigTest(unittest.TestCase):
    def test_default_request_length_follows_native_alignment(self):
        args = SimpleNamespace(tokens=1024, tokens_explicit=False)
        config = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=4352),
            model_config=SimpleNamespace(),
            scheduler_config=SimpleNamespace(),
        )
        resolve_native_length(args, config)
        self.assertEqual(args.tokens, 13056)
        self.assertEqual(config.model_config.max_model_len, 17408)
        self.assertEqual(config.scheduler_config.max_num_batched_tokens, 17408)
        args.tokens, args.tokens_explicit = 2048, True
        resolve_native_length(args, config)
        self.assertEqual(args.tokens, 2048)


if __name__ == "__main__":
    unittest.main()
