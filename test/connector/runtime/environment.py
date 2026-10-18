"""Software bootstrap and CPU-only device seams; no connector decisions are mocked."""

from __future__ import annotations

import importlib.machinery
import logging
import subprocess
import sys
import time
import types
from contextlib import ExitStack
from importlib import metadata, util
from importlib.machinery import PathFinder
from pathlib import Path
from unittest.mock import patch


def source_version(source, package, expected):
    """Use the actual release tag when setuptools' generated file is absent."""
    source = Path(source).resolve()
    tag = subprocess.check_output(
        ["git", "-C", str(source), "describe", "--tags", "--exact-match"],
        text=True,
    ).strip()
    if tag != "v" + expected:
        raise RuntimeError(f"Expected {package} tag v{expected}, got {tag}")
    status = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    )
    if status.strip():
        raise RuntimeError(f"Engine source has tracked modifications: {source}")
    name = package + "._version"
    generated = source / package / "_version.py"
    if not generated.exists():
        metadata = types.ModuleType(name)
        metadata.__version__ = expected
        metadata.__version_tuple__ = tuple(
            int(x) if x.isdigit() else x for x in expected.split(".")
        )
        sys.modules[name] = metadata
    return expected


def engine_version(package, source, expected):
    if source:
        return source_version(source, package, expected)
    option = "--ascend-source" if package == "vllm_ascend" else "--vllm-source"
    if util.find_spec(package) is None:
        raise RuntimeError(
            f"{package} is not installed; install it or specify {option}"
        )
    try:
        version = metadata.version(package)
    except metadata.PackageNotFoundError as error:
        raise RuntimeError(
            f"{package} has no installed version metadata; install it or specify {option}"
        ) from error
    if version.split("+")[0] != expected:
        raise RuntimeError(f"Requires {package} {expected}, installed {version}")
    return version


def validate_engine(module, source, version):
    if source and not Path(module.__file__).resolve().is_relative_to(Path(source)):
        raise RuntimeError(
            f"{module.__name__} was imported outside its source argument"
        )
    actual = vars(module).get("__version__")
    if actual is not None and actual.split("+")[0] != version.split("+")[0]:
        raise RuntimeError(
            f"{module.__name__} import/version mismatch: {actual}, {version}"
        )


def initialize_engines(args):
    version = engine_version("vllm", args.vllm_source, "0.30.0")
    ascend_version = None
    if args.backend == "ascend":
        ascend_version = engine_version("vllm_ascend", args.ascend_source, "0.30.0rc1")

    if args.backend == "cuda":
        from ..cache_setup.cuda import prepare_cuda_imports

        prepare_cuda_imports(args)

    class SourcePlatformFinder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname != "vllm.platforms":
                return None
            spec = PathFinder.find_spec(fullname, path)
            original = spec.loader

            class Loader:
                def create_module(self, spec):
                    return original.create_module(spec)

                def exec_module(self, module):
                    original.exec_module(module)
                    if args.backend == "cuda":
                        from ..cache_setup.cuda import select_cuda_platform

                        select_cuda_platform(module, args)
                    else:
                        module.builtin_platform_plugins = {
                            "ascend-source": lambda: "vllm_ascend.platform.NPUPlatform"
                        }

            spec.loader = Loader()
            return spec

    finder = SourcePlatformFinder()
    sys.meta_path.insert(0, finder)
    try:
        import vllm
    finally:
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)

    validate_engine(vllm, args.vllm_source, version)
    if args.backend == "cuda":
        from ..cache_setup.cuda import prepare_cuda_compute_imports

        prepare_cuda_compute_imports()
    if args.backend != "ascend":
        return

    import vllm.platforms
    import vllm_ascend

    validate_engine(vllm_ascend, args.ascend_source, ascend_version)
    # Register the real source platform callback explicitly; an uninstalled
    # plugin has no importlib.metadata entry point to discover.
    if vllm.platforms.current_platform.device_type != "npu":
        raise RuntimeError("Ascend bootstrap did not select NPUPlatform")

    # Same public callbacks as vLLM-Ascend's setup.py general entry points.
    for callback in (
        "register_connector",
        "register_model_loader",
        "register_service_profiling",
        "register_model",
    ):
        getattr(vllm_ascend, callback)()

    from ucm.integration.vllm.patch import apply_patch

    # Installed packages retain their real distribution version detector.
    if args.ascend_source:
        apply_patch._read_vllm_ascend_version_raw = lambda: ascend_version
    apply_patch.ENABLE_UCM_PATCH = True
    apply_patch.apply_all_patches()


def capture_engine_modules():
    """Read module dictionaries without triggering lazy module __getattr__."""
    extensions, substitutes = [], []
    for name, module in tuple(sys.modules.items()):
        data = vars(module) if isinstance(module, types.ModuleType) else {}
        if data.get("__ucm_unexecuted_extension__", False):
            substitutes.append(name)
        filename = data.get("__file__")
        if (
            name.startswith(("vllm.", "vllm_ascend."))
            and isinstance(filename, str)
            and filename.endswith(".so")
        ):
            extensions.append(filename)
    return {
        "loaded_engine_extensions": sorted(extensions),
        "unexecuted_extension_imports": sorted(substitutes),
    }


def register_fixture_model():
    """Declare only the synthetic fixture's model interfaces, without compute."""
    import torch
    from vllm.model_executor.models import ModelRegistry

    class UCMConnectorTestModel(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError(
                "The connector contract test must never execute forward"
            )

        def embed_input_ids(self, *args, **kwargs):
            raise AssertionError("No model compute in synthetic GQA fixture")

        def compute_logits(self, *args, **kwargs):
            raise AssertionError(
                "The connector contract test must never compute logits"
            )

    ModelRegistry.register_model("UCMConnectorTestModel", UCMConnectorTestModel)


def prepare_python_ucm():
    """Replace observability dependencies, then forbid all UCM native modules."""
    existing = sys.modules.get("ucm.logger")
    if existing is not None:
        if vars(existing).get("__ucm_test_python__"):
            return
        raise RuntimeError("Mock bootstrap must precede UCM logger/native imports")
    logger_module = types.ModuleType("ucm.logger")

    def init_logger(name):
        logger = logging.getLogger(name)
        for level in ("debug", "info", "warning", "error", "critical"):
            for suffix in ("once", "limit"):
                setattr(logger, level + "_" + suffix, getattr(logger, level))
        return logger

    logger_module.__ucm_test_python__ = True
    logger_module.init_logger = init_logger
    sys.modules["ucm.logger"] = logger_module
    metrics = types.ModuleType("ucm.shared.metrics")
    metrics.__path__ = []
    metrics.ucmmetrics = types.SimpleNamespace(
        update_stats=lambda *a, **k: None,
        get_all_stats_and_clear=lambda: ({}, {}, {}),
        set_up=lambda *a, **k: None,
        create_stats=lambda *a, **k: None,
    )
    sys.modules["ucm.shared.metrics"] = metrics

    class ForbidNative:
        def find_spec(self, fullname, path=None, target=None):
            if not fullname.startswith("ucm."):
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec and isinstance(
                spec.loader, importlib.machinery.ExtensionFileLoader
            ):
                raise ImportError(
                    "Mock test attempted to load a UCM native extension: " + fullname
                )
            if fullname == "ucm.store.pipeline.connector":
                raise ImportError(
                    "Mock test attempted to construct the real pipeline store"
                )
            return None

    sys.meta_path.insert(0, ForbidNative())


def require(condition, message):
    # Do not let python -O turn a validation run into a false PASS.
    if not condition:
        raise AssertionError(message)


def initialize_runtime(args):
    if args.store == "mock":
        prepare_python_ucm()
    initialize_engines(args)
    import torch
    import vllm

    require(vllm.__version__.startswith("0.30.0"), "Requires vLLM 0.30.0")
    torch.set_num_threads(1)

    def empty_uninitialized_accelerator_cache():
        require(
            not torch.cuda.is_initialized(), "Unexpected initialized CUDA allocator"
        )
        if "torch_npu" in sys.modules:
            require(
                not torch.npu.is_initialized(), "Unexpected initialized NPU allocator"
            )
        # Native EngineCore shutdown asks to empty accelerator caches even on
        # CPU-backed tests. There is no accelerator allocation to release.

    torch.accelerator.empty_cache = empty_uninitialized_accelerator_cache
    torch.accelerator.empty_host_cache = empty_uninitialized_accelerator_cache
    register_fixture_model()
    import ucm.integration.vllm.ucm_connector as connector_module

    control_directory = Path(args.output) / "control"
    control_directory.mkdir(exist_ok=True)
    # Preserve real worker/scheduler file exchange, while keeping its stale
    # file GC within this case instead of the machine-wide /dev/shm directory.
    connector_module._SHM_DIR = str(control_directory)
    if args.store == "real":
        import hashlib
        import ucm

        checkout = Path(__file__).resolve().parents[3] / "ucm"
        installed = Path(ucm.__file__).resolve().parent
        for relative in (
            "integration/vllm/ucm_connector.py",
            "integration/vllm/hla_connector.py",
            "integration/vllm/hma_connector.py",
            "integration/vllm/rank_consistency.py",
            "store/factory_v1.py",
        ):
            require(
                hashlib.sha256((checkout / relative).read_bytes()).digest()
                == hashlib.sha256((installed / relative).read_bytes()).digest(),
                f"Real simu UCM Python differs from checkout: {relative}",
            )


def cpu_worker_context(connector_module, rank):
    """Adapt hardware access only; distributed rank groups stay native."""
    import torch

    from ucm.integration.vllm.device import Device

    class CPUDevice(Device):
        def get_event_handle(self):
            return 0

        def synchronize(self):
            pass

        def record_timing_event(self):
            return time.perf_counter()

        def elapsed_time_ms(self, start_event, end_event):
            return (end_event - start_event) * 1000

        def destroy_event_handles(self):
            self.events.clear()

        def destroy_event_handle(self, event_handle):
            self.events.pop(event_handle, None)

        def get_cpu_affinity(self, local_rank):
            return None

    context = ExitStack()
    # A nonnegative device_id selects the real Cache transfer path. UCM must
    # be compiled with RUNTIME_ENVIRONMENT=simu, whose copies use CPU memory.
    context.enter_context(
        patch.object(connector_module, "get_current_device_id", return_value=rank)
    )
    context.enter_context(
        patch.object(
            connector_module,
            "get_ucm_worker_torch_device",
            return_value=torch.device("cpu"),
        )
    )
    context.enter_context(
        patch.object(connector_module, "create_device", side_effect=CPUDevice)
    )
    import ucm.integration.vllm.hla_connector as hla
    import ucm.integration.vllm.hma_connector as hma

    for module in (hla, hma):
        context.enter_context(
            patch.object(module, "create_device", connector_module.create_device)
        )
    return context


def loaded_ucm_extensions():
    return sorted(
        name
        for name, module in tuple(sys.modules.items())
        if name.startswith("ucm.")
        and isinstance(module, types.ModuleType)
        and str(vars(module).get("__file__", "")).endswith(".so")
    )


def close_message_queue(mq):
    """Native queue exposes shutdown; release its owned SHM before gc.freeze.

    MessageQueue.shutdown wakes readers but leaves buffers alive until GC.
    EngineCore freezes the heap, so explicitly drop that buffer reference.
    """
    mq.shutdown()
    mq.buffer = None
    for name in ("local_socket", "remote_socket"):
        socket = getattr(mq, name, None)
        if socket is not None:
            socket.close(linger=0)
    condition = mq._spin_condition
    if condition is not None:
        for name in (
            "local_notify_socket",
            "write_cancel_socket",
            "read_cancel_socket",
        ):
            socket = getattr(condition, name, None)
            if socket is not None:
                socket.close(linger=0)


def observe_owned_shared_memory(directory):
    """Record only native SHM created by this process for hard-timeout cleanup."""
    import json
    import os
    from multiprocessing import shared_memory

    original = shared_memory.SharedMemory.__init__

    def initialize(memory, *args, **kwargs):
        original(memory, *args, **kwargs)
        create = kwargs.get("create", args[1] if len(args) > 1 else False)
        if create:
            path = Path(directory) / f"owned-shm-{os.getpid()}.jsonl"
            with path.open("a") as stream:
                stream.write(
                    json.dumps({"name": memory.name, "pid": os.getpid()}) + "\n"
                )

    return patch.object(shared_memory.SharedMemory, "__init__", initialize)
