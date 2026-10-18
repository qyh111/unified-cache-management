"""pytest boundary. Collection and --help need neither torch nor vLLM."""

import json
import os
from pathlib import Path
import signal
import shutil
import tempfile
import subprocess
import sys
import uuid
from types import SimpleNamespace

import pytest


def cleanup_case_resources(root, ipc_directory):
    from multiprocessing import shared_memory

    removed, absent = [], []
    names = {
        json.loads(line)["name"]
        for path in Path(root).rglob("owned-shm-*.jsonl")
        for line in path.read_text().splitlines()
        if line
    }
    for name in sorted(names):
        try:
            memory = shared_memory.SharedMemory(name=name)
        except FileNotFoundError:
            absent.append(name)
            continue
        memory.close()
        try:
            memory.unlink()
        except FileNotFoundError:
            pass
        removed.append(name)
    shutil.rmtree(ipc_directory, ignore_errors=False)
    result = {
        "removed_shm": removed,
        "already_absent_shm": absent,
        "ipc_directory_removed": not Path(ipc_directory).exists(),
    }
    (Path(root) / "cleanup.json").write_text(json.dumps(result, indent=2))
    return result


def pytest_addoption(parser):
    group = parser.getgroup("connector")
    for name, choices, default in (
        ("backend", ("cuda", "ascend"), None),
        ("device", ("cpu", "cuda", "npu"), "cpu"),
        ("store", ("mock", "real"), "mock"),
        ("runner", ("v1", "v2"), "v1"),
        ("mode", ("bulk", "layerwise"), "bulk"),
    ):
        group.addoption("--" + name, choices=choices, default=default)
    for name, default in (("tp", 2), ("pp", 1), ("tokens", None)):
        group.addoption("--" + name, type=int, default=default)
    for name in (
        "model-config",
        "vllm-source",
        "ascend-source",
        "cuda-capability",
        "attention-backend",
        "output",
    ):
        group.addoption("--" + name)
    group.addoption("--worker-timeout", type=float, default=120)
    group.addoption("--case-timeout", type=float, default=600)
    group.addoption(
        "--kv-budget-mib",
        type=int,
        default=4096,
        help="Bounded per-rank memory budget replacing profiling",
    )


@pytest.fixture
def engine_case(request, tmp_path):
    config = request.config
    names = (
        "backend",
        "device",
        "store",
        "runner",
        "mode",
        "tp",
        "pp",
        "tokens",
        "model_config",
        "vllm_source",
        "ascend_source",
        "cuda_capability",
        "attention_backend",
        "kv_budget_mib",
    )
    options = {name: config.getoption(name) for name in names}
    if options["backend"] is None:
        pytest.fail(
            "Execution requires --backend cuda|ascend (collection needs no backend)"
        )
    if options["device"] != "cpu" or options["runner"] != "v1" or options["pp"] != 1:
        pytest.fail(
            "This core implementation requires --device cpu --runner v1 --pp 1; extensions are not implemented"
        )
    if options["tp"] not in (1, 2):
        pytest.fail("Core TP must be 1 or 2")
    if options["backend"] == "cuda" and not all(
        options[x] for x in ("cuda_capability", "attention_backend")
    ):
        pytest.fail("CUDA requires explicit --cuda-capability and --attention-backend")
    if options["ascend_source"] and options["backend"] != "ascend":
        pytest.fail("--ascend-source requires --backend ascend")
    if options["tokens"] is not None and options["tokens"] <= 0:
        pytest.fail("--tokens must be positive")
    if options["kv_budget_mib"] <= 0:
        pytest.fail("--kv-budget-mib must be positive")
    for name in ("vllm_source", "ascend_source", "model_config"):
        if options[name]:
            path = Path(options[name]).resolve()
            if name == "model_config" and path.name == "config.json":
                path = path.parent
            options[name] = str(path)
    options.update(
        timeout=config.getoption("worker_timeout"),
        tokens_explicit=options["tokens"] is not None,
    )
    timeout = config.getoption("case_timeout")
    if timeout <= 0 or options["timeout"] <= 0:
        pytest.fail("Timeouts must be positive")
    root = Path(config.getoption("output") or tmp_path) / request.node.name
    root.mkdir(parents=True, exist_ok=False)
    options["output"] = str(root.resolve())
    options["resolved_engine_id"] = "pytest-" + uuid.uuid4().hex
    config_path = root / "options.json"
    config_path.write_text(json.dumps(options, indent=2))
    env = os.environ.copy()
    repo = Path(__file__).resolve().parents[2]
    paths = [str(repo / "test")]
    if options["store"] == "mock":
        paths.append(str(repo))
    paths += [
        options[name] for name in ("vllm_source", "ascend_source") if options[name]
    ]
    paths += [env.get("PYTHONPATH", "")]
    env.update(
        PYTHONPATH=os.pathsep.join(paths),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        ENABLE_UCM_PATCH="0",
        UCM_CAPTURE_VLLM_LOG="0",
        VLLM_PLUGINS="",
        VLLM_CPU_AFFINITY="0",
        OMP_NUM_THREADS="1",
        VLLM_USE_V2_MODEL_RUNNER="0",
        CUDA_VISIBLE_DEVICES="",
        ASCEND_RT_VISIBLE_DEVICES="",
    )
    ipc_directory = tempfile.mkdtemp(prefix="ucm-pytest-")
    env["VLLM_RPC_BASE_PATH"] = ipc_directory
    env.pop("VLLM_TARGET_DEVICE", None)
    active = []

    def run(scenario, fault=None, timeout_override=None):
        command = [
            sys.executable,
            "-m",
            "connector.runtime.session",
            str(config_path),
            scenario,
        ]
        if fault:
            command.append(fault)
        (root / "command.json").write_text(
            json.dumps(
                {
                    "argv": command,
                    "case_timeout_seconds": timeout_override or timeout,
                    "environment": {
                        k: env[k]
                        for k in env
                        if k.startswith(
                            (
                                "VLLM",
                                "UCM",
                                "ENABLE_UCM",
                                "PYTHONPATH",
                                "CUDA_VISIBLE",
                                "ASCEND_RT",
                            )
                        )
                    },
                },
                indent=2,
            )
        )
        with (root / "driver.log").open("w") as log:
            process = subprocess.Popen(
                command,
                env=env,
                cwd=repo / "test",
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            active.append(process)
            try:
                code = process.wait(timeout=timeout_override or timeout)
            except subprocess.TimeoutExpired:
                pytest.fail(
                    f"Driver timed out after {timeout_override or timeout}s; see {root / 'driver.log'}"
                )
            finally:
                # Includes workers/Manager on driver exception or hard timeout.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                finally:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=10)
                    cleanup_case_resources(root, ipc_directory)
        report = root / "result.json"
        data = json.loads(report.read_text()) if report.exists() else {}
        if (
            code != 0
            and options["backend"] == "cuda"
            and "GLM-5.3-Flash CUDA KV cache layout is not supported"
            in data.get("error", "")
        ):
            pytest.xfail(
                "Expected mainline unsupported GLM CUDA native layout; no roundtrip PASS"
            )
        assert code == 0, (
            f"Driver exit={code}: {data.get('error', '')}; see {root / 'driver.log'}"
        )
        assert data["status"] == "PASS"
        return data

    yield SimpleNamespace(run=run, output=root, options=options)
    if not active and Path(ipc_directory).exists():
        shutil.rmtree(ipc_directory)
    for process in active:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
