"""Report package provenance without observing connector private state."""

import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from ..cache_setup.fixture import LAYERS, TOKENS
from ..run import parse_args
from .environment import loaded_ucm_extensions, require


def provenance(args):
    import torch
    import vllm
    from vllm.platforms import current_platform

    import ucm

    repo = Path(__file__).resolve().parents[3]
    ucm_root = Path(ucm.__file__).resolve().parent
    result = {
        "python": sys.executable,
        "torch": torch.__version__,
        "device": args.device,
        "backend": args.backend,
        "kv_backing_device": args.device,
        "vllm": vllm.__version__,
        "vllm_path": vllm.__file__,
        "platform": type(current_platform).__module__
        + "."
        + type(current_platform).__name__,
        "cuda_initialized": torch.cuda.is_initialized(),
        "target_cuda_capability": args.cuda_capability,
        "requested_attention_backend": args.attention_backend,
        "npu_initialized": (
            torch.npu.is_initialized() if "torch_npu" in sys.modules else None
        ),
        "ucm_path": ucm.__file__,
        "installed_engine_distributions": {},
        "engine_sources": {},
        "argv": sys.argv,
        "pid": os.getpid(),
        "adaptations": [
            "CPU device/events",
            "explicit worker rank context",
            "Python logger/metrics adapters in mock mode",
            (
                "native model KV and sampled token"
                if args.model_config
                else "synthetic FA fixture and sampled token"
            ),
            "explicit platform and Ascend plugin registration",
        ],
        "environment": {
            key: os.environ.get(key)
            for key in (
                "ENABLE_UCM_PATCH",
                "VLLM_PLUGINS",
                "UCM_CAPTURE_VLLM_LOG",
                "VLLM_CPU_AFFINITY",
                "HF_HUB_OFFLINE",
                "OMP_NUM_THREADS",
            )
        },
        "patch_modules": sorted(
            name
            for name in tuple(sys.modules)
            if name.startswith(("vllm_ascend.patch.", "ucm.integration.vllm.patch."))
        ),
        "native_libraries": {
            str(path.relative_to(ucm_root)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in ucm_root.rglob("*.so")
        },
    }
    if args.model_config:
        model_file = Path(args.model_config) / "config.json"
        result["model_config"] = str(model_file)
        result["model_config_sha256"] = hashlib.sha256(
            model_file.read_bytes()
        ).hexdigest()
    for distribution in ("vllm", "vllm-ascend"):
        try:
            result["installed_engine_distributions"][distribution] = (
                importlib.metadata.version(distribution)
            )
        except importlib.metadata.PackageNotFoundError:
            result["installed_engine_distributions"][distribution] = None
    for name in ("vllm", "vllm_ascend"):
        module = sys.modules.get(name)
        if module is None:
            continue
        source = Path(module.__file__).resolve().parents[1]
        source_arg = args.vllm_source if name == "vllm" else args.ascend_source
        identity = {
            "path": str(source),
            "mode": "source" if source_arg else "installed",
            "sha": None,
            "status": None,
        }
        if (source / ".git").exists() and shutil.which("git"):
            for field, command in (
                ("sha", ["rev-parse", "HEAD"]),
                ("status", ["status", "--porcelain"]),
            ):
                completed = subprocess.run(
                    ["git", "-C", str(source), *command], capture_output=True, text=True
                )
                identity[field] = (
                    completed.stdout.strip() if completed.returncode == 0 else None
                )
        result["engine_sources"][name] = identity
    if args.vllm_source or args.ascend_source:
        result["adaptations"].append("source release metadata from exact Git tags")
    from .environment import capture_engine_modules

    result.update(capture_engine_modules())
    try:
        result["ucm_distribution_version"] = importlib.metadata.version("uc-manager")
    except importlib.metadata.PackageNotFoundError:
        result["ucm_distribution_version"] = None
    result["loaded_ucm_extensions"] = loaded_ucm_extensions()
    result["ucm_python_modules"] = {
        name: hashlib.sha256(Path(filename).read_bytes()).hexdigest()
        for name, module in tuple(sys.modules.items())
        if name.startswith("ucm.")
        and isinstance(filename := vars(module).get("__file__"), str)
        and filename.endswith(".py")
    }
    require(
        args.store != "mock" or not result["loaded_ucm_extensions"],
        "Mock imported UCM native code",
    )
    for name, command in (
        ("source_sha", ["rev-parse", "HEAD"]),
        ("source_status", ["status", "--porcelain"]),
    ):
        completed = subprocess.run(
            ["git", "-C", str(repo), *command], capture_output=True, text=True
        )
        result[name] = completed.stdout.strip() if completed.returncode == 0 else None
    try:
        result["vllm_ascend"] = importlib.metadata.version("vllm-ascend")
    except importlib.metadata.PackageNotFoundError:
        result["vllm_ascend"] = None
    return result


def main(run_roundtrip):
    args = parse_args()
    root = Path(args.output).resolve()
    args.output = str(root)
    root.mkdir(parents=True, exist_ok=False)
    (root / "store").mkdir()
    if args.model_config:
        (root / "model-config.json").write_bytes(
            (Path(args.model_config) / "config.json").read_bytes()
        )
    else:
        (root / "fixture").mkdir()
        (root / "fixture/config.json").write_text(
            json.dumps(
                {
                    "model_type": "llama",
                    "architectures": ["UCMConnectorTestModel"],
                    "hidden_size": 512,
                    "intermediate_size": 1024,
                    "num_hidden_layers": LAYERS,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 4,
                    "head_dim": 128,
                    "vocab_size": 1024,
                    "max_position_embeddings": TOKENS * 2,
                    "rms_norm_eps": 1e-5,
                    "torch_dtype": "float16",
                },
                indent=2,
            )
        )
    started = time.monotonic()
    result = {"status": "FAIL", "store": args.store, "mode": args.mode, "tp": args.tp}
    code = 1
    try:
        result.update(run_roundtrip(args))
        code = 0
    except BaseException:
        result["error"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        try:
            result.update(provenance(args))
        except Exception:
            result["provenance_error"] = traceback.format_exc()
            result["status"] = "FAIL"
            code = 1
        result.update(exit_code=code, seconds=time.monotonic() - started)
        (root / "result.json").write_text(
            json.dumps(result, indent=2, default=str) + "\n"
        )
    print(
        f"connector-test: {result['status']}; report={root / 'result.json'}", flush=True
    )
    return code
