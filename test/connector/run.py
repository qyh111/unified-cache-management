"""Launch the connector tests with runtime settings applied before imports."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline CUDA/NPU connector tests: native KV layouts with --model-config, synthetic FA otherwise."
    )
    parser.add_argument("--tp", type=int, choices=(1, 2), default=2)
    parser.add_argument("--mode", choices=("bulk", "layerwise"), default="bulk")
    parser.add_argument(
        "--backend",
        choices=("cuda", "npu"),
        required=True,
        help="Native engine/layout backend; CPU attention layouts are not tested",
    )
    parser.add_argument(
        "--device",
        choices=("cpu",),
        default="cpu",
        help="KV tensor allocation device; this offline test supports CPU backing",
    )
    parser.add_argument(
        "--cuda-capability",
        help="Explicit target SM major.minor, e.g. 10.0; no device probe",
    )
    parser.add_argument(
        "--attention-backend", help="Explicit native CUDA attention backend enum name"
    )
    parser.add_argument(
        "--model-config",
        help="Local model directory or config.json; use native CUDA/NPU MRV1 KV layouts",
    )
    parser.add_argument(
        "--tokens", type=int, help="Synthetic request length for --model-config"
    )
    parser.add_argument(
        "--vllm-source",
        help="Optional vLLM v0.30.0 Git checkout; otherwise use installed vLLM",
    )
    parser.add_argument(
        "--ascend-source",
        help="Optional vLLM-Ascend v0.30.0rc1 checkout; otherwise use installed package",
    )
    parser.add_argument(
        "--store",
        choices=("mock", "real"),
        default="mock",
        help="Hash/replay Python store (default), or installed simu Cache|Posix",
    )
    parser.add_argument(
        "--output", required=True, help="New directory for stores, logs and result.json"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120,
        help="Per-command worker deadline in seconds",
    )
    args = parser.parse_args(argv)
    args.tokens_explicit = args.tokens is not None
    for option, package in (("vllm_source", "vllm"), ("ascend_source", "vllm_ascend")):
        value = getattr(args, option)
        if value:
            source = Path(value).resolve()
            if not (source / package / "__init__.py").is_file():
                parser.error(f"{value} does not contain {package} sources")
            setattr(args, option, str(source))
    if args.ascend_source and args.backend != "npu":
        parser.error("--ascend-source requires --backend npu")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.model_config:
        model = Path(args.model_config).resolve()
        if model.name == "config.json" and model.is_file():
            model = model.parent
        if not (model / "config.json").is_file():
            parser.error("--model-config must contain config.json")
        args.model_config = str(model)
    if args.backend == "cuda":
        if (
            not args.model_config
            or not args.cuda_capability
            or not args.attention_backend
        ):
            parser.error(
                "--backend cuda requires --model-config, --cuda-capability and --attention-backend"
            )
        parts = args.cuda_capability.split(".")
        if (
            len(parts) != 2
            or not all(p.isdigit() for p in parts)
            or int(parts[0]) < 8
            or int(parts[1]) > 9
        ):
            parser.error("--cuda-capability must be major.minor (major >= 8)")
    elif args.cuda_capability or args.attention_backend:
        parser.error("CUDA profile options require --backend cuda")
    if args.tokens is not None and (args.tokens <= 0 or not args.model_config):
        parser.error("--tokens requires --model-config and a positive value")
    return args


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    env = os.environ.copy()
    env.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        UCM_CAPTURE_VLLM_LOG="0",
        VLLM_CPU_AFFINITY="0",
        OMP_NUM_THREADS="1",
        VLLM_USE_V2_MODEL_RUNNER="0",
    )
    paths = [str(TEST_ROOT)]
    if args.store == "mock":
        paths.append(str(TEST_ROOT.parent))
    paths.extend(path for path in (args.vllm_source, args.ascend_source) if path)
    paths.append(env.get("PYTHONPATH", ""))
    env["PYTHONPATH"] = os.pathsep.join(paths)
    # Installed UCM's .pth must not initialize engine patches before source
    # version/platform bootstrapping. Ascend mode applies them explicitly later.
    env["ENABLE_UCM_PATCH"] = "0"
    env["VLLM_PLUGINS"] = ""
    env.pop("VLLM_TARGET_DEVICE", None)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["ASCEND_RT_VISIBLE_DEVICES"] = ""
    command = [
        sys.executable,
        "-m",
        "connector.scenario",
        "--backend",
        args.backend,
        "--device",
        args.device,
        "--tp",
        str(args.tp),
        "--mode",
        args.mode,
        "--store",
        args.store,
        "--timeout",
        str(args.timeout),
        "--output",
        str(Path(args.output).resolve()),
    ]
    if args.vllm_source:
        command += ["--vllm-source", args.vllm_source]
    if args.backend == "cuda":
        command += [
            "--cuda-capability",
            args.cuda_capability,
            "--attention-backend",
            args.attention_backend,
        ]
        env["CUDA_VISIBLE_DEVICES"] = ""
    if args.ascend_source:
        command += ["--ascend-source", args.ascend_source]
    if args.model_config:
        command += ["--model-config", args.model_config]
    if args.tokens is not None:
        command += ["--tokens", str(args.tokens)]
    # Mock uses checkout Python only; real mode retains the installed simu UCM.
    return subprocess.call(command, env=env, cwd=TEST_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
