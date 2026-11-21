# SPDX-License-Identifier: MIT
"""Pytest entry points for card-less MRV2 parallel verification (UCM v1 & v2)."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

RUNNER = Path(__file__).parent / "model_check" / "run.py"


def resolve_local_model(platform: str, model_id: str) -> Path:
    """Resolve and SHA256-verify a provisioned model snapshot from UCM_V2_MODEL_ROOT."""
    root_env = os.environ.get("UCM_V2_MODEL_ROOT")
    if not root_env:
        pytest.skip("Set UCM_V2_MODEL_ROOT to the provisioned model metadata directory")
    root = Path(root_env)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        pytest.fail(f"Model metadata manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    key = f"{platform}/{model_id}"
    if key not in manifest:
        pytest.skip(f"Model metadata manifest has no {key} entry")
    model_dir = root / platform / model_id
    files = manifest[key].get("files_sha256", {})
    if "config.json" not in files:
        pytest.fail(f"Model metadata manifest has no config.json hash for {key}")
    for filename, expected_hash in files.items():
        path = model_dir / filename
        if not path.is_file():
            pytest.fail(f"Model metadata file is missing: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_hash:
            pytest.fail(f"Model metadata hash mismatch: {path}")
    return model_dir


def offline_env() -> dict[str, str]:
    return {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "MODELSCOPE_OFFLINE": "1",
    }


def platform_flags(platform: str) -> list[str]:
    if platform == "cuda":
        cap = os.environ.get("UCM_MODEL_CHECK_CUDA_CAPABILITY", "9.0")
        return ["--platform", "cuda-sim", "--cuda-capability", cap]
    if platform == "ascend":
        soc = os.environ.get("UCM_MODEL_CHECK_ASCEND_SOC", "A2")
        return ["--platform", "ascend-sim", "--ascend-soc", soc]
    raise ValueError(f"Unsupported platform: {platform}")


def selected_connector() -> str:
    connector = os.environ.get("UCM_MODEL_CHECK_CONNECTOR", "v1")
    if connector not in ("v1", "v2"):
        raise ValueError(f"UCM_MODEL_CHECK_CONNECTOR must be v1 or v2, got {connector!r}")
    return connector


def selected_platform() -> str:
    platform = os.environ.get("UCM_MODEL_CHECK_PLATFORM", "")
    if not platform:
        pytest.skip("Set UCM_MODEL_CHECK_PLATFORM=ascend|cuda to run parallel matrix")
    if platform not in ("ascend", "cuda"):
        raise ValueError(f"Unsupported UCM_MODEL_CHECK_PLATFORM={platform!r}")
    return platform


# Parallel topologies for Connector v1 verification across UCMDirect/CP, FAWA
# (hma_connector), and HLA (hla_connector).
# (case_id, model_id, tp, pp, pcp, dcp, extra_args)
# Every case runs with --auto-align-chunks 4 so prompt length and per-step batch
# budget automatically align to the real runtime hit_alignment (including padded
# Mamba/KDA block sizes and FAWA chunks) and create a State checkpoint at 75%.
PARALLEL_CASES = [
    # ---- ucm_connector (GLM-5.3, MiniMax-M3) -------------------------------
    ("glm53-tp2", "glm-5.3", 2, 1, 1, 1, ["--block-size", "128"]),
    ("glm53-pp2-tp2", "glm-5.3", 2, 2, 1, 1, ["--block-size", "128"]),
    ("glm53-pcp2-tp2", "glm-5.3", 2, 1, 2, 1, ["--block-size", "128"]),
    (
        "glm53-dcp2-tp2",
        "glm-5.3",
        2,
        1,
        1,
        2,
        ["--block-size", "128", "--cp-kv-cache-interleave-size", "128"],
    ),
    (
        "glm53-pcp2-dcp2-tp2",
        "glm-5.3",
        2,
        1,
        2,
        2,
        ["--block-size", "128", "--cp-kv-cache-interleave-size", "128"],
    ),
    (
        "glm53-pp2-pcp2-dcp2-tp2",
        "glm-5.3",
        2,
        2,
        2,
        2,
        ["--block-size", "128", "--cp-kv-cache-interleave-size", "128"],
    ),
    # ---- MultiConnector ([MockPDHandshakeConnector, UCMConnector]) ---------
    # Control (PP=1) + PP handshake repro (PP=2), mimicking Ascend Mooncake+UCM.
    (
        "glm53-tp2-multi-pd-first",
        "glm-5.3",
        2,
        1,
        1,
        1,
        ["--block-size", "128", "--multi-connector", "pd-first"],
    ),
    (
        "glm53-pp2-tp2-multi-pd-first",
        "glm-5.3",
        2,
        2,
        1,
        1,
        ["--block-size", "128", "--multi-connector", "pd-first"],
    ),
    ("minimax-m3-tp2", "minimax-m3", 2, 1, 1, 1, ["--block-size", "128"]),
    ("minimax-m3-pp2-tp2", "minimax-m3", 2, 2, 1, 1, ["--block-size", "128"]),
    (
        "minimax-m3-dcp2-tp8",
        "minimax-m3",
        8,
        1,
        1,
        2,
        ["--block-size", "128", "--cp-kv-cache-interleave-size", "128"],
    ),
    # ---- hma_connector / FAWA (DeepSeek-V4-Flash, DeepSeek-V4.1-Flash) -----
    ("dsv4-flash-tp2", "deepseek-v4-flash", 2, 1, 1, 1, []),
    ("dsv4-flash-tp2-pcp2", "deepseek-v4-flash", 2, 1, 2, 1, []),
    ("dsv41-flash-tp2", "deepseek-v4.1-flash", 2, 1, 1, 1, ["--block-size", "128"]),
    (
        "dsv41-flash-tp2-pcp2",
        "deepseek-v4.1-flash",
        2,
        1,
        2,
        1,
        ["--block-size", "128"],
    ),
    # ---- hla_connector / Hybrid MLA+KDA/Mamba (use production TP to avoid
    #      oversized per-rank state padding) ---------------------------------
    ("glm53-flash-tp8", "glm-5.3-flash", 8, 1, 1, 1, ["--block-size", "128"]),
    ("glm53-flash-tp8-pcp2", "glm-5.3-flash", 8, 1, 2, 1, ["--block-size", "128"]),
    ("kimi-k3-tp8", "kimi-k3", 8, 1, 1, 1, ["--block-size", "128"]),
    ("kimi-k3-tp8-pcp2", "kimi-k3", 8, 1, 2, 1, ["--block-size", "128"]),
    ("qwen27b-tp2", "qwen3.8-27b", 2, 1, 1, 1, ["--block-size", "128"]),
    ("qwen27b-tp2-pcp2", "qwen3.8-27b", 2, 1, 2, 1, ["--block-size", "128"]),
    ("qwen-fn-tp4", "qwen3.8-flash-next", 4, 1, 1, 1, ["--block-size", "1536"]),
]

# Startup-rejection cases verifying unsupported v1 parallel configs fail fast
# inside UCM connectors instead of silently corrupting external KV cache.
# Only includes topologies that the engine itself permits (engine-banned combos
# such as DeepSeek-V4.1 PP/DCP, Qwen4Exp CP, GLM-Next KPool DCP, and SWA DCP
# are excluded).
# (case_id, model_id, tp, pp, pcp, dcp, extra_args, error_regex)
V1_REJECT_CASES = [
    (
        "hla-pp2-reject",
        "glm-5.3-flash",
        2,
        2,
        1,
        1,
        [],
        "Pipeline parallelism is not supported",
    ),
    (
        "fawa-v4-pp2-reject",
        "deepseek-v4-flash",
        2,
        2,
        1,
        1,
        [],
        "PP=DCP=1|Pipeline parallelism",
    ),
]


def _model_defaults(platform: str, model_id: str, model_dir: Path) -> list[str]:
    """Derive platform/checkpoint flags without clashing with checkpoint quant config."""
    if platform == "ascend":
        defaults = ["--trust-remote-code"]
        # Only pass --quantization ascend when the provisioned directory is an
        # Ascend ModelSlim checkpoint (carries quant_model_description.json).
        if (model_dir / "quant_model_description.json").is_file():
            defaults.extend(["--quantization", "ascend"])
        if model_id in ("glm-5.3", "glm-5.3-flash"):
            defaults.extend(["--kv-cache-dtype", "int8", "--indexer-kv-dtype", "int8"])
        return defaults

    defaults = ["--trust-remote-code"]
    if model_id in (
        "deepseek-v4-flash",
        "deepseek-v4.1-flash",
        "kimi-k3",
        "qwen3.8-flash-next",
    ):
        defaults.extend(["--kv-cache-dtype", "fp8"])
    if model_id == "qwen3.8-flash-next":
        defaults.extend(["--indexer-kv-dtype", "fp8"])
    if model_id == "kimi-k3":
        defaults.extend(["--mamba-ssm-cache-dtype", "bfloat16"])
    return defaults


@pytest.mark.stage(2)
@pytest.mark.feature("connector")
def test_custom_parallel_model_check():
    """Run one custom invocation using UCM_MODEL_CHECK_ARGS."""
    raw_args = os.environ.get("UCM_MODEL_CHECK_ARGS")
    if not raw_args:
        pytest.skip("Set UCM_MODEL_CHECK_ARGS to run a custom model-check command")
    subprocess.run(
        [sys.executable, str(RUNNER), *shlex.split(raw_args)],
        check=True,
        env=offline_env(),
    )


@pytest.mark.stage(2)
@pytest.mark.feature("connector")
@pytest.mark.parametrize(
    ("case_id", "model_id", "tp", "pp", "pcp", "dcp", "extra_args"),
    PARALLEL_CASES,
    ids=[case[0] for case in PARALLEL_CASES],
)
def test_parallel_roundtrip_matrix(
    case_id, model_id, tp, pp, pcp, dcp, extra_args
):
    platform = selected_platform()
    connector = selected_connector()
    model_dir = resolve_local_model(platform, model_id)
    cmd = [
        sys.executable,
        str(RUNNER),
        "--connector",
        connector,
        *platform_flags(platform),
        *_model_defaults(platform, model_id, model_dir),
        "--model",
        str(model_dir),
        "--tp",
        str(tp),
        "--pp",
        str(pp),
        "--pcp",
        str(pcp),
        "--dcp",
        str(dcp),
        "--tokens",
        os.environ.get("UCM_MODEL_CHECK_TOKENS", "4096"),
        "--max-num-batched-tokens",
        os.environ.get("UCM_MODEL_CHECK_BATCH_TOKENS", "2048"),
        "--auto-align-chunks",
        os.environ.get("UCM_MODEL_CHECK_AUTO_ALIGN_CHUNKS", "4"),
        *extra_args,
    ]
    subprocess.run(cmd, check=True, env=offline_env())


@pytest.mark.stage(2)
@pytest.mark.feature("connector")
@pytest.mark.parametrize(
    ("case_id", "model_id", "tp", "pp", "pcp", "dcp", "extra_args", "error_regex"),
    V1_REJECT_CASES,
    ids=[case[0] for case in V1_REJECT_CASES],
)
def test_v1_unsupported_parallel_rejection(
    case_id, model_id, tp, pp, pcp, dcp, extra_args, error_regex
):
    if selected_connector() != "v1":
        pytest.skip("Startup-rejection matrix applies to v1 connector")
    platform = selected_platform()
    model_dir = resolve_local_model(platform, model_id)
    cmd = [
        sys.executable,
        str(RUNNER),
        "--connector",
        "v1",
        *platform_flags(platform),
        *_model_defaults(platform, model_id, model_dir),
        "--model",
        str(model_dir),
        "--tp",
        str(tp),
        "--pp",
        str(pp),
        "--pcp",
        str(pcp),
        "--dcp",
        str(dcp),
        "--tokens",
        "1024",
        "--max-num-batched-tokens",
        "512",
        *extra_args,
        "--expect-reject",
        error_regex,
    ]
    subprocess.run(cmd, check=True, env=offline_env())
