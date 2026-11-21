# SPDX-License-Identifier: MIT
"""Provision offline metadata-only snapshots for Ascend A2/A3 ModelScope checkpoints.

Downloads only lightweight metadata files (config.json, quant_model_description.json,
tokenizer/processor configs, and custom modeling *.py files) -- never weight shards --
and writes a SHA256 manifest.json consumed by test_parallel_model_check.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

# Official Ascend A2/A3 ModelScope quantized checkpoints from
# vllm-ascend/docs/source/tutorials/models/ (v0.32.0rc).
ASCEND_A2_MODELSCOPE_CATALOG: dict[str, dict[str, str]] = {
    "deepseek-v4-flash": {
        "repo_id": "Eco-Tech/DeepSeek-V4-Flash-0731-w8a8",
        "quant_note": "A2/A3 W8A8 (ModelSlim)",
    },
    "deepseek-v4.1-flash": {
        "repo_id": "Eco-Tech/DeepSeek-V4.1-Flash-1017-w8a8",
        "quant_note": "A2/A3 W8A8 (ModelSlim)",
    },
    "glm-5.3": {
        "repo_id": "Eco-Tech/GLM-5.3-w8a8c8",
        "quant_note": "A2/A3 W8A8C8 (ModelSlim)",
    },
    "glm-5.3-flash": {
        "repo_id": "Eco-Tech/GLM-5.3-Flash-w8a8",
        "quant_note": "A2/A3 W8A8 (ModelSlim, dynamic INT8 KV)",
    },
    "kimi-k3": {
        "repo_id": "Eco-Tech/Kimi-K3-w4a8",
        "quant_note": "A2/A3 W4A8 (ModelSlim)",
    },
    "qwen3.8-flash-next": {
        "repo_id": "Eco-Tech/Qwen3.8-Flash-Next-w4a8",
        "quant_note": "A2/A3 W4A8 (ModelSlim)",
    },
    "qwen3.8-27b": {
        "repo_id": "Eco-Tech/Qwen3.8-27B-w8a8",
        "quant_note": "A2/A3 W8A8 (ModelSlim)",
    },
    "minimax-m3": {
        "repo_id": "Eco-Tech/MiniMax-M3-w8a8-0626",
        "quant_note": "A2/A3 W8A8 (ModelSlim)",
    },
    "minimax-m3.1-flash": {
        "repo_id": "Eco-Tech/Minimax-M3.1-Flash-w8a8",
        "quant_note": "A2/A3 W8A8 (ModelSlim)",
    },
}

METADATA_ALLOW_PATTERNS = [
    "*.json",
    "*.py",
    "*.jinja",
    "*.model",
    "*.tiktoken",
    "merges.txt",
    "vocab.txt",
]

METADATA_IGNORE_PATTERNS = [
    "*.safetensors",
    "*.safetensors.index.json",
    "*.bin",
    "*.bin.index.json",
    "*.pt",
    "*.pth",
    "*.gguf",
    "*.onnx",
]


def _is_metadata_file(rel_path: str) -> bool:
    lower = rel_path.lower()
    if lower.endswith(
        (
            ".safetensors",
            ".safetensors.index.json",
            ".bin",
            ".bin.index.json",
            ".pt",
            ".pth",
            ".gguf",
            ".onnx",
        )
    ):
        return False
    return lower.endswith(
        (".json", ".py", ".jinja", ".model", ".tiktoken")
    ) or Path(lower).name in ("merges.txt", "vocab.txt")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_via_modelscope_sdk(repo_id: str, target_dir: Path, revision: str) -> None:
    from modelscope.hub.snapshot_download import snapshot_download

    snapshot_download(
        model_id=repo_id,
        revision=revision,
        local_dir=str(target_dir),
        allow_patterns=METADATA_ALLOW_PATTERNS,
        ignore_patterns=METADATA_IGNORE_PATTERNS,
    )


def _download_via_http(repo_id: str, target_dir: Path, revision: str) -> None:
    api_url = (
        f"https://www.modelscope.cn/api/v1/models/{repo_id}/repo/files"
        f"?Revision={urllib.parse.quote(revision)}&Recursive=True"
    )
    req = urllib.request.Request(api_url, headers={"User-Agent": "ucm-model-check/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    files = (payload.get("Data") or {}).get("Files") or []
    downloaded = 0
    for entry in files:
        if entry.get("Type") != "blob":
            continue
        rel_path = entry.get("Path") or entry.get("Name") or ""
        if not rel_path or not _is_metadata_file(rel_path):
            continue
        encoded_path = urllib.parse.quote(rel_path, safe="/")
        file_url = (
            f"https://www.modelscope.cn/models/{repo_id}/resolve/"
            f"{urllib.parse.quote(revision)}/{encoded_path}"
        )
        dst = target_dir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        file_req = urllib.request.Request(
            file_url, headers={"User-Agent": "ucm-model-check/1.0"}
        )
        with urllib.request.urlopen(file_req, timeout=60) as stream:
            dst.write_bytes(stream.read())
        downloaded += 1
    if downloaded == 0:
        raise RuntimeError(f"No metadata files downloaded from ModelScope {repo_id}")


def fetch_one(
    model_id: str,
    repo_id: str,
    out_root: Path,
    *,
    platform: str,
    revision: str,
    use_http_only: bool,
) -> dict[str, object]:
    target_dir = out_root / platform / model_id
    target_dir.mkdir(parents=True, exist_ok=True)

    used_sdk = False
    if not use_http_only:
        try:
            _download_via_modelscope_sdk(repo_id, target_dir, revision)
            used_sdk = True
        except ImportError:
            used_sdk = False
    if not used_sdk:
        _download_via_http(repo_id, target_dir, revision)

    files_sha256: dict[str, str] = {}
    for file_path in sorted(target_dir.rglob("*")):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(target_dir).as_posix()
        if rel.startswith(".") or not _is_metadata_file(rel):
            continue
        files_sha256[rel] = _sha256_file(file_path)

    if "config.json" not in files_sha256:
        raise RuntimeError(f"{repo_id} did not provide config.json in {target_dir}")
    if "quant_model_description.json" not in files_sha256:
        print(
            f"[WARN] {model_id} ({repo_id}) has no quant_model_description.json; "
            "test_parallel_model_check will not force --quantization ascend.",
            flush=True,
        )

    return {
        "path": str(target_dir.resolve()),
        "resolved_path": str(target_dir.resolve()),
        "source": "modelscope",
        "repo_id": repo_id,
        "revision": revision,
        "files_sha256": files_sha256,
    }


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="output root directory (will hold ascend/<model_id>/ and manifest.json)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(ASCEND_A2_MODELSCOPE_CATALOG),
        default=sorted(ASCEND_A2_MODELSCOPE_CATALOG),
        help="subset of Ascend A2/A3 ModelScope quantized models to fetch",
    )
    parser.add_argument("--revision", default="master", help="ModelScope revision")
    parser.add_argument(
        "--http-only",
        action="store_true",
        help="skip modelscope SDK and fetch via direct ModelScope HTTP API",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out_root = args.out.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )

    failures: list[tuple[str, str]] = []
    for model_id in args.models:
        entry = ASCEND_A2_MODELSCOPE_CATALOG[model_id]
        repo_id = entry["repo_id"]
        print(f"Fetching ascend/{model_id} <- {repo_id} ({entry['quant_note']})...", flush=True)
        try:
            manifest[f"ascend/{model_id}"] = fetch_one(
                model_id,
                repo_id,
                out_root,
                platform="ascend",
                revision=args.revision,
                use_http_only=args.http_only,
            )
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(
                f"  OK: {len(manifest[f'ascend/{model_id}']['files_sha256'])} metadata files",
                flush=True,
            )
        except Exception as exc:
            failures.append((model_id, f"{type(exc).__name__}: {exc}"))
            print(f"  FAIL: {model_id} ({repo_id}): {exc}", flush=True)

    print(f"\nManifest written to: {manifest_path}", flush=True)
    if failures:
        summary = ", ".join(f"{m} ({err})" for m, err in failures)
        raise SystemExit(f"Some models failed to fetch: {summary}")


if __name__ == "__main__":
    main()
