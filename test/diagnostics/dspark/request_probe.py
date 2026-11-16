#!/usr/bin/env python3
import argparse
import json
import re
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

SPEC_COUNTERS = (
    "vllm:spec_decode_num_drafts",
    "vllm:spec_decode_num_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens",
    "vllm:corrupted_requests",
)


def get_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=1800) as response:
        return response.read().decode()


def post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.loads(response.read().decode())


def metric_value(metrics: str, name: str) -> float:
    total = 0.0
    pattern = re.compile(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([-+0-9.eE]+)\s*$", re.MULTILINE)
    for match in pattern.finditer(metrics):
        try:
            total += float(match.group(1))
        except ValueError:
            pass
    return total


def per_position(metrics: str) -> dict[int, float]:
    values: dict[int, float] = defaultdict(float)
    pattern = re.compile(
        r"^vllm:spec_decode_num_accepted_tokens_per_pos\{([^}]*)\}\s+([-+0-9.eE]+)\s*$",
        re.MULTILINE,
    )
    for match in pattern.finditer(metrics):
        labels = match.group(1)
        position = re.search(r'position="(\d+)"', labels)
        if position:
            values[int(position.group(1))] += float(match.group(2))
    return dict(values)


def snapshot(url: str) -> tuple[str, dict]:
    metrics = get_text(f"{url}/metrics")
    values = {name: metric_value(metrics, name) for name in SPEC_COUNTERS}
    values["accepted_per_pos"] = per_position(metrics)
    return metrics, values


def delta(before: dict, after: dict) -> dict:
    result = {name: after[name] - before[name] for name in SPEC_COUNTERS}
    positions = set(before["accepted_per_pos"]) | set(after["accepted_per_pos"])
    result["accepted_per_pos"] = {
        pos: after["accepted_per_pos"].get(pos, 0.0) - before["accepted_per_pos"].get(pos, 0.0)
        for pos in sorted(positions)
    }
    drafts = result["vllm:spec_decode_num_drafts"]
    draft_tokens = result["vllm:spec_decode_num_draft_tokens"]
    accepted = result["vllm:spec_decode_num_accepted_tokens"]
    result["acceptance_rate"] = accepted / draft_tokens if draft_tokens > 0 else None
    result["acceptance_per_pos"] = {
        pos: count / drafts for pos, count in result["accepted_per_pos"].items()
    } if drafts > 0 else {}
    result["mean_acceptance_length"] = 1 + accepted / drafts if drafts > 0 else None
    return result


def make_prompt() -> str:
    lines = [
        "Read the following records, then continue with one concise factual sentence.",
        "",
    ]
    for i in range(420):
        lines.append(
            f"Record {i:03d}: unified cache management stores attention state in blocks; "
            f"the reviewer checks tensor layout, stream order, and byte integrity before reuse."
        )
    lines.extend(["", "Continue:"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18080")
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.reset:
        reset_result = post_json(f"{args.url}/reset_prefix_cache", {})
        (out / f"{args.label}-reset.json").write_text(json.dumps(reset_result, indent=2))

    _, before = snapshot(args.url)
    payload = {
        "model": "dsv4",
        "prompt": make_prompt(),
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "seed": 42,
        "stream": False,
    }
    started = time.time()
    response = post_json(f"{args.url}/v1/completions", payload)
    elapsed = time.time() - started
    metrics_text, after = snapshot(args.url)
    result = delta(before, after)
    result["elapsed_seconds"] = elapsed
    result["response"] = response
    result["prompt_characters"] = len(payload["prompt"])
    (out / f"{args.label}-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    (out / f"{args.label}-metrics.txt").write_text(metrics_text)
    print(json.dumps({k: v for k, v in result.items() if k != "response"}, ensure_ascii=False, indent=2))
    for choice in response.get("choices", []):
        print("--- generated text ---")
        print(choice.get("text", ""))


if __name__ == "__main__":
    main()
