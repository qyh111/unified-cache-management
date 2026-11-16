#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize [DSPARK_DIAG] log lines")
    parser.add_argument("log", type=Path)
    parser.add_argument("--full", action="store_true", help="print all diagnostic lines")
    parser.add_argument("--id-limit", type=int, default=20)
    args = parser.parse_args()
    text = args.log.read_text(errors="replace")
    diagnostic = [line for line in text.splitlines() if "[DSPARK_DIAG]" in line]
    anomalies = [
        line
        for line in diagnostic
        if re.search(r"(?:^|\s)(?:nan|\+inf|-inf)=[1-9]\d*", line) or "failed:" in line or "invalid:" in line
    ]
    ids = [line for line in diagnostic if "sampler.target_argmax" in line or "sampler.draft_token_ids" in line]
    print(f"diagnostic_lines={len(diagnostic)} anomaly_lines={len(anomalies)} id_lines={len(ids)}")
    print("\n=== anomalies ===")
    print("\n".join(anomalies) if anomalies else "(none)")
    print("\n=== first target/draft IDs ===")
    print("\n".join(ids[: args.id_limit]))
    if args.full:
        print("\n=== all diagnostic lines ===")
        print("\n".join(diagnostic))


if __name__ == "__main__":
    main()
