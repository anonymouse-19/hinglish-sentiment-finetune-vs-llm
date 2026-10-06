"""Run every planned LLM baseline in order, resuming where the free-tier daily quota stopped the last run.

Usage (once a day until it prints "All runs complete"):
    python scripts/run_llm_all.py

Answered requests are cached, so each invocation only spends quota on what is still missing.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = [("zero", "val"), ("few", "val"), ("zero", "test"), ("few", "test"), ("zero", "test_youtube"), ("few", "test_youtube")]


def _metrics(mode: str, split: str) -> dict:
    path = ROOT / "reports" / "baselines" / "llm" / f"gpt-oss-120b_{mode}_{split}" / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def is_complete(mode: str, split: str) -> bool:
    return _metrics(mode, split).get("complete", False)


def main() -> None:
    for mode, split in RUNS:
        if is_complete(mode, split):
            print(f"✓ {mode}-shot {split} already complete")
            continue
        print(f"\n▶ {mode}-shot {split}", flush=True)
        subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / "run_llm_baseline.py"), "--mode", mode,
                        "--split", split], check=True)
        if not is_complete(mode, split):
            reason = _metrics(mode, split).get("stopped_early") or "interrupted"
            if reason.startswith("RateLimitError"):
                print("\nDaily token quota reached. Run this script again in ~24 h to continue.")
            else:
                print(f"\nStopped early ({reason[:120]}). Not a quota problem: check the internet connection "
                      "and run this script again; finished requests are cached.")
            return
    subprocess.run([sys.executable, str(ROOT / "scripts" / "compare_baselines.py")], check=True)
    print("\nAll runs complete.")


if __name__ == "__main__":
    main()
