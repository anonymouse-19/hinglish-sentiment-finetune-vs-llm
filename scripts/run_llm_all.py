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


def is_complete(mode: str, split: str) -> bool:
    path = ROOT / "reports" / "baselines" / "llm" / f"gpt-oss-120b_{mode}_{split}" / "metrics.json"
    return path.exists() and json.loads(path.read_text(encoding="utf-8")).get("complete", False)


def main() -> None:
    for mode, split in RUNS:
        if is_complete(mode, split):
            print(f"✓ {mode}-shot {split} already complete")
            continue
        print(f"\n▶ {mode}-shot {split}", flush=True)
        subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / "run_llm_baseline.py"), "--mode", mode,
                        "--split", split], check=True)
        if not is_complete(mode, split):
            print("\nDaily quota reached (or the run was interrupted). Run this script again tomorrow to continue.")
            return
    subprocess.run([sys.executable, str(ROOT / "scripts" / "compare_baselines.py")], check=True)
    print("\nAll runs complete.")


if __name__ == "__main__":
    main()
