"""Check the LLM API setup: key present, configured model available, one test request.

Usage:
    python scripts/check_llm.py [--config configs/llm.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_llm_baseline import load_api_key  # noqa: E402

from hinglish_sentiment.config import load_config  # noqa: E402
from hinglish_sentiment.llm.client import LLMClient  # noqa: E402
from hinglish_sentiment.llm.prompts import build_messages, parse_label  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/llm.yaml")
    cfg = load_config(parser.parse_args().config)
    client = LLMClient(cfg, load_api_key(cfg["api_key_env"]))

    models = sorted(m.id for m in client.sdk.models.list().data)
    print(f"Key OK. {len(models)} models available at {cfg['base_url']}:")
    for m in models:
        print(f"  {'*' if m == cfg['model'] else ' '} {m}")
    if cfg["model"] not in models:
        sys.exit(f"\nConfigured model {cfg['model']!r} is not available: pick one above and edit configs/llm.yaml")

    r = client.complete(build_messages("yaar ye movie toh ekdum bakwas thi, paisa barbaad"))
    print(f"\nTest request: answer={r.content!r} -> label id {parse_label(r.content)} (expected 0 = negative)")
    print(f"  {r.prompt_tokens} prompt + {r.completion_tokens} completion tokens "
          f"({r.reasoning_tokens} reasoning) · {r.latency_ms:.0f} ms · ${r.cost_usd:.6f}")


if __name__ == "__main__":
    main()
