"""Phase 5: publish the fine-tuned model to the Hugging Face Hub and deploy the Gradio Space.

Usage:
    # 1. on Colab, where the weights are (notebook section 9): model + card from the Phase 3 reports
    python scripts/publish.py model --model-dir <drive>/outputs/encoder/<best>/model \\
        --report-dir <drive>/reports --repo-id <user>/hinglish-sentiment-xlmr
    # 2. locally, after scripts/evaluate_all.py: refresh the card with the LLM comparison
    python scripts/publish.py card --repo-id <user>/hinglish-sentiment-xlmr
    # 3. locally: deploy app/ as a Space (GROQ_API_KEY from .env becomes a Space secret)
    python scripts/publish.py space --space-id <user>/hinglish-sentiment-demo --model-id <user>/hinglish-sentiment-xlmr

Add --dry-run to write everything to outputs/publish_preview/ without uploading.
The Hub token comes from HF_TOKEN (env, .env or Colab secret) or a prior `hf auth login`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hinglish_sentiment.config import PROJECT_ROOT, load_config, resolve  # noqa: E402
from hinglish_sentiment.ship.model_card import render_model_card  # noqa: E402

PREVIEW = resolve("outputs/publish_preview")
GITHUB_PKG = "git+https://github.com/anonymouse-19/hinglish-sentiment-finetune-vs-llm.git"


def _token() -> str | None:
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass
    return os.environ.get("HF_TOKEN")


def _card(repo_id: str, report_dir: Path) -> str:
    ft = json.loads((report_dir / "encoder" / "metrics.json").read_text(encoding="utf-8"))
    ev_path = resolve("reports/evaluation/summary.json")
    ev = json.loads(ev_path.read_text(encoding="utf-8")) if ev_path.exists() else None
    return render_model_card(repo_id, ft, ev)


def publish_model(args) -> None:
    from transformers import AutoConfig

    report_dir = Path(args.report_dir) if args.report_dir else resolve("reports/finetune")
    best = json.loads((report_dir / "encoder" / "metrics.json").read_text(encoding="utf-8"))["best_run"]
    stage = PREVIEW / "model"
    shutil.rmtree(stage, ignore_errors=True)
    shutil.copytree(args.model_dir, stage, ignore=shutil.ignore_patterns("training_args.bin", "checkpoint-*"))
    # record the preprocessing in the config, so the demo / any user can apply it (D-029)
    cfg = AutoConfig.from_pretrained(stage)
    cfg.demojize = bool(best["demojize"])
    cfg.save_pretrained(stage)
    (stage / "README.md").write_text(_card(args.repo_id, report_dir), encoding="utf-8")
    print(f"Staged {sum(f.stat().st_size for f in stage.rglob('*') if f.is_file()) / 1e6:.0f} MB in {stage}")
    if args.dry_run:
        return
    from huggingface_hub import HfApi
    api = HfApi(token=_token())
    api.create_repo(args.repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=args.repo_id, folder_path=str(stage), commit_message=f"Upload {best['name']}")
    print(f"https://huggingface.co/{args.repo_id}")


def publish_card(args) -> None:
    card = _card(args.repo_id, Path(args.report_dir) if args.report_dir else resolve("reports/finetune"))
    PREVIEW.mkdir(parents=True, exist_ok=True)
    (PREVIEW / "MODEL_CARD.md").write_text(card, encoding="utf-8")
    print(f"Card written to {PREVIEW / 'MODEL_CARD.md'}")
    if args.dry_run:
        return
    from huggingface_hub import HfApi
    HfApi(token=_token()).upload_file(path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md",
                                      repo_id=args.repo_id, commit_message="Update model card with the LLM comparison")
    print(f"https://huggingface.co/{args.repo_id}")


def publish_space(args) -> None:
    import pandas as pd

    stage = PREVIEW / "space"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    shutil.copy(resolve("app/app.py"), stage / "app.py")
    # The study's 9 few-shot demonstrations (train tweets, ids in configs/few_shot_ids.yaml) go to the Space only,
    # so the demo uses the exact frozen prompt; they are not committed to git (D-004, D-037).
    ids = load_config("configs/few_shot_ids.yaml")["ids"]
    train = pd.read_parquet(resolve("data/processed/train.parquet")).set_index("id").loc[ids].reset_index()
    (stage / "few_shot_examples.json").write_text(
        train[["id", "text", "label"]].to_json(orient="records", force_ascii=False, indent=1), encoding="utf-8")
    (stage / "requirements.txt").write_text(
        f"{GITHUB_PKG}\ntorch\ntransformers==5.18.0\nemoji==2.16.0\nopenai==3.24.0\n", encoding="utf-8")
    (stage / "README.md").write_text(f"""---
title: Hinglish Sentiment — Fine-tuned vs LLM
emoji: 🗣️
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: A fine-tuned small model vs a 117B LLM on Hinglish sentiment
---

Side-by-side demo for [{GITHUB_PKG.split('/')[-1][:-4]}]({GITHUB_PKG[4:-4]}): the fine-tuned model `{args.model_id}`
(on this Space's CPU) vs `openai/gpt-oss-120b` on Groq with the study's few-shot prompt.
""", encoding="utf-8")
    print(f"Staged Space files in {stage}")
    if args.dry_run:
        return
    from huggingface_hub import HfApi
    api = HfApi(token=_token())
    api.create_repo(args.space_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.add_space_variable(args.space_id, "MODEL_ID", args.model_id)
    if args.ft_usd_per_1k is not None:
        api.add_space_variable(args.space_id, "FT_USD_PER_1K", str(args.ft_usd_per_1k))
    if os.environ.get("GROQ_API_KEY"):
        api.add_space_secret(args.space_id, "GROQ_API_KEY", os.environ["GROQ_API_KEY"])
    else:
        print("GROQ_API_KEY not found: the LLM half of the demo will be disabled until you add the secret.")
    api.upload_folder(repo_id=args.space_id, repo_type="space", folder_path=str(stage), commit_message="Deploy demo")
    print(f"https://huggingface.co/spaces/{args.space_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("model")
    m.add_argument("--model-dir", required=True)
    m.add_argument("--repo-id", required=True)
    m.add_argument("--report-dir", default=None)
    c = sub.add_parser("card")
    c.add_argument("--repo-id", required=True)
    c.add_argument("--report-dir", default=None)
    s = sub.add_parser("space")
    s.add_argument("--space-id", required=True)
    s.add_argument("--model-id", required=True)
    s.add_argument("--ft-usd-per-1k", type=float, default=None, help="shown in the demo (from evaluate_all)")
    for p in (m, c, s):
        p.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    {"model": publish_model, "card": publish_card, "space": publish_space}[args.cmd](args)


if __name__ == "__main__":
    main()
