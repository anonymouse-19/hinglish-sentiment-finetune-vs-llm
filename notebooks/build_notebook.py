"""Generate notebooks/phase3_finetune_colab.ipynb (the notebook is built from this file so every cell is reviewable).

Usage:
    python notebooks/build_notebook.py
"""

import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


md("""
# Hinglish sentiment: fine-tune, evaluate, publish (free Colab T4)

**One `Runtime → Run all` does the whole GPU side of the project**, using the same code as the local scripts:

| Step | What | Time on a T4 |
|---|---|---|
| 1–5 | GPU check, Drive, code from GitHub, install, data rebuild, pipeline smoke test | ~5 min |
| 6 | **Option A:** XLM-R / MuRIL / MuRIL+emoji→text × 3 learning rates (9 runs) | ~40 min |
| 7 | **Option B:** Qwen3-1.7B QLoRA, LoRA rank × learning rate (4 runs) | ~1 h |
| 8 | Evaluate every model against the baselines, plus error-analysis files | ~2 min |
| 9 | Publish the best model to the Hugging Face Hub *(needs `HF_TOKEN`)* | ~3 min |
| 10 | Deploy the side-by-side demo to HF Spaces *(needs `HF_TOKEN` + `GROQ_API_KEY`)* | ~1 min |
| 11 | Download `finetune_results.zip`; unzip it into the project folder on your laptop | – |

**Before you start** (see `docs/OWNER_ACTIONS.md` for click-by-click steps):
1. `Runtime → Change runtime type → T4 GPU`.
2. Colab secrets (🔑 in the left sidebar, with **Notebook access** on): `WANDB_API_KEY` (recommended), `HF_TOKEN` (Write token) and `GROQ_API_KEY` (optional; for steps 9–10).

Everything is saved to **Google Drive**. If Colab disconnects or you run out of GPU time, just `Run all` again later: finished runs are skipped.
""")

code("""
GITHUB_REPO = "https://github.com/anonymouse-19/hinglish-sentiment-finetune-vs-llm.git"
DRIVE_DIR = "/content/drive/MyDrive/hinglish-ft"   # models + reports are stored here
SPACE_NAME = "hinglish-sentiment-demo"             # name of the demo Space (step 10)
""")

md("## 1. GPU, Drive, code")
code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
from google.colab import drive
drive.mount("/content/drive")
""")
code("""
import os
if os.path.exists("/content/repo"):
    !git -C /content/repo pull --ff-only
else:
    !git clone {GITHUB_REPO} /content/repo
%cd /content/repo
!git log -1 --format="code version: %h %s"
""")

md("## 2. Install (Colab's own CUDA torch is kept)")
code("""
!pip install -q -r requirements-colab.txt
!pip install -q -e . --no-deps
import torch, transformers, peft
print(torch.__version__, transformers.__version__, peft.__version__, "CUDA:", torch.cuda.is_available())
assert torch.cuda.is_available(), "No GPU: Runtime -> Change runtime type -> T4 GPU, then Run all again"
""")

md("## 3. Secrets\nMissing secrets don't stop the run: W&B falls back to offline logging, and steps 9–10 are skipped.")
code("""
from google.colab import userdata

def secret(name):
    try:
        return userdata.get(name)
    except Exception:
        return None

for name in ("WANDB_API_KEY", "HF_TOKEN", "GROQ_API_KEY"):
    value = secret(name)
    if value:
        os.environ[name] = value
    print(f"{name:14s}", "found" if value else "not set")

if os.environ.get("WANDB_API_KEY"):
    import wandb
    wandb.login()
else:
    os.environ["WANDB_MODE"] = "offline"   # still logs locally; nothing fails
""")

md("## 4. Rebuild the data (~1 min)\nThe data is never committed (no licence, D-004); this rebuilds the identical splits from the pinned source.")
code("""
!python scripts/prepare_data.py
""")

md("## 5. Pipeline check on tiny models (~1 min)\nCatches environment problems before spending GPU time.")
code("""
!python scripts/finetune_sweep.py --family encoder --smoke --only xlmr_lr2e-05
""")

md("""
## 6. Option A: encoder sweep (9 runs, ~40 min)
Each run: up to 4 epochs, early stopping on val macro-F1. Only the best-so-far model is kept on Drive.
Live curves: https://wandb.ai, project **hinglish-sentiment-finetune**.
""")
code("""
!python scripts/finetune_sweep.py --family encoder --output-root {DRIVE_DIR}/outputs --report-dir {DRIVE_DIR}/reports
""")

md("## 7. Option B: Qwen3-1.7B QLoRA sweep (4 runs, ~1 h)\nResumes from the last finished run if the session ends.")
code("""
!python scripts/finetune_sweep.py --family decoder --output-root {DRIVE_DIR}/outputs --report-dir {DRIVE_DIR}/reports
""")

md("""
## 8. Evaluate everything
The Phase 2 baseline results (TF-IDF, the LLM) are in the repo, so the fine-tuned models are compared on exactly the same tweets:
same-tweet tables, paired bootstrap tests in-domain and on YouTube, the headline, figures, and error-analysis files.
""")
code("""
!mkdir -p reports/finetune && cp -r {DRIVE_DIR}/reports/* reports/finetune/
!python scripts/evaluate_all.py
!for fam in encoder decoder; do [ -f reports/finetune/$fam/metrics.json ] && python scripts/error_analysis.py --model $fam --n 30; done
from IPython.display import Markdown, Image, display
display(Markdown(open("reports/evaluation/results.md", encoding="utf-8").read()))
for fig in ("cost_vs_f1", "robustness", "confusion_matrices"):
    display(Image(f"reports/figures/{fig}.png", width=760))
""")

md("""
## 9. Publish the best encoder to the Hugging Face Hub *(needs `HF_TOKEN`)*
Uploads the val-selected model (~1.1 GB) from Drive, with a model card generated from the results above.
Your username is read from the token.
""")
code("""
import json
REPO_ID = None
if os.environ.get("HF_TOKEN"):
    from huggingface_hub import HfApi
    HF_USERNAME = HfApi(token=os.environ["HF_TOKEN"]).whoami()["name"]
    best = open(f"{DRIVE_DIR}/outputs/encoder/BEST_RUN.txt").read().strip()
    REPO_ID = f"{HF_USERNAME}/hinglish-sentiment-{'muril' if best.startswith('muril') else 'xlmr'}"
    print("best run:", best, "->", REPO_ID)
    !python scripts/publish.py model --model-dir {DRIVE_DIR}/outputs/encoder/{best}/model --repo-id {REPO_ID}
else:
    print("HF_TOKEN not set: skipping. Add the secret and run this cell (and the next) later.")
""")

md("## 10. Deploy the side-by-side demo to HF Spaces *(needs `HF_TOKEN`, plus `GROQ_API_KEY` for the LLM half)*")
code("""
if REPO_ID:
    summary = json.load(open("reports/evaluation/summary.json", encoding="utf-8"))
    FT_COST = next(r["usd_per_1k"] for r in summary["tables"]["full_test"]["rows"] if r["key"] == "encoder")
    SPACE_ID = f"{HF_USERNAME}/{SPACE_NAME}"
    !python scripts/publish.py space --space-id {SPACE_ID} --model-id {REPO_ID} --ft-usd-per-1k {FT_COST}
    print("The Space takes ~5 min to build the first time.")
else:
    print("Skipped (no published model yet).")
""")

md("""
## 11. Bring the results home
Downloads `finetune_results.zip`: metrics, predictions, sweep tables, evaluation tables, figures and the error-analysis notes
(no model weights). **Unzip it into the project folder on your laptop** (it contains `reports/` and `outputs/` folders that
merge into the existing ones), then tell Claude "results are in".
""")
code("""
!rm -f /content/finetune_results.zip
!zip -qr /content/finetune_results.zip reports/finetune reports/evaluation reports/figures outputs/error_analysis_*.md
!unzip -l /content/finetune_results.zip | tail -3
from google.colab import files
files.download("/content/finetune_results.zip")
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = Path(__file__).parent / "phase3_finetune_colab.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"{out.name}: {len(cells)} cells")
