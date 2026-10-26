# Owner actions: what only you can do

> **Working alone? Follow [SOLO_TODO.md](SOLO_TODO.md)**: the full step-by-step list, including the parts Claude would normally do.
> Everything else is automated. This file is the single list of what needs **you** (accounts, secrets, the Colab run, decisions).
> Last updated: **2026-10-26**. Tick items off here as they're done.

## Where things stand

| Phase | Status |
|---|---|
| 1 · Data | ✅ done |
| 2 · Baselines (TF-IDF, Groq LLM zero-/few-shot) | ✅ done: all results in `reports/baselines/` |
| 3 · Fine-tuning (XLM-R / MuRIL, Qwen3 QLoRA) | ⏳ **code ready; needs your Colab run (step 3 below)** |
| 4 · Evaluation | ⏳ tooling ready; runs automatically at the end of the Colab notebook |
| 5 · Ship (HF Hub model + demo Space) | ⏳ tooling ready; runs automatically at the end of the Colab notebook if `HF_TOKEN` is set |

---

## Step 1: Hugging Face write token (≈ 5 min) ☐

1. Sign up at **https://huggingface.co/join**, or log in. Your username can be anything; the notebook reads it from the token.
2. Open **https://huggingface.co/settings/tokens** → **Create new token**.
3. Choose **Token type: Write**, name it `hinglish-sentiment`, then **Create token**.
4. **Copy the token** (starts with `hf_`). It is shown only once; keep it somewhere private until step 3.
5. Also paste it into the project's `.env` file as a new line, then save:
   ```
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
   ```
   (`.env` is git-ignored. Never paste tokens into chat, code or the notebook itself.)

## Step 2: Weights & Biases key (≈ 2 min) ☐ *(recommended, not required)*

1. Sign up at **https://wandb.ai** (free, "Personal" use).
2. Copy your API key from **https://wandb.ai/authorize**.

If you skip this, training still works; the charts are just saved offline instead of to a live dashboard.

## Step 3: Run the notebook on Colab (≈ 2 h, mostly unattended) ☐

1. Open **https://colab.research.google.com** → **File → Open notebook** → **GitHub** tab →
   type `anonymouse-19/hinglish-sentiment-finetune-vs-llm` → Enter → click **`notebooks/phase3_finetune_colab.ipynb`**.
2. **Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save.**
3. Click the **🔑 (Secrets)** icon in the left sidebar → **+ Add new secret**, three times. Switch **Notebook access** on for each:

   | Name | Value |
   |---|---|
   | `WANDB_API_KEY` | your W&B key (step 2) |
   | `HF_TOKEN` | your Hugging Face token (step 1) |
   | `GROQ_API_KEY` | the Groq key already in your `.env` (the line `GROQ_API_KEY=...`), for the demo's LLM half |

4. **Runtime → Run all.** When asked, click **Connect to Google Drive** and allow access.
5. Keep the tab open. You can watch training live at **https://wandb.ai** → project **hinglish-sentiment-finetune**.
   Section 6 takes ~40 min and section 7 ~1 h.
6. At the end the browser downloads **`finetune_results.zip`**.

**If something goes wrong**

| What you see | What to do |
|---|---|
| "No GPU" / `AssertionError` in step 2 | Runtime → Change runtime type → T4 GPU, then Run all again |
| Colab disconnects, or "You cannot currently connect to a GPU" (free-tier limit) | Come back later (usually next day) → Run all. Finished runs are skipped automatically |
| An error inside **section 7** (Qwen QLoRA) | Click the first cell of section 8 → **Runtime → Run after**, so you still get evaluation and publishing for the encoders. Then copy the error message to Claude |
| Any other red error | Copy the last ~20 lines of the error to Claude |

## Step 4: Bring the results home ☐

1. Find `finetune_results.zip` in your Downloads.
2. **Unzip it into the project folder** (`AIML — Fine-Tuned Small Model vs Large LLM (CostAccuracy Study)`), choosing **replace/merge** if asked.
   Afterwards these folders should exist: `reports\finetune\encoder\`, `reports\finetune\decoder\` and `reports\evaluation\`.
3. Tell Claude **"results are in"**, or continue yourself from [SOLO_TODO.md](SOLO_TODO.md) Part 3.3.

## Step 5: One decision ☐

**Licence for the code** (the public GitHub repo). It covers only the code, not the dataset. Reply with one word:
- **MIT** *(suggested)*: the most common for portfolio projects; anyone can reuse the code with attribution.
- **Apache-2.0**: like MIT, plus an explicit patent grant; common in ML libraries.

## Optional (≈ 20 min): blind relabelling, a great interview story ☐

C-010 found that the LLM often disagrees with the dataset's labels, and that the dataset is often the one that's wrong. You can measure this yourself as a Hindi speaker:

1. Open `outputs\relabel_sheet.csv` (Excel or VS Code). It has 40 test tweets and **no answers shown**.
2. Fill the `your_label` column with `negative`, `neutral` or `positive`, using your own judgement. Save.
3. In the VS Code terminal run: `.venv\Scripts\python scripts\relabel.py score`

It reports how often you agree with the dataset vs with the LLM on the tweets where they disagree. Half the rows are controls, so you can't tell which tweets are the disagreements.

---

## While you wait: read the interview-prep draft (optional)

[INTERVIEW_PREP.md](INTERVIEW_PREP.md) already has the pitches, ~19 likely questions with answers, and STAR stories. Only the ⟨TBD⟩ spots need the fine-tuning numbers. Ask Claude to explain any answer you can't yet say in your own words.

## What happens after the run (Claude, or you via [SOLO_TODO.md](SOLO_TODO.md) Parts 5–9)
- Sweep analysis: XLM-R vs MuRIL, emoji→text ablation, learning-rate sensitivity, encoder vs QLoRA.
- Final comparison tables and the headline, in-domain and out-of-domain (YouTube).
- 10 explained errors and the robustness analysis.
- README final pass, licence file, fill the ⟨TBD⟩ spots in `docs/INTERVIEW_PREP.md`.
- Close Phases 3–5 in the docs.
