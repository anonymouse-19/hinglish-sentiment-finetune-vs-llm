# Solo to-do list: finish the project without Claude

> Written **2026-10-26** against commit `398de36` (Phases 1–2 done; Phases 3–5 code complete, waiting on the Colab run).
> Work through it top to bottom and tick each ☐ as you go. Every command is for the **VS Code terminal (PowerShell)**, run
> from the project folder. Times are rough estimates.
>
> Golden rules:
> - **Never** paste a token into code, the notebook, a commit or a doc. Tokens go only in `.env` (git-ignored) and Colab secrets.
> - **Never** commit `data/` or `outputs/` (they quote tweets; no licence, D-004). `.gitignore` already blocks them; don't force-add.
> - If a number in a doc disagrees with `reports/evaluation/results.md`, **the report wins**. Copy numbers from it, don't retype from memory.

---

## Part 0. Check your laptop setup (10 min)

☐ **0.1 Open the project** in VS Code, then open a terminal (`` Ctrl+` ``).

☐ **0.2 Confirm the virtual environment works:**
```powershell
.venv\Scripts\python --version
.venv\Scripts\python -m pytest -q
```
You should see Python 3.13 and **70 passed**. If `pytest` isn't found: `.venv\Scripts\python -m pip install -r requirements.txt; .venv\Scripts\python -m pip install -e .`

☐ **0.3 Confirm git is clean and in sync with GitHub:**
```powershell
git status -sb
```
Expect `## main...origin/main` and nothing else. If it says `ahead`, run `git push`. **The Colab notebook clones from GitHub, so
whatever isn't pushed doesn't exist for Colab.**

☐ **0.4 Check `.env`** (open it in VS Code). It must contain `GROQ_API_KEY=gsk_...`. You'll add `HF_TOKEN` in Part 1.

---

## Part 1. Accounts and keys (10 min)

☐ **1.1 Hugging Face write token**
1. Log in / sign up at https://huggingface.co/join.
2. https://huggingface.co/settings/tokens → **Create new token** → Token type **Write** → name `hinglish-sentiment` → **Create token**.
3. Copy it (starts with `hf_`, shown only once).
4. Add a line to `.env` and save: `HF_TOKEN=hf_xxxxxxxx`

☐ **1.2 Weights & Biases key** (recommended)
1. Sign up at https://wandb.ai (Personal, free).
2. Copy the key from https://wandb.ai/authorize. (Optionally add `WANDB_API_KEY=...` to `.env` too; not needed locally.)

☐ **1.3 Groq key** — already in `.env`. Check it still works:
```powershell
.venv\Scripts\python scripts\check_llm.py
```
It should list models and print one test answer. If you get a 401, make a new key at https://console.groq.com/keys and replace the line in `.env`.

---

## Part 2. The Colab run (≈ 2 h, mostly waiting)

☐ **2.1 Open the notebook:** https://colab.research.google.com → **File → Open notebook** → **GitHub** tab →
type `anonymouse-19/hinglish-sentiment-finetune-vs-llm` → Enter → click `notebooks/phase3_finetune_colab.ipynb`.

☐ **2.2 GPU:** **Runtime → Change runtime type → T4 GPU → Save.**

☐ **2.3 Secrets:** click the 🔑 icon (left sidebar) → **+ Add new secret** three times, switch **Notebook access** ON for each:

| Name | Value |
|---|---|
| `WANDB_API_KEY` | from 1.2 |
| `HF_TOKEN` | from 1.1 |
| `GROQ_API_KEY` | the value after `GROQ_API_KEY=` in your `.env` |

☐ **2.4 Runtime → Run all.** Accept the Google Drive prompt (choose your account, allow all).

☐ **2.5 Watch it.** What each section should print:

| Section | Expected output | ~Time |
|---|---|---|
| 1 | `Tesla T4, 15360 MiB`, then `code version: 398de36 ...` (or newer) | 1 min |
| 2 | torch / transformers / peft versions and `CUDA: True` | 3 min |
| 3 | all three secrets `found` | – |
| 4 | data pipeline finishes; train 13,409 / val 2,367 / test 3,000 | 1 min |
| 5 | smoke run finishes without a red error | 1 min |
| 6 | 9 runs, each ending `-> val macro-F1 0.xxxx`; then a sweep table and test metrics | ~40 min |
| 7 | 4 QLoRA runs, same format | ~1 h |
| 8 | `results.md` tables with **encoder** and **decoder** rows added, plus 3 figures | 2 min |
| 9 | `best run: ... -> <you>/hinglish-sentiment-xlmr` (or `-muril`), then upload progress | 3 min |
| 10 | Space deployed, "takes ~5 min to build" | 1 min |
| 11 | browser downloads `finetune_results.zip` | – |

Live curves: https://wandb.ai → project **hinglish-sentiment-finetune**. **Keep the Colab tab open** (free Colab disconnects idle tabs after ~90 min of no interaction; clicking in the page now and then helps).

☐ **2.6 Write down** (you'll need these later; a text file in `outputs\` is fine):
- the **best encoder run name** (section 9 prints it, also in Drive `hinglish-ft/outputs/encoder/BEST_RUN.txt`),
- your **model repo id** (e.g. `yourname/hinglish-sentiment-xlmr`) and **Space id** (`yourname/hinglish-sentiment-demo`),
- anything odd you noticed (OOM errors, a run that scored much lower, re-runs). These become CHALLENGES_LOG entries (Part 8).

### If something goes wrong

| Symptom | Fix |
|---|---|
| `AssertionError: No GPU` in section 2 | Runtime → Change runtime type → T4 → **Run all** again |
| Disconnected / "cannot connect to a GPU" | Free-tier limit. Come back in a few hours or next day → **Run all**. Finished runs print `already done` and are skipped (results live on Drive) |
| `CUDA out of memory` in section 7 (QLoRA) | Edit `configs/finetune.yaml` on your laptop: under `decoder.train` set `batch_size: 8`. Commit + push (`git commit -am "QLoRA batch 8 for T4 memory"; git push`), then in Colab **Runtime → Restart session → Run all** (section 1 pulls the new code). Note it as a challenge |
| Any other error in section 7 | Click the first cell of section 8 → **Runtime → Run after**. You still get encoder evaluation + publishing; the decoder row will just be missing. Note the error text |
| Section 9 says `HF_TOKEN not set` | Secret name typo or Notebook access off. Fix in 🔑, then run sections 9–11 individually (click cell → Shift+Enter) |
| Section 9: `403 Forbidden` | Token is **Read** not **Write**. Make a new Write token (1.1), update the secret and `.env`, re-run section 9 |
| Section 10 fails | Skip it; you can deploy the Space from your laptop in Part 4.4 |
| Zip didn't download | Files icon (left sidebar) → `/content/finetune_results.zip` → ⋮ → Download. Or copy it from Drive by re-running section 11 |
| Unsure what an error means | Copy the last 20 lines into a web search with the library name (transformers / bitsandbytes / peft) |

---

## Part 3. Bring results home and verify (15 min)

☐ **3.1 Unzip** `finetune_results.zip` **into the project folder** (right-click → Extract All → choose the project folder; say **Replace** when asked). It merges into `reports\` and `outputs\`.

☐ **3.2 Check the files exist:**
```powershell
ls reports\finetune\encoder, reports\finetune\decoder
```
Each should have `sweep.csv`, `metrics.json`, `predictions_val.parquet`, `predictions_test.parquet`, `predictions_test_youtube.parquet`. (No `decoder` folder = section 7 failed; carry on with encoder only and say so honestly in the docs.)

☐ **3.3 Re-run the evaluation locally** (same code as Colab; confirms results reproduce on your machine):
```powershell
.venv\Scripts\python scripts\evaluate_all.py
.venv\Scripts\python scripts\error_analysis.py --model encoder --n 30
.venv\Scripts\python scripts\error_analysis.py --model decoder --n 30
.venv\Scripts\python -m pytest -q
```
Then open `reports\evaluation\results.md`. It now has encoder/decoder rows in tables A–D and, at the bottom, two **Headline** lines. Those are your project's main numbers.

☐ **3.4 Sanity-check the numbers** (anything outside these ranges deserves a second look before you publish claims):
- Encoder test macro-F1 roughly **0.68–0.75** (SemEval best was 0.750). Above 0.78 → suspect leakage; below 0.60 → a training run went wrong.
- Encoder p50 latency: tens of ms on the T4. Cost/1k: fractions of a cent.
- `n_invalid` is 0 for every non-LLM model.
- Paired test rows (table D) exist for: encoder vs TF-IDF, encoder vs LLM few-shot, on both test and YouTube.

☐ **3.5 Commit the results:**
```powershell
git add reports
git status            # check: only reports/ files, nothing from data/ or outputs/
git commit -m "Phase 3 results: encoder + QLoRA sweeps, final evaluation"
git push
```

---

## Part 4. Publishing checks (20 min)

☐ **4.1 Model page:** open `https://huggingface.co/<your-repo-id>`. The card should show results with CIs, a language-mix table and usage code.

☐ **4.2 Refresh the card** with the final local evaluation (adds the LLM comparison):
```powershell
.venv\Scripts\python scripts\publish.py card --repo-id <your-repo-id> --dry-run
```
Open `outputs\publish_preview\README.md` and read it. If it looks right, run the same command **without** `--dry-run`.

☐ **4.3 Test the model from code** (proves strangers can use it):
```powershell
.venv\Scripts\python -c "from transformers import pipeline; p=pipeline('text-classification', model='<your-repo-id>'); print(p('yaar ye movie toh ekdum bakwas thi'))"
```
Expect `negative` with a high score. (If the best run was `muril_demoji`, the demo applies demojize for you; plain `pipeline` doesn't. That's fine for this check.)

☐ **4.4 Demo Space:** open `https://huggingface.co/spaces/<your-space-id>`. Wait for **Running** (first build ~5 min; the **Logs** tab shows progress).
- Try 4–5 inputs: one clearly negative, one positive, one neutral news-like, one mostly English, one emoji-heavy.
- Both halves should answer: fine-tuned (probabilities + ms) and LLM (label, ms, tokens, $).
- **If section 10 failed or the Space shows an error**, redeploy from the laptop (the cost number comes from `summary.json`):
  ```powershell
  .venv\Scripts\python -c "import json;d=json.load(open('reports/evaluation/summary.json',encoding='utf-8'));print(next(r['usd_per_1k'] for r in d['tables']['full_test']['rows'] if r['key']=='encoder'))"
  .venv\Scripts\python scripts\publish.py space --space-id <your-space-id> --model-id <your-repo-id> --ft-usd-per-1k <number printed above>
  ```
- LLM half says error/no key → Space **Settings → Variables and secrets** → add secret `GROQ_API_KEY` → **Restart Space**.

☐ **4.5 Take 2 screenshots** of the demo (Win + Shift + S) for the README later. Save them as `reports\figures\demo.png` (and optionally `demo2.png`).

---

## Part 5. Analyse the sweeps (45 min, this is your own thinking)

The docs promise four analyses. Get the raw tables:
```powershell
.venv\Scripts\python -c "import pandas as pd; [print(f, '\n', pd.read_csv(f'reports/finetune/{f}/sweep.csv').to_string(), '\n') for f in ('encoder','decoder')]"
```
(First time, look at the column names it prints and use them below.)

☐ **5.1 XLM-R vs MuRIL.** Best val macro-F1 of `xlmr_*` runs vs best of `muril_*` runs. Write one sentence: which won, by how much, and is the gap bigger than ~0.01 (smaller than that is noise on 2,367 val tweets).

☐ **5.2 Emoji ablation (D-029, C-011).** Best `muril_*` vs best `muril_demoji_*`. Did rewriting emoji as text help MuRIL? This fills the ⟨TBD: result⟩ in INTERVIEW_PREP line ~95.

☐ **5.3 Learning-rate sensitivity.** For each model, the spread between its best and worst learning rate. Note whether 5e-5 diverged (very low F1) or the best epoch was always 1 (= overfits fast, labels are noisy).

☐ **5.4 Encoder vs QLoRA.** Best encoder vs best Qwen3 run on: test macro-F1, YouTube macro-F1, p50 latency, cost/1k, size, train minutes. Typical story: the 1.7B decoder is ~10× bigger and slower; is it more robust on YouTube? That's the interesting question.

☐ **5.5 In-domain vs OOD picture.** From table C: does the fine-tuned encoder keep TF-IDF's in-domain strength **and** recover on YouTube (TF-IDF 0.415, LLM 0.737)? Where between them does it land? This is the project's main finding; write it in 2–3 sentences.

Put the results of 5.1–5.5 into a new section **"Phase 3 (fine-tuning): final"** in `docs/PROJECT_CONTEXT.md` §9, in the same style as the Phase 2 block (a table, then bullets).

---

## Part 6. Error analysis: 10 explained examples (45 min)

☐ **6.1** Open `outputs\error_analysis_encoder.md`. It lists the model's most confident mistakes per true→predicted cell, with what every other model said.

☐ **6.2** Pick **10** that teach something, covering several cells. Good categories:
- the gold label looks wrong (you, a Hindi speaker, agree with the model) → label noise, links to C-008/C-010;
- sarcasm;
- truncated tweet (`…`) where the sentiment was cut off;
- neutral-vs-polarity convention (emotional tweet labelled neutral);
- a case the LLM got right and the encoder didn't, and vice versa.

☐ **6.3** Write them up in `docs/PROJECT_CONTEXT.md` §9 (or a short new `docs/ERROR_ANALYSIS.md`): for each, **paraphrase** the tweet (don't paste it verbatim; licence), gold label, prediction, other models, one-line explanation. End with the counts: how many were label noise vs genuine model error.

---

## Part 7. Optional: blind relabelling (20 min, great interview story)

☐ **7.1** Open `outputs\relabel_sheet.csv`, fill the `your_label` column (`negative` / `neutral` / `positive`) with your own judgement. Don't look anything up. Save as CSV.

☐ **7.2** Score it:
```powershell
.venv\Scripts\python scripts\relabel.py score
```
☐ **7.3** Record the result (how often you agree with gold vs the LLM where they disagree) as a bullet in PROJECT_CONTEXT §9 and as a CHALLENGES_LOG/INTERVIEW_PREP note. Mention n=40 is small.

---

## Part 8. Update every doc (1.5 h)

Copy numbers from `reports\evaluation\results.md` and `summary.json`. Keep the existing writing style: short sentences, numbers with CIs, links to D-/C- entries.

☐ **8.1 `README.md`**
- Status line: Phases 3–5 ✅.
- Results table: fill the **XLM-R / MuRIL** and **Qwen3 QLoRA** rows (test macro-F1 [CI], p50 latency, cost/1k, size). Use the best run's real name (e.g. "MuRIL + emoji→text, lr 3e-5").
- Replace the **Key insight** placeholder with the two Headline lines from `results.md` (in-domain and YouTube).
- Replace the "Where the fine-tuned models land is the question Phase 3 answers" sentence with your Part 5.5 finding.
- Add links: model `https://huggingface.co/<repo-id>`, demo `https://huggingface.co/spaces/<space-id>`, W&B project (make it public: W&B project → Settings → Visibility → Public), plus the demo screenshot (`![Demo](reports/figures/demo.png)`).
- Reproduce section: `pytest -q  # 62 tests` → **70** tests; add a "Phase 3: run `notebooks/phase3_finetune_colab.ipynb` on Colab" line.
- Add a **License** section (see 8.7).

☐ **8.2 `docs/INTERVIEW_PREP.md`** — fill every ⟨TBD⟩ (search the file for `TBD`):

| Line (approx.) | Fill with |
|---|---|
| 3 | change "Draft" note to "Final, <date>" |
| 15–16 (30-s pitch) | where the fine-tuned model lands in-domain and on YouTube; cost ratio from the headline |
| 38 (2-min pitch, result) | best model name, test F1, YouTube F1, p50 latency, cost/1k |
| 39 (takeaway) | the headline sentence: X% of LLM F1 at Y× lower cost; OOD keeps Z% |
| 95 (emoji lesson) | result of Part 5.2 |
| 100 (build vs buy) | cost ratio and latency ratio (LLM p50 ÷ encoder p50) |

Then read the whole file aloud once. Any answer you can't say in your own words → reread the linked D-/C- entry.

☐ **8.3 `docs/PROJECT_CONTEXT.md`**
- Top banner: "Last updated" date; phases 3–5 done.
- §6 tech stack: "*Planned:* Gradio + HF Spaces" → no longer planned.
- §9: the Phase 3 block (Part 5), error-analysis summary (Part 6), relabel result (Part 7).
- §10: mark Phases 3, 4, 5 ✅ with dates; remove "Open questions" about the licence.

☐ **8.4 `docs/OWNER_ACTIONS.md`** — tick steps 1–5 (☐ → ☑), update the status table, update "Last updated".

☐ **8.5 `docs/DECISION_LOG.md`** — add entries in the existing ADR format, numbering from **D-039**. Likely ones:
- D-039: which encoder was selected and why (val-selected; test touched once).
- D-040: the licence choice.
- Anything you changed during the run (e.g. QLoRA batch size).

☐ **8.6 `docs/CHALLENGES_LOG.md`** — add real problems from the run, numbering from **C-017**, in the same STAR format (Situation, Task, Action, Result). Mark owner-encountered ones with *(owner)* as in C-014. Don't invent challenges; if the run was smooth, say nothing.

☐ **8.7 LICENSE** (decision: MIT suggested)
1. GitHub repo page → **Add file → Create new file** → name it `LICENSE` → click **Choose a license template** → MIT → fill your name and year 2026 → **Commit changes**.
2. Back in VS Code: `git pull`.
3. In `pyproject.toml` under `[project]` add `license = "MIT"`.
4. README License section: "Code: MIT. The SentiMix data is **not** redistributed and has its own terms; the YouTube data is CC-BY-4.0."

☐ **8.8 `docs/DATA_CARD.md`** — only if something about the data changed (normally nothing). Skim it once.

---

## Part 9. Final checks and push (20 min)

☐ **9.1** Tests: `.venv\Scripts\python -m pytest -q` → all pass.

☐ **9.2** Leftover placeholders:
```powershell
git grep -n "TBD\|⟨\|X%\|Y×" -- README.md docs
```
Should return nothing (except intentional uses).

☐ **9.3** Secrets didn't leak:
```powershell
git grep -n "hf_\|gsk_\|wandb_" -- . ":!*.ipynb"
```
Must show no real tokens. Also `git status` must not list `.env`.

☐ **9.4** Commit and push:
```powershell
git add -A
git status          # read the list! nothing from data/, outputs/, .env
git commit -m "Close Phases 3-5: final results, docs, interview prep, licence"
git push
```

☐ **9.5** Look at the repo on GitHub as a stranger: README renders, images show, links to model / Space / W&B work, `results.md` opens.

☐ **9.6** Pin the repo on your GitHub profile (profile → Customize your pins) and add the project to your résumé/LinkedIn with the headline sentence and the demo link.

---

## Part 10. Interview practice (ongoing)

☐ Say the 30-second pitch out loud 5 times until it's natural.
☐ For each question in INTERVIEW_PREP, answer it aloud without reading, then check.
☐ Be able to draw the §5 architecture diagram from memory.
☐ Know these by heart: test F1 of each model, the YouTube flip (0.415 vs 0.737), the cost ratio, the 271 leaked tweets, the 35% conflicting-label rate.

---

### Quick reference

| Thing | Where |
|---|---|
| Final numbers | `reports/evaluation/results.md`, `summary.json` |
| Sweep tables | `reports/finetune/{encoder,decoder}/sweep.csv` |
| Models on Drive | `MyDrive/hinglish-ft/outputs/` (safe to delete after publishing, ~1–3 GB) |
| Re-run evaluation | `.venv\Scripts\python scripts\evaluate_all.py` |
| Rebuild notebook after editing `build_notebook.py` | `.venv\Scripts\python notebooks\build_notebook.py`, then commit + push |
