# Decision Log (ADR style)

Every non-trivial decision, including the ones the project owner made when answering questions. Format per entry: context → options (≥ 2) → decision → reasoning → trade-offs accepted → what would make us revisit.

| ID | Phase | Decision (one line) |
|---|---|---|
| D-001 | 1 | Task: Hinglish sentiment on SentiMix *(owner's choice)* |
| D-002 | 1 | LLM provider for baselines: Groq *(owner's choice)* |
| D-003 | 1 | Data source: HF mirror `RTT1/SentiMix` pinned to a commit |
| D-004 | 1 | Don't redistribute the data; rebuild it from source with a script |
| D-005 | 1 | Keep the official test set untouched; pool train+dev and re-split |
| D-006 | 1 | Stratify train/val on label × language-mix bucket, 15% val, seed 42 |
| D-007 | 1 | Headline metric: macro-F1 (plus weighted-F1 and accuracy) |
| D-008 | 1 | Clean at token level; mask `@user` / `http`; keep emoji and hashtags |
| D-009 | 1 | Repair mojibake first: `ftfy` plus a strict cp1252→UTF-8 re-decode |
| D-010 | 1 | Dedupe on content with char-n-gram TF-IDF cosine ≥ 0.9, clustered |
| D-011 | 1 | Duplicate resolution: drop test-leaking copies; majority vote, else drop |
| D-012 | 1 | Non-Hinglish filter: ≥ 3 accented letters (train/val only), not `langdetect` |
| D-013 | 1 | Robustness buckets from word-level language tags at an 80% threshold |
| D-014 | 1 | Add YouTube comments as an out-of-domain test set |
| D-015 | 1 | Project venv + pinned requirements + `src/` package with `pyproject.toml` |
| D-016 | 2 | LLM: `openai/gpt-oss-120b` on Groq, reasoning effort "low", temperature 0 |
| D-017 | 2 | TF-IDF baseline: word 1–2 + char 2–5-grams, LogReg; grid selected on val; refit on train only |
| D-018 | 2 | LLM output: one word + strict parser; prompts developed on a val subset, test run once |
| D-019 | 2 | Unparseable LLM answers count as wrong predictions, never dropped |
| D-020 | 2 | Few-shot: 9 fixed class-balanced train examples, filtered for label noise (out-of-fold p ≥ 0.6) |
| D-021 | 2 | LLM latency = client-side wall-clock of the successful call; sequential requests; disk cache |
| D-022 | 2 | LLM cost = billed tokens (incl. reasoning, cached-input discount) × dated prices in config |
| D-023 | 2 | LLM evaluation on the free tier with stratified subsets: val 200, test 600, YouTube 300 *(owner's choice)* |
| D-024 | 2 | Report a 95% bootstrap confidence interval for every macro-F1 |
| D-025 | 2 | Freeze the prompts after one val round (no further prompt tuning) |
| D-026 | 3 | Option A encoders: XLM-R base and MuRIL base (pinned) |
| D-027 | 3 | Experiment tracking: Weights & Biases *(owner's choice)* |
| D-028 | 3 | Code reaches Colab through a GitHub repo *(owner's choice)* |
| D-029 | 3 | MuRIL also run with emoji rewritten as text (demojize), as a sweep variant |
| D-030 | 3 | Option B decoder: Qwen3-1.7B (Apache-2.0) |
| D-031 | 3 | Decoder classifies through a classification head + QLoRA, not by generating the label |
| D-032 | 3 | Sweep: 9 encoder + 4 decoder runs, early stopping on val, winner reloaded and tested once |
| D-033 | 3 | Serving measurements: batch-1 p50 incl. tokenization on T4, throughput, three size measures |
| D-034 | 4 | Self-hosted cost/1k = AWS on-demand $/h ÷ measured batched throughput (g4dn.xlarge, c7i.large) |
| D-035 | 4 | Headline on identical tweet IDs for every model, with a paired bootstrap of the macro-F1 difference |
| D-036 | 4 | Error analysis: most-confident mistakes in every confusion cell; text stays out of git |
| D-037 | 5 | Shipping: publish the best encoder from Colab; Space = encoder on CPU + Groq few-shot, code installed from GitHub |

---

### D-001 — Task: Hinglish sentiment on SentiMix
- **Date / phase:** 2026-09-29 · Phase 1 · *decided by project owner*
- **Context:** The brief allowed either Hinglish sentiment/intent classification or resume-to-JD match scoring. The choice determines the data, the labels and the story.
- **Options considered:**
  1. **Hinglish sentiment, SentiMix** (SemEval-2020 Task 9): 20k real tweets, 3 classes, word-level language tags; no explicit licence.
  2. **Hinglish intent, Hinglish-TOP** (Google): Apache-2.0, 10.9k queries, 64 intents (22 with < 30 examples); the text is human *translations*, not natural Hinglish.
  3. **Resume–JD match**: MIT-licensed Djinni resumes and JDs with LLM-generated labels and owner spot-checks. The public ready-made set (`cnamuangtoun/resume-job-description-fit`) has only 642 unique resumes and 280 unique JDs across 8k pairs (leakage) and no licence.
- **Decision:** Option 1.
- **Reasoning:** It is real, naturally occurring code-mixed text. It is a recognised benchmark with a public leaderboard (best Hinglish F1 = 75.0, 61 teams), which gives an external anchor. The word-level language tags give the "code-mixed vs English vs Hindi" robustness split for free. And the LLM baseline isn't graded against LLM-written labels, which would be the circularity problem in option 3.
- **Trade-offs accepted:** No explicit licence (handled by D-004). Noisy social-media labels. The domain is dated (around 2019) and political.
- **Revisit if:** the licence becomes a blocker for publishing the model, or label noise makes model differences statistically indistinguishable.

### D-002 — LLM provider: Groq
- **Date / phase:** 2026-09-29 · Phase 1 (used in Phase 2) · *decided by project owner*
- **Context:** Zero-/few-shot baselines need a large LLM behind an API. The provider sets both the cost figures and the latency figures.
- **Options considered:** Anthropic (Claude), OpenAI (GPT), Google Gemini (free tier), **Groq** (hosts large open-weight models on custom LPU hardware, OpenAI-compatible API).
- **Decision:** Groq. The client will still be provider-agnostic (OpenAI-compatible `base_url` + model name in config).
- **Reasoning:** Owner's choice. Technically convenient: the OpenAI-compatible API means one client works for Groq and others, and it serves large open models (70B-class) cheaply, with a free tier.
- **Trade-offs accepted:** Groq's inference is unusually fast, so the *latency* gap between the LLM and the small model will look smaller than it would with a typical GPU-hosted API. That caveat must be stated next to the latency column. Free-tier rate limits may force evaluating the LLM on a stratified subset of test. The model catalogue changes often, so the exact model IDs are pinned in config and dated.
- **Revisit if:** rate limits make a full-test evaluation impractical, the chosen model is deprecated, or we want a second provider for comparison.

### D-003 — Data source: pinned HF mirror of SentiMix
- **Date / phase:** 2026-09-29 · Phase 1
- **Context:** The original SentiMix release was distributed via CodaLab for the competition. We need a source that works identically on a laptop and on Colab.
- **Options considered:**
  1. `RTT1/SentiMix` HF mirror (raw CoNLL train/dev/test + test labels).
  2. `Yaxin/SemEval2020Task9CodeSwitch`: a loading script, but it only exposes Spanglish.
  3. Manual download from the competition page (needs a login; not scriptable on Colab).
- **Decision:** Option 1, pinned to commit `205f0391…` via `hf_hub_download(revision=...)`.
- **Reasoning:** It is scriptable and cacheable. Its integrity was verified: 14k/3k/3k tweets, all 3,000 test labels join by uid, no uid overlaps, and the total of 20k matches the overview paper. Pinning the commit means the mirror owner can't silently change our data.
- **Trade-offs accepted:** We depend on a third-party mirror; if it disappears, the pinned files still need a new home. Its `openrail` licence tag is not authoritative.
- **Revisit if:** the mirror is deleted or altered, or an official HF release appears.

### D-004 — Don't redistribute the data
- **Date / phase:** 2026-09-29 · Phase 1
- **Context:** The SentiMix release has no explicit licence, and tweets are also subject to X/Twitter terms.
- **Options considered:** (a) publish the cleaned splits as an HF dataset; (b) commit the parquet files to the GitHub repo; (c) **publish only code; users rebuild the data with `scripts/prepare_data.py`**.
- **Decision:** (c). `data/` is git-ignored. Only models, metrics and aggregate statistics are published.
- **Reasoning:** This is the lowest-risk route that is still fully reproducible, because the source revision is pinned. It is also common practice for tweet datasets.
- **Trade-offs accepted:** The first run needs internet access and about a minute. The Gradio demo can't ship dataset examples verbatim, so it will use owner-written example sentences instead.
- **Revisit if:** the organisers publish a licence.

### D-005 — Official test set untouched; train+dev pooled and re-split
- **Date / phase:** 2026-09-29 · Phase 1
- **Context:** The brief asks for a stratified train/val/test split. SentiMix already ships official splits, and our cleaning changes train/dev (dedupe, filters).
- **Options considered:**
  1. Keep all three official splits as-is.
  2. Pool all 20k tweets and make a fresh stratified 70/15/15 split.
  3. **Keep the official test set exactly as released; pool the cleaned train+dev and re-split it stratified into train/val.**
- **Decision:** Option 3. Test rows are never dropped, only *flagged* (`no_content`: 10, `is_foreign`: 1, `test_internal_dup`: 34), so a "clean-test" sensitivity check is still possible.
- **Reasoning:** Keeping test identical to the release keeps our numbers comparable to the SemEval leaderboard (75.0 F1), which is the external credibility anchor. The official test set is already balanced by design (900/1,100/1,000). Re-splitting train+dev lets us stratify after cleaning.
- **Trade-offs accepted:** The test set keeps some known-noisy rows. Leakage is fixed by removing *train/dev* copies (D-011) rather than touching test.
- **Revisit if:** we stop caring about leaderboard comparability, e.g. if the owner prefers a fully re-split benchmark.

### D-006 — Stratify on label × language-mix bucket; 15% validation; seed 42
- **Date / phase:** 2026-09-29 · Phase 1
- **Context:** The validation set drives early stopping, hyperparameter sweeps and prompt design. It should look like test on every axis we report.
- **Options considered:** stratify on label only · **stratify on label × mix bucket** · random split.
- **Decision:** `train_test_split(stratify=label|mix_bucket, test_size=0.15, random_state=42)` → 13,409 train / 2,367 val.
- **Reasoning:** Language mix is confounded with sentiment (D-013), so stratifying on both keeps per-bucket validation numbers stable. 15% of about 15.7k gives around 2.4k validation tweets, close to the official dev size (3k), and leaves the most data for training.
- **Trade-offs accepted:** One fixed split. Variance across splits isn't measured; seed variance will be measured at training time instead.
- **Revisit if:** validation scores disagree with test by more than a couple of points, or we need k-fold CV for small-data experiments.

### D-007 — Headline metric: macro-F1
- **Date / phase:** 2026-09-29 · Phase 1 (applies to Phases 2–4)
- **Context:** A comparison table needs one headline number. The classes are mildly imbalanced (neutral 37.5%, negative 30%).
- **Options considered:** accuracy · weighted-F1 · **macro-F1**.
- **Decision:** Macro-F1 is the headline, as the brief also requires. Accuracy and weighted-F1 are reported alongside. The SemEval overview abstract just says "F1", so we report both averages to compare fairly with its 75.0.
- **Reasoning:** Macro-F1 weights each class equally, so a model can't score well by favouring the majority `neutral` class. It is also more robust to the YouTube set's different class prior.
- **Trade-offs accepted:** Macro-F1 is less intuitive to non-ML stakeholders than accuracy, so the README reports accuracy too.
- **Revisit if:** a stakeholder cares about one class specifically, e.g. negative-sentiment recall for moderation. Then per-class recall becomes the headline.

### D-008 — Token-level cleaning; mask mentions/URLs; keep emoji and hashtags
- **Date / phase:** 2026-09-29 · Phase 1
- **Context:** The original tokenizer split Twitter entities apart (`@ BTS _ army _ Fin`, `https // t . co / x`, `# videsh _ mantri`), and the word-level language tagger then labelled those fragments as Hin/Eng: about 28.8k handle tokens and at least 11k URL fragments.
- **Options considered:**
  1. No cleaning; let models cope with the raw text.
  2. Regex over the joined text.
  3. **A token-level state machine that keeps each token aligned with its language tag, plus a small regex safety net for tokens glued to neighbours.**
- **Decision:** Option 3. Mentions become `@user` and URLs become `http` (the TweetEval / cardiffnlp convention, which Twitter-trained models expect). Hashtags are re-joined and **kept** (e.g. `#JaiShriRam` carries stance). Emoji are **kept**, as strong sentiment cues. The leading `RT` is dropped.
- **Reasoning:** Staying at token level keeps language tags aligned, so language statistics exclude handles and URLs (option 2 would lose that alignment). Masking stops models from memorising user names (spurious features), and it is what made duplicate detection possible (D-010).
- **Trade-offs accepted:** Heuristics. The ≤ 15-character handle rule can swallow one real word after an `_`, and 2 split mentions remain (`@ @user` edge cases). Both are tested in `tests/test_clean.py`.
- **Revisit if:** error analysis shows masked mentions mattered, e.g. sentiment towards a named politician's handle.

### D-009 — Repair mojibake first: `ftfy` plus a strict cp1252→UTF-8 re-decode
- **Date / phase:** 2026-09-29 · Phase 1
- **Context:** 2,119 tweets (10.6%) contained UTF-8 text mis-decoded as cp1252 (`ðŸ˜…` = 😅, `â€¦` = …).
- **Options considered:** leave as-is · only hand-written `.encode("cp1252").decode("utf-8")` round-trips · only `ftfy.fix_text` · **both: a strict re-decode, then `ftfy`, then the strict re-decode again**.
- **Decision:** The combined approach, on every token and again on the joined text, before any other heuristic.
  - The strict re-decode only replaces a span if it decodes as valid UTF-8, so legitimate accents (café…, Señor, naïve) are untouched; this is unit-tested.
  - Residuals are counted with the classic mojibake signature (a Latin-1 letter followed by a cp1252 0x80–0xBF character) as well as `ftfy.badness.is_bad`: **0 remain**.
  - As a final check, all 86 remaining texts containing any Latin-1 letter were read by hand. All were genuine accents or the × sign.
- **Reasoning:**
  - `ftfy` handles the general and mixed cases well, but it is deliberately conservative. It left 4 tweets unfixed: short sequences like `ÄŸ` (ğ) and `ã…‹` (ㅋ).
  - It also straightens curly quotes, which destroyed one mojibake sequence (`Û”` → `Û"`) before it could be repaired. Hence the strict pass *before* `ftfy` as well as after.
  - Doing all this first mattered. Before repair, an accented-letter heuristic flagged 2,209 tweets as foreign; after repair, about 98 contained accented letters (CHALLENGES_LOG C-001).
- **Trade-offs accepted:** `ftfy` straightens curly quotes (’ becomes '). That is harmless, and the detokenizer re-joins `aren ' t` → `aren't`. The extra pass costs negligible time.
- **Revisit if:** we find text the repair changed incorrectly.

### D-010 — Content-based dedupe: char 3–5-gram TF-IDF cosine ≥ 0.9, clustered
- **Date / phase:** 2026-09-29 · Phase 1
- **Context:** Exact string matching found only 4 duplicate rows, but the copies differed only in their @handle and t.co link. See CHALLENGES_LOG C-002.
- **Options considered:**
  1. Exact match on the raw text: 4 rows found.
  2. Exact match on masked, punctuation-free content: 658 groups.
  3. MinHash-LSH near-duplicates: scales to millions, but approximate and an extra dependency.
  4. **Char 3–5-gram TF-IDF cosine similarity, chunked matrix product, then connected components.**
- **Decision:** Option 4 at threshold **0.9**. Exact duplicates are a subset (cosine = 1).
- **Reasoning:** At 20k tweets an exact, chunked similarity computation takes about 25 seconds, so approximation isn't needed. The threshold table below was measured during exploration, before the final filters; the final run gives the same 271 leakage rows. Character n-grams are robust to Hinglish spelling variation (nhi/nahi) and punctuation differences. Thresholds were compared empirically:

  | Threshold | Train/dev rows leaking into test | Largest cluster | Observation |
  |---|---|---|---|
  | 0.95 | 181 | 8 | misses clear copies that differ only in punctuation |
  | **0.90** | **271** | **8** | every sampled pair in 0.90–0.95 was the same tweet |
  | 0.85 | 322 | 15 | clusters start chaining templated-but-different tweets |

- **Trade-offs accepted:** Connected components are transitive, so A~B~C clusters A with C even if they aren't similar to each other. The largest cluster size (8) shows this stays contained. The 99 content-free tweets (`@user @user… http`) are excluded from the similarity search; they otherwise form one giant clique.
- **Revisit if:** the dataset grows past about 200k rows (switch to MinHash-LSH), or error analysis finds near-duplicate pairs we missed.

### D-011 — Duplicate resolution policy
- **Date / phase:** 2026-09-29 · Phase 1
- **Context:** 849 near-duplicate clusters, 323 with conflicting labels. 235 clusters mix test and train/dev tweets.
- **Options considered:** (a) keep everything; (b) drop every tweet in any conflicting cluster; (c) **cluster-level policy: drop train/dev members of any cluster that contains a test tweet; otherwise keep one row if the labels agree; majority-vote if there is a strict majority; drop the whole cluster on a tie**.
- **Decision:** (c). This removed 271 leakage rows, 403 redundant copies and 436 conflict rows.
- **Reasoning:** Leakage inflates test scores and must go. Keeping one copy of an agreeing cluster keeps the signal without over-weighting spammy templates. A strict majority is the best available estimate of the true label. A tie means we genuinely don't know, so the row is dropped.
- **Trade-offs accepted:** We lose 7.2% of train+dev (1,224 of 17,000 rows, all filters combined). Majority vote on clusters of 3 is still noisy.
- **Revisit if:** the model underfits and more data would help; relaxing the tie rule would recover a few hundred rows.

### D-012 — Non-Hinglish filter: ≥ 3 accented letters (Unicode decomposition), train/val only
- **Date / phase:** 2026-09-29 · Phase 1
- **Context:** The "Hinglish" corpus contains Czech, Turkish, Polish, Norwegian, Vietnamese, Spanish, Portuguese and French tweets. The very first training tweet is Czech.
- **Options considered:**
  1. Keep everything.
  2. **`langdetect`, tested on 300 mostly-Hindi training tweets: it said Indonesian 46%, Somali 17%, Swahili 9%, Estonian 8%, Tagalog 7%, and Hindi 0 times.** Unusable: it would delete most real Hinglish.
  3. **Count "accented letters", i.e. characters whose Unicode decomposition (NFD) is an ASCII letter plus combining diacritics (é, ş, ğ, å…).** Hinglish is typed in plain ASCII. Threshold 2 or 3.
- **Decision:** Option 3 with **threshold 3**, applied to train/val only (25 rows removed). Test rows are flagged, not removed (1 row).
- **Reasoning:** It is transparent and cheap. The threshold was chosen by **reading every tweet the rule removes**, not a sample:
  - Threshold 2 removed 42 tweets, 8 of which were not foreign: English or Urdu with a single foreign word (Whānau, Māori, Topkapı, "Í sey Skyr", an Ahmed Faraz couplet), plus the IAST-Hindi tweet. That's about 81% precision.
  - Threshold 3 removes 25, and the only non-foreign one is a Hindi tweet written in IAST transliteration. That's roughly 96% precision.
  - Wrongly removing English tweets hurts more than keeping a few foreign ones, because English tweets feed the smallest robustness bucket (`mostly_english`).
- **Trade-offs accepted:** About 8 short Spanish, Swedish or Turkish tweets with exactly 2 accents stay in train. The rule also misses foreign languages typed in plain ASCII (Indonesian, Tagalog).
- **Implementation note:** An earlier version counted characters whose Unicode *name* contains "LATIN". That silently matched the invisible TAG LATIN letters in flag emoji (🏴 England) and decorative small-caps fonts, and dropped real Hinglish tweets. The NFD definition fixes this, and regression tests cover it (CHALLENGES_LOG C-006).
- **Revisit if:** error analysis surfaces many plain-ASCII foreign tweets. Then try a code-mixing-aware language-ID model.

### D-013 — Robustness buckets: word-level language share, 80% threshold
- **Date / phase:** 2026-09-29 · Phase 1 (used in Phase 4)
- **Context:** The brief asks for performance on "code-mixed vs pure-English/Hindi subsets".
- **Options considered:**
  1. Sentence-level language ID (e.g. `langdetect`): fails on romanized Hindi (D-012).
  2. "Pure" buckets at a 90% threshold: only 112 mostly-English test tweets, 16 of them negative. Too small for stable per-class F1.
  3. **80% threshold on the dataset's own word-level tags**, equivalent to Code-Mixing Index ≤ 20. It gives test buckets of mostly-Hindi 1,189 / code-mixed 1,431 / mostly-English 370 (37 negative).
- **Decision:** Option 3, named `mostly_*` rather than `pure_*` to be honest about what it measures.
- **Reasoning:** Uses the human-provided word tags instead of an unreliable detector, and gives usable subset sizes.
- **Trade-offs accepted:** The tags themselves are noisy. The mostly-English test bucket is still small (370), so its confidence intervals will be wide and will be reported. **Language mix is confounded with sentiment** (mostly-Hindi is 44% negative; mostly-English is 56% positive), so per-bucket analysis must look at per-class recall too, not just aggregate F1.
- **Revisit if:** confidence intervals on the mostly-English bucket are too wide to conclude anything. Then consider bootstrapping or pooling val and test for the robustness analysis only.

### D-014 — Add an out-of-domain (OOD) test set: YouTube comments
- **Date / phase:** 2026-09-29 · Phase 1
- **Context:** A model fine-tuned on 2019 political tweets may not generalise. An LLM is usually more robust to domain shift, which is a key part of the cost/accuracy trade-off story.
- **Options considered:** no OOD set · Hinglish-TOP (different task and labels; unusable) · **`shae2977/hinglish-youtube-sentiments-dataset`** (CC-BY-4.0, 3,190 comments, same 3 labels).
- **Decision:** Use the YouTube set as test-only (3,168 after removing 22 duplicates).
- **Reasoning:** Same label space, different platform, topics and time. It is exactly the check that exposes brittle fine-tuned models.
- **Trade-offs accepted:** Annotation details are thin (number of annotators and agreement are not reported). There are only 23 videos. The label prior differs (45% negative).
- **Revisit if:** its labels look unreliable in error analysis.

### D-015 — Environment: project venv, pinned requirements, `src/` package
- **Date / phase:** 2026-09-29 · Phase 1
- **Context:** Code has to run locally (Windows, Python 3.13, no GPU) and on Colab (Linux, T4 GPU, Colab's Python).
- **Options considered:** use the global Python install · conda env · **`.venv` + pinned `requirements.txt` + `pyproject.toml` (`pip install -e .` on Colab)**.
- **Decision:** `.venv` locally, with exact versions pinned for the phase-1 numbers. Code lives in `src/hinglish_sentiment/`, and thin CLI scripts live in `scripts/`.
- **Reasoning:** Reproducible numbers, and the same code imports in notebooks and scripts. The `src/` layout avoids accidentally importing from the working directory.
- **Trade-offs accepted:** Colab's pre-installed torch and transformers versions differ from local ones, so training dependencies get their own pins in Phase 3.
- **Revisit if:** dependency conflicts on Colab. Then consider `uv` with a lockfile.

---

## Phase 2: Baselines

### D-016 — LLM: `openai/gpt-oss-120b` on Groq, reasoning effort "low", temperature 0
- **Date / phase:** 2026-10-05 · Phase 2
- **Context:** D-002 picked Groq, assuming it serves 70B-class open models on the free tier. On 2026-10-05 Groq's free tier lists only `openai/gpt-oss-120b`, `openai/gpt-oss-20b` and `qwen/qwen3.8-27b` (preview). Llama 3.3 70B is listed as an enterprise-tier model (see C-009).
- **Options considered:**
  1. **`openai/gpt-oss-120b`**: the largest free-tier model, a production model, $0.15 / $0.60 per 1M input/output tokens; a *reasoning* model.
  2. `openai/gpt-oss-20b`: cheaper ($0.075 / $0.30) and faster, but it is "a large LLM" only loosely.
  3. `qwen/qwen3.8-27b`: preview status (may disappear without notice) and about 5× pricier on output ($0.80 / $4.00).
  4. `llama-3.3-70b-versatile`: a non-reasoning 70B model, but not on the free tier.
- **Decision:** Option 1, with `reasoning_effort: low` and `temperature: 0`. All of this lives in `configs/llm.yaml`, so swapping the model is a config change.
- **Reasoning:** The project needs the strongest "buy" option that is realistically available. 120B parameters is about 400–1,000× the size of the models we'll fine-tune, which makes the comparison meaningful. It is a production model (stable ID). "Low" reasoning effort keeps the hidden reasoning tokens (which are billed) and the latency down, which is how one would deploy a classifier.
- **Trade-offs accepted:** Reasoning tokens are invisible but billed, so cost per request is higher than the visible one-word answer suggests; we log them separately. Temperature 0 does not guarantee identical outputs on every call, but the disk cache makes *our* reported numbers reproducible.
- **Revisit if:** the model is deprecated, or results suggest "medium" reasoning would change the conclusion. That could be tested as an ablation on val. A second, cheaper LLM (gpt-oss-20b) would also make a useful extra row.

### D-017 — TF-IDF baseline: word + char n-grams, logistic regression, selected on val
- **Date / phase:** 2026-10-05 · Phase 2
- **Context:** We need a cheap, classical lower bound that any fine-tuned model must beat.
- **Options considered:**
  - Features: word 1–2-grams only · char 2–5-grams (`char_wb`) only · **both (FeatureUnion)**.
  - Classifier: **logistic regression** · linear SVM (no calibrated probabilities) · Naive Bayes (usually weaker on short noisy text).
  - Selection: cross-validation on train · **a single held-out val split** (the same protocol the fine-tuned models will use).
  - Final fit: on train+val · **on train only**.
- **Decision:** Grid search over feature set × C ∈ {0.1, 0.3, 1, 3, 10} × class weight ∈ {none, balanced} (30 configs), selected on val macro-F1. The winner is refit on train only and scores test/YouTube once. The winner: word+char, C = 0.3, no class weighting (val 0.642).
- **Reasoning:**
  - Char n-grams handle Hinglish's free spelling ("nahi / nhi / nahin") and capture emoji; word n-grams capture phrases. The sweep confirms it: char-only 0.640 > word-only 0.614, and both 0.642.
  - Logistic regression gives probabilities, which we reuse elsewhere (D-020).
  - Training on train only keeps the protocol identical to the fine-tuned models, which need val for early stopping.
- **Trade-offs accepted:** About 15% fewer training examples than a train+val refit. Single-split selection is noisier than CV; the val CI is about ±0.02, larger than the gap between the top few configs.
- **Revisit if:** we need the strongest possible classical number (then refit on train+val, or use CV).

### D-018 — LLM output format: one word plus a strict parser; prompt developed on val, test run once
- **Date / phase:** 2026-10-05 · Phase 2
- **Context:** LLM answers are free text and have to become a label id. Prompt tweaking on the test set would be test-set overfitting.
- **Options considered:**
  1. **Ask for exactly one word; parse with a whole-word regex**: invalid if no label, or more than one distinct label, is named.
  2. JSON mode or structured outputs (`{"label": ...}`): guaranteed format, but more output tokens, and support varies by model and provider.
  3. Score the three label words with log-probabilities: elegant, but not reliably exposed for reasoning models on Groq.
- **Decision:** Option 1. Prompt changes are evaluated on a fixed 300-tweet stratified val subset. Test and YouTube are run once, with the final prompt.
- **Reasoning:** Cheapest (1–2 visible output tokens) and portable to any OpenAI-compatible provider. Strictness means sloppy answers are counted, not guessed (D-019).
- **Trade-offs accepted:** A small number of answers may be unparseable. We report the count.
- **Revisit if:** more than about 1% of answers are invalid. Then switch to structured outputs.

### D-019 — Unparseable LLM answers count as wrong
- **Date / phase:** 2026-10-05 · Phase 2
- **Context:** What to do when the LLM answers "mixed", refuses, or runs out of tokens.
- **Options considered:** drop them from scoring · map them to "neutral" · **score them as errors (predicted label = "invalid")**.
- **Decision:** Score them as errors, reported in an extra "invalid" column of the confusion matrix and as `n_invalid`.
- **Reasoning:** In production, an unusable answer is a failure. Dropping hard examples would flatter the LLM, and mapping them to neutral would hide the failure inside a real class.
- **Trade-offs accepted:** None meaningful; it is the conservative choice.
- **Revisit if:** never for the headline. A "parse-repaired" score could be shown as a secondary number.

### D-020 — Few-shot: 9 fixed, class-balanced train examples, filtered for label noise
- **Date / phase:** 2026-10-05 · Phase 2
- **Context:** In the first dry run, randomly drawn train examples included obvious mislabels, e.g. an insulting tweet labelled *neutral* (see C-008). Phase 1 found 35% of duplicate groups carry conflicting labels.
- **Options considered:**
  1. Random class-balanced examples from train.
  2. **Random class-balanced examples from train, after removing likely mislabels**: drop tweets whose out-of-fold TF-IDF probability of their own label is < 0.6.
  3. Hand-picked examples (subjective, hard to defend as fair, unreproducible).
  4. Dynamic kNN few-shot (retrieve the most similar train tweets per query): usually stronger, but the prompt differs per request, which defeats prompt caching and raises cost.
- **Decision:** Option 2: 3 per class = 9 examples (seed 42), interleaved neg/neu/pos, placed in the system prompt. Only the IDs are saved, in `configs/few_shot_ids.yaml` (no tweet text is committed; D-004).
- **Reasoning:** Fair to the LLM: it isn't taught wrong labels. The selection is automatic, reproducible and uses only training data. The out-of-fold probability comes from a model that never saw that tweet, so it flags examples that disagree with the rest of the data. A fixed prefix lets Groq's prompt cache bill repeated input at a discount.
- **Trade-offs accepted:** The filter favours "easy, prototypical" examples. Only 521 neutral tweets pass, against 1,759 positive, which itself shows how ambiguous "neutral" is. Nine examples is a small sample of the label definition.
- **Revisit if:** few-shot underperforms zero-shot on val (then try kNN few-shot as an ablation), or the owner's review of the 9 examples finds a mislabel.

### D-021 — LLM latency definition, sequential requests, disk cache
- **Date / phase:** 2026-10-05 · Phase 2
- **Context:** "p50 latency" must mean the same thing for every model.
- **Options considered:**
  - Latency: server-reported processing time · **client-side wall-clock of the successful HTTP call** · end-to-end including retries and rate-limit waits.
  - Concurrency: parallel requests (fast runs, but queueing inflates latency) · **sequential with client-side pacing**.
  - Re-runs: re-query the API · **cache every response on disk**.
- **Decision:** Client-side wall-clock time of the successful attempt, from the owner's machine (so network round-trip is included, as a real client would see). Requests are sequential, paced at 25 per minute. Retries and waits are logged separately. Every response is appended to `outputs/llm_cache/responses.jsonl`, keyed by a hash of (model, messages, params); cache hits keep their original latency.
- **Reasoning:** This is what a user of the API experiences. Excluding *our* rate-limit pacing keeps us from penalising the LLM for free-tier limits. Caching makes interrupted runs resumable and re-analysis free.
- **Trade-offs accepted:** Latency includes our network distance to Groq's data centres, which is real for us but varies by location. Groq's LPU hardware is unusually fast (D-002 caveat). Local models will be timed as model-only CPU/GPU inference, so the LLM figure carries a network round-trip the local ones don't; this is stated next to the table.
- **Revisit if:** network latency dominates, e.g. p50 far above Groq's compute time. Then also report server-side timing.

### D-022 — LLM cost = billed tokens × dated prices
- **Date / phase:** 2026-10-05 · Phase 2
- **Context:** "Cost per 1k predictions" needs a precise definition.
- **Options considered:** visible tokens only · **all billed tokens: prompt (cached share at its discount) + completion including hidden reasoning** · the actual invoice (not available per request).
- **Decision:** Cost per request = uncached prompt × input price + cached prompt × cached price + completion (incl. reasoning) × output price. Prices live in `configs/llm.yaml`, with an `as_of` date and source.
- **Reasoning:** It matches how the provider bills, and makes reasoning-token overhead visible.
- **Trade-offs accepted:** Prices are list prices as of 2026-10-05 and will change; the cost ratio vs local models is what should be quoted. The cached-input price ($0.075, half of input) is an assumption to verify on the console. On the free tier we pay $0, but we report what the same traffic *would* cost on the paid tier.
- **Revisit if:** Groq changes its pricing before the final run. Re-check `as_of`.

### D-023 — LLM evaluation on the free tier, with stratified subsets
- **Date / phase:** 2026-10-05 (decided 2026-10-06) · Phase 2 · *decided by project owner*
- **Context:** Free tier per model = 30 requests/min, 1,000 requests/day, **200K tokens/day**. Planned runs: val 300 + test 3,000 + YouTube 1,000, each zero- and few-shot = 8,600 requests ≈ 4M tokens (estimated: ~300 tokens per zero-shot request, ~650 per few-shot, including reasoning). That is about 20 days on the free tier, or about **$1–3** on Groq's pay-as-you-go Developer tier.
- **Options considered:**
  1. Developer tier with a spend cap (≈ $5): full test set, done in hours.
  2. Free tier with smaller stratified subsets (e.g. test 600, YouTube 300): about 5 days, and wider confidence intervals (about ±0.04 instead of ±0.017 on macro-F1).
- **Decision:** Option 2 (owner's choice). Stratified subsets (label × language mix; label only for YouTube, seed 42): **val 200** (prompt development), **test 600**, **YouTube 300**, each run zero- and few-shot. The client paces at ≤ 25 requests/min and ≤ 7,000 tokens/min. Daily-quota stops are resumed the next day from the cache.
- **Reasoning:** No spending is required. The pre-flight call measured 271 tokens for a zero-shot request (14 of them reasoning), so the plan is roughly 2,200 requests and about 1M tokens, which is about 5 days of free quota. 600 test tweets still give a usable confidence interval.
- **Trade-offs accepted:**
  - The LLM is scored on a 600-tweet subset while the local models are scored on all 3,000. For a fair head-to-head, Phase 4 reports **every model on the same 600 IDs** too.
  - Wider CIs (≈ ±0.04): LLM-vs-fine-tuned gaps smaller than that can't be called. A paired test on the same 600 tweets is more sensitive than comparing two CIs (D-024).
  - Calendar time: about 5 days.
  - Cost per 1k is still computed from list prices (D-022), as if we were paying.
- **Revisit if:** the subset CI is too wide to support the headline claim. Then extend the test subset over more free days (the cache keeps earlier answers), or spend about $1 on the paid tier.
- **Update 2026-10-06:** the daily token limit behaves as a rolling 24-hour window, not a calendar day. Quota spent at ~02:00 was usable again by ~15:30, so runs can be scheduled roughly every 24 h after the previous one.

### D-024 — Report 95% bootstrap confidence intervals for macro-F1
- **Date / phase:** 2026-10-05 · Phase 2
- **Context:** Several models may land within a couple of F1 points of each other, and the LLM may be scored on a subset.
- **Options considered:** point estimates only · **percentile bootstrap (1,000 resamples)** · McNemar or permutation tests for each pair of models.
- **Decision:** Bootstrap CI on every reported macro-F1. Paired tests may be added in Phase 4 for the headline comparison.
- **Reasoning:** Cheap, and it makes "is this gap real?" answerable at a glance. Example: TF-IDF test 0.687 [0.669, 0.703].
- **Trade-offs accepted:** The bootstrap treats test tweets as independent, which near-duplicates inside the test set slightly violate (Phase 1 flagged them).
- **Revisit if:** the headline claim depends on a gap smaller than the CI width. Then use a paired test.

### D-025 — Freeze the prompts after one validation round
- **Date / phase:** 2026-10-06 · Phase 2
- **Context:** On the 200-tweet val subset, zero-shot scored 0.602 and few-shot 0.674 macro-F1, against TF-IDF's 0.688 on the same tweets. The obvious next move is to keep rewording the prompt, especially the "neutral" definition, until the LLM does better.
- **Options considered:**
  1. Iterate on prompt wording against val until the score stops improving.
  2. **Freeze both prompts as they are and run test once.**
  3. Write the prompt from the official SentiMix annotation guidelines. We don't have them in enough detail to quote faithfully, and inventing them would be worse than not using them.
- **Decision:** Option 2.
- **Reasoning:**
  - The val subset's 95% CI is about ±0.06 wide, so prompt variants differing by a few points can't be told apart. Iterating would mostly fit noise in 200 tweets, and the "improved" val score would overstate test performance.
  - Each prompt variant costs a day's worth of free quota.
  - The gap has an explainable cause (the LLM's idea of "neutral" differs from the annotators'; see C-010), and few-shot already demonstrates the standard fix: showing the model the dataset's convention.
- **Trade-offs accepted:** The LLM numbers are a reasonable effort, not the best achievable. A determined prompt engineer could probably gain a few points; we state this as a limitation.
- **Revisit if:** we get a larger labelled dev set to tune prompts on without overfitting, or the final comparison hinges on a gap smaller than about 3 points.

---

## Phase 3: Fine-tuning

### D-026 — Option A encoders: XLM-R base and MuRIL base
- **Date / phase:** 2026-10-06 · Phase 3
- **Context:** We need a multilingual encoder that understands romanised Hindi, fits a free T4, and can be published.
- **Options considered:**
  1. **`FacebookAI/xlm-roberta-base`** (278M parameters, 100 languages, MIT licence).
  2. **`google/muril-base-cased`** (~237M parameters; trained on 17 Indian languages *including transliterated* text, so romanised Hindi is in-distribution; Apache-2.0).
  3. `cardiffnlp/twitter-xlm-roberta-base` (XLM-T: XLM-R further pre-trained on tweets, a strong domain match) has no licence on its Hub page, so we couldn't publish a model fine-tuned from it.
  4. XLM-R large / MuRIL large: about 3× slower to train, which strains a free Colab session for a 9-run sweep.
  5. mBERT: generally weaker than XLM-R on code-mixed benchmarks.
- **Decision:** Options 1 and 2, both in the same sweep, pinned to commits.
- **Reasoning:** XLM-R is the standard multilingual baseline. MuRIL is the "built for Indian languages" candidate. Comparing them answers a question interviewers ask: does a region-specific pre-trained model beat a general multilingual one?
- **Trade-offs accepted:** No tweet-domain pre-training. Base-size models only.
- **Revisit if:** both encoders land well below the LLM. Then try XLM-T (for internal comparison only) or a large model with gradient accumulation.

### D-027 — Experiment tracking: Weights & Biases
- **Date / phase:** 2026-10-06 · Phase 3 · *decided by project owner*
- **Context:** The brief asked for W&B or MLflow to track the sweep.
- **Options considered:**
  - **W&B**: hosted dashboards, live curves from Colab, shareable links; needs an account and API key.
  - **MLflow**: no account needed, but on Colab the tracking store must be written to Drive and viewed locally with `mlflow ui`; nothing is shareable without hosting.
- **Decision:** W&B. Project `hinglish-sentiment-finetune`, one group per model family, with each run's hyperparameters, parameter counts, training time and peak GPU memory logged. `report_to: none` is a config switch.
- **Reasoning:** Owner's choice. Live curves during Colab training and a link that can go in the README or résumé.
- **Trade-offs accepted:** Depends on a third-party service, and its free tier has storage limits. The key metrics are also written to `reports/` so the repo doesn't depend on W&B.
- **Revisit if:** W&B's free tier changes, or the owner wants fully local tracking.

### D-028 — Getting code onto Colab: a GitHub repository
- **Date / phase:** 2026-10-06 · Phase 3 · *decided by project owner*
- **Options considered:** **GitHub repo cloned in the notebook** · zip of the code uploaded to Google Drive.
- **Decision:** GitHub. A local git repo was created (data, `.env`, outputs and model weights are git-ignored). The notebook `git clone`s the repo, or `git pull`s on re-runs.
- **Reasoning:** Owner's choice. Every Colab run is tied to a commit (the notebook prints it), code changes need no re-upload, and Phase 5 needs a public repo anyway.
- **Trade-offs accepted:** The repo must be public, or Colab needs a token for private clones. Nothing sensitive is committed (no data, no secrets).
- **Revisit if:** the owner wants the repo private until Phase 5. Then use a fine-grained read-only token as a Colab secret.

### D-029 — MuRIL with emoji rewritten as text, as a sweep variant
- **Date / phase:** 2026-10-06 · Phase 3
- **Context:** MuRIL's vocabulary contains no emoji. **19.4% of training tweets** lose emoji such as 😂 🙏 😍 😭 to `[UNK]`, and emoji carry sentiment (C-011). XLM-R's vocabulary has them (0.03% unknown tokens).
- **Options considered:**
  1. Ignore it.
  2. Add emoji to MuRIL's vocabulary as new tokens: their embeddings would start random and only ~13k tweets would train them.
  3. **"Demojize"**: replace each emoji with its English name (😂 → `:face_with_tears_of_joy:`) using the `emoji` library, so MuRIL reads it as words it knows.
- **Decision:** Run MuRIL **both ways** in the sweep (`muril` and `muril_demoji`, 3 learning rates each), and let val macro-F1 decide.
- **Reasoning:** Option 3 is the standard, cheap fix. Keeping the raw variant turns an assumption into a measured ablation for the cost of 3 runs (~12 min of T4 time).
- **Trade-offs accepted:** Emoji names are English. That fits Hinglish text but adds tokens (longer inputs; still well under the 128-token limit). The same demojization must run at inference time, so it is stored as part of the run config and applied by `predict_proba`.
- **Revisit if:** results show no difference. Then report "emoji are recoverable from context" as the finding.

### D-030 — Option B decoder: Qwen3-1.7B
- **Date / phase:** 2026-10-06 · Phase 3
- **Context:** The brief asks for LoRA/QLoRA on a ~1–3B open decoder, on a free T4 (16 GB, no bf16).
- **Options considered:**
  1. **`Qwen/Qwen3-1.7B`**: Apache-2.0, text-only, 2.0B parameters including embeddings. Qwen models are noted for multilingual strength, and transformers has a ready `Qwen3ForSequenceClassification`.
  2. `Qwen/Qwen3.5-2B`: newer, but natively multimodal (`Qwen3_5ForConditionalGeneration`), so a classification head and 4-bit loading are less well-trodden.
  3. `HuggingFaceTB/SmolLM3-3B`: Apache-2.0 and strong, but 1.5× the parameters, so slower QLoRA runs on a T4.
  4. Llama 3.2 1B/3B: gated, with a custom licence that complicates publishing. Gemma: custom terms of use.
- **Decision:** Option 1, pinned to commit `70d244cc…`.
- **Reasoning:** The best licence, tooling and size combination for a T4. At ~2B parameters it is about 7× an encoder but ~60× smaller than the 120B LLM, a useful middle point on the size axis.
- **Trade-offs accepted:** Not the newest model family. Qwen is trained mostly on English and Chinese, so romanised Hindi coverage is uncertain, which is part of what we're measuring.
- **Revisit if:** QLoRA training is unstable in fp16 on the T4 (Qwen3 is released in bf16), or results lag the encoders badly. Then try SmolLM3-3B.

### D-031 — The decoder classifies through a head, with QLoRA
- **Date / phase:** 2026-10-06 · Phase 3
- **Context:** A decoder LLM can be fine-tuned to *generate* the label word, or to *score* the classes with a classification head on its last token.
- **Options considered:**
  1. **Sequence-classification head + QLoRA**: base frozen in 4-bit NF4 with double quantization; LoRA (r ∈ {8, 16}, alpha = 2r, dropout 0.05) on all attention and MLP projections; the new 3-way `score` head trained in full.
  2. Generative instruction tuning (SFT): train it to output "positive" etc., then parse the text. That needs generation at inference (slower) and answer parsing, and yields no calibrated probabilities.
  3. Full fine-tuning: impossible on a 16 GB T4 at 2B parameters with Adam.
- **Decision:** Option 1.
- **Reasoning:** One forward pass per prediction, with real class probabilities, no parsing failures, and the same evaluation code as the encoders. QLoRA trains only ~1% of the weights, and 4-bit storage cuts the frozen base from ~4 GB (fp16) to ~1.2 GB, which fits the T4 with room for activations. LoRA and head weights are kept in fp32, with fp16 autocast for computation.
- **Trade-offs accepted:** It doesn't use the model's instruction-following ability (no prompt). The classification head starts random. Results from 4-bit training may differ slightly from 16-bit.
- **Revisit if:** the head approach underperforms the encoders by a wide margin. A generative-SFT variant would show whether "speaking the label" helps.

### D-032 — Sweep protocol: small grids, early stopping on val, one test evaluation, keep only the best model
- **Date / phase:** 2026-10-06 · Phase 3
- **Context:** The brief asks for a hyperparameter sweep on a small grid that fits free Colab.
- **Options considered:**
  - Search: random or Bayesian search (W&B Sweeps) · **a small explicit grid**.
  - Final evaluation: score every run on test · **only the val-selected winner**.
  - Storage: keep every model · **keep only the best so far**.
- **Decision:**
  - Encoder grid: {XLM-R, MuRIL, MuRIL-demojized} × learning rate {2e-5, 3e-5, 5e-5}. Up to 4 epochs, batch 32, warmup 10%, weight decay 0.01, early stopping (patience 2) on val macro-F1. 9 runs.
  - Decoder grid: LoRA r {8, 16} × learning rate {1e-4, 2e-4}. 2 epochs, batch 16, gradient checkpointing, patience 1. 4 runs.
  - Every run keeps its best epoch. The winner is chosen on val macro-F1, **reloaded from disk**, and scored once on val/test/YouTube with the shared metrics code.
  - Only the best-so-far weights are kept on Drive; every run's metrics are kept.
- **Reasoning:**
  - Learning rate (and model or rank) is the hyperparameter that matters most for fine-tuning. A 9 + 4 grid fits in roughly 2 GPU-hours, and an explicit grid is easy to explain and reproduce.
  - Selecting on val only, as for TF-IDF (D-017), keeps test honest.
  - Reloading the saved model proves the published artefact reproduces the reported numbers.
  - Pruning keeps 9 × 1.1 GB of encoders off a 15 GB Drive.
  - The sweep is resumable (a finished run's `result.json` is skipped), so a Colab disconnect loses at most one run.
- **Trade-offs accepted:** No seed repeats, so run-to-run variance (often ±0.5–1 F1 for fine-tuning) isn't measured, and the winner may be partly lucky. Other hyperparameters (batch size, epochs, warmup) are fixed at common defaults.
- **Revisit if:** the top runs are within about 1 F1 of each other. Then re-run the top 2 configs with 3 seeds before declaring a winner (time permitting).

### D-033 — Serving measurements for fine-tuned models
- **Date / phase:** 2026-10-06 · Phase 3
- **Context:** The final table needs p50 latency, cost per 1k and model size defined consistently with the baselines (D-021, D-022).
- **Options considered:**
  - Latency: model forward pass only · **request-like: tokenization + forward pass + softmax at batch size 1, after warm-up, with CUDA synchronised**.
  - Size: parameter count only · **parameters (total and trained) + saved artefact size + loaded memory footprint**.
- **Decision:**
  - On the Colab T4: batch-1 p50/p95 over 500 test tweets, plus throughput at batch 64. Encoders run in fp16; the decoder runs 4-bit with fp16 autocast.
  - Size: parameter counts, artefact MB (for QLoRA, the adapter alone, with the 4-bit base stated separately) and the in-memory footprint.
  - Cost per 1k is computed in Phase 4 from measured throughput × a stated GPU or CPU hourly price.
- **Reasoning:** It mirrors what an API client experiences (minus the network), and it separates "what you download" from "what must fit in memory".
- **Trade-offs accepted:** Colab T4s are shared and timings vary between sessions. The LLM latency includes a network round-trip; the local models' latency does not (stated next to the table).
- **Revisit if:** timings vary more than about 20% between sessions. Then report the median of several timing passes.

---

## Phase 4: Evaluation

### D-034 — Cost per 1k predictions for self-hosted models: cloud instance price ÷ measured throughput
- **Date / phase:** 2026-10-06 · Phase 4
- **Context:** LLM cost comes from billed tokens (D-022). Self-hosted models have no per-call bill, so the "Y× cheaper" headline needs an equivalent.
- **Options considered:**
  1. **On-demand cloud price of the hardware the model was timed on ÷ measured batched throughput**: a machine kept busy.
  2. Price × batch-1 latency: assumes one request at a time on a dedicated machine. That is far more expensive per prediction, and nobody serves at volume that way.
  3. Zero, because "it's my laptop": not a fair comparison.
  4. Serverless GPU pricing (per second): varies a lot by vendor and includes cold starts.
- **Decision:** Option 1, with AWS us-east-1 on-demand prices as of 2026-10-06 in `configs/cost.yaml`:
  - **g4dn.xlarge** (1× T4, the GPU the fine-tuned models are timed on): $0.526/h;
  - **c7i.large** (2 vCPU): $0.0892/h, for TF-IDF.

  Cost per 1k = $/h ÷ (predictions/s × 3600) × 1000.
- **Reasoning:** One provider and one pricing model for all self-hosted rows. Throughput at batch 64 reflects how a classifier serves real traffic (micro-batching).
- **Trade-offs accepted:**
  - It assumes full utilisation; at low traffic an idle GPU still costs money, and the LLM API doesn't.
  - TF-IDF throughput was measured on the owner's laptop, not on a c7i.large.
  - Spot instances or reserved pricing would be 2–3× cheaper.

  The README states the assumption, and the cost *ratio* spans several orders of magnitude, so none of these change the conclusion.
- **Revisit if:** the fine-tuned and LLM costs come within ~10× of each other. Then model utilisation explicitly (cost at e.g. 10% load).

### D-035 — Headline comparison on identical tweets, with a paired bootstrap
- **Date / phase:** 2026-10-06 · Phase 4
- **Context:** The LLM is scored on a 600-tweet stratified subset of test (D-023), the local models on all 3,000. Comparing numbers from different samples would mix model differences with sample differences.
- **Options considered:**
  - Sample: compare each model on its own sample · **restrict every model to the LLM's exact tweet IDs for the headline** (and report local models on the full 3,000 separately).
  - Significance: overlapping-CI eyeballing · McNemar's test (accuracy only, not macro-F1) · **paired bootstrap of the macro-F1 difference** (resample tweets; score both models on each resample).
- **Decision:**
  - Table A = every model on the same IDs.
  - Table B = self-hosted models on the full test set.
  - Robustness by language-mix bucket uses the same IDs; YouTube uses the LLM's 300-comment subset.
  - Table D = paired bootstrap (2,000 resamples) for best fine-tuned vs best LLM, best fine-tuned vs TF-IDF, and best LLM vs TF-IDF. It reports the difference with its 95% CI and the share of resamples where A is not better.
- **Reasoning:** Pairing removes the shared "difficulty of this sample", so real differences show up with far fewer tweets than two independent CIs need. Macro-F1 is our headline metric (D-007), so the test must be on macro-F1, which rules out McNemar.
- **Trade-offs accepted:** Three comparisons with no multiple-comparison correction; they are reported as estimates with intervals, not as pass/fail tests. Mix buckets on 600 tweets are small (mostly-English ≈ 75), so per-bucket gaps there are indicative only.
- **Revisit if:** a claim rests on a bucket-level difference. Then compute paired CIs per bucket on the full 3,000 for the local models.

### D-036 — Error analysis: sample confident mistakes across all six confusion types
- **Date / phase:** 2026-10-06 · Phase 4
- **Context:** The brief asks for 10 real misclassified examples, explained.
- **Options considered:** a random sample of mistakes (dominated by neutral↔polar confusions, which are the most common) · **the most confident mistakes in each true→predicted cell** · hand-picking "interesting" ones (cherry-picking).
- **Decision:** `scripts/error_analysis.py` takes up to 4 of the most confident mistakes per confusion cell, with every other model's prediction alongside. The 10 explained examples are chosen from that pool to cover the cells. The output quotes tweet text, so it goes to git-ignored `outputs/`. Only the 10 discussed tweets appear in the docs, as short quotations for analysis (as research papers do), which is compatible with D-004.
- **Reasoning:** Confident mistakes are the most informative: either the model learned something wrong, or the gold label is questionable (C-010 showed many are). Covering every cell avoids a story told only about neutral.
- **Trade-offs accepted:** Explanations are one person's judgement. Where the gold label itself looks wrong, we say so rather than inventing a model failure. The owner's blind relabelling of a sample would strengthen this (offered in Phase 4).
- **Revisit if:** the owner relabels a sample. Then report the share of "model right, gold wrong" cases.

---

## Phase 5: Ship

### D-037 — How the model and demo are shipped
- **Date / phase:** 2026-10-06 · Phase 5 (prepared while Phases 2–4 wait on external runs)
- **Context:** The brief asks for the model on the HF Hub with a proper card, and a Gradio demo on HF Spaces showing the fine-tuned model and the LLM side by side.
- **Options considered:**
  - Which model to publish and demo: **best encoder** · QLoRA adapter (it needs the 2B base model and, on a free CPU-only Space, can't use 4-bit bitsandbytes: ~8 GB fp32 and seconds per prediction) · both.
  - Where to push from: download 1.1 GB from Drive and push locally · **push from Colab, where the weights already are**.
  - How the Space gets the code: copy `prompts.py` and `client.py` into the Space (two copies to keep in sync) · **`pip install` the project package from GitHub in the Space's requirements**.
  - The model card: hand-written · **generated from `reports/` (`ship/model_card.py`)**, refreshed locally once the LLM comparison is final (`publish.py card`).
- **Decision:**
  - Publish and demo the val-selected encoder; the QLoRA adapter stays in the study.
  - The notebook's section 8 pushes the model from Colab, using an `HF_TOKEN` Colab secret.
  - `config.demojize` is written into the uploaded config, so the demo and users apply the same preprocessing.
  - The Space runs the encoder on CPU, plus Groq `gpt-oss-120b` with the *frozen few-shot prompt*. Its 9 example tweets go into the Space as `few_shot_examples.json` (not into git), and the Groq key is a Space secret.
- **Reasoning:**
  - The encoder is the model a team would actually deploy cheaply, and it fits a free CPU Space.
  - Generated cards can't drift from the reported numbers.
  - Installing from GitHub means the demo runs exactly the evaluated prompt, parser and cost code.
- **Trade-offs accepted:**
  - The demo's CPU latency is a shared free Space, not the T4 numbers in the table (labelled as such in the UI).
  - The Space shares one Groq free-tier key, so heavy traffic hits rate limits; the UI reports failures gracefully.
  - Publishing 9 training tweets in the Space is a small quotation, like the 10 in the error analysis (D-036).
  - The Space depends on the GitHub repo staying public.
- **Revisit if:** the QLoRA model wins clearly (then also publish the adapter, demoed via a GPU Space or a merged fp16 model), or the Groq key gets abused (then disable the LLM half and show cached example outputs).
