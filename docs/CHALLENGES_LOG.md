# Challenges Log

Only real difficulties that actually happened, each with a STAR version for interviews. Entries marked *(owner)* come from the project owner's answers at the end of each phase.

| ID | Phase | Title |
|---|---|---|
| C-001 | 1 | Mojibake in ~10% of tweets, which also fooled the first foreign-language heuristic |
| C-002 | 1 | Hidden duplicates and train/test leakage masked by per-copy handles and links |
| C-003 | 1 | Tokenizer-split Twitter entities, with language tags on handles and URLs |
| C-004 | 1 | Non-Hinglish tweets, and the off-the-shelf language detector that failed |
| C-005 | 1 | Windows console crashes when printing emoji and Devanagari |
| C-006 | 1 | Flag emoji silently deleted real Hinglish tweets; `ftfy` left 5 tweets broken |
| C-007 | 2 | The official test set scores about 5 points higher than validation |
| C-008 | 2 | Randomly chosen few-shot examples taught the LLM wrong labels |
| C-009 | 2 | The provider catalogue and SDK had changed from what the plan assumed |
| C-010 | 2 | Wrong assumption: the 120B LLM would beat a TF-IDF model (zero-shot lost by 0.09 F1) |
| C-011 | 3 | MuRIL can't see emoji: 19% of tweets lose them to `[UNK]` |
| C-012 | 3 | Mixed-precision dtype mismatch in the QLoRA classifier, caught by a CPU smoke test |
| C-013 | 3 | Installing transformers silently downgraded a pinned library |

---

### C-001 — Mojibake in ~10% of tweets, which also fooled the first foreign-language heuristic
- **Phase:** 1 (Data)
- **What happened:** Profiling non-ASCII characters showed `HORIZONTAL`, `EURO SIGN` and `BROKEN BAR` among the most frequent Unicode character names. Tweets contained sequences like `ðŸ˜…` and `â€¦`. A first heuristic for "tweets containing accented Latin letters" (meant to find foreign languages) flagged **2,209 tweets**, far more than expected.
- **Root cause:** Upstream, UTF-8 bytes had been decoded as Windows-1252 (cp1252), turning 😅 into `ðŸ˜…` and … into `â€¦`. The characters `â` and `Ã` are themselves accented Latin letters, so the mojibake inflated the foreign-language heuristic.
- **What was tried:**
  - Counting Unicode character-name prefixes to see *what* the non-ASCII text actually was. This worked, and it's how the cause was found.
  - Running the accented-letter heuristic on the raw text. It did **not** work: it measured mojibake, not language.
- **Final fix:** Repair encoding on every token *before* any other heuristic. It repaired 2,119 tweets. After the repair, only about 98 tweets contained accented letters, and those were overwhelmingly foreign-language tweets. `ftfy` alone was not quite enough; see C-006 for the last 5 tweets it missed.
- **Lesson learned:** Fix encoding first. Every downstream heuristic (language rules, dedupe keys, tokenizer vocabularies) silently depends on it. Profile *what* characters exist before writing rules about them.
- **STAR:**
  - *S:* While cleaning a 20k-tweet Hinglish sentiment dataset, my foreign-language filter flagged 2,209 tweets, which was suspiciously many.
  - *T:* I needed to find out whether the data really was that multilingual or my check was wrong.
  - *A:* I profiled the non-ASCII characters by Unicode name. The top ones were "EURO SIGN" and "BROKEN BAR", the signature of UTF-8 text decoded as cp1252. I repaired it with `ftfy` before any other processing and added a residual-mojibake check.
  - *R:* 2,119 tweets (10.6%) were repaired, with emoji restored as sentiment features, and the true foreign-language count dropped to 98. That's a 20× over-count avoided.

### C-002 — Hidden duplicates and train/test leakage masked by per-copy handles and links
- **Phase:** 1 (Data)
- **What happened:** Exact-string deduplication found only **4 duplicate rows** in 20k tweets. After masking `@handles` and URLs, **658 duplicate groups (1,336 rows)** appeared. 315 groups spanned the official train/dev/test splits, and 231 had conflicting labels. The first near-duplicate search then reported about 5.6k similar pairs at 0.9, which looked alarming.
- **Root cause:**
  - Copies of the same tweet (retweets, copy-paste campaigns, templated greetings) each carry a different @handle and a unique t.co short link, so exact matching never sees them as equal.
  - The alarming pair count had its own cause. 99 tweets consist only of `@user @user … http`; after masking they are near-identical and form an almost complete clique. **4,656 of the 5,652 pairs** involved these tweets (a full clique would be 99×98/2 = 4,851).
- **What was tried:**
  1. Exact dedupe on the raw text. It did **not** work (4 rows).
  2. Exact dedupe on masked, punctuation-free content. It works, but misses copies that differ in punctuation or truncation.
  3. Char 3–5-gram TF-IDF cosine similarity. The first run was dominated by the empty-tweet clique, and counting *pairs* was the wrong unit.
  4. Excluding content-free tweets, grouping pairs into clusters with connected components, and comparing thresholds 0.85 / 0.90 / 0.95 by leakage count, largest cluster size and hand-read samples in each similarity band.
- **Final fix:** Similarity threshold 0.9 (largest cluster 8, no chaining), with a cluster-level policy (DECISION_LOG D-010, D-011). This removed **271 train/dev rows that were near-copies of test tweets**, 403 redundant copies and 436 label-conflict rows. The test set itself is untouched.
- **Lesson learned:** On social-media data, dedupe on *content*, not the raw string. Count clusters, not pairs. Always check for leakage against the test set, even in a published benchmark.
- **STAR:**
  - *S:* A public SemEval benchmark appeared to have almost no duplicates; exact matching found 4 rows.
  - *T:* Make sure the evaluation wasn't inflated by train/test leakage.
  - *A:* I masked user handles and URLs, which differ on every copy of a tweet, then ran character-n-gram TF-IDF similarity and clustered the matches with connected components. I chose the threshold empirically by inspecting samples and cluster sizes.
  - *R:* I found 271 training tweets that were near-copies of test tweets and removed them. I also found that 35% of duplicate groups carry conflicting labels, which quantifies the dataset's label noise.

### C-003 — Tokenizer-split Twitter entities, with language tags on handles and URLs
- **Phase:** 1 (Data)
- **What happened:** Joining the CoNLL tokens produced text like `@ BTS _ army _ Fin`, `https // t . co / OwhP` and `# videsh _ mantri`. The word-level language tags labelled **about 28.8k handle tokens and at least 11k URL fragments** as Hindi or English. Code-mixing statistics computed naively would have counted `tco` as an English word.
- **Root cause:** The release's tokenizer split on non-alphanumeric characters, and the language tagger ran on every token, entities included.
- **What was tried:**
  1. Considered regex cleaning on the joined text, but rejected it: it breaks the one-to-one alignment between tokens and language tags that the robustness analysis needs.
  2. A token-level state machine that merges mentions, URLs and hashtags. After the first pass, **47 hashtags, 5 mentions and 4 URLs were still split**, because the entity was glued to a neighbouring token (`Superstar…#`, `Satsanghttps //`, `@ shahidkapoor's`).
  3. Added a small regex safety net in the detokenizer for exactly those glued cases.
- **Final fix:** Token-level normalisation plus a safety net. Residuals are **2 mentions** (`@ @user` edge cases), 0 URLs and 0 hashtags. Unit tests in `tests/test_clean.py` cover each tricky shape. Language statistics now count only real words.
- **Lesson learned:** Inspect *residuals* after each cleaning pass. The first version fixed 99% of cases, and the rest needed specific handling. And check what your annotations were computed on before trusting derived metrics.
- **STAR:**
  - *S:* The dataset's word-level language tags were the basis for my code-mixed vs English vs Hindi robustness analysis.
  - *T:* Compute a trustworthy code-mixing score per tweet.
  - *A:* I noticed the tagger had labelled user handles and URL fragments as Hindi or English, about 40k tokens. I wrote a token-level normaliser that merges and masks those entities while keeping tags aligned, then iterated on the residual cases with a regex safety net and unit tests.
  - *R:* The robustness buckets are now based only on real words, and 0.01% of tweets still have a split entity.

### C-004 — Non-Hinglish tweets, and the off-the-shelf language detector that failed
- **Phase:** 1 (Data)
- **What happened:** The very first training tweet is Czech ("nenávist bolest vztek smutek…"), with its accented letters split into separate tokens. More sampling found Turkish, Polish, Norwegian, Swedish, Vietnamese, Portuguese and French tweets inside the "Hinglish" corpus.
- **Root cause:** Not documented by the organisers. Most likely the collection filter let some non-Hinglish tweets through and they were labelled anyway.
- **What was tried:** The obvious tool, `langdetect`, tested on 300 romanized-Hindi training tweets. It did **not** work: it said Indonesian 46%, Somali 17%, Swahili 9%, Estonian 8%, and **Hindi 0 times**. It would have deleted most of the real data.
- **Final fix:** A transparent rule. A tweet with ≥ 3 accented letters (an ASCII letter plus a diacritic) is treated as foreign, because Hinglish is typed in plain ASCII. It is applied only to train/val (25 removed); test rows are flagged (1). Every removed tweet was read by hand: 24 are foreign, and 1 is Hindi written in IAST transliteration. The first version of this rule had a bug that deleted real Hinglish tweets (C-006).
- **Lesson learned:** Off-the-shelf NLP tools are trained on monolingual, native-script text. Validate them on a sample of *your* data before using them as a filter.
- **STAR:**
  - *S:* The Hinglish training set contained Turkish, Polish and Czech tweets.
  - *T:* Remove foreign-language noise without losing real Hinglish.
  - *A:* I first tested `langdetect` on 300 known Hinglish tweets. It classified 46% as Indonesian and none as Hindi, so I rejected it. Instead I wrote a simple rule based on accented Latin letters, which Hinglish typed in ASCII never uses, and spot-checked its hits.
  - *R:* 25 tweets were removed, 24 of them genuinely foreign after a full manual review. Blindly using the standard tool would have destroyed most of the dataset.

### C-005 — Windows console crashes when printing emoji and Devanagari
- **Phase:** 1 (Data), environment
- **What happened:** Printing sample rows raised `UnicodeEncodeError: 'charmap' codec can't encode characters in position 80-82`.
- **Root cause:** On Windows, Python's stdout used the legacy cp1252 code page, which can't represent emoji or Devanagari.
- **What was tried / final fix:** Set `PYTHONIOENCODING=utf-8` for scripts that print text. All file I/O in the pipeline already passes `encoding="utf-8"` explicitly, so data files were never affected, only console output.
- **Lesson learned:** For multilingual NLP on Windows, set UTF-8 I/O explicitly, and never rely on platform-default encodings when reading or writing files.
- **STAR:**
  - *S:* The data-exploration scripts crashed on Windows when printing Hinglish tweets with emoji.
  - *T:* Make the tooling work on Windows as well as on Linux and Colab.
  - *A:* I traced it to the Windows console's legacy cp1252 encoding and forced UTF-8 console output. I also confirmed that every file read and write specifies UTF-8 explicitly.
  - *R:* The same scripts now run on Windows and Colab, with no silent data corruption.

### C-006 — Flag emoji silently deleted real Hinglish tweets; `ftfy` left 5 tweets broken
- **Phase:** 1 (Data)
- **What happened:** Before signing off Phase 1, I read *every* tweet the foreign-language filter removed, not just a sample. Several were clearly not foreign:
  - *"shabaash Stokes buss issi tarhaan World Cup jeetna hai come on England [England flag emoji]"* (Hinglish)
  - *"All the best to all teams … Jo b team best dega wahi jeetegi #CWC19"* (Hinglish)
  - *"Saturday class done ʟɪᴋᴇ ᴛᴀɢ ғʀɪᴇɴᴅ"* (English)

  The same review found tweets still containing mojibake, e.g. `ã…‹ã…‹` (Korean ㅋㅋ), `ÄŸ` (Turkish ğ) and `Û"Û"`, even though the residual check reported 0.
- **Root cause:** Three separate problems.
  1. My "accented letter" test checked whether the character's Unicode *name* contains "LATIN". Subdivision flag emoji (England, Scotland) are built from invisible characters named `TAG LATIN SMALL LETTER G`, `…B`, `…E`…, so every England flag counted as 5 accented letters. Decorative "small caps" fonts (`LATIN LETTER SMALL CAPITAL L`) matched the same way.
  2. `ftfy` is deliberately conservative. It won't "fix" short sequences like `ÄŸ` that could be legitimate, and its `is_bad` check (my residual check) doesn't flag them either.
  3. `ftfy` straightens curly quotes. In `Û”` (the mojibake of the Urdu full stop ۔) it turned `”` into `"` *before* the sequence could be repaired, so the damage became unrecoverable by `ftfy`.
- **What was tried:**
  - Reviewing a sample of ~12 removed tweets. It did **not** catch the problem, because the sample happened to be all foreign.
  - `ftfy` alone plus `is_bad` as the residual check. It did **not** catch the short sequences.
  - A strict cp1252→UTF-8 re-decode applied only *after* `ftfy`. It did **not** fix `Û”`, because the quote was already straightened.
- **Final fix:**
  1. Redefined "accented letter" through canonical decomposition: NFD must give an ASCII letter plus combining marks. Tag characters, small caps and look-alike letters have no such decomposition.
  2. Mojibake repair became strict re-decode → `ftfy` → strict re-decode. The re-decode only replaces a span if it forms valid UTF-8, so legitimate accents (café…, Señor) are untouched; unit-tested.
  3. The residual check now uses the mojibake byte signature, and all 86 remaining Latin-1 texts were read by hand: 0 mojibake left.
  4. Compared variants by reading every removal:
     - Name-based rule at threshold 2: 50 removals, 14 not foreign (about 72% precision).
     - NFD rule at threshold 2: 42 removals, 8 not foreign (about 81%).
     - NFD rule at threshold 3: 25 removals, 1 not foreign (about 96%).

     Chose NFD at 3 (D-012).
  5. Regression tests for flag emoji, small caps, `ÄŸ`, `ã…‹`, `Û”`, and "legit text untouched".

  Net effect: 22 more tweets kept in train/val (15,754 → 15,776), and 5 more mojibake tweets repaired.
- **Lesson learned:** Validate filters by reading **all** of what they remove when the count is small (here, dozens). A 12-item sample hid a systematic bug. Unicode property checks should use decomposition and categories, not name substrings. Order matters in cleaning pipelines: a "harmless" normalisation (quote straightening) can destroy information a later step needs.
- **STAR:**
  - *S:* My foreign-language filter for a Hinglish dataset looked fine on a spot-check of about 12 removed tweets.
  - *T:* Before finalising the dataset, confirm the filter wasn't deleting real data.
  - *A:* I read all 50 removals and found Hinglish cricket tweets deleted because the England flag emoji is made of invisible "TAG LATIN" Unicode characters, which my name-based check counted as accented letters. I rewrote the check using Unicode decomposition, fixed a related ordering bug in mojibake repair, added regression tests, and re-tuned the threshold by reviewing every removal.
  - *R:* Filter precision rose from about 72% to about 96%, 22 more tweets were kept, and residual mojibake went to zero, all verified by hand rather than by a proxy metric.

---

## Phase 2: Baselines

### C-007 — The official test set scores about 5 points higher than validation
- **Phase:** 2 (Baselines)
- **What happened:** The selected TF-IDF model scored **0.642 macro-F1 on val but 0.687 on test**. Usually test is equal to or a bit *below* val, because val was used for model selection. A gap in the "wrong" direction can signal a bug, such as a leak between train and test.
- **Root cause:** Not fully determined. What was ruled out:
  - *Not a split artifact:* val tweets that came from the official **dev** set score the same as those from the official **train** set (0.641 vs 0.642).
  - *Not luck in the val split:* 5-fold cross-validation on train gives 0.60–0.63, consistent with val.
  - *Not leakage:* Phase 1 already removed all train near-copies of test tweets (C-002).
  - *Not surface differences:* truncation (55% vs 55%), retweet share (15% vs 17%) and length (99 vs 100 chars) match.
  - *Label noise (inconclusive):* conflicting-label rates inside duplicate groups are 34% (train), 33% (dev) and 27% (test). But test has only 11 such groups, too few to conclude anything.

  The working hypothesis is that the official test set is somewhat cleaner or easier than train/dev, a property of the benchmark rather than of our pipeline.
- **What was tried:** Breaking val down by origin, 5-fold CV on train, comparing surface statistics, and comparing label-conflict rates per split (all above).
- **Final fix:** Nothing to "fix". It is documented. The protocol stays: select on val, report test once. Every model will show the same offset, so *comparisons between models* remain valid, but val scores should not be read as test predictions.
- **Lesson learned:** When a number moves in an unexpected direction, rule out bugs systematically before accepting it. And write down "not determined" honestly instead of inventing a cause.
- **STAR:**
  - *S:* My baseline scored 0.687 macro-F1 on the official test set but only 0.642 on my validation split, the opposite of the usual pattern.
  - *T:* Make sure this wasn't a bug or leakage before trusting any comparison built on it.
  - *A:* I broke val down by data origin, ran 5-fold CV on train, and compared length, truncation, retweet share and label-conflict rates across splits.
  - *R:* I ruled out split artifacts and leakage. The gap is a property of the benchmark's test set, and since every model faces the same offset, model comparisons stay valid. I documented it so nobody misreads val scores as test predictions.

### C-008 — Randomly chosen few-shot examples taught the LLM wrong labels
- **Phase:** 2 (Baselines)
- **What happened:** A dry run of the few-shot prompt printed the 9 randomly drawn training examples. One clearly insulting tweet ("…Modi ki gulam biki hue or tum muslim khalana ka layak Nahi hoo…") was labelled **neutral**, and another negative-sounding example was also labelled neutral. Those examples would have been shown to the LLM as the definition of "neutral".
- **Root cause:** SentiMix's training labels are noisy (Phase 1: 35% of duplicate groups carry conflicting labels), so a random sample of 9 is likely to include a mislabelled one. It matters more in a prompt than in training: one bad example out of 9 is about 11% of the LLM's "training data".
- **What was tried:**
  1. Random class-balanced sampling. It produced mislabelled demonstrations.
  2. Considered hand-picking examples. Rejected as subjective and hard to defend as a fair comparison.
  3. Computed each train tweet's **out-of-fold** probability of its own label with the TF-IDF model (5-fold cross-validation, so the model never saw the tweet it scores). The median is only 0.48, which confirms how noisy the labels are. Sampled only from tweets with p ≥ 0.6, and read the result.
- **Final fix:** Option 3 (D-020). The 9 IDs are saved in `configs/few_shot_ids.yaml` and are reproducible with `scripts/select_few_shot.py`. Only 521 of 5,029 neutral train tweets (10%) pass the filter, against 1,759 of 4,368 positive (40%), which is itself evidence that "neutral" is the noisiest class.
- **Lesson learned:** Few-shot examples are training data for an LLM, so they deserve the same label-quality scrutiny. Confident-learning-style filtering (out-of-fold probabilities) is a cheap, automatic way to find likely mislabels.
- **STAR:**
  - *S:* Building a few-shot prompt for an LLM baseline on a noisy Hinglish sentiment dataset, I noticed a randomly picked example was mislabelled: an insult tagged as "neutral".
  - *T:* Give the LLM correct demonstrations without hand-picking them, which would bias the comparison.
  - *A:* I computed out-of-fold label probabilities for every training tweet with cross-validated TF-IDF, kept only tweets whose model-estimated probability of their own label was ≥ 0.6, and sampled class-balanced examples from that pool. I saved the IDs for reproducibility.
  - *R:* Clean, reproducible demonstrations chosen by an automatic rule. As a side result, I quantified that the "neutral" class is by far the noisiest (only 10% of neutral tweets pass, against 40% of positive).

### C-009 — The provider catalogue and SDK had changed from what the plan assumed
- **Phase:** 2 (Baselines)
- **What happened:**
  1. The Phase 1 plan (D-002) assumed Groq would serve a 70B-class Llama model to free-tier users. Groq's docs on 2026-10-05 list only `gpt-oss-120b`, `gpt-oss-20b` and a preview Qwen model under free-tier rate limits. Llama 3.3 70B now sits in an enterprise tier.
  2. Groq's pricing page no longer shows per-token prices; they had to be pieced together from the models page and price trackers.
  3. A new unit test failed at import with `ModuleNotFoundError: No module named 'httpx'`. The installed `openai` SDK (3.24) had replaced `httpx` with `httpx2`, so the usual way of building a fake `RateLimitError` from an `httpx.Response` broke.
- **Root cause:** Fast-moving external dependencies: LLM provider catalogues and pricing change monthly, and a major SDK version changed its HTTP layer.
- **What was tried:** Read the live rate-limit and model docs instead of relying on memory, and inspected the SDK's constructor signature (`response: 'httpx2.Response'`).
- **Final fix:**
  1. Chose the model from what is actually available (D-016), and put model ID, prices, `as_of` date and source in `configs/llm.yaml`.
  2. Added `scripts/check_llm.py`, which lists the models the key can reach before any run.
  3. Made the test build the error from a minimal stand-in object, so it depends on neither HTTP library.
- **Lesson learned:** Treat model IDs, prices and SDK internals as configuration that expires. Date them, verify them at run time, and keep tests independent of a library's private dependencies.
- **STAR:**
  - *S:* My plan assumed a 70B Llama model on Groq's free tier; when I started the LLM baseline, the catalogue had changed, and the newest OpenAI SDK had swapped its HTTP library, breaking my tests.
  - *T:* Keep the baseline credible and the code robust to a moving external API.
  - *A:* I re-checked the live docs, chose the strongest available model and documented why, moved model, prices and dates into config with a pre-flight check script, and rewrote the tests to use a minimal stand-in instead of SDK internals.
  - *R:* The LLM can be swapped by editing one YAML line, prices are auditable by date, and the test suite (62 tests) runs offline in about 6 seconds with no API key.

### C-010 — Wrong assumption: the 120B LLM would beat a TF-IDF model
- **Phase:** 2 (Baselines)
- **What happened:** The plan treated the large LLM as the accuracy *ceiling* that a small fine-tuned model would try to approach. On the first 200 val tweets, zero-shot `gpt-oss-120b` scored **0.602 macro-F1, against 0.688 for TF-IDF + logistic regression** on the same tweets. Its "neutral" F1 was only 0.40: of 75 tweets labelled neutral, it called 33 negative and 23 positive.
- **Root cause:** Mostly a **label-convention mismatch**, plus some genuine LLM errors. I read all 56 neutral tweets the LLM got "wrong":
  - Many carry obvious sentiment that the annotators nonetheless labelled neutral, e.g. *"…Boring Matches 💔😩"*, or insults ending in an expletive. On these, the LLM's answer is what most people would say, and TF-IDF is "right" only because it learned the annotators' broad notion of neutral. TF-IDF got 32 of the 56 right.
  - Some are real LLM mistakes: sarcasm read literally (*"sasural walo ne 70 saal me world class ke hospital banaye"* → positive), and promotional tweets ("Earn points… Free hotel stays") called positive where the convention is neutral.
- **What was tried:**
  1. Error reading (above), rather than immediately rewording the prompt.
  2. Few-shot prompting with 9 label-noise-filtered training examples, 3 of them neutral, so the model sees the dataset's convention. Neutral F1 rose from 0.40 to 0.54 and macro-F1 from 0.602 to **0.674**, within noise of TF-IDF.
  3. Considered tuning the prompt wording further on val. Rejected (D-025): the val CI is ±0.06, so I would mostly be fitting noise.
- **Final fix:** Not a bug to fix; it is a finding. Prompts are frozen, and test is run once. The headline framing changes from "how close does the small model get to the LLM?" to "which approach learns the *task as defined by the labels*, at what cost?". Phase 4 error analysis will check how often the LLM is arguably right and the gold label wrong. A good candidate: the owner, as a Hindi speaker, blind-labels a sample of disagreements.
- **Lesson learned:** "Bigger model = higher score" assumes the model shares the dataset's label definitions. Supervised models learn the annotation convention, including its quirks; prompted models bring their own. Read the errors before tuning anything.
- **STAR:**
  - *S:* I expected a 120B-parameter LLM to be the accuracy ceiling in my Hinglish sentiment study, but zero-shot it scored 0.60 macro-F1, below a TF-IDF baseline's 0.69.
  - *T:* Work out whether this was a broken prompt, a parsing bug, or something real, without overfitting the prompt to a small validation set.
  - *A:* I read every misclassified "neutral" tweet and found that the dataset labels many clearly emotional tweets as neutral. The LLM applied common sense; the trained model had learned the annotators' convention. I then added nine label-noise-filtered training examples to the prompt and froze it.
  - *R:* Few-shot closed most of the gap (0.674, neutral F1 0.40 → 0.54) at about $0.10 per 1,000 predictions. It reframed the project's key insight: fine-tuning's real advantage is learning *your* label definition, not just being cheaper.

---

## Phase 3: Fine-tuning

### C-011 — MuRIL can't see emoji: 19% of tweets lose them to `[UNK]`
- **Phase:** 3 (Fine-tuning)
- **What happened:** While choosing `max_length`, I tokenized the training set with each candidate model and also measured the unknown-token rate. MuRIL produced `[UNK]` for **0.8% of tokens**, against 0.03% for XLM-R. Listing the words that produced `[UNK]` showed they were almost all emoji (😂 ×190, 😂😂 ×105, 🙏 ×89, 😍 ×75, ❤️ ×63, …). In total, **2,605 training tweets (19.4%)** had at least one emoji erased.
- **Root cause:** MuRIL's WordPiece vocabulary was built from Indian-language text (Wikipedia, CommonCrawl, transliterated text) and contains no emoji characters. Emoji are a strong sentiment signal in tweets, and the TF-IDF baseline's char n-grams use them.
- **What was tried:** Counting `[UNK]`-producing words per model, and considering adding emoji as new vocabulary tokens. That was rejected: the new embeddings would start random with only ~13k tweets to learn them (D-029).
- **Final fix:** A `demojize` option (😂 → `:face_with_tears_of_joy:`), applied identically in training and inference. MuRIL runs in the sweep both raw and demojized, so the effect is measured, not assumed.
- **Lesson learned:** Check tokenizer coverage of *your* data before choosing a pre-trained model. A model built "for Indian languages" can still be blind to a key feature of Indian *social media* text.
- **STAR:**
  - *S:* I was choosing between XLM-R and MuRIL, a model pre-trained specifically for Indian languages, to fine-tune for Hinglish tweet sentiment.
  - *T:* Make sure the tokenizer could actually represent the data before spending GPU time.
  - *A:* I measured each tokenizer's unknown-token rate on the training set and listed which words became `[UNK]`. MuRIL turned every emoji into `[UNK]`, affecting 19% of tweets. I added emoji-to-text conversion and ran MuRIL both ways in the sweep.
  - *R:* A silent loss of a major sentiment signal was caught before training, and turned into a measured ablation (raw vs demojized MuRIL) instead of an untested assumption.

### C-012 — Mixed-precision dtype mismatch in the QLoRA classifier, caught by a CPU smoke test
- **Phase:** 3 (Fine-tuning)
- **What happened:** The first local smoke run of the decoder pipeline (a tiny random Qwen3, CPU) crashed in the classification head: `RuntimeError: expected m1 and m2 to have the same dtype, but got: struct c10::BFloat16 != float`.
- **Root cause:** For stable mixed-precision training, I upcast all *trainable* weights (LoRA adapters and the new `score` head) to fp32. The base model was loaded in its checkpoint's default dtype (bf16), so the bf16 hidden states met an fp32 head. On Colab the same mismatch would occur (fp16 base, fp32 head). During training, Trainer's fp16 autocast hides it, but my evaluation function runs *outside* the Trainer, so it would have crashed after a full training run.
- **What was tried:** Reading the stack trace (it fails at `self.score(hidden_states)`) and reasoning through which code paths use autocast: training yes, my `predict_proba` no.
- **Final fix:**
  1. Without quantization (CPU), load the decoder explicitly in fp32.
  2. On GPU, `predict_proba` runs under `torch.autocast("cuda", dtype=float16)`, the same as training.
  3. Added a check that the reloaded adapter's `score` head equals the saved one exactly (it does), so the final evaluation can't silently use a random head.
- **Lesson learned:** Smoke-test the *whole* pipeline (train, save, reload, predict, time) on tiny models before using GPU time. Code paths outside the Trainer don't inherit its mixed-precision context.
- **STAR:**
  - *S:* I was preparing a QLoRA fine-tuning pipeline for a 2B-parameter model to run on a free Colab GPU.
  - *T:* Avoid losing an hour of GPU time to a bug that only shows up after training.
  - *A:* I ran the entire pipeline locally on a tiny random Qwen3 on CPU. It exposed a dtype mismatch between the fp32 classification head and the half-precision base that only occurs outside the Trainer's autocast. I fixed precision handling for both paths and added a check that the saved classification head reloads exactly.
  - *R:* The bug was caught in a ~1-minute CPU run instead of after a ~15-minute GPU run, and the reload check guarantees the published adapter reproduces the reported scores.

### C-013 — Installing transformers silently downgraded a pinned library
- **Phase:** 3 (Fine-tuning)
- **What happened:** After installing the training stack, `pip list` showed `huggingface_hub 1.33.0`, though `requirements.txt` pinned `2.0.0` (the version used for Phase 1). pip had downgraded it without an error, because `transformers 5.18` requires `huggingface_hub < 2`. The new `requirements-colab.txt` also pinned 2.0.0, which would have produced an unsatisfiable combination on Colab.
- **Root cause:** Phase 1's pins were chosen before the training libraries existed in the environment. The two sets of pins conflict, and pip resolves the conflict by quietly changing the earlier choice.
- **What was tried:** `pip check` (no broken requirements after the downgrade) and comparing installed versions against the pins.
- **Final fix:** Re-pinned `huggingface_hub==1.33.0` in both requirement files, then **re-ran the Phase 1 pipeline under the new version**: `reports/data_stats.json` came out identical and all split sizes matched, so the data are unaffected.
- **Lesson learned:** When adding a dependency group, re-check that earlier pins still hold, and re-run earlier pipeline stages to confirm results are unchanged, not just that installation succeeded.
- **STAR:**
  - *S:* Adding the fine-tuning libraries to a project whose data pipeline was already finished and pinned.
  - *T:* Keep the environment reproducible and the earlier results valid.
  - *A:* I noticed pip had silently downgraded a pinned library to satisfy transformers. I updated the pins in both the local and Colab requirement files and re-ran the full data pipeline under the new version.
  - *R:* The data statistics came out byte-identical, so the earlier results stand, and the Colab install file can no longer request an impossible combination.
