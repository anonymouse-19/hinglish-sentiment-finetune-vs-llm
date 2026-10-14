# Interview prep: Fine-tuned small model vs large LLM (Hinglish sentiment)

> **Draft, 2026-10-15.** Everything about data, baselines and design is final. Spots marked **⟨TBD⟩** are filled in once the
> Colab fine-tuning results arrive (Phase 3). Every claim points to an entry in [DECISION_LOG](DECISION_LOG.md) (D-xxx) or
> [CHALLENGES_LOG](CHALLENGES_LOG.md) (C-xxx), so you can go one level deeper when an interviewer pushes.

---

## 30-second pitch

"I measured, on real data, when a team should fine-tune a small model instead of paying for a large LLM. The task was sentiment
in Hinglish, the Hindi–English mix that hundreds of millions of Indians type online. I compared a TF-IDF baseline,
a 117-billion-parameter LLM prompted zero- and few-shot, and fine-tuned XLM-R, MuRIL and a QLoRA'd 2B model, all on the
same tweets, on accuracy, latency and cost. The surprise: the giant LLM *didn't* win on the benchmark. A 6 MB model matched it
at 1/20,000 of the cost. But on YouTube comments the LLM won by 32 points. The fine-tuned model ⟨TBD: lands where on both⟩,
at ⟨TBD⟩× lower cost than the LLM."

## 2-minute pitch

1. **Problem.** Classifying Hinglish text is a real build-vs-buy decision: call an LLM API (no training, pay per request), or fine-tune a small model (needs labels, but cheap and fast to serve). I wanted numbers, not opinions.
2. **Data.** SemEval-2020 SentiMix: 20k tweets with 3 sentiment labels. Cleaning it turned up three things:
   - **hidden train/test leakage**: 271 training tweets were near-copies of test tweets, invisible to exact matching because every copy has a different @handle and link (C-002);
   - mojibake in 10% of tweets (C-001);
   - labels that conflict in 35% of duplicate groups.

   I added YouTube comments as an out-of-domain test (D-014).
3. **Baselines.**
   - TF-IDF + logistic regression: 0.69 macro-F1.
   - `gpt-oss-120b` on Groq, zero-shot: 0.64, *significantly worse* by paired bootstrap.
   - Few-shot: 0.66, a statistical tie, at about $0.11 per 1,000 predictions vs effectively $0.

   Reading the errors showed why: the dataset labels many clearly emotional tweets "neutral", so the LLM applies common sense while a trained model learns the annotators' convention (C-010).
4. **But out of domain** (YouTube) the order flips: LLM 0.73 vs TF-IDF 0.42. The classical model calls most negative comments "neutral" because it has never seen their words.
5. **Fine-tuning.** On a free Colab T4:
   - a 9-run sweep over XLM-R and MuRIL, including an ablation, because MuRIL's vocabulary turns every emoji into an unknown token (C-011);
   - a 4-run QLoRA sweep on Qwen3-1.7B: a 4-bit frozen base plus LoRA adapters and a classification head (D-030, D-031).

   Result: ⟨TBD: best model, test F1, YouTube F1, p50 latency, cost/1k⟩.
6. **Takeaway.** ⟨TBD headline, e.g. "the fine-tuned encoder reaches X% of the LLM's accuracy in-domain at Y× lower cost; out of domain it keeps Z%"⟩. Fine-tuning's real advantage isn't only cost: it learns *your* label definition. Its weakness is domain shift, so the right architecture is fine-tuned plus monitoring, or an LLM fallback for out-of-distribution traffic.
7. **Shipped.** Model on the Hugging Face Hub with a generated model card; a Gradio Space comparing it side by side with the LLM; reproducible scripts, 70 tests, and a decision log.

---

## Likely questions and strong answers

### Data and evaluation

**1. How did you know your test results weren't inflated?**
I assumed they might be. Exact dedupe found 4 duplicates in 20k tweets, which is implausibly low for Twitter. I masked @handles and URLs, ran character-n-gram TF-IDF cosine similarity, and clustered matches with connected components. That found 658 duplicate groups, 315 crossing splits, and 271 train tweets that were near-copies of test tweets. I removed those from training and kept the official test set untouched, so the scores stay comparable to the leaderboard. *(C-002, D-010, D-011)*

**2. Why macro-F1 and not accuracy?**
Three imbalanced classes, and "neutral" is the hard, interesting one. Accuracy lets a model coast on the frequent classes; macro-F1 weights each class equally. I also report accuracy, per-class F1 and bootstrap CIs. *(D-007, D-024)*

**3. The LLM was only scored on 600 tweets. Isn't that unfair?**
The free API quota forced a subset, so I made the comparison fair rather than big. The subset is stratified by label × language mix, and *every* model is scored on the same 600 IDs for the headline. Significance comes from a **paired** bootstrap, which removes the shared difficulty of the sample. As a sanity check, TF-IDF scores 0.690 on the subset vs 0.687 on all 3,000. *(D-023, D-035)*

**4. Your test score was higher than validation. Isn't that suspicious?**
Yes, so I investigated: val tweets from the official dev and train sets score the same, 5-fold CV agrees with val, and leakage was already removed. The official test set seems slightly easier. Every model sees the same offset, so comparisons hold; I just don't read val as a test forecast. *(C-007)*

**5. How do you know a difference is real?**
A paired bootstrap of the macro-F1 difference: resample tweets 2,000 times and score both models on each resample. For example, zero-shot LLM − TF-IDF = −0.053 [−0.099, −0.009], which is real; few-shot − TF-IDF = −0.034 [−0.078, +0.011], which is a tie. *(D-035)*

### LLM baseline

**6. Why did a 117B LLM lose to TF-IDF?**
It's a label-convention mismatch more than a capability gap. I read all 56 "neutral" tweets the LLM got wrong; many are obviously emotional ("Boring Matches 💔😩") but labelled neutral. A supervised model learns that convention; a prompted LLM brings its own. Few-shot examples teach part of it (neutral F1 0.40 → 0.54). Lesson: "bigger is better" assumes the model shares your label definition. *(C-010)*

**7. Why didn't you keep tuning the prompt?**
On 200 validation tweets the CI is ±0.06, so prompt tweaks would mostly fit noise and inflate the reported score. I froze the prompts and ran test once. *(D-025)*

**8. How did you choose few-shot examples?**
Not at random, and not by hand. Random picks included mislabelled tweets (C-008). I computed each training tweet's *out-of-fold* probability of its own label with cross-validated TF-IDF and sampled class-balanced examples from tweets with p ≥ 0.6. That is automatic, reproducible, and uses training data only. *(D-020)*

**9. How did you measure LLM cost and latency?**
- Cost = billed tokens × dated list prices, including the *hidden reasoning tokens*: about 80% of output tokens for gpt-oss.
- Latency = client-side wall-clock of the successful call, excluding my own rate-limit waits.

Every response is cached, so reruns are free and reproducible. *(D-021, D-022)*

### Fine-tuning

**10. Why XLM-R and MuRIL? Why not a tweet-specific model?**
XLM-R is the multilingual standard; MuRIL was pre-trained on Indian languages including transliterated text. The tweet-pretrained XLM-T had no licence on its model page, so I couldn't publish a derivative. *(D-026)*

**11. What's QLoRA, and why use it?**
Load the 2B base model in 4-bit and freeze it, then train small low-rank adapter matrices (LoRA) plus a classification head: about 1% of the parameters. That fits a free 16 GB T4. I used a classification head rather than generating the label word: one forward pass, real probabilities, no parsing failures. *(D-030, D-031)*

**12. How did you choose hyperparameters without overfitting?**
A small explicit grid (learning rate × model, and rank × learning rate for QLoRA), early stopping on val macro-F1, and the winner chosen on val only. The winner is then *reloaded from disk* and scored on test once, which also proves the published artefact reproduces the numbers. *(D-032)*

**13. Tell me about a bug you caught before it cost you.**
A mixed-precision dtype mismatch in the QLoRA classification head. It only happens outside the Trainer's autocast, so it would have crashed *after* an hour of GPU training. A 1-minute CPU smoke test on a tiny random model caught it. *(C-012)*

**14. What did the MuRIL emoji issue teach you?**
Check tokenizer coverage on *your* data before choosing a model. A model "built for Indian languages" turned 19% of tweets' emoji into `[UNK]`. I made it an ablation (raw vs demojized) instead of an assumption. ⟨TBD: result⟩ *(C-011, D-029)*

### Business and engineering

**15. So, should a team fine-tune or use the LLM?**
⟨TBD with final numbers⟩. Roughly: if your traffic looks like your labelled data and volume is high, fine-tune: cheaper by ⟨TBD⟩× and ~⟨TBD⟩× faster, and it learns your label definition. If the inputs drift or you have no labels, the LLM is far more robust. In production: fine-tuned model first, drift monitoring, LLM fallback for low-confidence or out-of-distribution inputs.

**16. How did you compute cost for a self-hosted model?**
AWS on-demand price of the hardware it was timed on (a T4 instance at $0.526/h) divided by measured batched throughput. That assumes the machine stays busy; at low traffic the API's pay-per-call model looks better, which I state as a caveat. *(D-034)*

**17. Why Groq?**
The owner's choice: an OpenAI-compatible API, a free tier, and a fast open 120B model. The client is provider-agnostic, so it's a config change. Caveat: Groq's hardware is unusually fast, which makes the LLM's latency look better than a typical API's. *(D-002, D-016)*

**18. How did you handle the free tier's limits?**
I measured tokens per request, sized stratified subsets to the quota, and paced by requests *and* tokens per minute. Every response is cached, and the job resumes after crashes. It survived a network outage (C-015) and a Windows restart (C-016) without losing any paid work.

**19. What would you do differently?** See the last section.

---

## The 3 best STAR stories

**1. Hidden leakage in a public benchmark** *(C-002)*
- *S:* A SemEval benchmark appeared to contain almost no duplicates (4 exact matches in 20k tweets).
- *T:* Make sure my evaluation wasn't inflated by train/test leakage.
- *A:* I masked the per-copy handles and URLs, used char-n-gram TF-IDF similarity with connected-component clustering, and chose the threshold by inspecting samples.
- *R:* I found and removed 271 training tweets that were near-copies of test tweets, and quantified label noise at 35% of duplicate groups.

**2. The giant LLM lost, and I found out why** *(C-010)*
- *S:* I expected a 117B LLM to be the accuracy ceiling, but it scored 0.60 vs a TF-IDF model's 0.69.
- *T:* Find out whether this was a bug or something real, without overfitting the prompt.
- *A:* I read every misclassified "neutral" tweet and found the annotators labelled clearly emotional tweets neutral. I fixed it with label-noise-filtered few-shot examples, then froze the prompt.
- *R:* The gap closed to a statistical tie, and it reframed the project: fine-tuning learns *your* label definition. Out of domain, the LLM won by 32 points, which made the trade-off concrete.

**3. A bug caught before the GPU run** *(C-012)*
- *S:* I was preparing a QLoRA pipeline for a 2B model on a free Colab GPU.
- *T:* Avoid losing GPU time to a late-stage failure.
- *A:* I smoke-tested train → save → reload → predict on a tiny random model on CPU, found a dtype mismatch outside the Trainer's autocast, fixed it, and added an exact head-reload check.
- *R:* The crash was caught in 1 minute instead of after an hour, and the published adapter is guaranteed to reproduce its scores.

*(Backup stories: C-006 (reading every filtered tweet raised filter precision from 72% to 96%), C-011 (MuRIL emoji blindness), C-015/C-016 (resumable jobs survived an outage and a restart).)*

---

## What I'd do differently / next, with more time

- **Seeds:** 3 seeds for the top configs; with one seed, run-to-run variance isn't measured (D-032).
- **More LLM data:** score the LLM on the full 3,000 test tweets (about $1 on the paid tier) to tighten CIs, and try `reasoning_effort: medium` and a kNN (retrieved) few-shot prompt (D-016, D-020).
- **Labels:** relabel a sample with multiple annotators to separate model error from label noise (`scripts/relabel.py` is a start).
- **Domain adaptation:** fine-tune on a small labelled YouTube sample, or use LLM-labelled in-domain data (distillation) to fix the out-of-domain gap cheaply.
- **Production shape:** a confidence-based router (fine-tuned model first, LLM for low-confidence inputs), with cost and accuracy measured on mixed traffic.
- **Serving:** ONNX/int8 export of the encoder for CPU serving, and measured CPU latency next to the T4 numbers.
