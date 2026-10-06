"""Gradio demo: the fine-tuned small model and the large LLM, side by side, on the same Hinglish text.

Runs on a free Hugging Face Space (CPU) or locally:
    MODEL_ID=<hub id or local dir> GROQ_API_KEY=... python app/app.py

- Left: the fine-tuned encoder, running on this machine's CPU (class probabilities, latency).
- Right: gpt-oss-120b on Groq with the study's frozen few-shot prompt (answer, latency, billed tokens, cost).
The LLM prompt, answer parser and cost accounting are imported from the project package, so the demo uses
exactly the code that produced the reported numbers.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import gradio as gr
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from hinglish_sentiment import LABELS
from hinglish_sentiment.finetune.data import preprocess
from hinglish_sentiment.llm.client import LLMClient
from hinglish_sentiment.llm.prompts import build_messages, parse_label

HERE = Path(__file__).parent
MODEL_ID = os.environ.get("MODEL_ID", "anonymouse-19/hinglish-sentiment-xlmr")
LLM_CFG = {
    "model": "openai/gpt-oss-120b", "base_url": "https://api.groq.com/openai/v1",
    "prices": {"input_per_million": 0.15, "cached_input_per_million": 0.075, "output_per_million": 0.60},
    "generation": {"temperature": 0, "max_completion_tokens": 1024, "reasoning_effort": "low", "extra_body": {}},
    "client": {"requests_per_minute": 30, "max_retries": 2, "timeout_s": 30},
}
# Self-hosted cost shown per prediction: from the study's measured throughput (configs/cost.yaml, D-034).
FT_USD_PER_1K = float(os.environ.get("FT_USD_PER_1K", "nan"))

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()
DEMOJIZE = bool(getattr(model.config, "demojize", False))  # stored in the config at upload time (MuRIL variant)
torch.set_num_threads(max(1, os.cpu_count() or 1))

few_shot_path = HERE / "few_shot_examples.json"
FEW_SHOT = pd.DataFrame(json.loads(few_shot_path.read_text(encoding="utf-8"))) if few_shot_path.exists() else None
llm = LLMClient(LLM_CFG, os.environ["GROQ_API_KEY"]) if os.environ.get("GROQ_API_KEY") else None


@torch.no_grad()
def fine_tuned(text: str):
    start = time.perf_counter()
    enc = tokenizer(preprocess(text, DEMOJIZE), truncation=True, max_length=128, return_tensors="pt")
    probs = torch.softmax(model(**enc).logits, dim=-1)[0].tolist()
    ms = (time.perf_counter() - start) * 1000
    cost = "" if FT_USD_PER_1K != FT_USD_PER_1K else f" · ≈ ${FT_USD_PER_1K / 1000:.8f} per prediction at scale"
    return dict(zip(LABELS, probs)), f"{ms:.0f} ms on this Space's CPU{cost}"


def large_llm(text: str):
    if llm is None:
        return None, "LLM disabled: set the GROQ_API_KEY secret on the Space."
    try:
        r = llm.complete(build_messages(text, FEW_SHOT))
    except Exception as exc:  # rate limits on a shared free key, network errors
        return None, f"LLM call failed: {type(exc).__name__}"
    label = parse_label(r.content)
    probs = {l: float(i == label) for i, l in enumerate(LABELS)} if label >= 0 else {"(unparseable)": 1.0}
    return probs, (f"{r.latency_ms:.0f} ms via Groq API · {r.prompt_tokens} in + {r.completion_tokens} out tokens "
                   f"({r.reasoning_tokens} hidden reasoning) · ${r.cost_usd:.6f} for this call")


def compare(text: str):
    text = (text or "").strip()
    if not text:
        return None, "", None, ""
    return (*fine_tuned(text), *large_llm(text))


EXAMPLES = [  # written for the demo (not from the dataset)
    "yaar ye movie toh ekdum bakwas thi, paisa barbaad 😤",
    "Congratulations bhai! Bahut proud feel ho raha hai aaj 🎉🙏",
    "kal office jaana hai, meeting 10 baje hai",
    "match dekh ke maza aa gaya, Kohli ne kya shot maara! 😍",
    "Sarkar ne phir se petrol ke daam badha diye, wah kya vikas hai 👏",
]

with gr.Blocks(title="Hinglish sentiment: fine-tuned vs LLM") as demo:
    gr.Markdown(
        "# Hinglish sentiment: a fine-tuned small model vs a 117B LLM\n"
        "Type Hindi–English code-mixed text. The same input goes to a fine-tuned encoder running on this CPU "
        "and to `gpt-oss-120b` on Groq with the study's few-shot prompt. "
        "[Code, data card & full results](https://github.com/anonymouse-19/hinglish-sentiment-finetune-vs-llm)")
    inp = gr.Textbox(label="Hinglish text", lines=3, placeholder="e.g. yaar ye movie toh ekdum bakwas thi")
    btn = gr.Button("Classify with both", variant="primary")
    with gr.Row():
        with gr.Column():
            gr.Markdown(f"### Fine-tuned small model\n`{MODEL_ID}`")
            ft_out, ft_info = gr.Label(label="Class probabilities"), gr.Markdown()
        with gr.Column():
            gr.Markdown("### Large LLM (few-shot prompt)\n`openai/gpt-oss-120b` on Groq")
            llm_out, llm_info = gr.Label(label="Answer"), gr.Markdown()
    gr.Examples(EXAMPLES, inputs=inp)
    btn.click(compare, inputs=inp, outputs=[ft_out, ft_info, llm_out, llm_info])
    inp.submit(compare, inputs=inp, outputs=[ft_out, ft_info, llm_out, llm_info])

if __name__ == "__main__":
    demo.launch()
