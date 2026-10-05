"""Train one sweep run, reload a saved model, predict, and time inference.

Shared by scripts/finetune_sweep.py and the Colab notebook, so local and Colab runs execute the same code.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from transformers import DataCollatorWithPadding, EarlyStoppingCallback, Trainer, TrainingArguments

from hinglish_sentiment.eval.metrics import classification_metrics, latency_summary
from hinglish_sentiment.finetune.data import TweetDataset, preprocess
from hinglish_sentiment.finetune.modeling import build_encoder, build_qlora_decoder, count_parameters


def _compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    logits = logits[0] if isinstance(logits, tuple) else logits
    m = classification_metrics(labels, np.argmax(logits, axis=-1))
    return {"macro_f1": m["macro_f1"], "accuracy": m["accuracy"]}


def build_for_run(run: dict, cfg: dict, family: str, hf_id: str, revision: str | None):
    if family == "encoder":
        model, tokenizer = build_encoder(hf_id, revision)
        return model, tokenizer, {"quantized_4bit": False}
    dec = cfg["decoder"]
    model, tokenizer, quantized = build_qlora_decoder(
        hf_id, revision, run["lora_r"], dec["lora"], dec["quantization"], dec["train"].get("gradient_checkpointing", False))
    return model, tokenizer, {"quantized_4bit": quantized}


def train_run(run: dict, cfg: dict, family: str, hf_id: str, revision: str | None, splits: dict, out_dir: Path,
              report_to: str = "none") -> dict:
    """Fine-tune one configuration; keep the best epoch (by val macro-F1) in out_dir/model."""
    tcfg = cfg[family]["train"]
    torch.manual_seed(cfg["seed"])
    model, tokenizer, info = build_for_run(run, cfg, family, hf_id, revision)
    params = count_parameters(model)
    ds = {name: TweetDataset(df.text, df.label_id, tokenizer, cfg["max_length"], run.get("demojize", False))
          for name, df in splits.items() if name in ("train", "val")}

    use_fp16 = bool(tcfg.get("fp16")) and torch.cuda.is_available()
    if report_to == "wandb":
        import wandb
        os.environ.setdefault("WANDB_PROJECT", cfg["tracking"]["project"])
        wandb.init(project=cfg["tracking"]["project"], name=run["name"], group=family, reinit="finish_previous",
                   config={**run, "family": family, "hf_id": hf_id, "revision": revision, **params, **info})
    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"), run_name=run["name"], report_to=report_to,
        learning_rate=run["learning_rate"], num_train_epochs=tcfg["epochs"],
        per_device_train_batch_size=tcfg["batch_size"], per_device_eval_batch_size=tcfg["batch_size"] * 2,
        weight_decay=tcfg["weight_decay"], warmup_steps=tcfg["warmup"], lr_scheduler_type="linear",
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="macro_f1", greater_is_better=True, save_total_limit=1, save_only_model=True,
        logging_steps=25, fp16=use_fp16, seed=cfg["seed"], data_seed=cfg["seed"],
        dataloader_num_workers=0, use_cpu=not torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["val"], processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=tcfg["early_stopping_patience"])],
    )
    start = time.perf_counter()
    trainer.train()
    train_minutes = (time.perf_counter() - start) / 60
    val = trainer.evaluate()
    best_epoch = next((h["epoch"] for h in trainer.state.log_history
                       if h.get("eval_macro_f1") == trainer.state.best_metric), None)

    trainer.save_model(str(out_dir / "model"))  # full model (encoder) or adapter + head (QLoRA)
    tokenizer.save_pretrained(str(out_dir / "model"))
    shutil.rmtree(out_dir / "checkpoints", ignore_errors=True)
    peak_gpu_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
    if report_to == "wandb":
        import wandb
        wandb.summary.update({"best_val_macro_f1": val["eval_macro_f1"], "best_epoch": best_epoch,
                              "train_minutes": train_minutes, "peak_gpu_gb": peak_gpu_gb})
        wandb.finish()
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return {**run, "family": family, "hf_id": hf_id, "val_macro_f1": val["eval_macro_f1"],
            "val_accuracy": val["eval_accuracy"], "best_epoch": best_epoch, "train_minutes": round(train_minutes, 2),
            "peak_gpu_gb": peak_gpu_gb, "params_total": params["total"], "params_trainable": params["trainable"],
            **info}


def load_trained(model_dir: Path, family: str, cfg: dict, hf_id: str, revision: str | None):
    """Reload a saved run for inference (also proves the saved artefact is usable on its own)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    if family == "encoder":
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        if torch.cuda.is_available():
            model = model.half().cuda()  # fp16 inference on GPU, as it would be served
    else:
        from peft import PeftModel

        base = _load_quantized_base(hf_id, revision, cfg["decoder"]["quantization"], tokenizer)
        model = PeftModel.from_pretrained(base, str(model_dir))
    model.eval()
    return model, tokenizer


def _load_quantized_base(hf_id: str, revision: str | None, quant_cfg: dict, tokenizer):
    from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

    from hinglish_sentiment import ID2LABEL, LABEL2ID

    kwargs = {}
    if quant_cfg.get("load_in_4bit") and torch.cuda.is_available():
        kwargs = {"quantization_config": BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"], bnb_4bit_compute_dtype=torch.float16),
            "dtype": torch.float16, "device_map": {"": 0}}
    else:
        kwargs = {"dtype": torch.float32}
    base = AutoModelForSequenceClassification.from_pretrained(
        hf_id, revision=revision, num_labels=len(LABEL2ID), id2label=ID2LABEL, label2id=LABEL2ID, **kwargs)
    base.config.pad_token_id = tokenizer.pad_token_id
    return base


@torch.no_grad()
def predict_proba(model, tokenizer, texts, demojize: bool, max_length: int, batch_size: int = 64) -> np.ndarray:
    texts = [preprocess(t, demojize) for t in texts]
    device = next(model.parameters()).device
    # fp16 autocast on GPU, as in training: the 4-bit base computes in fp16 while LoRA/head weights are fp32
    autocast = torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda")
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], truncation=True, max_length=max_length, padding=True,
                        return_tensors="pt").to(device)
        with autocast:
            logits = model(**enc).logits
        out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def time_inference(model, tokenizer, texts, demojize: bool, max_length: int, n: int, warmup: int,
                   batch_size: int) -> dict:
    """p50/p95 latency at batch size 1 (an online request) and throughput at a larger batch size."""
    texts = list(texts)
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    for t in texts[:warmup]:
        predict_proba(model, tokenizer, [t], demojize, max_length, 1)
    lat = []
    for t in texts[:n]:
        sync()
        start = time.perf_counter()
        predict_proba(model, tokenizer, [t], demojize, max_length, 1)  # includes tokenization, like a real request
        sync()
        lat.append((time.perf_counter() - start) * 1000)
    sync()
    start = time.perf_counter()
    predict_proba(model, tokenizer, texts, demojize, max_length, batch_size)
    sync()
    throughput = len(texts) / (time.perf_counter() - start)
    device = next(model.parameters()).device
    return {
        "latency_batch1": latency_summary(lat),
        f"throughput_batch{batch_size}_per_s": float(throughput),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "memory_footprint_mb": float(model.get_memory_footprint() / 1e6) if hasattr(model, "get_memory_footprint") else None,
    }
