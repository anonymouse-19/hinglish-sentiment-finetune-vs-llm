"""Build (model, tokenizer) for the two fine-tuning options.

A (encoder): every weight of XLM-R / MuRIL is trained, plus a fresh 3-way classification head.
B (decoder, QLoRA): Qwen3's weights are loaded in 4-bit and frozen; small low-rank adapter matrices (LoRA)
   on every linear layer, plus the classification head, are the only trained parameters (D-030, D-031).
   The class is predicted from the hidden state of the last real token (how *ForSequenceClassification works
   for decoders), so no text generation or answer parsing is needed.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

from hinglish_sentiment import ID2LABEL, LABEL2ID


def count_parameters(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable), "trainable_share": trainable / total}


def build_encoder(hf_id: str, revision: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(hf_id, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_id, revision=revision, num_labels=len(LABEL2ID), id2label=ID2LABEL, label2id=LABEL2ID)
    return model, tokenizer


def build_qlora_decoder(hf_id: str, revision: str | None, lora_r: int, lora_cfg: dict, quant_cfg: dict | None,
                        gradient_checkpointing: bool):
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    tokenizer = AutoTokenizer.from_pretrained(hf_id, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantize = bool(quant_cfg and quant_cfg.get("load_in_4bit")) and torch.cuda.is_available()
    kwargs = {}
    if quantize:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"], bnb_4bit_compute_dtype=torch.float16)
        kwargs["dtype"] = torch.float16
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["dtype"] = torch.float32  # CPU / no quantization: full precision (checkpoints may default to bf16)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_id, revision=revision, num_labels=len(LABEL2ID), id2label=ID2LABEL, label2id=LABEL2ID, **kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id  # needed to find the last real token in a padded batch
    if quantize:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    elif gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # only target modules that exist (tiny smoke-test models may lack some projections)
    present = {name.split(".")[-1] for name, _ in model.named_modules()}
    targets = [m for m in lora_cfg["target_modules"] if m in present]
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=lora_r, lora_alpha=lora_cfg["lora_alpha_over_r"] * lora_r,
        lora_dropout=lora_cfg["lora_dropout"], target_modules=targets, bias="none")
    model = get_peft_model(model, peft_config)  # SEQ_CLS keeps the new "score" head trainable
    # LoRA and head weights in fp32 for stable fp16 mixed-precision training
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return model, tokenizer, quantize
