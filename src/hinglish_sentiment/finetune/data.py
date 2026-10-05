"""Turn the Phase 1 parquet splits into tokenized datasets for the HF Trainer."""

from __future__ import annotations

import emoji
import torch


def preprocess(text: str, demojize: bool) -> str:
    """Optionally rewrite emoji as text (😂 -> ":face_with_tears_of_joy:") for vocabularies without emoji (D-029)."""
    return " ".join(emoji.demojize(text, delimiters=(" :", ": ")).split()) if demojize else text


class TweetDataset(torch.utils.data.Dataset):
    """Pre-tokenized examples without padding; DataCollatorWithPadding pads each batch to its longest item."""

    def __init__(self, texts, labels, tokenizer, max_length: int, demojize: bool = False):
        texts = [preprocess(t, demojize) for t in texts]
        self.encodings = tokenizer(texts, truncation=True, max_length=max_length)
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        item = {k: v[i] for k, v in self.encodings.items()}
        item["labels"] = int(self.labels[i])
        return item
