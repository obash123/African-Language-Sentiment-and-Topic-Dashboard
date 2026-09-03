"""Loads the fine-tuned checkpoint and batch-predicts sentiment for text."""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "models" / "sentiment-distilbert"


class SentimentClassifier:
    def __init__(self, checkpoint_dir: str | Path = CHECKPOINT_DIR):
        checkpoint_dir = str(checkpoint_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
        self.model.eval()
        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def predict(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, max_length=96, return_tensors="pt")
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            top = probs.argmax(dim=-1)
            for j, label_id in enumerate(top.tolist()):
                results.append(
                    {
                        "sentiment": self.id2label[label_id],
                        "confidence": float(probs[j, label_id]),
                    }
                )
        return results
