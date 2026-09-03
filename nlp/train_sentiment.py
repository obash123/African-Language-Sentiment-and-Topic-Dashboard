"""Fine-tunes distilbert-base-multilingual-cased into a 3-class sentiment
classifier (negative/neutral/positive) across five African languages, using
the real masakhane/afrisenti dataset (Hausa, Yoruba, Igbo, Nigerian Pidgin,
Swahili configs).

Usage:
    python -m nlp.train_sentiment --epochs 3 --max-train-per-lang 4000

Writes the checkpoint to models/sentiment-distilbert/ and a metrics.json
with the real per-language and overall eval accuracy achieved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset, concatenate_datasets, ClassLabel
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "distilbert-base-multilingual-cased"
LANG_CONFIGS = ["hau", "yor", "ibo", "pcm", "swa"]
LABEL_NAMES = ["negative", "neutral", "positive"]
LABEL2ID = {name: idx for idx, name in enumerate(LABEL_NAMES)}
OUT_DIR = Path(__file__).resolve().parent.parent / "models" / "sentiment-distilbert"


def load_afrisenti(max_train_per_lang: int | None, seed: int = 42):
    train_parts, test_parts = [], []
    for lang in LANG_CONFIGS:
        ds = load_dataset("masakhane/afrisenti", lang)
        train = ds["train"]
        test = ds["test"] if "test" in ds else ds["validation"]

        # Normalize label representation to a plain int column matching
        # LABEL_NAMES, regardless of how the underlying feature is typed.
        def to_int_label(example):
            label = example["label"]
            if isinstance(label, str):
                label = LABEL2ID[label]
            elif isinstance(label, int) and isinstance(train.features["label"], ClassLabel):
                name = train.features["label"].int2str(label)
                label = LABEL2ID[name]
            return {"label": label, "language": lang}

        train = train.map(to_int_label)
        test = test.map(to_int_label)

        if max_train_per_lang is not None and len(train) > max_train_per_lang:
            train = train.shuffle(seed=seed).select(range(max_train_per_lang))

        train_parts.append(train)
        test_parts.append(test)

    train_all = concatenate_datasets(train_parts).shuffle(seed=seed)
    test_all = concatenate_datasets(test_parts)
    return train_all, test_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train-per-lang", type=int, default=1600)
    parser.add_argument("--max-length", type=int, default=72)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    args = parser.parse_args()

    print("[train] loading masakhane/afrisenti for:", LANG_CONFIGS)
    train_ds, test_ds = load_afrisenti(args.max_train_per_lang)
    print(f"[train] train={len(train_ds)} test={len(test_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["tweet"], truncation=True, max_length=args.max_length)

    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    keep_cols = ["input_ids", "attention_mask", "label"]
    train_ds = train_ds.remove_columns(
        [c for c in train_ds.column_names if c not in keep_cols]
    )
    test_ds_full = test_ds  # keep "language" for per-language breakdown
    test_ds_model = test_ds.remove_columns(
        [c for c in test_ds.column_names if c not in keep_cols]
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABEL_NAMES), id2label=dict(enumerate(LABEL_NAMES)), label2id=LABEL2ID
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
        }

    training_args = TrainingArguments(
        output_dir=str(OUT_DIR / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=32,
        learning_rate=args.lr,
        eval_strategy="no",  # full test-set eval only once, after training
        save_strategy="no",
        logging_steps=50,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds_model,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    per_language = {}
    predictions = trainer.predict(test_ds_model)
    overall = {
        "eval_accuracy": accuracy_score(test_ds_model["label"], np.argmax(predictions.predictions, axis=-1)),
        "eval_f1_macro": f1_score(test_ds_model["label"], np.argmax(predictions.predictions, axis=-1), average="macro"),
    }
    print("[train] overall eval:", overall)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = np.array(test_ds_full["label"])
    langs = np.array(test_ds_full["language"])
    for lang in LANG_CONFIGS:
        mask = langs == lang
        if mask.sum() == 0:
            continue
        per_language[lang] = {
            "n": int(mask.sum()),
            "accuracy": float(accuracy_score(labels[mask], preds[mask])),
            "f1_macro": float(f1_score(labels[mask], preds[mask], average="macro")),
        }
    print("[train] per-language eval:", json.dumps(per_language, indent=2))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))

    metrics = {
        "base_model": MODEL_NAME,
        "dataset": "masakhane/afrisenti",
        "languages": LANG_CONFIGS,
        "train_examples": len(train_ds),
        "test_examples": len(test_ds_model),
        "epochs": args.epochs,
        "overall": {
            "accuracy": overall.get("eval_accuracy"),
            "f1_macro": overall.get("eval_f1_macro"),
        },
        "per_language": per_language,
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[train] saved checkpoint + metrics.json -> {OUT_DIR}")


if __name__ == "__main__":
    main()
