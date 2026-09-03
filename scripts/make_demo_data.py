"""Builds the dashboard's demo dataset.

IMPORTANT — data provenance: this does NOT scrape YouTube. It takes real
comment/tweet text from the masakhane/afrisenti test split (the same real
dataset the sentiment model was fine-tuned on, held-out portion), runs it
through the ACTUAL fine-tuned checkpoint and the ACTUAL topic model, and
attaches synthetic-but-labeled metadata (country/region bucket, a spread of
dates) so the dashboard has language x region x time dimensions to filter
and chart. The scraper in scraper/ is real and ready to point at live
YouTube videos — running it live was intentionally out of scope for this
build (see README).

Usage:
    python -m scripts.make_demo_data --n-per-lang 300
"""
from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from nlp.infer_sentiment import SentimentClassifier
from nlp.topics import fit_topics_per_language

LANG_CONFIGS = ["hau", "yor", "ibo", "pcm", "swa"]

LANG_META = {
    "hau": {"name": "Hausa", "region": "West Africa", "country": "Nigeria"},
    "yor": {"name": "Yoruba", "region": "West Africa", "country": "Nigeria"},
    "ibo": {"name": "Igbo", "region": "West Africa", "country": "Nigeria"},
    "pcm": {"name": "Nigerian Pidgin", "region": "West Africa", "country": "Nigeria"},
    "swa": {"name": "Swahili", "region": "East Africa", "country": "Kenya/Tanzania"},
}

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "demo_comments.csv"


def load_sample_text(n_per_lang: int, seed: int = 7) -> pd.DataFrame:
    rows = []
    for lang in LANG_CONFIGS:
        ds = load_dataset("masakhane/afrisenti", lang)
        test = ds["test"] if "test" in ds else ds["validation"]
        test = test.shuffle(seed=seed).select(range(min(n_per_lang, len(test))))
        for row in test:
            rows.append({"text": row["tweet"], "language": lang})
    return pd.DataFrame(rows)


def attach_synthetic_metadata(df: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    rng = random.Random(seed)
    df = df.copy()
    df["region"] = df["language"].map(lambda l: LANG_META[l]["region"])
    df["country"] = df["language"].map(lambda l: LANG_META[l]["country"])
    df["language_name"] = df["language"].map(lambda l: LANG_META[l]["name"])

    start = datetime(2026, 1, 1)
    df["date"] = [start + timedelta(days=rng.randint(0, 240)) for _ in range(len(df))]
    df["likes"] = [rng.randint(0, 500) for _ in range(len(df))]
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-lang", type=int, default=300)
    args = parser.parse_args()

    print("[demo-data] sampling AfriSenti test text...")
    df = load_sample_text(args.n_per_lang)
    print(f"[demo-data] {len(df)} rows across {df['language'].nunique()} languages")

    print("[demo-data] running fine-tuned sentiment model...")
    clf = SentimentClassifier()
    preds = clf.predict(df["text"].tolist())
    df["sentiment"] = [p["sentiment"] for p in preds]
    df["confidence"] = [p["confidence"] for p in preds]

    print("[demo-data] fitting per-language topic models (BERTopic)...")
    df = fit_topics_per_language(df)

    df = attach_synthetic_metadata(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"[demo-data] wrote {len(df)} rows -> {OUT_PATH}")
    print(df["sentiment"].value_counts())


if __name__ == "__main__":
    main()
