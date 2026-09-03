"""Per-language topic extraction with BERTopic over a multilingual
sentence-transformer embedding, so Yoruba/Hausa/Igbo/Pidgin/Swahili text all
share one semantic space without needing per-language topic models.
"""
from __future__ import annotations

import os

# Must be set before numba (pulled in by umap, which BERTopic uses) picks its
# threading layer. Without this, umap's numba-jitted code segfaults on macOS
# arm64 when torch has already been loaded in the same process (their OpenMP
# runtimes collide) — this forces numba onto a threading backend that doesn't
# conflict with torch's.
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def fit_topics_per_language(
    df: pd.DataFrame,
    text_col: str = "text",
    lang_col: str = "language",
    min_topic_size: int = 8,
) -> pd.DataFrame:
    """Adds `topic_id` and `topic_label` columns, fitting one BERTopic model
    per language (so topics don't get muddled across languages)."""
    df = df.copy()
    df["topic_id"] = -1
    df["topic_label"] = "misc"

    embedder = SentenceTransformer(EMBED_MODEL)

    for lang, group in df.groupby(lang_col):
        texts = group[text_col].astype(str).tolist()
        if len(texts) < min_topic_size * 2:
            # Too few docs for a stable topic model in this language.
            continue
        embeddings = embedder.encode(texts, show_progress_bar=False)
        topic_model = BERTopic(
            embedding_model=embedder,
            min_topic_size=min_topic_size,
            verbose=False,
        )
        topic_ids, _ = topic_model.fit_transform(texts, embeddings)
        info = topic_model.get_topic_info().set_index("Topic")["Name"]
        labels = [info.get(t, "misc") for t in topic_ids]

        df.loc[group.index, "topic_id"] = topic_ids
        df.loc[group.index, "topic_label"] = labels

    df["topic_label"] = df["topic_label"].where(df["topic_id"] != -1, "misc / outlier")
    return df
