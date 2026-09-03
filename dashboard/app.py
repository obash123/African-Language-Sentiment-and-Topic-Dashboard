"""African Language Sentiment & Topic Model Dashboard.

Reads data/processed/demo_comments.csv (see scripts/make_demo_data.py for how
it's built — real text + real model predictions, synthetic region/date
metadata; not a live YouTube scrape). Run with:

    streamlit run dashboard/app.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "demo_comments.csv"

# --- palette (from the project's validated design-system reference) -------
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

SENTIMENT_COLORS = {"positive": "#2a78d6", "neutral": "#c3c2b7", "negative": "#e34948"}
SENTIMENT_ORDER = ["negative", "neutral", "positive"]

LANG_ORDER = ["hau", "yor", "ibo", "pcm", "swa"]
LANG_LABELS = {
    "hau": "Hausa", "yor": "Yoruba", "ibo": "Igbo", "pcm": "Nigerian Pidgin", "swa": "Swahili",
}
LANG_COLORS = {  # fixed categorical order: blue, orange, aqua, yellow, magenta
    "hau": "#2a78d6", "yor": "#eb6834", "ibo": "#1baf7a", "pcm": "#eda100", "swa": "#e87ba4",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    font=dict(color=INK_SECONDARY, family="system-ui, -apple-system, Segoe UI, sans-serif"),
    margin=dict(l=10, r=10, t=40, b=10),
)


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df["language_name"] = df["language"].map(LANG_LABELS)
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    return df


def kpi_row(df: pd.DataFrame) -> None:
    total = len(df)
    pos_pct = (df["sentiment"] == "positive").mean() * 100
    neg_pct = (df["sentiment"] == "negative").mean() * 100
    n_langs = df["language"].nunique()

    cols = st.columns(4)
    cols[0].metric("Comments in view", f"{total:,}")
    cols[1].metric("Positive", f"{pos_pct:.0f}%")
    cols[2].metric("Negative", f"{neg_pct:.0f}%")
    cols[3].metric("Languages", n_langs)


def sentiment_heatmap(df: pd.DataFrame) -> go.Figure:
    """Language x month, colored by net sentiment score (positive% - negative%)."""
    grouped = (
        df.groupby(["language", "month"])["sentiment"]
        .apply(lambda s: (s == "positive").mean() - (s == "negative").mean())
        .reset_index(name="net_sentiment")
    )
    pivot = grouped.pivot(index="language", columns="month", values="net_sentiment")
    pivot = pivot.reindex([l for l in LANG_ORDER if l in pivot.index])
    pivot.index = [LANG_LABELS[l] for l in pivot.index]

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[d.strftime("%b %Y") for d in pivot.columns],
            y=pivot.index,
            colorscale=[[0, "#e34948"], [0.5, "#f0efec"], [1, "#2a78d6"]],
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Net sentiment", tickformat=".0%"),
            hovertemplate="%{y} · %{x}<br>Net sentiment: %{z:.0%}<extra></extra>",
        )
    )
    fig.update_layout(**PLOTLY_LAYOUT, title="Sentiment trend by language", height=320)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def topic_trend_chart(df: pd.DataFrame) -> go.Figure:
    top_topics = (
        df[df["topic_label"] != "misc / outlier"]
        .groupby("topic_label")
        .size()
        .sort_values(ascending=False)
        .head(10)
        .index.tolist()
    )
    sub = df[df["topic_label"].isin(top_topics)]
    counts = sub.groupby(["topic_label", "sentiment"]).size().reset_index(name="n")

    fig = go.Figure()
    for sentiment in SENTIMENT_ORDER:
        part = counts[counts["sentiment"] == sentiment]
        fig.add_bar(
            y=part["topic_label"],
            x=part["n"],
            name=sentiment.capitalize(),
            orientation="h",
            marker_color=SENTIMENT_COLORS[sentiment],
        )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="stack",
        title="Top topics by sentiment mix",
        height=420,
        yaxis=dict(categoryorder="total ascending", gridcolor=GRIDLINE),
        xaxis=dict(gridcolor=GRIDLINE, title="Comments"),
        legend=dict(orientation="h", y=1.08),
    )
    return fig


def region_sentiment_bars(df: pd.DataFrame) -> go.Figure:
    counts = df.groupby(["region", "sentiment"]).size().reset_index(name="n")
    fig = go.Figure()
    for sentiment in SENTIMENT_ORDER:
        part = counts[counts["sentiment"] == sentiment]
        fig.add_bar(x=part["region"], y=part["n"], name=sentiment.capitalize(), marker_color=SENTIMENT_COLORS[sentiment])
    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="stack",
        title="Sentiment by region",
        height=320,
        yaxis=dict(gridcolor=GRIDLINE, title="Comments"),
        xaxis=dict(gridcolor=GRIDLINE),
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="African Language Sentiment Dashboard", layout="wide")
    st.title("African Language Sentiment & Topic Model Dashboard")
    st.caption(
        "Yoruba · Hausa · Igbo · Nigerian Pidgin · Swahili — fine-tuned multilingual "
        "DistilBERT sentiment + per-language BERTopic topics."
    )

    if not DATA_PATH.exists():
        st.error(
            f"No demo data found at `{DATA_PATH.relative_to(DATA_PATH.parent.parent)}`. "
            "Run `python -m scripts.make_demo_data` first."
        )
        st.stop()

    df = load_data()

    with st.sidebar:
        st.header("Filters")
        langs = st.multiselect(
            "Language",
            options=[l for l in LANG_ORDER if l in df["language"].unique()],
            default=list(df["language"].unique()),
            format_func=lambda l: LANG_LABELS[l],
        )
        regions = st.multiselect("Region", options=sorted(df["region"].unique()), default=list(df["region"].unique()))
        sentiments = st.multiselect(
            "Sentiment", options=SENTIMENT_ORDER, default=SENTIMENT_ORDER, format_func=str.capitalize
        )
        min_date, max_date = df["date"].min().date(), df["date"].max().date()
        date_range = st.slider("Date range", min_value=min_date, max_value=max_date, value=(min_date, max_date))

    filtered = df[
        df["language"].isin(langs)
        & df["region"].isin(regions)
        & df["sentiment"].isin(sentiments)
        & (df["date"].dt.date >= date_range[0])
        & (df["date"].dt.date <= date_range[1])
    ]

    if filtered.empty:
        st.warning("No comments match the current filters.")
        st.stop()

    kpi_row(filtered)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.plotly_chart(sentiment_heatmap(filtered), use_container_width=True)
    with col2:
        st.plotly_chart(region_sentiment_bars(filtered), use_container_width=True)

    st.plotly_chart(topic_trend_chart(filtered), use_container_width=True)

    st.subheader("Comment explorer")
    display_cols = ["date", "language_name", "region", "sentiment", "confidence", "topic_label", "text"]
    st.dataframe(
        filtered[display_cols].sort_values("date", ascending=False).rename(columns={"language_name": "language"}),
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()
