# African Language Sentiment & Topic Model Dashboard

A multilingual sentiment/topic dashboard for Yoruba, Hausa, Igbo, Nigerian
Pidgin, and Swahili social comments — YouTube-comment scraping, a fine-tuned
DistilBERT sentiment classifier, per-language topic modeling, and a Streamlit
dashboard, containerized for Google Cloud Run.


| Piece | Status |
|---|---|
| YouTube comment scraper (`scraper/`) | **Built and unit-tested**, not run against live YouTube at scale — see below |
| Sentiment model (`nlp/train_sentiment.py`) | **Actually fine-tuned** on the real [`masakhane/afrisenti`](https://huggingface.co/datasets/masakhane/afrisenti) dataset — real accuracy in `models/sentiment-distilbert/metrics.json` |
| Topic modeling (`nlp/topics.py`) | **Actually run** (BERTopic, multilingual sentence-transformer) to produce the demo data |
| Dashboard (`dashboard/app.py`) | **Built and run locally**, verified in-browser |
| Cloud Run deployment (`deploy/`) | **Prepared, not deployed** — Dockerfile/cloudbuild/deploy.sh are ready to run; deploying needs your own GCP project + billing + the `gcloud` CLI |



## Architecture

```
YouTube video IDs
      │
      ▼
scraper/pipeline.py ──(youtube-comment-downloader, Selenium fallback)──▶ data/raw/*.csv
      │
      ▼
nlp/lang_filter.py  (tags each comment: hau / yor / ibo / pcm / swa)
      │
      ▼
nlp/infer_sentiment.py  (fine-tuned DistilBERT: negative / neutral / positive)
      │
      ▼
nlp/topics.py  (BERTopic per language, shared multilingual embedding space)
      │
      ▼
dashboard/app.py  (Streamlit + Plotly: heatmap, topic trends, comment explorer)
      │
      ▼
deploy/  (Dockerfile → Cloud Run)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train the sentiment model

```bash
python -m nlp.train_sentiment --epochs 3 --max-train-per-lang 1600
```

Fine-tunes `distilbert-base-multilingual-cased` into a 3-class classifier
across the five AfriSenti language configs, evaluates on the full held-out
test split per language, and writes the checkpoint + `metrics.json` to
`models/sentiment-distilbert/`. `--max-train-per-lang` and `--epochs` trade
runtime for accuracy — the defaults finish in well under an hour on CPU.

## Build the demo dataset

```bash
python -m scripts.make_demo_data --n-per-lang 300
```

## Run the dashboard

```bash
streamlit run dashboard/app.py
```

## Scrape real YouTube comments (not run in this session — do this yourself)

```bash
python -m scraper.pipeline --video-ids VIDEO_ID_1 VIDEO_ID_2 --limit 300 --out data/raw/comments.csv
# or
python -m scraper.pipeline --urls-file videos.txt --out data/raw/comments.csv
```

Uses `youtube-comment-downloader` (no API key) as the primary path. For
videos it can't handle, `scraper/youtube_scraper.py` has a Selenium-based
fallback (`scrape_video(video_id)`) that drives a real Chrome instance and
scrolls the comments panel — call it directly for the shortlist `pipeline.py`
reports as failed. Offline parsing checks: `python -m scraper.test_parsing`.


## Deploy to Cloud Run (not executed — needs your GCP project)

```bash
# Install the gcloud CLI first: https://cloud.google.com/sdk/docs/install
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

PROJECT_ID=YOUR_PROJECT_ID REGION=us-central1 ./deploy/deploy.sh
```

This builds the image from `deploy/Dockerfile` via Cloud Build and deploys it
to Cloud Run with 2 vCPU / 2 GiB memory (the multilingual model + BERTopic's
sentence-transformer need real headroom). Adjust `--memory`/`--cpu` in
`deploy/deploy.sh` if needed.

## Project layout

```
african-sentiment-dashboard/
├── scraper/          # YouTube comment scraping (downloader + Selenium fallback)
├── nlp/               # language tagging, sentiment fine-tuning/inference, topics
├── dashboard/         # Streamlit app
├── scripts/           # demo-data generation
├── models/            # fine-tuned checkpoint + metrics.json (gitignored, regenerate via training)
├── data/               # raw scrapes (gitignored) + processed demo data
└── deploy/            # Dockerfile, cloudbuild.yaml, deploy.sh
```
