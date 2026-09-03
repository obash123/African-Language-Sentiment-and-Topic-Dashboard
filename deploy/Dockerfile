FROM python:3.11-slim

WORKDIR /app

# System deps for torch/sentence-transformers wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dashboard/ dashboard/
COPY nlp/ nlp/
COPY models/sentiment-distilbert/ models/sentiment-distilbert/
COPY data/processed/ data/processed/

ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Cloud Run injects $PORT; Streamlit must bind to it.
CMD streamlit run dashboard/app.py --server.port=${PORT:-8080} --server.address=0.0.0.0
