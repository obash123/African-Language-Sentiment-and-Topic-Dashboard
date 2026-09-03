#!/usr/bin/env bash
# Builds and deploys the dashboard to Google Cloud Run.
#
# NOT executed automatically — this is a documented, ready-to-run script for
# you to invoke once you have the gcloud CLI installed and are authenticated
# (`gcloud auth login`, `gcloud config set project <PROJECT_ID>`).
#
# Usage:
#   PROJECT_ID=my-gcp-project REGION=us-central1 ./deploy/deploy.sh

set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID, e.g. PROJECT_ID=my-gcp-project ./deploy/deploy.sh}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-african-sentiment-dashboard}"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"

echo "Building ${IMAGE}..."
gcloud builds submit --project "${PROJECT_ID}" \
  --config deploy/cloudbuild.yaml \
  --substitutions=_IMAGE="${IMAGE}" \
  .

echo "Deploying to Cloud Run (${SERVICE_NAME} in ${REGION})..."
gcloud run deploy "${SERVICE_NAME}" \
  --project "${PROJECT_ID}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300

echo "Done. Service URL:"
gcloud run services describe "${SERVICE_NAME}" \
  --project "${PROJECT_ID}" --region "${REGION}" \
  --format="value(status.url)"
