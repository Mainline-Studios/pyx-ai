#!/usr/bin/env bash
# Pyx-ai deploy: Cloud Run (API) and/or Firebase Hosting.
# Usage:
#   npm run deploy          # API + Hosting (full stack)
#   npm run deploy:api      # Cloud Run only (app.py / backend)
#   npm run deploy:hosting  # Firebase Hosting only (public/, firebase.json)
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TARGET="${1:-all}"
case "$TARGET" in
  all|full) TARGET="all" ;;
  api|run|cloud-run) TARGET="api" ;;
  hosting|host) TARGET="hosting" ;;
  *)
    echo "Usage: $0 [all|api|hosting]" >&2
    exit 1
    ;;
esac

REGION="${GCP_REGION:-us-central1}"
SERVICE="${CLOUD_RUN_SERVICE:-pyxaiapi}"
PROJECT="${GCP_PROJECT:-pyx-ai}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/cloud-run-source-deploy/${SERVICE}"

deploy_api() {
  echo "==> Cloud Build + Cloud Run (${SERVICE})"
  gcloud builds submit . --config=cloudbuild.yaml
  gcloud run deploy "${SERVICE}" \
    --image="${IMAGE}" \
    --region="${REGION}" \
    --platform=managed \
    --allow-unauthenticated \
    --quiet
}

deploy_hosting() {
  echo "==> npm build + Firebase Hosting"
  npm run build
  firebase deploy --only hosting
}

case "$TARGET" in
  all)
    deploy_api
    deploy_hosting
    ;;
  api)
    deploy_api
    ;;
  hosting)
    deploy_hosting
    ;;
esac

echo "Done."
