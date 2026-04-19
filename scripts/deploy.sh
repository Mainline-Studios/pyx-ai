#!/usr/bin/env bash
# Pyx-ai deploy: Cloud Run (API) and/or Firebase Hosting.
# Usage:
#   npm run deploy          # API + Hosting (full stack)
#   npm run deploy:api      # Cloud Run only (app.py / backend)
#   npm run deploy:hosting  # Firebase Hosting only (public/, firebase.json)
#
# Commits must be saved first; then this pushes the current branch to origin
# before Cloud Build / Firebase. Skip with: DEPLOY_SKIP_GIT_PUSH=1 npm run deploy
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

git_push() {
  if [[ "${DEPLOY_SKIP_GIT_PUSH:-}" == "1" ]]; then
    echo "==> (DEPLOY_SKIP_GIT_PUSH=1, skip git push)"
    return 0
  fi
  if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "==> (not a git repo, skip git push)"
    return 0
  fi
  local br
  br="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  if [[ -z "$br" || "$br" == "HEAD" ]]; then
    echo "==> (detached or unknown branch, skip git push)"
    return 0
  fi
  if ! git remote get-url origin >/dev/null 2>&1; then
    echo "==> (no git remote 'origin', skip git push)"
    return 0
  fi
  echo "==> git push origin ${br}"
  git push origin "$br"
}

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

git_push

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
