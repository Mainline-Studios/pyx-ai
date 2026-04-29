#!/usr/bin/env bash
# Pyx-ai deploy: Cloud Run (API) and/or Firebase Hosting.
# Usage:
#   npm run deploy          # API + Hosting (full stack)
#   npm run deploy:api      # Cloud Run only (app.py / backend)
#   npm run deploy:hosting  # Firebase Hosting only (public/, firebase.json)
#
# npm run deploy runs git-deploy-commit first when there are changes. Optional reason for GitHub:
#   npm run deploy -- warm min-instances + pixel defaults
#   npm run deploy:api -- fix Groq 429 handling
# deploy:githubmessage — commit only (same optional reason after --).
# Skip auto-commit: DEPLOY_SKIP_GIT_COMMIT=1 npm run deploy
# Commits must be saved first; then this pushes the current branch to origin
# before Cloud Build / Firebase. Skip push: DEPLOY_SKIP_GIT_PUSH=1 npm run deploy
#
# API deploy defaults: min-instances=1, 2Gi RAM, 900s timeout, --no-cpu-throttling, --cpu-boost.
# CPU throttling OFF by default (warm instances still throttle CPU between requests unless disabled — a major 503 cause).
# Opt in to idle CPU throttling (save $): CLOUD_RUN_CPU_THROTTLING=1 npm run deploy:api
# Scale to zero: CLOUD_RUN_MIN_INSTANCES=0 npm run deploy:api
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
  local min_inst="${CLOUD_RUN_MIN_INSTANCES:-1}"
  local mem="${CLOUD_RUN_MEMORY:-2Gi}"
  local tmo="${CLOUD_RUN_TIMEOUT:-900s}"
  local -a run_args=(
    --image="${IMAGE}"
    --region="${REGION}"
    --platform=managed
    --allow-unauthenticated
    --quiet
    --min-instances="${min_inst}"
    --memory="${mem}"
    --timeout="${tmo}"
    --cpu-boost
  )
  echo "==> Cloud Run min-instances=${min_inst} memory=${mem} timeout=${tmo} --cpu-boost"
  if [[ "${CLOUD_RUN_CPU_THROTTLING:-}" == "1" ]]; then
    echo "==> Cloud Run CPU throttling enabled (idle savings; may increase 503s)"
  else
    run_args+=(--no-cpu-throttling)
    echo "==> Cloud Run --no-cpu-throttling (default)"
  fi
  gcloud run deploy "${SERVICE}" "${run_args[@]}"
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
