#!/usr/bin/env bash
# Usage: bash scripts/run-deploy.sh <all|api|hosting> [-- optional reason...]
# npm forwards args after -- to this script, e.g.:
#   npm run deploy -- fix Groq rate limits
#   npm run deploy:api -- bump Pyx Talk prompt
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "Syncing local branch with remote..."
git pull

TARGET="${1:?usage: run-deploy.sh all|api|hosting [reason...]}"
shift

export DEPLOY_COMMIT_TAG=""
case "$TARGET" in
  all) ;;
  api) DEPLOY_COMMIT_TAG="api" ;;
  hosting) DEPLOY_COMMIT_TAG="hosting" ;;
  *)
    echo "run-deploy.sh: target must be all, api, or hosting" >&2
    exit 1
    ;;
esac

bash scripts/git-deploy-commit.sh "$@"
bash scripts/deploy.sh "${TARGET}"
