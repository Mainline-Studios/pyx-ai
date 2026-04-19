#!/usr/bin/env bash
# Stage all changes and commit with a deploy message. No-op if clean or skipped.
#
# Usage:
#   bash scripts/git-deploy-commit.sh [optional reason words...]
#   npm run deploy:githubmessage -- "why you are deploying"
#
# Message rules:
#   - If you pass a reason:  chore: deploy: <reason>
#   - Else if DEPLOY_COMMIT_TAG is set (api / hosting):  chore: deploy (<tag>)
#   - Else:                   chore: deploy (api + hosting)
#
# Env: DEPLOY_SKIP_GIT_COMMIT=1 — skip (only run deploy)
#      DEPLOY_COMMIT_TAG=api|hosting — used when reason is empty (set by run-deploy.sh)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ "${DEPLOY_SKIP_GIT_COMMIT:-}" == "1" ]]; then
  echo "==> (DEPLOY_SKIP_GIT_COMMIT=1, skip git commit)"
  exit 0
fi
if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "==> (not a git repo, skip git commit)"
  exit 0
fi

REASON="${*}"
REASON="${REASON//$'\n'/ }"

if [[ -n "$REASON" ]]; then
  MSG="chore: deploy: ${REASON}"
elif [[ -n "${DEPLOY_COMMIT_TAG:-}" ]]; then
  MSG="chore: deploy (${DEPLOY_COMMIT_TAG})"
else
  MSG="chore: deploy (api + hosting)"
fi

if [[ -z "$(git status --porcelain 2>/dev/null)" ]]; then
  echo "==> Working tree clean — no deploy commit."
  exit 0
fi

git add -A
git commit -m "${MSG}"
echo "==> Committed: ${MSG}"
