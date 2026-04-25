#!/usr/bin/env bash
# Pyx 1.5 launcher — run Pyx against a local OpenAI-compatible model server.
#
# Default assumes Ollama on http://127.0.0.1:11434. Override any of:
#   PYX_TALK_LLM_URL          (default http://127.0.0.1:11434/v1/chat/completions)
#   PYX_TALK_MODEL_FAST       (default llama3.1:8b-instruct)
#   PYX_TALK_MODEL_SMART      (default llama3.3:70b-instruct)
#   PYX_TALK_MODEL_THINKING   (default llama3.3:70b-instruct)
#   PYX_CODE_MODEL            (default gpt-oss:20b)
#   PYX_PIXEL_MODEL           (default gpt-oss:20b)
#   PYX_PORT                  (default 8080)
#
# Usage:
#   bash scripts/pyx-local.sh                 # start gunicorn + (if available) ollama
#   SKIP_OLLAMA=1 bash scripts/pyx-local.sh   # don't touch ollama, just run Pyx
#
# Installs: https://ollama.com/download   Weights: see PYX_1_5_LOCAL.md

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYX_TALK_LLM_URL="${PYX_TALK_LLM_URL:-http://127.0.0.1:11434/v1/chat/completions}"
unset PYX_TALK_LLM_KEY 2>/dev/null || true
export PYX_TALK_MODEL_FAST="${PYX_TALK_MODEL_FAST:-llama3.1:8b-instruct}"
export PYX_TALK_MODEL_SMART="${PYX_TALK_MODEL_SMART:-llama3.3:70b-instruct}"
export PYX_TALK_MODEL_THINKING="${PYX_TALK_MODEL_THINKING:-llama3.3:70b-instruct}"
export PYX_CODE_MODEL="${PYX_CODE_MODEL:-gpt-oss:20b}"
export PYX_PIXEL_MODEL="${PYX_PIXEL_MODEL:-gpt-oss:20b}"
export PYX_TALK_TIMEOUT="${PYX_TALK_TIMEOUT:-600}"
PYX_PORT="${PYX_PORT:-8080}"

echo "==> Pyx 1.5 local"
echo "    URL    : $PYX_TALK_LLM_URL"
echo "    fast   : $PYX_TALK_MODEL_FAST"
echo "    smart  : $PYX_TALK_MODEL_SMART"
echo "    think  : $PYX_TALK_MODEL_THINKING"
echo "    code   : $PYX_CODE_MODEL"
echo "    pixel  : $PYX_PIXEL_MODEL"

if [[ "${SKIP_OLLAMA:-}" != "1" ]] && command -v ollama >/dev/null 2>&1; then
  if ! curl -sS -m 2 "${PYX_TALK_LLM_URL%/v1/chat/completions}/api/tags" >/dev/null 2>&1; then
    echo "==> starting ollama serve (background)"
    (ollama serve >/tmp/pyx-ollama.log 2>&1 &) || true
    for i in 1 2 3 4 5 6 7 8 9 10; do
      if curl -sS -m 2 "${PYX_TALK_LLM_URL%/v1/chat/completions}/api/tags" >/dev/null 2>&1; then
        echo "==> ollama up"
        break
      fi
      sleep 1
    done
  else
    echo "==> ollama already serving"
  fi
  # Pull any model we reference but don't have yet (best-effort; ignore failures).
  for m in "$PYX_TALK_MODEL_FAST" "$PYX_TALK_MODEL_SMART" "$PYX_TALK_MODEL_THINKING" \
           "$PYX_CODE_MODEL" "$PYX_PIXEL_MODEL"; do
    if ! ollama list 2>/dev/null | awk 'NR>1{print $1}' | grep -qx "$m"; then
      echo "==> ollama pull $m"
      ollama pull "$m" || echo "   (pull failed for $m — continuing)"
    fi
  done
elif [[ "${SKIP_OLLAMA:-}" != "1" ]]; then
  echo "==> ollama not on PATH — install from https://ollama.com/download (or set SKIP_OLLAMA=1)"
fi

# Virtualenv (best-effort)
if [[ ! -d .venv ]]; then
  echo "==> creating .venv"
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install -r requirements.txt
else
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "==> gunicorn on 127.0.0.1:${PYX_PORT}"
exec gunicorn app:app \
  --bind "127.0.0.1:${PYX_PORT}" \
  --worker-class gthread \
  --workers 1 \
  --threads 16 \
  --timeout 600 \
  --graceful-timeout 60 \
  --keep-alive 75
