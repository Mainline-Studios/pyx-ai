"""
Pyx HTTP function for Firebase (Hosting + Cloud Functions).

Deploy with Firebase so Pixel Place can call Pyx at your Hosting URL (e.g. /api/score).
No server 24/7 — runs only when a request comes in.

Before first deploy: copy Pyx_ai_moderator.py and data/ into functions/ (see DEPLOY_FIREBASE.md).
"""

import json
import sys
from pathlib import Path
from typing import Tuple

# Allow importing Pyx_ai_moderator from repo root (local dev) or from functions/ (after copy for deploy)
_functions_dir = Path(__file__).resolve().parent
if str(_functions_dir.parent) not in sys.path:
    sys.path.insert(0, str(_functions_dir.parent))
try:
    from Pyx_ai_moderator import PyxAI, BAN_LINE, censor_letters
except ImportError:
    sys.path.insert(0, str(_functions_dir))
    from Pyx_ai_moderator import PyxAI, BAN_LINE, censor_letters

from firebase_functions import https_fn, options

# One Pyx instance per container (reused across invocations)
pyx = PyxAI()


def _score_body(body: dict) -> Tuple[dict, int]:
    """Parse body, score text, return (response_dict, status_code)."""
    text = body.get("text")
    if text is None:
        return ({"error": "Missing \"text\" in body"}, 400)
    if not isinstance(text, str):
        return ({"error": "\"text\" must be a string"}, 400)
    if len(text) > 1_000_000:
        return ({"error": "Text too long"}, 413)
    score = pyx.score(text)
    bad = score >= BAN_LINE
    censored = censor_letters(text) if bad else text
    return (
        {
            "score": round(score, 4),
            "bad": bad,
            "censored": censored,
        },
        200,
    )


@https_fn.on_request(cors=options.CorsOptions(cors_origins="*", cors_methods=["get", "post", "options"]))
def pyxscore(req: https_fn.Request) -> https_fn.Response:
    """
    POST /score — body: {"text": "..."} → {"score", "bad", "censored"}
    GET  /      — health: {"status": "ok", "service": "pyx"}
    """
    # CORS preflight
    if req.method == "OPTIONS":
        return https_fn.Response("", status=204)

    # GET = health
    if req.method == "GET":
        return https_fn.Response(
            json.dumps({"status": "ok", "service": "pyx"}),
            status=200,
            headers={"Content-Type": "application/json; charset=utf-8"},
        )

    # POST = score
    if req.method != "POST":
        return https_fn.Response(
            json.dumps({"error": "Method not allowed"}),
            status=405,
            headers={"Content-Type": "application/json; charset=utf-8"},
        )

    try:
        body = req.get_json(silent=True) or {}
    except Exception:
        body = {}
    if not body and req.get_data(as_text=True):
        try:
            body = json.loads(req.get_data(as_text=True))
        except json.JSONDecodeError:
            return https_fn.Response(
                json.dumps({"error": "Invalid JSON"}),
                status=400,
                headers={"Content-Type": "application/json; charset=utf-8"},
            )

    data, status = _score_body(body)
    return https_fn.Response(
        json.dumps(data),
        status=status,
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
