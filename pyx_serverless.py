"""
Pyx serverless — run only when called (no server 24/7).

Deploy this as AWS Lambda (or Google Cloud Functions) so Pixel Place can call
Pyx without running a dedicated server. It runs only when a request comes in.

  AWS Lambda: set handler to pyx_serverless.handler
  Request: POST with body {"text": "..."}
  Response: {"score": float, "bad": bool, "censored": "..."}
"""

import json
import base64
from typing import Tuple

from Pyx_ai_moderator import PyxAI, BAN_LINE, censor_letters

# One Pyx instance per container (reused across invocations)
pyx = PyxAI()

CORS_HEADERS = {
    "Content-Type": "application/json; charset=utf-8",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}


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


def handler(event, context):
    """
    AWS Lambda handler (API Gateway HTTP API or REST API).
    Also works with Google Cloud Functions (event = Flask request object or dict).
    """
    # API Gateway HTTP API (v2) or REST API
    if isinstance(event.get("body"), str):
        raw = event["body"]
        if event.get("isBase64Encoded"):
            raw = base64.b64decode(raw).decode("utf-8")
        try:
            body = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            return {
                "statusCode": 400,
                "headers": CORS_HEADERS,
                "body": json.dumps({"error": "Invalid JSON"}),
            }
    else:
        body = event.get("body") or {}

    # OPTIONS (CORS preflight)
    if event.get("requestContext", {}).get("http", {}).get("method") == "OPTIONS":
        return {"statusCode": 204, "headers": CORS_HEADERS, "body": ""}
    if event.get("httpMethod") == "OPTIONS":
        return {"statusCode": 204, "headers": CORS_HEADERS, "body": ""}

    # GET /health
    method = event.get("requestContext", {}).get("http", {}).get("method") or event.get("httpMethod") or ""
    if method == "GET":
        return {
            "statusCode": 200,
            "headers": CORS_HEADERS,
            "body": json.dumps({"status": "ok", "service": "pyx"}),
        }

    # POST /score
    data, status = _score_body(body)
    return {
        "statusCode": status,
        "headers": CORS_HEADERS,
        "body": json.dumps(data),
    }
