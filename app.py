"""
Pyx API — WSGI app for gunicorn / Cloud Run (pyxaiapi).

Optional API key: set PYX_API_KEY or PYX_API_KEYS (comma-separated) in the environment.
Clients send the key in header X-API-Key or Authorization: Bearer <key>.
If no keys are set, the API works without auth (open).
"""

import json
import os

from flask import Flask, request, jsonify

from Pyx_ai_moderator import PyxAI, BAN_LINE, censor_letters
from Pyx_ai_code import complete as code_complete, explain as code_explain, refactor as code_refactor, health as code_health
from Pyx_ai_check import check_code, check_three_js, __version__ as check_version
from Pyx_ai_analyze import analyze_code, analyze_three_js, __version__ as analyze_version

app = Flask(__name__)
pyx = PyxAI()

# Optional API key auth: if set, requests must include a valid key
_API_KEYS: set = set()
_raw = os.environ.get("PYX_API_KEY") or os.environ.get("PYX_API_KEYS") or ""
if _raw:
    _API_KEYS = {k.strip() for k in _raw.split(",") if k.strip()}
_REQUIRE_API_KEY = len(_API_KEYS) > 0


def _get_api_key_from_request():
    """Read API key from X-API-Key or Authorization: Bearer <key>."""
    key = request.headers.get("X-API-Key", "").strip()
    if not key and request.headers.get("Authorization", "").startswith("Bearer "):
        key = request.headers.get("Authorization", "Bearer ").replace("Bearer ", "", 1).strip()
    return key


def _require_api_key():
    if not _REQUIRE_API_KEY:
        return None
    if request.method == "OPTIONS":
        return None
    # Allow GET /health and GET / without a key so load balancers can check
    if request.method == "GET" and request.path in ("/", "/health"):
        return None
    key = _get_api_key_from_request()
    if key and key in _API_KEYS:
        return None
    return jsonify({"error": "Missing or invalid API key"}), 401


@app.before_request
def check_api_key():
    r = _require_api_key()
    if r is not None:
        return r


@app.after_request
def cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-API-Key, Authorization"
    return response


@app.route("/health")
@app.route("/")
def health():
    firebase_connected = bool(getattr(pyx, "_db", None))
    return jsonify({
        "status": "ok",
        "services": {
            "pyx_moderator": "ok",
            "pyx_code": "ok",
            "pyx_check": "ok",
            "pyx_analyze": "ok",
            "firebase": "connected" if firebase_connected else "offline",
        },
        "firebase_connected": firebase_connected,
    })


@app.route("/score", methods=["GET", "POST", "OPTIONS"])
def score():
    if request.method == "OPTIONS":
        return "", 204
    if request.method != "POST":
        return jsonify({"error": "Method not allowed"}), 405
    data = request.get_json(silent=True) or {}
    text = data.get("text")
    if text is None:
        return jsonify({"error": "Missing \"text\" in body"}), 400
    if not isinstance(text, str):
        return jsonify({"error": "\"text\" must be a string"}), 400
    if len(text) > 1_000_000:
        return jsonify({"error": "Text too long"}), 413
    s = pyx.score(text)
    bad = s >= BAN_LINE
    censored = censor_letters(text) if bad else text
    return jsonify({
        "score": round(s, 4),
        "bad": bad,
        "censored": censored,
    })


@app.route("/ai-decide", methods=["POST", "OPTIONS"])
def ai_decide():
    """Same as /score but also trains and writes to Firestore. Use for game AI decisions so Pyx learns."""
    if request.method == "OPTIONS":
        return "", 204
    if request.method != "POST":
        return jsonify({"error": "Method not allowed"}), 405
    data = request.get_json(silent=True) or {}
    text = data.get("text")
    if text is None:
        return jsonify({"error": "Missing \"text\" in body"}), 400
    if not isinstance(text, str):
        return jsonify({"error": "\"text\" must be a string"}), 400
    if len(text) > 1_000_000:
        return jsonify({"error": "Text too long"}), 413
    category = data.get("category", "phrases")
    if not isinstance(category, str):
        category = "phrases"
    try:
        safe, s = pyx.ai_decide(text, category=category)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    bad = not safe
    censored = censor_letters(text) if bad else text
    return jsonify({
        "score": round(s, 4),
        "bad": bad,
        "censored": censored,
        "safe": safe,
    })


@app.route("/feedback", methods=["POST", "OPTIONS"])
def feedback():
    """Send a label so Pyx learns and (if Firestore is configured) syncs to the database."""
    if request.method == "OPTIONS":
        return "", 204
    if request.method != "POST":
        return jsonify({"error": "Method not allowed"}), 405
    data = request.get_json(silent=True) or {}
    text = data.get("text")
    safe = data.get("safe")
    if text is None:
        return jsonify({"error": "Missing \"text\" in body"}), 400
    if not isinstance(text, str):
        return jsonify({"error": "\"text\" must be a string"}), 400
    if safe is None:
        return jsonify({"error": "Missing \"safe\" in body (true or false)"}), 400
    if not isinstance(safe, bool):
        return jsonify({"error": "\"safe\" must be a boolean"}), 400
    if len(text) > 1_000_000:
        return jsonify({"error": "Text too long"}), 413
    category = data.get("category", "phrases")
    if not isinstance(category, str):
        category = "phrases"
    try:
        message = pyx.set_label(text, safe, category=category)
        return jsonify({"ok": True, "message": message})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ---- Pyx Code ----
@app.route("/code/health")
def code_health_route():
    h = code_health()
    return jsonify({"service": "pyx_code", "status": "ok", **h})


@app.route("/code/complete", methods=["POST", "OPTIONS"])
def code_complete_route():
    if request.method == "OPTIONS":
        return "", 204
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 256)
    if not isinstance(prompt, str):
        return jsonify({"error": "\"prompt\" must be a string"}), 400
    try:
        out = code_complete(prompt, max_tokens=max_tokens)
        return jsonify({"completion": out})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/code/explain", methods=["POST", "OPTIONS"])
def code_explain_route():
    if request.method == "OPTIONS":
        return "", 204
    data = request.get_json(silent=True) or {}
    snippet = data.get("snippet", "")
    if not isinstance(snippet, str):
        return jsonify({"error": "\"snippet\" must be a string"}), 400
    try:
        out = code_explain(snippet)
        return jsonify({"explanation": out})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/code/refactor", methods=["POST", "OPTIONS"])
def code_refactor_route():
    if request.method == "OPTIONS":
        return "", 204
    data = request.get_json(silent=True) or {}
    snippet = data.get("snippet", "")
    instruction = data.get("instruction")
    if not isinstance(snippet, str):
        return jsonify({"error": "\"snippet\" must be a string"}), 400
    try:
        out = code_refactor(snippet, instruction=instruction)
        return jsonify({"refactored": out})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---- Pyx Check ----
@app.route("/check/health")
def check_health_route():
    return jsonify({"service": "pyx_check", "status": "ok", "version": check_version})


@app.route("/check", methods=["POST", "OPTIONS"])
@app.route("/check/three", methods=["POST", "OPTIONS"])
def check_route():
    if request.method == "OPTIONS":
        return "", 204
    data = request.get_json(silent=True) or {}
    source = data.get("source", "")
    language = data.get("language", "javascript")
    if not isinstance(source, str):
        return jsonify({"error": "\"source\" must be a string"}), 400
    try:
        if request.path.endswith("/three"):
            out = check_three_js(source, data.get("options"))
        else:
            out = check_code(source, language=language, options=data.get("options"))
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---- Pyx Analyze ----
@app.route("/analyze/health")
def analyze_health_route():
    return jsonify({"service": "pyx_analyze", "status": "ok", "version": analyze_version})


@app.route("/analyze", methods=["POST", "OPTIONS"])
@app.route("/analyze/three", methods=["POST", "OPTIONS"])
def analyze_route():
    if request.method == "OPTIONS":
        return "", 204
    data = request.get_json(silent=True) or {}
    source = data.get("source", "")
    language = data.get("language", "javascript")
    use_filter = data.get("use_content_filter", True)
    if not isinstance(source, str):
        return jsonify({"error": "\"source\" must be a string"}), 400
    try:
        if request.path.endswith("/three"):
            out = analyze_three_js(source, use_content_filter=use_filter)
        else:
            out = analyze_code(source, language=language, use_content_filter=use_filter)
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
