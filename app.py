"""
Pyx API — WSGI app for gunicorn / Cloud Run (pyxaiapi).

Optional API key: set PYX_API_KEY or PYX_API_KEYS (comma-separated) in the environment.
Clients send the key in header X-API-Key or Authorization: Bearer <key>.
If no keys are set, the API works without auth (open).
"""

import json
import os
import urllib.error
import urllib.request

from flask import Flask, request, jsonify

from Pyx_ai_moderator import PyxAI, BAN_LINE, censor_letters
from Pyx_ai_code import complete as code_complete, explain as code_explain, refactor as code_refactor, health as code_health
from Pyx_ai_check import check_code, check_three_js, __version__ as check_version
from Pyx_ai_analyze import analyze_code, analyze_three_js, __version__ as analyze_version

app = Flask(__name__)
pyx = PyxAI()

# Pyx Talk (Llama-class chat via OpenAI-compatible API, e.g. Groq)
_TALK_MAX_MSG_LEN = 4000
_TALK_MAX_MESSAGES = 24
_TALK_SYSTEM = os.environ.get(
    "PYX_TALK_SYSTEM",
    "You are Pyx Talk, a helpful, friendly assistant. Keep answers concise and clear. "
    "Stay safe for general audiences; refuse harmful or explicit requests briefly and offer something helpful instead.",
)

# Reasoning modes: Llama on Groq — fast (8B instant), smart / thinking (70B versatile + prompts).
_TALK_MODES = frozenset({"fast", "smart", "thinking"})
_TALK_MODE_SPECS = {
    "fast": {
        "model_env": "PYX_TALK_MODEL_FAST",
        "default_model": "llama-3.1-8b-instant",
        "max_tokens": 384,
        "temperature": 0.55,
        "system_suffix": " Mode: fast. Prefer short, direct answers. Skip long preambles unless the user asks for depth.",
    },
    "smart": {
        "model_env": "PYX_TALK_MODEL_SMART",
        "default_model": "llama-3.3-70b-versatile",
        "max_tokens": 1024,
        "temperature": 0.5,
        "system_suffix": " Mode: smart. Prioritize correctness and clarity. Structure longer answers when it helps (brief setup, then the answer).",
    },
    "thinking": {
        "model_env": "PYX_TALK_MODEL_THINKING",
        "default_model": "llama-3.3-70b-versatile",
        "max_tokens": 2048,
        "temperature": 0.35,
        "system_suffix": (
            " Mode: thinking. For anything non-trivial, reason step by step first "
            '(use a short heading like "Reasoning:" with numbered steps), then give a final concise answer under '
            '"Answer:". For trivial greetings or one-word factual lookups, answer directly without the full template.'
        ),
    },
}


def _normalize_talk_messages(raw):
    if not isinstance(raw, list):
        return None, '"messages" must be a list'
    if len(raw) > _TALK_MAX_MESSAGES:
        raw = raw[-_TALK_MAX_MESSAGES:]
    out = []
    for m in raw:
        if not isinstance(m, dict):
            return None, "each message must be an object"
        role = m.get("role")
        content = m.get("content")
        if role not in ("user", "assistant"):
            return None, '"role" must be "user" or "assistant"'
        if not isinstance(content, str):
            return None, '"content" must be a string'
        content = content.strip()
        if len(content) > _TALK_MAX_MSG_LEN:
            return None, "message too long"
        if not content:
            return None, "empty message"
        out.append({"role": role, "content": content})
    if not out or out[-1]["role"] != "user":
        return None, "last message must be from user"
    return out, None


def _normalize_talk_mode(raw):
    """Return (mode_str or None, error_str or None). Default mode is fast."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return "fast", None
    if not isinstance(raw, str):
        return None, '"mode" must be a string'
    m = raw.strip().lower()
    aliases = {
        "quick": "fast",
        "speed": "fast",
        "deep": "thinking",
        "reason": "thinking",
        "reasoning": "thinking",
    }
    m = aliases.get(m, m)
    if m not in _TALK_MODES:
        return None, '"mode" must be "fast", "smart", or "thinking"'
    return m, None


def _groq_openai_chat(messages_for_api, mode="fast"):
    """Call OpenAI-compatible chat completions. Returns (reply_text, model_id) or raises.
    If PYX_TALK_LLM_KEY is unset, returns (None, None) without calling the network."""
    key = os.environ.get("PYX_TALK_LLM_KEY", "").strip()
    if not key:
        return None, None
    url = os.environ.get(
        "PYX_TALK_LLM_URL",
        "https://api.groq.com/openai/v1/chat/completions",
    ).strip()
    spec = _TALK_MODE_SPECS.get(mode) or _TALK_MODE_SPECS["fast"]
    model = (os.environ.get(spec["model_env"]) or "").strip() or spec["default_model"]
    max_tokens = min(max(spec["max_tokens"], 64), 2048)
    temperature = float(spec["temperature"])
    system_content = _TALK_SYSTEM + spec["system_suffix"]
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system_content}] + messages_for_api,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + key,
        },
        method="POST",
    )
    # Fast mode: short timeout so failures return quickly; larger models may need longer.
    timeout_s = 32 if mode == "fast" else 90
    try:
        timeout_s = max(8, min(int(os.environ.get("PYX_TALK_TIMEOUT", str(timeout_s))), 120))
    except ValueError:
        pass
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("LLM returned no choices")
    msg = choices[0].get("message") or {}
    content = (msg.get("content") or "").strip()
    if not content:
        raise ValueError("empty LLM content")
    return content, model

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
            "pyx_talk": "ok",
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


@app.route("/talk", methods=["POST", "OPTIONS"])
def talk():
    """Pyx Talk: user text is not blocked by Pyx (still scored for `score` in the response).
    Assistant replies are scored; inappropriate model output is replaced with a safe message."""
    if request.method == "OPTIONS":
        return "", 204
    if request.method != "POST":
        return jsonify({"error": "Method not allowed"}), 405
    data = request.get_json(silent=True) or {}
    mode, mode_err = _normalize_talk_mode(data.get("mode"))
    if mode_err:
        return jsonify({"error": mode_err}), 400
    messages, err = _normalize_talk_messages(data.get("messages"))
    if err:
        return jsonify({"error": err}), 400
    last_user = messages[-1]["content"]
    u_score = None
    try:
        u_score = pyx.score(last_user)
    except Exception:
        pass
    llm_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
    try:
        reply, model_used = _groq_openai_chat(llm_messages, mode=mode)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:800]
        return jsonify({"error": "LLM request failed", "status": e.code, "detail": detail}), 502
    except urllib.error.URLError as e:
        return jsonify({"error": "LLM network error", "detail": str(e.reason)}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 502
    if reply is None:
        reply = (
            "Hi — I’m Pyx Talk. An unknown error occurred. Please try again later, or submit an issue on Github."
            "Llama isn’t wired up on this server yet: set PYX_TALK_LLM_KEY (e.g. Groq) "
            "and optional PYX_TALK_MODEL on the Pyx API (Cloud Run) to get full replies."
        )
        model_used = "pyx-fallback"
    reply_blocked = False
    try:
        r_score = pyx.score(reply)
        if pyx.memory.is_banned(r_score):
            reply_blocked = True
            reply = (
                "Oops! I am not comfortable with that question or topic. Lets change the topic."
            )
    except Exception:
        pass
    out = {
        "bad": False,
        "reply": reply,
        "model": model_used or "unknown",
        "mode": mode,
        "reply_moderated": reply_blocked,
    }
    if u_score is not None:
        out["score"] = round(u_score, 4)
    return jsonify(out)


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
