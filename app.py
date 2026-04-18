"""
Pyx API — WSGI app for gunicorn / Cloud Run (pyxaiapi).

Optional API key: set PYX_API_KEY or PYX_API_KEYS (comma-separated) in the environment.
Clients send the key in header X-API-Key or Authorization: Bearer <key>.
If no keys are set, the API works without auth (open).
"""

import html
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from flask import Flask, request, jsonify, Response, stream_with_context

from Pyx_ai_moderator import PyxAI, BAN_LINE, censor_letters
from Pyx_ai_code import complete as code_complete, explain as code_explain, refactor as code_refactor, health as code_health
from Pyx_ai_check import check_code, check_three_js, __version__ as check_version
from Pyx_ai_analyze import analyze_code, analyze_three_js, __version__ as analyze_version

app = Flask(__name__)
pyx = PyxAI()

# Pyx Talk (Llama-class chat via OpenAI-compatible API, e.g. Groq or local Ollama)
_GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
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
            ' Mode: Pyx Talk Reasoning 1.0 — reasoning is mandatory on every reply, no exceptions. '
            'Always write two sections in this exact order: (1) Start with a line **Reasoning:** then numbered '
            "step-by-step working (even for greetings or tiny questions — at least one short step stating what you're doing). "
            '(2) Then a blank line and a line **Answer:** followed by the final user-facing reply (clear and concise). '
            "Never skip the Reasoning block. Never put the final answer before **Answer:**."
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


def _as_bool(v):
    if v is True:
        return True
    if v is False or v is None:
        return False
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def _mentions_space_or_mission(t: str) -> bool:
    """Topics where training data is often wrong about dates / flight status."""
    low = (t or "").lower()
    if any(
        x in low
        for x in (
            "artemis",
            "orion",
            "splashdown",
            "spacex",
            "starship",
            "falcon heavy",
            "falcon 9",
            "crew dragon",
            "dragon capsule",
            "sls",
            "moon mission",
            "lunar mission",
            "gateway",
            "hubble",
            "james webb",
            "jwst",
            "mars rover",
            "perseverance",
            "curiosity rover",
        )
    ):
        return True
    if "nasa" in low:
        return True
    if " iss" in low or low.startswith("iss ") or "international space station" in low:
        return True
    if re.search(r"\b(has|have)\s+.+\s+launched\b", low):
        return True
    if re.search(r"\b(splash(?:ed)?\s+down|splashed\s+down)\b", low):
        return True
    return False


def _needs_live_web(user_text: str) -> bool:
    """Always fetch web for these asks — training cutoffs miss crewed flight status, etc."""
    return _mentions_space_or_mission(user_text)


def _web_auto_trigger(user_text: str) -> bool:
    """Heuristic: turn on web search without explicit user toggle."""
    t = (user_text or "").lower()
    if len(t) < 6:
        return False
    if _mentions_space_or_mission(user_text):
        return True
    for y in ("2024", "2025", "2026"):
        if y in t:
            return True
    needles = (
        "latest", "news", "today", "right now", "current events", "breaking",
        "who won", "stock price", "weather in", "release date", "announced",
        "how much does", "what happened",
    )
    if any(n in t for n in needles):
        return True
    if "?" in user_text and len(user_text) > 12:
        return any(
            k in t
            for k in ("when", "where", "who is", "what is the", "how many", "why did", "current ")
        )
    return False


def _strip_html_fragment(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    return html.unescape(re.sub(r"\s+", " ", s).strip())


def _ddg_is_ad_link(href: str) -> bool:
    h = (href or "").lower()
    return "duckduckgo.com/y.js" in h or "ad_provider=" in h or "ad_domain=" in h


def _unwrap_duck_redirect(href: str) -> str:
    href = (href or "").strip()
    if not href:
        return ""
    href = html.unescape(href)
    if href.startswith("//"):
        href = "https:" + href
    if "duckduckgo.com/l/?" in href or "duckduckgo.com/l?" in href:
        try:
            q = urllib.parse.urlparse(href).query
            params = urllib.parse.parse_qs(q)
            if "uddg" in params:
                return urllib.parse.unquote(params["uddg"][0])
        except Exception:
            pass
    return href


def _local_web_search_snippets(query: str):
    """Pyx local web search: DuckDuckGo HTML (no API keys). Returns (text, provider, error)."""
    query = (query or "").strip()[:500]
    if not query:
        return "", None, "empty query"
    max_results = 6
    try:
        max_results = max(1, min(int(os.environ.get("PYX_TALK_WEB_MAX_RESULTS", "6")), 12))
    except ValueError:
        pass
    cap = min(max(int(os.environ.get("PYX_TALK_WEB_CONTEXT_CHARS", "8000")), 2000), 32000)
    timeout = max(5, min(int(os.environ.get("PYX_TALK_WEB_TIMEOUT", "22")), 45))

    ddg_url = (os.environ.get("PYX_TALK_WEB_HTML_URL") or "https://html.duckduckgo.com/html/").strip()
    ua = (os.environ.get("PYX_TALK_USER_AGENT") or "").strip() or (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )
    form = urllib.parse.urlencode({"q": query, "b": ""}).encode("utf-8")
    req = urllib.request.Request(
        ddg_url,
        data=form,
        headers={
            "User-Agent": ua,
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Content-Type": "application/x-www-form-urlencoded",
            "Referer": "https://duckduckgo.com/",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            page = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        return "", "local-web", str(e)[:300]

    # DDG: result__a (title+link), optional result__snippet in the same result block
    blocks = re.findall(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>(?s:.*?)class="result__snippet"[^>]*>(.*?)</a>',
        page,
        re.IGNORECASE | re.DOTALL,
    )
    lines = []
    for href, title_html, snip_html in blocks:
        if _ddg_is_ad_link(href):
            continue
        url = _unwrap_duck_redirect(href)
        if _ddg_is_ad_link(url):
            continue
        title = _strip_html_fragment(title_html)
        snippet = _strip_html_fragment(snip_html)
        if not title and not snippet:
            continue
        lines.append(f"- {title}\n  {url}\n  {snippet[:1200]}")
        if len(lines) >= max_results:
            break

    if not lines:
        # Fallback: titles/links only (older or alternate markup)
        for m in re.finditer(
            r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            page,
            re.IGNORECASE | re.DOTALL,
        ):
            href = m.group(1)
            if _ddg_is_ad_link(href):
                continue
            url = _unwrap_duck_redirect(href)
            if _ddg_is_ad_link(url):
                continue
            title = _strip_html_fragment(m.group(2))
            if title or url:
                lines.append(f"- {title}\n  {url}\n  ")
            if len(lines) >= max_results:
                break

    text = "\n".join(lines).strip()
    if not text:
        return "", "local-web", "no results (blocked, empty query, or page layout changed)"
    return text[:cap], "local-web", None


def _talk_web_snippets(query: str):
    """Local-first web search for Pyx Talk (no third-party search API keys)."""
    return _local_web_search_snippets(query)


def _enhance_talk_search_query(user_text: str) -> str:
    """Bias DuckDuckGo toward recent pages (news, games, human spaceflight, etc.)."""
    q = (user_text or "").strip()
    if not q:
        return q
    low = q.lower()
    now = datetime.now(timezone.utc)
    year = str(now.year)
    prev_y = str(now.year - 1)
    if year in q or prev_y in q:
        return q

    recency_hints = (
        "news", "latest", "current", "today", "breaking",
        "announcement", "released", "release", "launch", "patch", "update",
        "dlc", "trailer", "delay", "rumor", "rumour",
    )
    wants_year = any(h in low for h in recency_hints) or _mentions_space_or_mission(user_text)

    if not wants_year:
        return q

    if _mentions_space_or_mission(user_text):
        mission_flight = any(
            x in low
            for x in (
                "artemis",
                "orion",
                "splashdown",
                "crew dragon",
                "dragon capsule",
                "starship",
                "sls",
                "moon mission",
                "lunar mission",
                "gateway",
            )
        )
        mission_flight = mission_flight or " iss" in low or low.startswith("iss ")
        mission_flight = mission_flight or "international space station" in low
        mission_flight = mission_flight or re.search(r"\b(has|have)\s+.+\s+launched\b", low)
        mission_flight = mission_flight or re.search(
            r"\b(splash(?:ed)?\s+down|splashed\s+down)\b", low
        )
        if mission_flight:
            return f"{q} {year} mission flight status".strip()
        return f"{q} {year} latest".strip()

    return f"{q} {year}".strip()


def _groq_openai_prepare(messages_for_api, mode="fast", web_context="", ground_web=False):
    """Build shared OpenAI-compatible chat request pieces. Returns None if Groq is selected but no API key."""
    key = os.environ.get("PYX_TALK_LLM_KEY", "").strip()
    url = os.environ.get("PYX_TALK_LLM_URL", _GROQ_CHAT_COMPLETIONS_URL).strip()
    url_norm = url.rstrip("/").lower()
    groq_norm = _GROQ_CHAT_COMPLETIONS_URL.rstrip("/").lower()
    if not key and url_norm == groq_norm:
        return None
    spec = _TALK_MODE_SPECS.get(mode) or _TALK_MODE_SPECS["fast"]
    model = (os.environ.get(spec["model_env"]) or "").strip() or spec["default_model"]
    max_tokens = min(max(spec["max_tokens"], 64), 2048)
    temperature = float(spec["temperature"])
    if ground_web:
        try:
            cap_t = float(os.environ.get("PYX_TALK_WEB_TEMP_CAP", "0.38"))
        except ValueError:
            cap_t = 0.38
        temperature = max(0.12, min(temperature, cap_t))
    web_block = ""
    ctx = (web_context or "").strip()
    if ctx:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        web_block = (
            f"\n\nToday’s date (UTC) is {today}. The user asked for web-grounded information.\n"
            "RULES — follow strictly:\n"
            "1) For releases, delays, trailers, DLC, patches, and news, ONLY state facts that are directly supported by "
            "the search snippets below. If a detail is not in the snippets, do not state it.\n"
            "2) Do NOT supplement with memorized hype, old marketing cycles, or plausible-sounding but uncited claims from training data.\n"
            "3) If snippets look outdated vs today’s date, or contradict each other, say that clearly and tell the user to verify on a live source.\n"
            "4) Cite the site name or URL from the snippet for each major claim.\n"
            "5) If snippets are empty or useless, say search didn’t return enough fresh info—do not invent a news roundup.\n"
            "6) Human spaceflight (Artemis, Orion, Crew Dragon, ISS crews, splashdowns, etc.): do NOT guess launch, in-flight, "
            "or recovery status from memory. Training data is often wrong here. Only state status if snippets explicitly support it "
            "(with timing). If snippets are thin, say you can’t confirm from search alone and point to NASA / the operator’s official site.\n\n"
            "--- Web search snippets ---\n" + ctx
        )
    system_content = _TALK_SYSTEM + spec["system_suffix"] + web_block
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system_content}] + messages_for_api,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    ua = (os.environ.get("PYX_TALK_USER_AGENT") or "").strip() or (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )
    headers = {
        "Content-Type": "application/json",
        "User-Agent": ua,
    }
    if key:
        headers["Authorization"] = "Bearer " + key
    timeout_s = 32 if mode == "fast" else 90
    if url_norm != groq_norm:
        timeout_s = max(timeout_s, 120)
    try:
        cap = 600 if url_norm != groq_norm else 120
        timeout_s = max(8, min(int(os.environ.get("PYX_TALK_TIMEOUT", str(timeout_s))), cap))
    except ValueError:
        pass
    return {
        "url": url,
        "headers": headers,
        "body": body,
        "model": model,
        "timeout_s": timeout_s,
    }


def _groq_openai_stream_deltas(prep):
    """Stream assistant content tokens from OpenAI-compatible SSE. Yields str fragments."""
    if prep is None:
        return
    body = {**prep["body"], "stream": True}
    headers = {**prep["headers"], "Accept": "text/event-stream"}
    req = urllib.request.Request(
        prep["url"],
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=prep["timeout_s"]) as resp:
        while True:
            raw = resp.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            for ch in obj.get("choices") or []:
                delta = ch.get("delta") or {}
                piece = delta.get("content")
                if piece:
                    yield piece


def _groq_openai_chat(messages_for_api, mode="fast", web_context="", ground_web=False):
    """Call OpenAI-compatible chat completions. Returns (reply_text, model_id) or raises.
    Groq requires PYX_TALK_LLM_KEY. For a custom PYX_TALK_LLM_URL (e.g. local Ollama), the key may be omitted."""
    prep = _groq_openai_prepare(messages_for_api, mode, web_context, ground_web)
    if prep is None:
        return None, None
    headers = {**prep["headers"], "Accept": "application/json"}
    req = urllib.request.Request(
        prep["url"],
        data=json.dumps(prep["body"]).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=prep["timeout_s"]) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("LLM returned no choices")
    msg = choices[0].get("message") or {}
    content = (msg.get("content") or "").strip()
    if not content:
        raise ValueError("empty LLM content")
    return content, prep["model"]

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


def _talk_sse_event(obj: dict) -> str:
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


@app.route("/talk", methods=["POST", "OPTIONS"])
def talk():
    """Pyx Talk: LLM chat with optional built-in web search (`use_web`, `use_web_auto`, local HTML fetch)."""
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

    use_web = _as_bool(data.get("use_web"))
    use_web_auto = _as_bool(data.get("use_web_auto"))
    want_stream = _as_bool(data.get("stream"))
    # Space / mission questions: always search so answers aren’t stuck at training cutoff.
    do_web = use_web or (use_web_auto and _web_auto_trigger(last_user)) or _needs_live_web(last_user)
    web_meta = {"used": False, "provider": None, "error": None}
    web_context = ""
    if do_web:
        search_query = _enhance_talk_search_query(last_user)
        web_meta["query"] = search_query
        snippets, provider, werr = _talk_web_snippets(search_query)
        web_meta["used"] = True
        web_meta["provider"] = provider
        web_meta["error"] = werr
        if snippets:
            web_context = snippets
        elif werr:
            web_context = f"(Search note: {werr})"

    llm_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
    ground_web = bool(
        web_context.strip()
        and not web_context.strip().lower().startswith("(search note:")
    )

    if want_stream:
        prep = _groq_openai_prepare(llm_messages, mode, web_context, ground_web)

        def generate():
            meta = {"type": "meta", "mode": mode, "web_search": web_meta, "bad": False}
            if prep:
                meta["model"] = prep["model"]
            else:
                meta["model"] = "pyx-fallback"
            if u_score is not None:
                meta["score"] = round(u_score, 4)
            yield _talk_sse_event(meta)
            if prep is None:
                fb = (
                    "Hi — I’m Pyx Talk. No LLM is configured for this server yet. "
                    "For Groq, set PYX_TALK_LLM_KEY. For a local OpenAI-compatible API (e.g. Ollama), "
                    "set PYX_TALK_LLM_URL to your /v1/chat/completions endpoint (key optional). "
                    "If the API runs on your computer, run Pyx there too or use a tunnel — Cloud Run cannot reach your localhost."
                )
                yield _talk_sse_event({"type": "delta", "t": fb})
                yield _talk_sse_event({"type": "done", "model": "pyx-fallback"})
                return
            try:
                for piece in _groq_openai_stream_deltas(prep):
                    yield _talk_sse_event({"type": "delta", "t": piece})
                yield _talk_sse_event({"type": "done", "model": prep["model"]})
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", errors="replace")[:800]
                yield _talk_sse_event(
                    {
                        "type": "error",
                        "message": "LLM request failed",
                        "status": e.code,
                        "detail": detail,
                    }
                )
            except urllib.error.URLError as e:
                yield _talk_sse_event({"type": "error", "message": "LLM network error", "detail": str(e.reason)})
            except Exception as e:
                yield _talk_sse_event({"type": "error", "message": str(e)})

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    try:
        reply, model_used = _groq_openai_chat(
            llm_messages,
            mode=mode,
            web_context=web_context,
            ground_web=ground_web,
        )
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:800]
        return jsonify({"error": "LLM request failed", "status": e.code, "detail": detail}), 502
    except urllib.error.URLError as e:
        return jsonify({"error": "LLM network error", "detail": str(e.reason)}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 502
    if reply is None:
        reply = (
            "Hi — I’m Pyx Talk. No LLM is configured for this server yet. "
            "For Groq, set PYX_TALK_LLM_KEY. For a local OpenAI-compatible API (e.g. Ollama), "
            "set PYX_TALK_LLM_URL to your /v1/chat/completions endpoint (key optional). "
            "If the API runs on your computer, run Pyx there too or use a tunnel — Cloud Run cannot reach your localhost."
        )
        model_used = "pyx-fallback"
    out = {
        "bad": False,
        "reply": reply,
        "model": model_used or "unknown",
        "mode": mode,
        "web_search": web_meta,
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
