"""
Pyx API — WSGI app for gunicorn / Cloud Run (pyxaiapi).

Optional API key: set PYX_API_KEY or PYX_API_KEYS (comma-separated) in the environment.
Clients send the key in header X-API-Key or Authorization: Bearer <key>.
If no keys are set, the API works without auth (open).
"""

import html
import json
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from flask import Flask, request, jsonify, Response, stream_with_context

from Pyx_ai_moderator import PyxAI, BAN_LINE, censor_letters
from Pyx_ai_check import check_code, check_three_js, __version__ as check_version
from Pyx_ai_analyze import analyze_code, analyze_three_js, __version__ as analyze_version
from werkzeug.exceptions import HTTPException

import pyx13_preview

try:
    from Pyx_ai_code import (
        complete as code_complete,
        explain as code_explain,
        refactor as code_refactor,
        health as code_health,
    )
except Exception:
    def _code_module_unavailable(*_args, **_kwargs):
        raise RuntimeError("Pyx_ai_code module unavailable on this deployment.")

    code_complete = _code_module_unavailable
    code_explain = _code_module_unavailable
    code_refactor = _code_module_unavailable

    def code_health():
        return {"status": "error", "error": "Pyx_ai_code module unavailable"}

app = Flask(__name__)


def _json_safe_score(x):
    """Float safe for JSON (NaN/Inf break jsonify). Returns rounded float or None."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return round(v, 4)
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

# Reasoning modes: Groq defaults align with Pyx Mini (Llama-class fast), Pyx 1.5 + Reasoning (Llama 4 Scout).
_TALK_MODES = frozenset({"fast", "smart", "thinking"})
_TALK_MODE_SPECS = {
    "fast": {
        "model_env": "PYX_TALK_MODEL_FAST",
        "default_model": "llama-3.1-8b-instant",
        "max_tokens": 384,
        "temperature": 0.55,
        "system_suffix": (
            " Mode: Pyx Mini — fast. Prefer short, direct answers. "
            "(Cloud Groq uses Llama 3.1 8B Instant for this slot; for Llama 2 use Ollama with PYX_TALK_MODEL_FAST=llama2:7b.)"
        ),
    },
    "smart": {
        "model_env": "PYX_TALK_MODEL_SMART",
        "default_model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "max_tokens": 1024,
        "temperature": 0.5,
        "system_suffix": " Mode: Pyx 1.5 (Llama 4 Scout). Prioritize correctness and clarity. Structure longer answers when it helps.",
    },
    "thinking": {
        "model_env": "PYX_TALK_MODEL_THINKING",
        "default_model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "max_tokens": 2048,
        "temperature": 0.35,
        "system_suffix": (
            ' Mode: Pyx Reasoning 1.5 (Llama 4 Scout) — reasoning is mandatory on every reply, no exceptions. '
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


def _normalize_preview_messages(raw):
    """Preview-safe normalizer: trims oversized history instead of rejecting."""
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
        if not content:
            return None, "empty message"
        # 1.3 preview can emit long code blocks; clip history instead of failing.
        limit = 4000 if role == "user" else 20000
        if len(content) > limit:
            content = content[:limit]
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
        # Groq streams can run several minutes (thinking / long replies); 120s cap caused dropped streams.
        cap = 600
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
            err = obj.get("error")
            if err:
                msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                raise RuntimeError(msg)
            for ch in obj.get("choices") or []:
                delta = ch.get("delta") or {}
                piece = delta.get("content")
                if not piece and isinstance(delta.get("reasoning_content"), str):
                    piece = delta.get("reasoning_content")
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


# --- Pyx Speak TTS (OpenAI-compatible / Groq audio endpoint) ---
_GROQ_SPEECH_URL = "https://api.groq.com/openai/v1/audio/speech"
_SPEAK_TTS_FORMATS = {"mp3": "audio/mpeg", "wav": "audio/wav", "flac": "audio/flac", "opus": "audio/ogg"}
_ORPHEUS_ENGLISH_MODEL = "canopylabs/orpheus-v1-english"
_ORPHEUS_ARABIC_MODEL = "canopylabs/orpheus-arabic-saudi"
_DEPRECATED_TTS_MODEL_REPLACEMENTS = {
    "playai-tts": _ORPHEUS_ENGLISH_MODEL,
    "playai-tts-arabic": _ORPHEUS_ARABIC_MODEL,
}
_ORPHEUS_DEFAULT_VOICE = {
    _ORPHEUS_ENGLISH_MODEL: "austin",
    _ORPHEUS_ARABIC_MODEL: "fahad",
}
_ORPHEUS_INPUT_MAX = 200


def _is_orpheus_model(model: str) -> bool:
    m = (model or "").strip().lower()
    return m in (_ORPHEUS_ENGLISH_MODEL, _ORPHEUS_ARABIC_MODEL)


def _normalize_tts_model(model: str) -> str:
    m = (model or "").strip().lower()
    return _DEPRECATED_TTS_MODEL_REPLACEMENTS.get(m, m)


def _speak_tts_model_candidates(requested_model: str | None = None) -> list[str]:
    seen = set()
    out = []

    def _add(m):
        n = _normalize_tts_model(m or "")
        if n and n not in seen:
            seen.add(n)
            out.append(n)

    _add(requested_model)
    _add(os.environ.get("PYX_SPEAK_TTS_MODEL"))
    _add(_ORPHEUS_ENGLISH_MODEL)
    return out


def _speak_direction_tag(instructions: str | None) -> str:
    if not instructions:
        return ""
    t = re.sub(r"[^a-zA-Z0-9 \-]", " ", instructions).strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"^(speak|style)\s+", "", t)
    if not t:
        return ""
    # Orpheus directions work best with concise 1-2 word descriptors.
    words = t.split(" ")
    return " ".join(words[:3])[:40].strip()


def _prepare_orpheus_input(text: str, instructions: str | None) -> str:
    body = (text or "").strip()
    tag = _speak_direction_tag(instructions)
    if tag:
        return f"[{tag}] {body}".strip()
    return body


def _prepare_speak_tts(
    text: str,
    voice: str | None,
    fmt: str,
    speed: float,
    instructions: str | None = None,
    model_override: str | None = None,
):
    key = (os.environ.get("PYX_SPEAK_TTS_KEY") or os.environ.get("PYX_TALK_LLM_KEY") or "").strip()
    url = (os.environ.get("PYX_SPEAK_TTS_URL") or _GROQ_SPEECH_URL).strip()
    if not url:
        return None
    if url.rstrip("/").lower() == _GROQ_SPEECH_URL.rstrip("/").lower() and not key:
        return None
    model = _normalize_tts_model(model_override or os.environ.get("PYX_SPEAK_TTS_MODEL") or _ORPHEUS_ENGLISH_MODEL)
    is_orpheus = _is_orpheus_model(model)
    input_text = _prepare_orpheus_input(text, instructions) if is_orpheus else text
    use_fmt = (fmt or "mp3").strip().lower()
    if use_fmt not in _SPEAK_TTS_FORMATS:
        use_fmt = "mp3"
    if is_orpheus:
        use_fmt = "wav"
    use_voice = (voice or "").strip()
    if is_orpheus and not use_voice:
        use_voice = _ORPHEUS_DEFAULT_VOICE.get(model, "austin")
    payload = {
        "model": model,
        "input": input_text,
        "response_format": use_fmt,
        "speed": max(0.7, min(float(speed), 1.35)),
    }
    if use_voice:
        payload["voice"] = use_voice[:64]
    headers = {
        "Content-Type": "application/json",
        "Accept": _SPEAK_TTS_FORMATS.get(use_fmt, "audio/mpeg"),
        "User-Agent": (os.environ.get("PYX_TALK_USER_AGENT") or "PyxSpeak/1.0").strip(),
    }
    if key:
        headers["Authorization"] = "Bearer " + key
    timeout_s = 120
    try:
        timeout_s = max(20, min(int(os.environ.get("PYX_SPEAK_TTS_TIMEOUT", str(timeout_s))), 240))
    except ValueError:
        pass
    return {
        "url": url,
        "headers": headers,
        "payload": payload,
        "format": use_fmt,
        "timeout_s": timeout_s,
        "is_orpheus": is_orpheus,
    }


# --- Pyx AI Code (Groq GPT-OSS via OpenAI-compatible API) ---
_CODE_MODEL_DEFAULT = "openai/gpt-oss-20b"
_CODE_SYSTEM = os.environ.get(
    "PYX_CODE_SYSTEM",
    "You are Pyx AI Code, an expert programming assistant running on OpenAI GPT-OSS via Groq. "
    "Give precise, runnable code when the user asks for implementation. "
    "Use fenced markdown code blocks with language tags. Keep prose short; put the answer in code when appropriate. "
    "Call out security and performance pitfalls briefly when relevant.",
)
_CODE_SYSTEM_AGENT = os.environ.get(
    "PYX_CODE_SYSTEM_AGENT",
    "You are Pyx AI Code in AGENT mode. The user’s source is in the editor context you receive — you must NOT paste the "
    "entire file back. Never dump a full rewritten file unless the user explicitly asks for the whole file. "
    "Prefer minimal edits: explain briefly (1–3 sentences), then output a JSON object in a ```json fenced block with this shape:\n"
    '{"patches":[{"search":"exact substring to find once in the editor","replace":"replacement text"}, ...]}\n'
    "Rules: each `search` must match exactly ONE occurrence in the current editor buffer (copy from context). "
    "Use multiple patches for multiple edits. If you cannot produce safe patches, explain and give only a small illustrative snippet — "
    "still do not paste the whole file.",
)


def _groq_code_prepare(messages_for_api, language="auto", agent=False):
    """OpenAI-compatible chat request for coding (GPT-OSS on Groq by default). Returns None if Groq selected but no key."""
    key = os.environ.get("PYX_TALK_LLM_KEY", "").strip()
    url = (
        os.environ.get("PYX_CODE_LLM_URL", "").strip()
        or os.environ.get("PYX_TALK_LLM_URL", _GROQ_CHAT_COMPLETIONS_URL).strip()
    )
    url_norm = url.rstrip("/").lower()
    groq_norm = _GROQ_CHAT_COMPLETIONS_URL.rstrip("/").lower()
    if not key and url_norm == groq_norm:
        return None
    model = (os.environ.get("PYX_CODE_MODEL") or "").strip() or _CODE_MODEL_DEFAULT
    lang = language if isinstance(language, str) else "auto"
    lang = (lang or "auto").strip() or "auto"
    lang_hint = ""
    if lang.lower() not in ("auto", "plain", ""):
        lang_hint = f"\n\nEditor / stack context: {lang}. Prefer idioms, build tools, and libraries typical for this environment."
    if agent:
        system_base = _CODE_SYSTEM_AGENT
        try:
            max_tokens = int(os.environ.get("PYX_CODE_AGENT_MAX_TOKENS", "3072"))
        except ValueError:
            max_tokens = 3072
        max_tokens = max(256, min(max_tokens, 8192))
        try:
            temperature = float(os.environ.get("PYX_CODE_AGENT_TEMPERATURE", "0.18"))
        except ValueError:
            temperature = 0.18
        temperature = max(0.05, min(temperature, 0.9))
    else:
        system_base = _CODE_SYSTEM
        try:
            max_tokens = int(os.environ.get("PYX_CODE_MAX_TOKENS", "4096"))
        except ValueError:
            max_tokens = 4096
        max_tokens = max(256, min(max_tokens, 16384))
        try:
            temperature = float(os.environ.get("PYX_CODE_TEMPERATURE", "0.22"))
        except ValueError:
            temperature = 0.22
        temperature = max(0.05, min(temperature, 1.2))
    system_content = system_base + lang_hint
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
    try:
        timeout_s = max(15, min(int(os.environ.get("PYX_CODE_TIMEOUT", "120")), 600))
    except ValueError:
        timeout_s = 120
    if url_norm != groq_norm:
        timeout_s = max(timeout_s, 120)
    return {
        "url": url,
        "headers": headers,
        "body": body,
        "model": model,
        "timeout_s": timeout_s,
    }


def _groq_code_chat(messages_for_api, language="auto", agent=False):
    """Non-streaming code assistant. Returns (reply_text, model_id) or (None, None) if unconfigured."""
    prep = _groq_code_prepare(messages_for_api, language, agent=agent)
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


# --- Pyx pixel art (LLM emits a small grid; optionally upscaled for export / UI) ---
_PIXEL_LINE_RE = re.compile(r"^\s*px(\d+)\s*=\s*#([0-9A-Fa-f]{6})\s*$", re.I | re.M)


def _parse_px_lines(text: str, n_expected: int):
    """Parse pxK=#RRGGBB lines; fill missing indices by carrying the last color."""
    found = {}
    for m in _PIXEL_LINE_RE.finditer(text or ""):
        try:
            idx = int(m.group(1))
        except ValueError:
            continue
        hx = "#" + m.group(2).upper()
        if 1 <= idx <= n_expected:
            found[idx] = hx
    out = []
    last = "#2D2D2D"
    for i in range(1, n_expected + 1):
        if i in found:
            last = found[i]
        out.append(last)
    return out


def _upscale_nearest(px: list, gw: int, gh: int, W: int, H: int):
    """Nearest-neighbor upscale row-major grid gw×gh → W×H."""
    out = []
    for Y in range(H):
        for X in range(W):
            sx = min(gw - 1, (X * gw) // W) if W else 0
            sy = min(gh - 1, (Y * gh) // H) if H else 0
            out.append(px[sy * gw + sx])
    return out


def _groq_pixel_art_completion(user_prompt: str, gen_w: int, gen_h: int):
    """Single non-streaming completion for pixel-line output. Returns (text, model) or (None, None)."""
    key = os.environ.get("PYX_TALK_LLM_KEY", "").strip()
    url = (
        os.environ.get("PYX_PIXEL_LLM_URL", "").strip()
        or os.environ.get("PYX_CODE_LLM_URL", "").strip()
        or os.environ.get("PYX_TALK_LLM_URL", _GROQ_CHAT_COMPLETIONS_URL).strip()
    )
    url_norm = url.rstrip("/").lower()
    groq_norm = _GROQ_CHAT_COMPLETIONS_URL.rstrip("/").lower()
    if not key and url_norm == groq_norm:
        return None, None
    n = gen_w * gen_h
    model = (os.environ.get("PYX_PIXEL_MODEL") or "").strip() or "openai/gpt-oss-20b"
    # Hard ceiling keeps Groq TPM low (pixel_art was the main 502 path). Override with PYX_PIXEL_COMPLETION_CEILING.
    try:
        _cap = int(os.environ.get("PYX_PIXEL_COMPLETION_CEILING", "350"))
    except ValueError:
        _cap = 350
    _cap = max(64, min(_cap, 8192))
    try:
        max_tokens = int(os.environ.get("PYX_PIXEL_MAX_TOKENS", str(_cap)))
    except ValueError:
        max_tokens = _cap
    # Tight per-grid estimate; never exceed _cap.
    ceil_for_grid = min(_cap, max(32, min(n * 3 + 200, _cap)))
    max_tokens = max(32, min(max_tokens, _cap, ceil_for_grid))
    try:
        temperature = float(os.environ.get("PYX_PIXEL_TEMPERATURE", "0.35"))
    except ValueError:
        temperature = 0.35
    temperature = max(0.1, min(temperature, 0.9))
    system = (
        "You are a pixel-art engine. Output ONLY pixel color lines — no markdown fences, no explanations, no blank lines "
        f"before or after the data. The grid is exactly {gen_w} columns × {gen_h} rows ({n} pixels), row-major: "
        "px1 is top-left, px2 is one step right on the same row, and so on. "
        f"Emit EXACTLY {n} lines. Line format MUST be: pxK=#RRGGBB (uppercase hex, six digits). "
        f"K must be ONLY from 1 to {n} inclusive — never use px indices above {n} or below 1. "
        f"List px1 through px{n} in order (one line per pixel). "
        f"Subject / scene to draw: interpret the user’s request vividly but keep the output strictly to those {n} lines."
    )
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": (user_prompt or "abstract pattern").strip()[:2000]},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    ua = (os.environ.get("PYX_TALK_USER_AGENT") or "").strip() or (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": ua,
    }
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    timeout_s = 90
    if url_norm != groq_norm:
        timeout_s = 180
    try:
        timeout_s = max(20, min(int(os.environ.get("PYX_PIXEL_TIMEOUT", str(timeout_s))), 300))
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
    response.headers["Access-Control-Allow-Headers"] = (
        "Content-Type, X-API-Key, Authorization, X-Requested-With, Accept"
    )
    response.headers["Access-Control-Max-Age"] = "86400"
    return response


@app.errorhandler(Exception)
def handle_unexpected_exception(e):
    """Avoid HTML 500 for API clients; jsonify rejects NaN which caused opaque 500 pages."""
    if isinstance(e, HTTPException):
        return e
    if request.path.startswith("/api/") or request.method in ("POST", "PUT", "PATCH", "DELETE"):
        app.logger.exception("Unhandled error: %s", request.path)
        return jsonify(
            {
                "error": "Internal server error",
                "detail": str(e),
                "error_code": "internal_error",
            }
        ), 500
    raise


def _talk_backend_info():
    """Classify the configured LLM backend for UI badges and /health.
    - cloud (Groq)         : default PYX_TALK_LLM_URL + key present
    - cloud (Groq, unset)  : default URL + no key  (talk_llm_configured=False)
    - local (ollama)       : URL host is 127.0.0.1 / localhost (11434 by default)
    - local (lmstudio)     : URL host localhost on 1234
    - custom               : any other PYX_TALK_LLM_URL
    """
    key = os.environ.get("PYX_TALK_LLM_KEY", "").strip()
    url = os.environ.get("PYX_TALK_LLM_URL", _GROQ_CHAT_COMPLETIONS_URL).strip()
    un = url.rstrip("/").lower()
    gn = _GROQ_CHAT_COMPLETIONS_URL.rstrip("/").lower()
    if un == gn:
        return {
            "backend": "groq",
            "backend_kind": "cloud",
            "label": "Pyx 1.0 (Groq cloud)",
            "configured": bool(key),
            "url_host": "api.groq.com",
        }
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        parsed = None
    host = (parsed.hostname if parsed else "") or ""
    port = parsed.port if parsed else None
    local = host in ("localhost", "127.0.0.1", "0.0.0.0", "::1")
    kind = "local" if local else "custom"
    backend = "custom"
    label = "Pyx 1.5 (custom)"
    if local:
        if port == 11434 or "ollama" in host:
            backend = "ollama"
            label = "Pyx 1.5 (local · Ollama)"
        elif port == 1234:
            backend = "lmstudio"
            label = "Pyx 1.5 (local · LM Studio)"
        elif port == 8080:
            backend = "llama.cpp"
            label = "Pyx 1.5 (local · llama.cpp)"
        elif port == 8000:
            backend = "vllm"
            label = "Pyx 1.5 (local · vLLM)"
        else:
            backend = "local"
            label = "Pyx 1.5 (local)"
    return {
        "backend": backend,
        "backend_kind": kind,
        "label": label,
        "configured": True,  # non-Groq URLs treat key as optional
        "url_host": host + (f":{port}" if port else ""),
    }


@app.route("/health")
def health():
    firebase_connected = bool(getattr(pyx, "_db", None))
    backend = _talk_backend_info()
    return jsonify({
        "status": "ok",
        "services": {
            "pyx_moderator": "ok",
            "pyx_code": "ok",
            "pyx_check": "ok",
            "pyx_analyze": "ok",
            "pyx_talk": "ok",
            "pyx_13_preview": "ok",
            "pyx_ai_code": "ok",
            "pyx_pixel_art": "ok",
            "firebase": "connected" if firebase_connected else "offline",
        },
        "firebase_connected": firebase_connected,
        "talk_llm_configured": bool(backend.get("configured")),
        "backend": backend,
    })


@app.route("/")
def homepage():
    try:
        return send_from_directory("public", "index.html")
    except Exception:
        return jsonify({"status": "ok", "service": "pyx"}), 200


@app.route("/score", methods=["GET", "POST", "OPTIONS"])
@app.route("/api/score", methods=["GET", "POST", "OPTIONS"])
def score():
    if request.method == "OPTIONS":
        return "", 204
    # Match legacy Cloud Function: GET on /api/score = health (Hosting rewrites forward /api/score to Run)
    if request.method == "GET":
        return jsonify({"status": "ok", "service": "pyx"}), 200
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
    sf = _json_safe_score(s)
    if sf is None:
        sf = 0.0
    bad = sf >= BAN_LINE
    censored = censor_letters(text) if bad else text
    return jsonify({
        "score": sf,
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
    sf = _json_safe_score(s)
    if sf is None:
        sf = 0.0
    return jsonify({
        "score": sf,
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
@app.route("/api/talk", methods=["POST", "OPTIONS"])
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
        try:
            prep = _groq_openai_prepare(llm_messages, mode, web_context, ground_web)
        except Exception as e:
            app.logger.exception("talk: _groq_openai_prepare failed")
            return (
                jsonify(
                    {
                        "error": "Failed to prepare LLM request",
                        "detail": str(e),
                        "error_code": "talk_prepare",
                    }
                ),
                500,
            )

        def generate():
            meta = {
                "type": "meta",
                "mode": mode,
                "web_search": web_meta,
                "bad": False,
                "backend": _talk_backend_info(),
            }
            if prep:
                meta["model"] = prep["model"]
            else:
                meta["model"] = "pyx-fallback"
            us = _json_safe_score(u_score)
            if us is not None:
                meta["score"] = us
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
    us = _json_safe_score(u_score)
    if us is not None:
        out["score"] = us
    return jsonify(out)


@app.route("/pyx13-preview/chat", methods=["POST", "OPTIONS"])
@app.route("/api/pyx13-preview/chat", methods=["POST", "OPTIONS"])
def pyx13_preview_chat():
    """Pyx 1.3 **website** preview: Markov + optional DDG snippets — no GGUF, no cloud LLM."""
    if request.method == "OPTIONS":
        return "", 204
    if request.method != "POST":
        return jsonify({"error": "Method not allowed"}), 405
    data = request.get_json(silent=True) or {}
    messages, err = _normalize_preview_messages(data.get("messages"))
    if err:
        return jsonify({"error": err}), 400
    last_user = messages[-1]["content"]
    u_score = None
    try:
        u_score = pyx.score(last_user)
    except Exception:
        pass
    us = _json_safe_score(u_score)
    use_web = _as_bool(data.get("use_web"))
    use_web_auto = _as_bool(data.get("use_web_auto"))

    # For website preview, always answer so users can test the model flow.
    # (We still include score metadata for debugging/telemetry.)

    reply, meta = pyx13_preview.build_preview_reply(
        messages,
        use_web=use_web,
        use_web_auto=use_web_auto,
    )
    web = meta.get("web") or {}
    out = {
        "bad": False,
        "reply": reply,
        "model": "Pyx 1.3 Preview 4.26",
        "engine": meta.get("engine", "pyx-1.3-preview"),
        "web_search": {
            "used": bool(web.get("used")),
            "provider": web.get("provider"),
            "error": web.get("error"),
            "query": web.get("query"),
        },
    }
    if meta.get("in_chat_app"):
        out["in_chat_app"] = meta["in_chat_app"]
    if us is not None:
        out["score"] = us
    return jsonify(out)


@app.route("/speak/tts", methods=["POST", "OPTIONS"])
@app.route("/api/speak/tts", methods=["POST", "OPTIONS"])
def speak_tts():
    """Generate high-quality speech audio via OpenAI-compatible TTS (Groq default)."""
    if request.method == "OPTIONS":
        return "", 204
    if request.method != "POST":
        return jsonify({"error": "Method not allowed"}), 405
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    if not isinstance(text, str):
        return jsonify({"error": '"text" must be a string'}), 400
    text = text.strip()
    if not text:
        return jsonify({"error": "empty text"}), 400
    if len(text) > 3200:
        return jsonify({"error": "text too long"}), 413
    voice = data.get("voice")
    if voice is not None and not isinstance(voice, str):
        return jsonify({"error": '"voice" must be a string'}), 400
    fmt = data.get("format", "mp3")
    if fmt is not None and not isinstance(fmt, str):
        return jsonify({"error": '"format" must be a string'}), 400
    try:
        speed = float(data.get("speed", 1.0))
    except (TypeError, ValueError):
        return jsonify({"error": '"speed" must be a number'}), 400
    instructions = data.get("instructions")
    if instructions is not None and not isinstance(instructions, str):
        return jsonify({"error": '"instructions" must be a string'}), 400
    model = data.get("model")
    if model is not None and not isinstance(model, str):
        return jsonify({"error": '"model" must be a string'}), 400

    prep = None
    last_http = None
    last_url = None
    last_err = None
    audio = None
    content_type = "audio/wav"

    for candidate in _speak_tts_model_candidates(model):
        prep = _prepare_speak_tts(
            text=text,
            voice=voice,
            fmt=fmt or "wav",
            speed=speed,
            instructions=instructions,
            model_override=candidate,
        )
        if prep is None:
            continue
        if prep.get("is_orpheus") and len((prep["payload"].get("input") or "")) > _ORPHEUS_INPUT_MAX:
            return jsonify({
                "error": "text too long for current cloud TTS model",
                "limit": _ORPHEUS_INPUT_MAX,
                "hint": "Use shorter text/chunks for Groq Orpheus (200 chars max per request).",
            }), 413
        req = urllib.request.Request(
            prep["url"],
            data=json.dumps(prep["payload"]).encode("utf-8"),
            headers=prep["headers"],
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=prep["timeout_s"]) as resp:
                audio = resp.read()
                content_type = (resp.headers.get("Content-Type") or _SPEAK_TTS_FORMATS.get(prep["format"], "audio/wav")).split(";")[0]
            break
        except urllib.error.HTTPError as e:
            last_http = e
            last_url = prep["url"]
            last_err = e.read().decode("utf-8", errors="replace")[:800]
            low = (last_err or "").lower()
            # Retry next model when provider reports deprecation/decommission.
            if any(k in low for k in ("decommission", "deprecated", "no longer supported")):
                continue
            return jsonify({"error": "TTS provider request failed", "status": e.code, "detail": last_err}), 502
        except urllib.error.URLError as e:
            return jsonify({"error": "TTS provider network error", "detail": str(e.reason)}), 502
        except Exception as e:
            return jsonify({"error": str(e)}), 502

    if prep is None:
        return jsonify({
            "error": "TTS backend not configured",
            "hint": "Set PYX_SPEAK_TTS_KEY or PYX_TALK_LLM_KEY for Groq.",
        }), 503
    if audio is None:
        status = getattr(last_http, "code", 502)
        return jsonify({
            "error": "TTS provider request failed",
            "status": status,
            "detail": last_err or "No response from TTS provider.",
            "provider_url": last_url,
        }), 502

    return Response(
        audio,
        mimetype=content_type,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.route("/code_chat", methods=["POST", "OPTIONS"])
@app.route("/api/code_chat", methods=["POST", "OPTIONS"])
def code_chat():
    """Pyx AI Code: GPT-OSS on Groq (default) — coding assistant with streaming SSE."""
    if request.method == "OPTIONS":
        return "", 204
    if request.method != "POST":
        return jsonify({"error": "Method not allowed"}), 405
    data = request.get_json(silent=True) or {}
    messages, err = _normalize_talk_messages(data.get("messages"))
    if err:
        return jsonify({"error": err}), 400
    last_user = messages[-1]["content"]
    u_score = None
    try:
        u_score = pyx.score(last_user)
    except Exception:
        pass

    language = data.get("language", "auto")
    if language is not None and not isinstance(language, str):
        return jsonify({"error": '"language" must be a string'}), 400
    language = (language or "auto").strip() or "auto"

    agent = _as_bool(data.get("agent"))
    want_stream = _as_bool(data.get("stream"))
    llm_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

    if want_stream:
        try:
            prep = _groq_code_prepare(llm_messages, language, agent=agent)
        except Exception as e:
            app.logger.exception("code_chat: _groq_code_prepare failed")
            return (
                jsonify(
                    {
                        "error": "Failed to prepare LLM request",
                        "detail": str(e),
                        "error_code": "code_prepare",
                    }
                ),
                500,
            )

        def generate_code():
            meta = {
                "type": "meta",
                "kind": "code",
                "language": language,
                "agent": agent,
                "bad": False,
                "backend": _talk_backend_info(),
            }
            if prep:
                meta["model"] = prep["model"]
            else:
                meta["model"] = "pyx-fallback"
            us = _json_safe_score(u_score)
            if us is not None:
                meta["score"] = us
            yield _talk_sse_event(meta)
            if prep is None:
                fb = (
                    "Pyx AI Code needs an LLM on the server. Set PYX_TALK_LLM_KEY for Groq. "
                    "Optional: PYX_CODE_MODEL (default openai/gpt-oss-20b), PYX_TALK_LLM_URL for a custom OpenAI-compatible host."
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
            stream_with_context(generate_code()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    try:
        reply, model_used = _groq_code_chat(llm_messages, language=language, agent=agent)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:800]
        return jsonify({"error": "LLM request failed", "status": e.code, "detail": detail}), 502
    except urllib.error.URLError as e:
        return jsonify({"error": "LLM network error", "detail": str(e.reason)}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 502
    if reply is None:
        reply = (
            "Pyx AI Code needs an LLM on the server. Set PYX_TALK_LLM_KEY for Groq. "
            "Optional: PYX_CODE_MODEL (default openai/gpt-oss-20b)."
        )
        model_used = "pyx-fallback"
    out = {
        "bad": False,
        "reply": reply,
        "model": model_used or "unknown",
        "language": language,
        "agent": agent,
    }
    us = _json_safe_score(u_score)
    if us is not None:
        out["score"] = us
    return jsonify(out)


@app.route("/pixel_art", methods=["POST", "OPTIONS"])
@app.route("/api/pixel_art", methods=["POST", "OPTIONS"])
def pixel_art():
    """LLM draws a small grid (pxK=#RRGGBB); server may nearest-neighbor upscale for JSON + UI (defaults: 10×10, no upscale)."""
    if request.method == "OPTIONS":
        return "", 204
    if request.method != "POST":
        return jsonify({"error": "Method not allowed"}), 405
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt") or data.get("q") or ""
    if not isinstance(prompt, str) or not prompt.strip():
        return jsonify({"error": '"prompt" must be a non-empty string'}), 400
    try:
        gw = int(os.environ.get("PYX_PIXEL_GEN_W", os.environ.get("PYX_PIXEL_GEN_GRID", "10")))
    except ValueError:
        gw = 10
    try:
        gh = int(os.environ.get("PYX_PIXEL_GEN_H", str(gw)))
    except ValueError:
        gh = gw
    gw = max(8, min(gw, 64))
    gh = max(8, min(gh, 64))
    try:
        out_w = int(os.environ.get("PYX_PIXEL_OUT_W", str(gw)))
    except ValueError:
        out_w = gw
    try:
        out_h = int(os.environ.get("PYX_PIXEL_OUT_H", str(gh)))
    except ValueError:
        out_h = gh
    out_w = max(8, min(out_w, 256))
    out_h = max(8, min(out_h, 256))

    try:
        raw, model_used = _groq_pixel_art_completion(prompt.strip(), gw, gh)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:1200]
        status = 502
        err_code = "llm_http_error"
        if e.code == 429:
            status = 429
            err_code = "rate_limited"
        elif e.code in (400, 413):
            status = 400
            err_code = "bad_request"
        return jsonify(
            {
                "error": "LLM request failed",
                "status": e.code,
                "detail": detail,
                "error_code": err_code,
            }
        ), status
    except urllib.error.URLError as e:
        return jsonify({"error": "LLM network error", "detail": str(e.reason)}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 502

    if raw is None:
        # 422 (not 503) so this isn’t confused with Cloud Run / proxy “Service Unavailable”.
        return jsonify(
            {
                "error": (
                    "No LLM configured for pixel art. On Cloud Run set secret/env PYX_TALK_LLM_KEY "
                    "(Groq). Optional: PYX_PIXEL_MODEL, PYX_TALK_LLM_URL for a custom OpenAI-compatible host."
                ),
                "error_code": "llm_not_configured",
            }
        ), 422

    n = gw * gh
    try:
        base_px = _parse_px_lines(raw, n)
        pixels = _upscale_nearest(base_px, gw, gh, out_w, out_h)
    except Exception as e:
        return jsonify({"error": "Failed to build pixel grid", "detail": str(e), "error_code": "pixel_parse"}), 500
    return jsonify(
        {
            "ok": True,
            "pixels": pixels,
            "width": out_w,
            "height": out_h,
            "gen_w": gw,
            "gen_h": gh,
            "model": model_used or "unknown",
        }
    )


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
