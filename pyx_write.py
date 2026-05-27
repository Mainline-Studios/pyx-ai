"""Pyx Write — instrumental composition via GPT-OSS (symbolic score JSON)."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

_GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
_WRITE_MODEL_DEFAULT = "openai/gpt-oss-20b"

ALLOWED_INSTRUMENTS = frozenset(
    {
        "piano",
        "synth",
        "bass",
        "drums",
        "guitar",
        "strings",
        "brass",
        "pad",
        "bells",
        "organ",
    }
)

_PITCH_RE = re.compile(r"^[A-Ga-g](#|b|bb)?-?\d{1,2}$")

_WRITE_SYSTEM = """You are Pyx Write, an expert instrumental composer. You output ONLY valid JSON (no markdown prose).

The user wants original INSTRUMENTAL music (no vocals, no lyrics). Compose a short piece as a multi-track score.

JSON schema (strict):
{
  "title": "short title",
  "bpm": 60-180,
  "key": "e.g. C minor, F# major",
  "time_signature": [4, 4],
  "bars": 8-24,
  "tracks": [
    {
      "instrument": one of: piano, synth, bass, drums, guitar, strings, brass, pad, bells, organ,
      "notes": [
        {"pitch": "C4", "start": 0.0, "duration": 0.5, "velocity": 0.75}
      ]
    }
  ]
}

Rules:
- Use 2-5 tracks from the instruments the user requested (or pick fitting ones).
- "start" and "duration" are in quarter-note beats (4/4: one beat = quarter note).
- Keep each track under 80 notes; total piece within "bars" (last note start+duration <= bars*4 beats in 4/4).
- Use pitch like C4, D#3, Bb2 (drums: C1 kick, D1 snare, F#1 hat, etc.).
- velocity 0.35-1.0. Make melodies and rhythms musical — not random notes.
- Match the user's mood/style (tempo, density, harmony implied by pitch choices).
- Output ONLY the JSON object, no ``` fences, no explanation."""


def _write_llm_prepare(user_content: str) -> dict[str, Any] | None:
    key = os.environ.get("PYX_TALK_LLM_KEY", "").strip()
    url = (
        os.environ.get("PYX_WRITE_LLM_URL", "").strip()
        or os.environ.get("PYX_TALK_LLM_URL", _GROQ_CHAT_COMPLETIONS_URL).strip()
    )
    url_norm = url.rstrip("/").lower()
    groq_norm = _GROQ_CHAT_COMPLETIONS_URL.rstrip("/").lower()
    if not key and url_norm == groq_norm:
        return None
    model = (os.environ.get("PYX_WRITE_MODEL") or "").strip() or _WRITE_MODEL_DEFAULT
    try:
        max_tokens = int(os.environ.get("PYX_WRITE_MAX_TOKENS", "4096"))
    except ValueError:
        max_tokens = 4096
    max_tokens = max(512, min(max_tokens, 8192))
    try:
        temperature = float(os.environ.get("PYX_WRITE_TEMPERATURE", "0.55"))
    except ValueError:
        temperature = 0.55
    temperature = max(0.2, min(temperature, 1.0))
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _WRITE_SYSTEM},
            {"role": "user", "content": user_content[:4000]},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    ua = (os.environ.get("PYX_TALK_USER_AGENT") or "").strip() or "PyxWrite/1.0"
    headers = {"Content-Type": "application/json", "User-Agent": ua}
    if key:
        headers["Authorization"] = "Bearer " + key
    try:
        timeout_s = max(20, min(int(os.environ.get("PYX_WRITE_TIMEOUT", "120")), 300))
    except ValueError:
        timeout_s = 120
    return {"url": url, "headers": headers, "body": body, "model": model, "timeout_s": timeout_s}


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("LLM did not return a JSON object")
    return json.loads(raw[start : end + 1])


def _normalize_pitch(pitch: Any) -> str | None:
    if isinstance(pitch, (int, float)):
        midi = int(pitch)
        if midi < 0 or midi > 127:
            return None
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        return names[midi % 12] + str(midi // 12 - 1)
    s = str(pitch or "").strip()
    if not s:
        return None
    s = s.replace("♯", "#").replace("♭", "b")
    if _PITCH_RE.match(s):
        return s[0].upper() + s[1:]
    return None


def validate_and_normalize_score(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("score must be an object")
    title = str(data.get("title") or "Untitled").strip()[:120] or "Untitled"
    try:
        bpm = int(float(data.get("bpm", 120)))
    except (TypeError, ValueError):
        bpm = 120
    bpm = max(60, min(bpm, 180))
    key = str(data.get("key") or "C major").strip()[:40]
    ts = data.get("time_signature")
    if not isinstance(ts, list) or len(ts) < 2:
        ts = [4, 4]
    try:
        beats_per_bar = max(1, int(ts[0]))
    except (TypeError, ValueError):
        beats_per_bar = 4
    try:
        bars = int(data.get("bars", 16))
    except (TypeError, ValueError):
        bars = 16
    bars = max(4, min(bars, 32))
    max_beats = float(bars * beats_per_bar)

    tracks_in = data.get("tracks")
    if not isinstance(tracks_in, list) or not tracks_in:
        raise ValueError("score needs at least one track")
    out_tracks: list[dict[str, Any]] = []
    for tr in tracks_in[:6]:
        if not isinstance(tr, dict):
            continue
        inst = str(tr.get("instrument") or "piano").strip().lower()
        if inst not in ALLOWED_INSTRUMENTS:
            inst = "piano"
        notes_in = tr.get("notes")
        if not isinstance(notes_in, list):
            continue
        out_notes: list[dict[str, Any]] = []
        for n in notes_in[:96]:
            if not isinstance(n, dict):
                continue
            pitch = _normalize_pitch(n.get("pitch"))
            if not pitch:
                continue
            try:
                start = float(n.get("start", 0))
                duration = float(n.get("duration", 0.25))
                velocity = float(n.get("velocity", 0.75))
            except (TypeError, ValueError):
                continue
            if duration <= 0 or start < 0 or start + duration > max_beats + 0.01:
                continue
            duration = min(duration, 8.0)
            velocity = max(0.2, min(velocity, 1.0))
            out_notes.append(
                {
                    "pitch": pitch,
                    "start": round(start, 4),
                    "duration": round(duration, 4),
                    "velocity": round(velocity, 3),
                }
            )
        if out_notes:
            out_tracks.append({"instrument": inst, "notes": out_notes})
    if not out_tracks:
        raise ValueError("no valid notes in score")
    out_tracks.sort(key=lambda t: (0 if t["instrument"] == "drums" else 1, t["instrument"]))
    return {
        "title": title,
        "bpm": bpm,
        "key": key,
        "time_signature": [beats_per_bar, int(ts[1]) if len(ts) > 1 else 4],
        "bars": bars,
        "tracks": out_tracks,
    }


def compose_instrumental(
    *,
    prompt: str,
    style: str = "",
    instruments: list[str] | None = None,
    bars: int = 16,
) -> tuple[dict[str, Any], str]:
    """Returns (score_dict, model_id). Raises ValueError on bad input/parse."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("prompt required")
    style = (style or "").strip()[:200]
    bars = max(8, min(int(bars), 24))
    inst_list = []
    for i in instruments or []:
        s = str(i).strip().lower()
        if s in ALLOWED_INSTRUMENTS and s not in inst_list:
            inst_list.append(s)
    if not inst_list:
        inst_list = ["piano", "bass", "pad"]
    user_lines = [
        f"Creative brief: {prompt}",
        f"Target length: {bars} bars in 4/4.",
        f"Use these instruments: {', '.join(inst_list)}.",
    ]
    if style:
        user_lines.append(f"Style/mood: {style}.")
    user_content = "\n".join(user_lines)

    prep = _write_llm_prepare(user_content)
    if prep is None:
        raise RuntimeError(
            "Write LLM not configured — set PYX_TALK_LLM_KEY for Groq GPT-OSS."
        )
    headers = {**prep["headers"], "Accept": "application/json"}
    req = urllib.request.Request(
        prep["url"],
        data=json.dumps(prep["body"]).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=prep["timeout_s"]) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Compose provider HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Compose network error: {e.reason}") from e

    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("LLM returned no choices")
    content = (choices[0].get("message") or {}).get("content") or ""
    if not str(content).strip():
        raise ValueError("empty LLM content")
    raw_score = _extract_json_object(str(content))
    score = validate_and_normalize_score(raw_score)
    return score, str(prep.get("model") or _WRITE_MODEL_DEFAULT)
