"""Pyx Write 0.5 — instrumental composition via GPT-OSS (symbolic score JSON)."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

from pyx_llm_utils import (
    choice_message_text,
    groq_gpt_oss_body_extras,
    groq_oss_token_fields,
    is_gpt_oss_model,
)

_GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
_WRITE_MODEL_DEFAULT = "openai/gpt-oss-20b"
_WRITE_025_MODEL_DEFAULT = "llama-3.1-8b-instant"

WRITE_PROFILES: dict[str, dict[str, Any]] = {
    "1.0": {
        "version": "1.0",
        "engine": "pyx-write-1.0",
        "model_env": "PYX_WRITE_10_MODEL",
        "default_model": _WRITE_MODEL_DEFAULT,
        "max_tokens_env": "PYX_WRITE_10_MAX_TOKENS",
        "default_max_tokens": 4608,
        "timeout_env": "PYX_WRITE_10_TIMEOUT",
        "default_timeout": 140,
        "temperature_env": "PYX_WRITE_10_TEMPERATURE",
        "default_temperature": 0.6,
        "user_agent": "PyxWrite/1.0",
        "vocal": True,
    },
    "0.5": {
        "version": "0.5",
        "engine": "pyx-write-0.5",
        "model_env": "PYX_WRITE_MODEL",
        "default_model": _WRITE_MODEL_DEFAULT,
        "max_tokens_env": "PYX_WRITE_MAX_TOKENS",
        "default_max_tokens": 4096,
        "timeout_env": "PYX_WRITE_TIMEOUT",
        "default_timeout": 120,
        "temperature_env": "PYX_WRITE_TEMPERATURE",
        "default_temperature": 0.55,
        "user_agent": "PyxWrite/0.5",
        "vocal": False,
    },
    "0.25": {
        "version": "0.25",
        "engine": "pyx-write-0.25",
        "model_env": "PYX_WRITE_025_MODEL",
        "default_model": _WRITE_025_MODEL_DEFAULT,
        "max_tokens_env": "PYX_WRITE_025_MAX_TOKENS",
        "default_max_tokens": 2048,
        "timeout_env": "PYX_WRITE_025_TIMEOUT",
        "default_timeout": 45,
        "temperature_env": "PYX_WRITE_025_TEMPERATURE",
        "default_temperature": 0.42,
        "user_agent": "PyxWrite/0.25",
    },
}


def normalize_write_profile(raw: Any) -> str:
    s = str(raw or "0.5").strip().lower()
    if s in ("0.25", "025", "0_25", "fast", "lite", "quick"):
        return "0.25"
    if s in ("1", "1.0", "10", "1_0", "voice", "vocal", "sing", "singing"):
        return "1.0"
    return "0.5"


def write_profile_meta(profile_id: str) -> dict[str, str]:
    spec = WRITE_PROFILES[normalize_write_profile(profile_id)]
    return {
        "write_profile": normalize_write_profile(profile_id),
        "version": str(spec["version"]),
        "engine": str(spec["engine"]),
    }

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
        "voice",
    }
)

# Singable syllables the browser formant voice can render (mapped to vowel formants client-side).
ALLOWED_SYLLABLES = frozenset(
    {"ah", "aa", "aah", "ee", "oo", "ooh", "oh", "eh", "ih", "la", "na", "doo", "doot", "mm", "hmm", "ya", "ba", "da"}
)
_DEFAULT_SYLLABLE = "ah"

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


_WRITE_VOCAL_SYSTEM = """You are Pyx Write, an expert songwriter and composer. You output ONLY valid JSON (no markdown prose).

The user wants an original song with a SINGING lead voice that sings REAL WORDS, plus instrumental backing. A browser voice synthesizer sings the words you write (it sings them on pitch, it does not speak), so write actual lyrics and put ONE sung syllable on each vocal note.

JSON schema (strict):
{
  "title": "short title",
  "bpm": 60-170,
  "key": "e.g. C minor, F# major",
  "time_signature": [4, 4],
  "bars": 8-20,
  "tracks": [
    {
      "instrument": "voice",
      "notes": [
        {"pitch": "C4", "start": 0.0, "duration": 0.5, "velocity": 0.8, "text": "shine"},
        {"pitch": "E4", "start": 0.5, "duration": 0.5, "velocity": 0.8, "text": "on"}
      ]
    },
    {
      "instrument": "piano",
      "notes": [ {"pitch": "C3", "start": 0.0, "duration": 1.0, "velocity": 0.6} ]
    }
  ]
}

Rules:
- ALWAYS include exactly ONE track with "instrument": "voice" — this is the lead singer. Put it first.
- The voice is MONOPHONIC: voice notes must NOT overlap (each starts at or after the previous one ends).
- Every voice note MUST have a "text" field: ONE sung syllable of the lyrics (lowercase, letters only, no punctuation). Split multi-syllable words across consecutive notes (e.g. "for" then "ev" then "er"). Use simple, clearly pronounceable English words so the synthesizer can sing them. Hold vowels on long notes.
- Write a coherent, emotional lyric (a verse and a hook/chorus) that matches the user's theme — real words, not gibberish, not random syllables.
- Keep the voice in a comfortable singing range C4-A5. Give sustained notes (0.5-2 beats) on important words for an expressive, lyrical melody.
- Add 1-4 backing instrument tracks (piano, bass, drums, pad, guitar, strings, brass, bells, organ, synth) that support the vocal melody and its key.
- "start" and "duration" are in quarter-note beats (4/4: one beat = quarter note). Keep each track under 80 notes; whole piece within "bars".
- velocity 0.35-1.0. Make it musical and emotional — a real song. Match the user's mood/style.
- Output ONLY the JSON object, no ``` fences, no explanation."""


def _write_llm_prepare(user_content: str, write_profile: str = "0.5") -> dict[str, Any] | None:
    profile_id = normalize_write_profile(write_profile)
    spec = WRITE_PROFILES[profile_id]
    key = os.environ.get("PYX_TALK_LLM_KEY", "").strip()
    url = (
        os.environ.get("PYX_WRITE_LLM_URL", "").strip()
        or os.environ.get("PYX_TALK_LLM_URL", _GROQ_CHAT_COMPLETIONS_URL).strip()
    )
    url_norm = url.rstrip("/").lower()
    groq_norm = _GROQ_CHAT_COMPLETIONS_URL.rstrip("/").lower()
    if not key and url_norm == groq_norm:
        return None
    model = (os.environ.get(spec["model_env"]) or "").strip() or str(spec["default_model"])
    try:
        max_tokens = int(
            os.environ.get(spec["max_tokens_env"], str(spec["default_max_tokens"]))
        )
    except ValueError:
        max_tokens = int(spec["default_max_tokens"])
    max_tokens = max(512, min(max_tokens, 8192))
    try:
        temperature = float(
            os.environ.get(spec["temperature_env"], str(spec["default_temperature"]))
        )
    except ValueError:
        temperature = float(spec["default_temperature"])
    temperature = max(0.2, min(temperature, 1.0))
    system_prompt = _WRITE_VOCAL_SYSTEM if spec.get("vocal") else _WRITE_SYSTEM
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content[:4000]},
        ],
        "temperature": temperature,
    }
    if is_gpt_oss_model(model):
        body.update(groq_oss_token_fields(model, max_tokens))
        body.update(groq_gpt_oss_body_extras(model))
    else:
        body["max_tokens"] = max_tokens
    ua = (os.environ.get("PYX_TALK_USER_AGENT") or "").strip() or str(spec["user_agent"])
    headers = {"Content-Type": "application/json", "User-Agent": ua}
    if key:
        headers["Authorization"] = "Bearer " + key
    try:
        timeout_s = max(
            15,
            min(int(os.environ.get(spec["timeout_env"], str(spec["default_timeout"]))), 300),
        )
    except ValueError:
        timeout_s = int(spec["default_timeout"])
    return {
        "url": url,
        "headers": headers,
        "body": body,
        "model": model,
        "timeout_s": timeout_s,
        "write_profile": profile_id,
    }


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


def _normalize_lyric_text(value: Any) -> str:
    """One sung syllable: lowercase letters only, capped length."""
    s = str(value or "").strip().lower()
    s = re.sub(r"[^a-z]", "", s)
    return s[:14]


def _normalize_syllable(value: Any) -> str:
    s = str(value or "").strip().lower()
    if not s:
        return _DEFAULT_SYLLABLE
    s = re.sub(r"[^a-z]", "", s)
    if s in ALLOWED_SYLLABLES:
        return s
    # Map common variants / real words to the nearest singable vowel.
    if s.startswith("ah") or s.startswith("aa") or s.startswith("a"):
        return "ah"
    if s.startswith("oo") or s.startswith("u"):
        return "oo"
    if s.startswith("oh") or s.startswith("o"):
        return "oh"
    if s.startswith("ee") or s.startswith("i"):
        return "ee"
    if s.startswith("eh") or s.startswith("e"):
        return "eh"
    if s.startswith("la") or s.startswith("l"):
        return "la"
    if s.startswith("na") or s.startswith("n"):
        return "na"
    if s.startswith("m") or s.startswith("hm"):
        return "mm"
    return _DEFAULT_SYLLABLE


def _make_monophonic(notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Trim overlaps so a lead voice never sings two notes at once."""
    if not notes:
        return notes
    ordered = sorted(notes, key=lambda n: n["start"])
    out: list[dict[str, Any]] = []
    for n in ordered:
        if out:
            prev = out[-1]
            prev_end = prev["start"] + prev["duration"]
            if n["start"] < prev_end:
                trimmed = round(n["start"] - prev["start"], 4)
                if trimmed <= 0.01:
                    continue
                prev["duration"] = trimmed
        out.append(n)
    return out


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
        if inst in ("vocal", "vocals", "vox", "lead vocal", "singer", "sing"):
            inst = "voice"
        if inst not in ALLOWED_INSTRUMENTS:
            inst = "piano"
        is_voice = inst == "voice"
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
            note_out: dict[str, Any] = {
                "pitch": pitch,
                "start": round(start, 4),
                "duration": round(duration, 4),
                "velocity": round(velocity, 3),
            }
            if is_voice:
                text = _normalize_lyric_text(n.get("text") or n.get("word") or n.get("lyric"))
                if text:
                    note_out["text"] = text
                note_out["syllable"] = _normalize_syllable(
                    n.get("syllable") or n.get("vowel") or text
                )
            out_notes.append(note_out)
        if is_voice:
            out_notes = _make_monophonic(out_notes)
        if out_notes:
            out_tracks.append({"instrument": inst, "notes": out_notes})
    if not out_tracks:
        raise ValueError("no valid notes in score")
    # Voice first, then drums, then other instruments.
    def _track_order(t: dict[str, Any]) -> tuple:
        if t["instrument"] == "voice":
            return (0, "")
        if t["instrument"] == "drums":
            return (1, "")
        return (2, t["instrument"])

    out_tracks.sort(key=_track_order)
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
    write_profile: str = "0.5",
    lyric_theme: str = "",
) -> tuple[dict[str, Any], str, dict[str, str]]:
    """Returns (score_dict, provider_model_id, profile_meta). Raises ValueError on bad input/parse."""
    profile_id = normalize_write_profile(write_profile)
    is_vocal = bool(WRITE_PROFILES[profile_id].get("vocal"))
    meta = write_profile_meta(profile_id)
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("prompt required")
    style = (style or "").strip()[:200]
    lyric_theme = (lyric_theme or "").strip()[:200]
    bars_cap = 16 if profile_id == "0.25" else 20 if is_vocal else 24
    bars = max(8, min(int(bars), bars_cap))
    inst_list = []
    for i in instruments or []:
        s = str(i).strip().lower()
        if s in ALLOWED_INSTRUMENTS and s not in inst_list:
            inst_list.append(s)
    if is_vocal:
        # Voice is the lead; ensure it is requested plus a couple of backing instruments.
        inst_list = [i for i in inst_list if i != "voice"]
        if not inst_list:
            inst_list = ["piano", "bass", "drums"]
        inst_list = ["voice"] + inst_list
    elif not inst_list:
        inst_list = ["piano", "bass", "pad"]
    user_lines = [
        f"Creative brief: {prompt}",
        f"Target length: {bars} bars in 4/4.",
        f"Use these instruments: {', '.join(inst_list)}.",
    ]
    if is_vocal:
        user_lines.append(
            "Include a lead 'voice' track that sings a memorable melody using only the allowed syllables."
        )
        if lyric_theme:
            user_lines.append(
                f"Vocal mood/feel to evoke through the melody and syllable choices: {lyric_theme}."
            )
    if style:
        user_lines.append(f"Style/mood: {style}.")
    user_content = "\n".join(user_lines)

    prep = _write_llm_prepare(user_content, profile_id)
    if prep is None:
        raise RuntimeError(
            "Write LLM not configured — set PYX_TALK_LLM_KEY for Groq."
        )
    headers = {**prep["headers"], "Accept": "application/json"}
    model = str(prep.get("model") or WRITE_PROFILES[profile_id]["default_model"])
    token_limits = [
        int(
            prep["body"].get("max_tokens")
            or prep["body"].get("max_completion_tokens")
            or WRITE_PROFILES[profile_id]["default_max_tokens"]
        )
    ]
    if is_gpt_oss_model(model) and token_limits[0] < 8192:
        token_limits.append(min(8192, token_limits[0] * 2))
    elif not is_gpt_oss_model(model) and profile_id == "0.25":
        token_limits = [min(token_limits[0], 2048)]

    content = ""
    last_finish = ""
    for limit in token_limits:
        body = {**prep["body"]}
        if is_gpt_oss_model(model):
            body.update(groq_oss_token_fields(model, limit))
            if limit == token_limits[-1] and len(token_limits) > 1:
                body.update(groq_gpt_oss_body_extras(model, reasoning_format=""))
            else:
                body.update(groq_gpt_oss_body_extras(model))
        else:
            body["max_tokens"] = limit
        req = urllib.request.Request(
            prep["url"],
            data=json.dumps(body).encode("utf-8"),
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
        choice = choices[0]
        content = choice_message_text(choice)
        last_finish = str(choice.get("finish_reason") or "")
        if content.strip():
            break

    if not content:
        detail = f" (finish_reason={last_finish})" if last_finish else ""
        raise ValueError("empty LLM content" + detail)
    raw_score = _extract_json_object(content)
    score = validate_and_normalize_score(raw_score)
    return score, str(prep.get("model") or WRITE_PROFILES[profile_id]["default_model"]), meta
