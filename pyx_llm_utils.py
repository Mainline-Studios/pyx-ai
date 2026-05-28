"""Shared helpers for Groq / OpenAI-compatible chat responses (GPT-OSS reasoning)."""

from __future__ import annotations

import re
from typing import Any

_THINK_RE = re.compile(
    r"<\s*(?:think|redacted_reasoning)\s*>(.*?)</\s*(?:think|redacted_reasoning)\s*>",
    re.IGNORECASE | re.DOTALL,
)


def is_gpt_oss_model(model: str | None) -> bool:
    return "gpt-oss" in (model or "").lower()


def groq_gpt_oss_body_extras(model: str | None, *, reasoning_format: str = "hidden") -> dict[str, Any]:
    """
    GPT-OSS on Groq often leaves message.content empty and puts text in message.reasoning.
    hidden → final answer in content (required for pixel lines, JSON scores, code).
    On retry, pass reasoning_format="" to read message.reasoning via message_text().
    """
    if not is_gpt_oss_model(model):
        return {}
    if reasoning_format:
        return {"reasoning_format": reasoning_format}
    return {}


def groq_oss_token_fields(model: str | None, max_tokens: int) -> dict[str, Any]:
    """Groq GPT-OSS respects max_completion_tokens; keep max_tokens for other hosts."""
    if not is_gpt_oss_model(model):
        return {"max_tokens": max_tokens}
    return {
        "max_tokens": max_tokens,
        "max_completion_tokens": max_tokens,
        "reasoning_effort": "low",
    }


def message_text(msg: Any) -> str:
    """Extract assistant text from a chat completion message object."""
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    if isinstance(content, str) and content.strip():
        return _strip_reasoning_wrappers(content.strip())
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                t = block.get("text") or block.get("content")
                if isinstance(t, str) and t.strip():
                    parts.append(t.strip())
            elif isinstance(block, str) and block.strip():
                parts.append(block.strip())
        if parts:
            return _strip_reasoning_wrappers("\n".join(parts).strip())
    reasoning = msg.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        return _strip_reasoning_wrappers(reasoning.strip())
    if isinstance(reasoning, dict):
        for key in ("content", "text", "reasoning"):
            t = reasoning.get(key)
            if isinstance(t, str) and t.strip():
                return _strip_reasoning_wrappers(t.strip())
    rc = msg.get("reasoning_content")
    if isinstance(rc, str) and rc.strip():
        return _strip_reasoning_wrappers(rc.strip())
    return ""


def _strip_reasoning_wrappers(text: str) -> str:
    """Drop redacted_thinking wrappers if the provider inlined them."""
    t = text.strip()
    m = _THINK_RE.search(t)
    if m and m.group(1).strip():
        return m.group(1).strip()
    return t


def choice_message_text(choice: Any) -> str:
    if not isinstance(choice, dict):
        return ""
    msg = choice.get("message") or {}
    text = message_text(msg)
    if text:
        return text
    legacy = choice.get("text")
    if isinstance(legacy, str) and legacy.strip():
        return _strip_reasoning_wrappers(legacy.strip())
    return ""
