"""
Pyx AI Check — Check code and give tips.
=========================================
Version 0.5: Regex-based checks so it somewhat works.
Extend with linters (ESLint, Pylint) or your LLM for deeper tips.
"""

from typing import List, Dict, Any, Optional
import re

__version__ = "0.5"


def _line_number_at(source: str, pos: int) -> int:
    """Approximate 1-based line number for position in source."""
    return source[:pos].count("\n") + 1


def check_code(
    source: str,
    language: str = "javascript",
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Check code and return tips. v0.5: regex-based; add linters for more.
    """
    options = options or {}
    tips: List[Dict[str, Any]] = []
    lines = source.split("\n")

    # Long lines
    for i, line in enumerate(lines, 1):
        if len(line) > 120:
            tips.append({"line": i, "message": "Line longer than 120 characters", "severity": "warning"})

    # TODO / FIXME / HACK
    for m in re.finditer(r"(?i)\b(TODO|FIXME|HACK|XXX)\b\s*:?\s*(.*)", source):
        tips.append({
            "line": _line_number_at(source, m.start()),
            "message": f"{m.group(1)}: {m.group(2).strip() or 'item'}"[:80],
            "severity": "info",
        })

    # JS: possible assignment in condition (if (x = 1))
    if language in ("javascript", "js", "typescript", "ts"):
        for m in re.finditer(r"\bif\s*\(\s*(\w+)\s*=\s*[^=]", source):
            tips.append({
                "line": _line_number_at(source, m.start()),
                "message": "Possible assignment in condition (use == or ===?)",
                "severity": "warning",
            })
        # == vs ===
        for m in re.finditer(r"[^=!]==(?!=)[^=]", source):
            tips.append({
                "line": _line_number_at(source, m.start()),
                "message": "Consider === for strict equality",
                "severity": "info",
            })

    # Python: use of == None
    if language in ("python", "py"):
        for m in re.finditer(r"==\s*None\b", source):
            tips.append({
                "line": _line_number_at(source, m.start()),
                "message": "Prefer 'is None' instead of '== None'",
                "severity": "info",
            })

    # Empty catch / except
    for m in re.finditer(r"(?m)^\s*(catch|except)\s*\([^)]*\)\s*\{\s*\}|\s*:\s*pass\s*$", source):
        tips.append({
            "line": _line_number_at(source, m.start()),
            "message": "Empty catch/except block",
            "severity": "warning",
        })

    return {
        "tips": tips,
        "language": language,
        "checked": True,
        "version": __version__,
    }


def check_three_js(source: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Check three.js / WebGL-related code. v0.5: runs JS checks + three.js hints."""
    out = check_code(source, language="javascript", options=options)
    # three.js: deprecated or common gotchas
    if "THREE." in source and "Geometry" in source and "BufferGeometry" not in source:
        for i, line in enumerate(source.split("\n"), 1):
            if "Geometry()" in line and "BufferGeometry" not in line:
                out["tips"].append({
                    "line": i,
                    "message": "Consider BufferGeometry for better performance (legacy Geometry is deprecated)",
                    "severity": "info",
                })
                break
    return out
