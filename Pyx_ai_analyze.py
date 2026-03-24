"""
Pyx AI Analyze — Analyze code and three.js for inappropriate content.
========================================================================
Pre-Build: Extracts strings/comments/URLs and runs Pyx AI Moderator on them.
"""

import re
from typing import List, Dict, Any, Optional

__version__ = "Pre-Build"

# Optional: use Pyx AI Moderator (content filter) for extracted text
try:
    from Pyx_ai_moderator import PyxAI, BAN_LINE
    _pyx: Optional[PyxAI] = None

    def _get_pyx() -> PyxAI:
        global _pyx
        if _pyx is None:
            _pyx = PyxAI()
        return _pyx
except ImportError:
    def _get_pyx():
        return None
    BAN_LINE = 0.7


def _extract_strings_and_comments(source: str) -> List[str]:
    """Extract string literals, comments, and URLs from JS-like code."""
    snippets: List[str] = []
    # Double-quoted strings
    for m in re.finditer(r'"([^"\\]*(?:\\.[^"\\]*)*)"', source):
        snippets.append(m.group(1).replace("\\n", "\n").replace('\\"', '"'))
    # Single-quoted strings
    for m in re.finditer(r"'([^'\\]*(?:\\.[^'\\]*)*)'", source):
        snippets.append(m.group(1).replace("\\n", "\n").replace("\\'", "'"))
    # Line comments
    for m in re.finditer(r"//\s*(.+)$", source, re.MULTILINE):
        snippets.append(m.group(1).strip())
    # Block comments
    for m in re.finditer(r"/\*\*?(.*?)\*/", source, re.DOTALL):
        snippets.append(m.group(1).strip())
    # URLs (http/https)
    for m in re.finditer(r"https?://[^\s\"'>)\]]+", source):
        snippets.append(m.group(0))
    return [s for s in snippets if s and len(s.strip()) > 0]


def analyze_code(
    source: str,
    language: str = "javascript",
    use_content_filter: bool = True,
) -> Dict[str, Any]:
    """
    Analyze code (and three.js) for inappropriate content. Scaffold + basic extraction.
    Extracts strings/comments and optionally runs Pyx content filter on them.
    """
    extracted = _extract_strings_and_comments(source)
    flagged: List[Dict[str, Any]] = []
    pyx = _get_pyx() if use_content_filter else None

    for snippet in extracted:
        if not snippet.strip():
            continue
        score = 0.0
        if pyx:
            score = pyx.score(snippet)
        if pyx and score >= BAN_LINE:
            flagged.append({
                "snippet": snippet[:200] + ("..." if len(snippet) > 200 else ""),
                "score": round(score, 4),
                "reason": "content filter",
            })

    return {
        "safe": len(flagged) == 0,
        "flagged": flagged,
        "extracted_count": len(extracted),
        "language": language,
        "version": __version__,
    }


def analyze_three_js(source: str, use_content_filter: bool = True) -> Dict[str, Any]:
    """Analyze three.js / WebGL code for inappropriate content."""
    return analyze_code(source, language="javascript", use_content_filter=use_content_filter)
