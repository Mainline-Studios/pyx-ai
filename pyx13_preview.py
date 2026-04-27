"""
Pyx 1.3 **website preview** — pure Python, no GGUF, no external LLM APIs.

Uses a tiny from-scratch **bigram Markov** model over an embedded corpus, plus
optional **DuckDuckGo HTML** snippets (same keyless approach as Pyx Talk web search).
Replies are **generated** from the Markov chain; web text is only prepended as context,
not echoed verbatim as the whole answer.
"""

from __future__ import annotations

import html
import random
import re
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from typing import Any

# --- Embedded training text (synthetic assistant voice + Pyx product facts) ---
_CORPUS = """
Pyx is an assistant family built for creators students and builders who want clear answers
without unnecessary fluff. Pyx Talk helps you chat plan and learn. Pyx Code helps you write
and refactor software in many languages. Pyxel makes tiny pixel art from short prompts.
Pyx 1.0 on the public site uses fast cloud models through the Pyx API. Pyx 1.3 is the
local-first line you can download as a starter kit wire into your own UI and run on your
own hardware when you are ready. This preview runs entirely without a weight file so you
can try the shape of the product on the website. Ask about features roadmaps or how to
integrate Pyx into a classroom or studio workflow. I can explain moderation safety filters
and how messages are scored before they are shown. I can outline how web search snippets
can ground a reply while the model still writes fresh sentences. Keep questions specific
for the best experience. If you need code samples ask for Python JavaScript or pseudocode.
If you want study tips ask for a short study plan or a checklist. If you are curious about
pixel art ask for grid sizes color palettes or prompt ideas. Pyx values concise helpful tone
and honest limits when information is uncertain. Welcome to Pyx we are glad you are here.
The local starter repository includes a template UI and a downloadable core you can extend.
Teachers can use Pyx to brainstorm lesson hooks and rubrics. Developers can use Pyx to draft
commit messages tests and comments. Artists can use Pyxel for quick sprite experiments.
Privacy matters for local Pyx because your weights stay on your machine in full local mode.
Cloud Pyx relies on the service provider you configure. Always review terms for your org.
For troubleshooting check your network your API keys for cloud routes and your model path
for local routes. Pyx grows with feedback from the community. Share feature ideas politely.
We try to keep latency low and errors readable. Retry when a gateway times out. Rotate keys
if you hit rate limits on cloud tiers. For learning read the docs and try small prompts first.
Celebrate small wins when your first integration works end to end. Pyx is a tool not a replacement
for human judgment verify critical facts especially for medical legal or safety topics.
When web search is enabled snippets may be recent but not perfect compare sources when it matters.
Enjoy exploring Pyx ask what you want to build next.
"""

_TOKEN_RE = re.compile(r"[a-z0-9']+|[.!?]", re.I)


def _tokens(text: str) -> list[str]:
    return [t.lower() if t.isalpha() or "'" in t else t for t in _TOKEN_RE.findall(text)]


def _train_bigram(token_list: list[str]) -> dict[str, Counter[str]]:
    g: dict[str, Counter[str]] = defaultdict(Counter)
    for i in range(len(token_list) - 1):
        g[token_list[i]][token_list[i + 1]] += 1
    return g


_BIGRAM: dict[str, Counter[str]] | None = None
_VOCAB: list[str] | None = None


def _model() -> tuple[dict[str, Counter[str]], list[str]]:
    global _BIGRAM, _VOCAB
    if _BIGRAM is None:
        toks = _tokens(_CORPUS)
        _BIGRAM = _train_bigram(toks)
        _VOCAB = list({t for t in toks if t not in ".!?"})
    assert _BIGRAM is not None and _VOCAB is not None
    return _BIGRAM, _VOCAB


def _weighted_next(counter: Counter[str]) -> str | None:
    if not counter:
        return None
    pop = list(counter.elements())
    return random.choice(pop)


def _markov_generate(seed_words: list[str], max_tokens: int = 72) -> str:
    bigram, vocab = _model()
    if not seed_words:
        seed_words = ["pyx"]
    out: list[str] = []
    cur = seed_words[-1] if seed_words[-1] in bigram else random.choice(vocab)
    out.append(cur)
    for _ in range(max_tokens - 1):
        nxt = _weighted_next(bigram.get(cur, Counter()))
        if nxt is None:
            cur = random.choice(vocab)
            continue
        out.append(nxt)
        cur = nxt
    # Light punctuation cleanup
    s = " ".join(out)
    s = re.sub(r"\s+([.!?])", r"\1", s)
    return s.strip().capitalize() + ("." if s and s[-1] not in ".!?" else "")


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


def preview_web_snippets(query: str, *, max_results: int = 5, timeout: int = 18) -> tuple[str, str | None, str | None]:
    """Keyless DDG HTML search; returns (text, provider, error)."""
    query = (query or "").strip()[:500]
    if not query:
        return "", None, "empty query"
    ddg_url = "https://html.duckduckgo.com/html/"
    ua = (
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

    blocks = re.findall(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>(?s:.*?)class="result__snippet"[^>]*>(.*?)</a>',
        page,
        re.I | re.S,
    )
    lines: list[str] = []
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
        lines.append(f"• {title} — {snippet[:280]}")
        if len(lines) >= max_results:
            break

    if not lines:
        for m in re.finditer(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', page, re.I | re.S):
            href = m.group(1)
            if _ddg_is_ad_link(href):
                continue
            url = _unwrap_duck_redirect(href)
            title = _strip_html_fragment(m.group(2))
            if title or url:
                lines.append(f"• {title} ({url})")
            if len(lines) >= max_results:
                break

    text = "\n".join(lines).strip()
    if not text:
        return "", "local-web", "no results"
    return text[:6000], "local-web", None


def _web_auto_trigger(user_text: str) -> bool:
    t = (user_text or "").strip().lower()
    if len(t) < 8:
        return False
    needles = (
        "latest", "news", "today", "2024", "2025", "2026", "price", "release",
        "who won", "weather",
    )
    if any(n in t for n in needles):
        return True
    if "?" in user_text and len(t) > 12:
        return any(k in t for k in ("when", "where", "who is", "what is the", "how many"))
    return False


def _normalize_messages(raw: Any) -> tuple[list[dict[str, str]], str | None]:
    if not isinstance(raw, list) or not raw:
        return [], "messages must be a non-empty list"
    out: list[dict[str, str]] = []
    for m in raw[-24:]:
        if not isinstance(m, dict):
            return [], "invalid message"
        role = (m.get("role") or "").strip()
        content = (m.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            return [], "invalid message shape"
        if len(content) > 4000:
            content = content[:4000]
        out.append({"role": role, "content": content})
    if not out or out[-1]["role"] != "user":
        return [], "last message must be user"
    return out, None


def build_preview_reply(
    messages: list[dict[str, str]],
    *,
    use_web: bool = False,
    use_web_auto: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Returns (reply_text, meta)."""
    last_user = messages[-1]["content"]
    history_seed: list[str] = []
    for m in messages[:-1][-6:]:
        history_seed.extend(_tokens(m["content"])[:12])

    do_web = use_web or (use_web_auto and _web_auto_trigger(last_user))
    meta: dict[str, Any] = {
        "engine": "pyx-1.3-preview",
        "model": "markov-bigram+optional-web",
        "web": {"used": do_web, "provider": None, "error": None, "query": None},
    }

    web_block = ""
    if do_web:
        meta["web"]["query"] = last_user[:500]
        snippets, provider, werr = preview_web_snippets(last_user)
        meta["web"]["provider"] = provider
        meta["web"]["error"] = werr
        if snippets:
            web_block = (
                "Here is a quick pulse from web snippets (verify important facts yourself):\n"
                f"{snippets}\n\n"
            )
        elif werr:
            web_block = f"(Web search note: {werr})\n\n"

    user_toks = _tokens(last_user)
    seed = (history_seed + user_toks)[-3:] or ["pyx"]
    # Bias first token toward something in vocabulary
    _, vocab = _model()
    if seed[-1] not in _model()[0]:
        seed[-1] = random.choice(vocab)

    body = _markov_generate(seed, max_tokens=64)
    # Second short paragraph for variety (different seed)
    tail_seed = [random.choice(vocab), user_toks[0] if user_toks else "pyx"]
    tail = _markov_generate(tail_seed, max_tokens=48)
    if tail.lower().startswith(body[:20].lower()):
        tail = _markov_generate(["build", "with", "pyx"], max_tokens=40)

    reply = (web_block + body + " " + tail).strip()
    reply = re.sub(r"\s+", " ", reply)
    return reply, meta
