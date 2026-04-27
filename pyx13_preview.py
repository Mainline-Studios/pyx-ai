"""
Pyx 1.3 **website preview** — pure Python, no GGUF, no external LLM APIs.

- Prose: short ranked sentences from a small library.
- **Code**: long procedural synthesis (hash-derived names / stage counts), not canned paragraphs.
- **In-chat App**: Pyxel grid hint (`in_chat_app` JSON) so the Talk UI can auto-run `pixel_art`.
- Optional DuckDuckGo HTML snippets (same idea as Pyx Talk).
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import random
import re
import textwrap
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

# --- Curated lines: written as normal sentences (no Markov word salad) ---
# Each entry: (sentence, optional topic tags for light routing)
_LIBRARY: list[tuple[str, frozenset[str]]] = [
    (
        "Pyx is an assistant family for creators, students, and builders who want clear answers without unnecessary fluff.",
        frozenset(),
    ),
    ("Pyx Talk helps you chat, plan, and learn.", frozenset({"talk", "general"})),
    ("Pyx Code helps you write and refactor software in many languages.", frozenset({"code"})),
    ("Pyxel makes tiny pixel art from short prompts — handy for sprites and quick experiments.", frozenset({"pixel"})),
    (
        "Pyx 1.0 on the public site uses fast cloud models through the Pyx API; Pyx 1.3 is the local-first line you can download and wire into your own UI.",
        frozenset({"version", "download", "local"}),
    ),
    (
        "This website preview runs without a weight file so you can try the product shape before you set up local models.",
        frozenset({"preview", "general"}),
    ),
    (
        "Ask what you want to extend: features, classroom workflows, or how moderation and scoring work before messages are shown.",
        frozenset({"extend", "moderation"}),
    ),
    (
        "If you are curious about pixel art, ask for grid sizes, color palettes, or prompt ideas — I can outline practical starting points.",
        frozenset({"pixel"}),
    ),
    (
        "If you want study tips, ask for a short study plan or a checklist and keep the topic specific.",
        frozenset({"study"}),
    ),
    (
        "Developers often use Pyx to draft commit messages, tests, and comments; share feature ideas politely and we can reason about tradeoffs.",
        frozenset({"code", "community"}),
    ),
    (
        "Privacy matters: in full local mode your weights stay on your machine; cloud routes depend on the provider you configure — always review terms for your org.",
        frozenset({"privacy", "cloud"}),
    ),
    (
        "Web search snippets can ground an answer when you enable search; treat them as hints and verify anything critical.",
        frozenset({"web"}),
    ),
    (
        "For troubleshooting, check your network, API keys for cloud routes, and model paths for local routes; retry once if a gateway times out.",
        frozenset({"troubleshoot"}),
    ),
    (
        "Pyx grows with feedback — try small prompts first, celebrate when your first integration works end to end, and use Pyx as a tool alongside human judgment.",
        frozenset({"general", "community"}),
    ),
    (
        "I am not a replacement for professional advice on medical, legal, or safety topics — double-check important facts, especially when search is on.",
        frozenset({"safety"}),
    ),
]

_STOP = frozenset(
    "a an the is are was were be been being i you we it its this that these those "
    "to of in on for with as at by from or if do does did so than then too very just "
    "what which who how when where why can could should would about into out up down "
    "me my your our their them they he she his her not no yes ok okay".split()
)

_TOPIC_TRIGGERS: dict[str, tuple[str, ...]] = {
    "pixel": ("pixel", "pyxel", "sprite", "palette", "16x16", "32x32", "tile", "bitmap"),
    "code": (
        "code",
        "python",
        "javascript",
        "typescript",
        "program",
        "refactor",
        "bug",
        "api",
        "git",
        "commit",
        "implement",
        "function",
        "class",
        "script",
    ),
    "study": ("study", "exam", "homework", "class", "school", "learn", "notes", "quiz"),
    "privacy": ("privacy", "local", "on device", "on-device", "data", "gdpr", "hipaa"),
    "cloud": ("cloud", "groq", "hosted", "api key", "rate limit"),
    "download": ("download", "install", "starter", "zip", "repo", "1.3"),
    "moderation": ("moderat", "safe", "filter", "ban", "score", "censor"),
    "web": ("web search", "search the web", "ddg", "duck", "snippet", "news", "latest"),
    "troubleshoot": ("error", "timeout", "502", "503", "broken", "fix", "not work"),
    "extend": ("extend", "integrat", "workflow", "classroom", "studio", "plugin"),
    "safety": ("medical", "legal", "danger", "harm", "suicide", "violence"),
    "version": ("1.0", "1.3", "pyx 1", "difference", "which pyx"),
    "community": ("feature request", "feedback", "idea", "roadmap"),
    "talk": ("talk", "chat", "conversation"),
    "general": (),
}

_GREETING_RE = re.compile(r"^\s*(hi|hello|hey|good\s+(morning|afternoon|evening)|greetings)\b", re.I)
_FAREWELL_RE = re.compile(r"\b(bye|goodbye|see you|cya)\b", re.I)
_CODE_INTENT_RE = re.compile(
    r"\b(?:"
    r"(?:write|generate|implement|create|build|show)\s+(?:me\s+)?(?:a\s+)?"
    r"(?:long\s+)?(?:full\s+)?(?:working\s+)?"
    r"(?:code|script|module|function|class|program|snippet|boilerplate|scaffold)"
    r"|source\s*code"
    r"|code\s+example"
    r"|example\s+code"
    r")\b",
    re.I,
)


def _content_words(text: str) -> set[str]:
    return {
        w
        for w in re.findall(r"[a-z0-9']+", (text or "").lower())
        if len(w) > 1 and w not in _STOP
    }


def _detect_topics(text: str) -> frozenset[str]:
    t = (text or "").lower()
    out: set[str] = set()
    for topic, needles in _TOPIC_TRIGGERS.items():
        if not needles:
            continue
        if any(n in t for n in needles):
            out.add(topic)
    if "search" in t and "web" in t:
        out.add("web")
    return frozenset(out)


def _sentence_score(sentence: str, sent_topics: frozenset[str], pool: set[str], topics: frozenset[str]) -> float:
    sw = _content_words(sentence)
    if not pool:
        overlap = 0.0
    else:
        overlap = len(pool & sw) / max(1, len(pool))
    topic_bonus = 0.15 * len(sent_topics & topics)
    # Light preference for shorter, clearer lines when scores tie
    brevity = max(0, 1.0 - len(sentence) / 450)
    return overlap * 2.0 + topic_bonus + brevity * 0.05


def _pick_sentences(
    last_user: str,
    history_text: str,
    topics: frozenset[str],
    max_sentences: int = 3,
) -> list[str]:
    pool = _content_words(last_user) | _content_words(history_text)
    scored: list[tuple[float, str]] = []
    for sentence, st in _LIBRARY:
        scored.append((_sentence_score(sentence, st, pool, topics), sentence))
    scored.sort(key=lambda x: -x[0])
    chosen: list[str] = []
    seen_lower: set[str] = set()
    for score, sentence in scored:
        if score < 0.08 and len(chosen) >= 2:
            break
        key = sentence[:48].lower()
        if key in seen_lower:
            continue
        seen_lower.add(key)
        chosen.append(sentence)
        if len(chosen) >= max_sentences:
            break
    if not chosen:
        chosen = [t[0] for t in _LIBRARY[:2]]
    return chosen


def _compose_body(last_user: str, messages: list[dict[str, str]], topics: frozenset[str]) -> str:
    hist_bits = []
    for m in messages[:-1][-4:]:
        if m.get("role") == "user":
            hist_bits.append(m.get("content", ""))
    history_text = " ".join(hist_bits)

    if _FAREWELL_RE.search(last_user):
        return "Goodbye — thanks for trying Pyx. Come back anytime if you want to plan your next build."

    opener = ""
    if _GREETING_RE.search(last_user) or last_user.strip().lower() in {"hi", "hello", "hey"}:
        opener = "Hi — thanks for trying Pyx. "

    dynamic_head = "Good question. " if ("?" in last_user or " is " in (" " + last_user.lower() + " ")) else "Got it. "

    sentences = _pick_sentences(last_user, history_text, topics, max_sentences=3)
    body = opener + dynamic_head + " ".join(sentences)

    if "code" in topics and "pixel" not in topics:
        body += " If you want help with implementation, tell me your language and target output and I will draft steps in plain language first."
    if "pixel" in topics:
        body += " For Pyxel, include subject, mood, and grid size (for example 10x10 or 16x16) and I can propose prompt and palette options."

    if last_user.strip().endswith("?"):
        body += " If that does not quite answer your question, rephrase with one concrete detail (goal, stack, or timeline) and I will narrow it down."

    body += " What would you like to build or learn next with Pyx?"
    return re.sub(r"\s+", " ", body).strip()


def _wants_long_code(text: str) -> bool:
    """Heuristic: user is asking for substantive generated source, not product Q&A."""
    raw = text or ""
    t = raw.lower()
    if "```" in raw:
        return True
    if "long code" in t or "lots of code" in t or "big script" in t or "full script" in t:
        return True
    if len(t) < 14:
        return False
    prog = (
        "python",
        "javascript",
        "typescript",
        ".py",
        ".ts",
        ".js",
        "node",
        "nodejs",
        "rust",
        "golang",
        " go ",
        "java",
        "kotlin",
        "swift",
        "c++",
        "csharp",
        "ruby",
    )
    if any(p in t for p in prog) and any(k in t for k in ("code", "script", "module", "function", "class", "snippet", "implement")):
        return True
    codeish = (
        "code",
        "script",
        "snippet",
        "boilerplate",
        "scaffold",
        "function",
        "class ",
        "implement",
        "program",
        "write a ",
        "write me",
    )
    if any(w in t for w in codeish) and _CODE_INTENT_RE.search(raw):
        return True
    if "code" in t and len(t) > 28 and any(v in t for v in ("write", "show", "generate", "create", "build", "implement")):
        return True
    return _CODE_INTENT_RE.search(raw) is not None and (
        "def " in t or "class " in t or "import " in t or "async " in t or "function " in t
    )


def _extract_pyxel_subject(text: str) -> str | None:
    """Match Pyx Talk pixel intents + loose Pyxel wording; returns subject for pixel_art API."""
    s = (text or "").strip()
    if len(s) < 4:
        return None
    patterns = [
        r"(?is)^\s*(?:generate|make|create)\s+(?:a\s+)?(?:an\s+)?image\s+(?:of\s+)?(.+)$",
        r"(?is)^\s*(?:pixel\s*art|pix\s*art)\s+(?:of\s+)?(.+)$",
        r"(?is)^\s*draw\s+(?:a\s+)?(?:pixel\s*art\s+)?(?:of\s+)?(.+)$",
    ]
    for p in patterns:
        m = re.match(p, s)
        if m:
            sub = (m.group(1) or "").strip()
            return sub[:500] if sub else None
    low = s.lower()
    if any(k in low for k in ("pyxel", "pixel art", "pix art", "sprite", "10x10")):
        sub = re.sub(
            r"(?i)^\s*(?:make|create|draw|generate|show|open|start|run)\s+(?:me\s+)?(?:a\s+)?(?:the\s+)?(?:some\s+)?",
            "",
            s,
        ).strip()
        sub = re.sub(r"(?i)^(pyxel|pixel\s*art|pix\s*art)\s*(?:of|:)?\s*", "", sub).strip()
        return (sub or "custom prompt")[:500]
    return None


def _wants_pyxel_in_chat(text: str) -> bool:
    return _extract_pyxel_subject(text) is not None


try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


_LM_PATH = Path(__file__).resolve().parent / "models" / "pyx13_fflm_v1.npz"
_LM_CACHE: dict[str, Any] | None = None


def _lm_tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+|[.,!?;:]", (text or "").lower())


def _lm_corpus_sentences() -> list[list[str]]:
    out: list[list[str]] = []
    for sentence, _tags in _LIBRARY:
        toks = _lm_tokenize(sentence)
        if toks:
            out.append(toks)
    out.extend(
        [
            _lm_tokenize("Pyx 1.3 preview should respond in natural language with coherent complete sentences."),
            _lm_tokenize("Ask direct questions and Pyx will answer clearly without random word salad."),
            _lm_tokenize("For coding help, explicitly ask for code and include language plus goal."),
            _lm_tokenize("For Pyxel, ask for a subject and Pyx can launch the in-chat app grid automatically."),
        ]
    )
    out.extend(
        [
            _lm_tokenize("Pyx should answer in plain language first unless the user explicitly asks for code."),
            _lm_tokenize("A strong answer starts with intent, gives steps, and ends with one practical next action."),
            _lm_tokenize("When users ask if Pyx is good, respond honestly and explain strengths plus limits."),
            _lm_tokenize("If users ask for code, generate complete code blocks with clear structure and comments."),
            _lm_tokenize("For debugging requests, suggest checks, likely root causes, and one direct fix path."),
            _lm_tokenize("If a request is vague, ask one focused clarification question and offer a default approach."),
        ]
    )
    return [s for s in out if len(s) > 2]


def _lm_build_dataset(ctx: int = 6) -> tuple[list[str], Any, Any]:
    if np is None:
        raise RuntimeError("numpy unavailable")
    sents = _lm_corpus_sentences()
    counts: dict[str, int] = {}
    for s in sents:
        for t in s:
            counts[t] = counts.get(t, 0) + 1
    vocab = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"] + sorted(counts.keys())
    tok2id = {t: i for i, t in enumerate(vocab)}
    bos = tok2id["<BOS>"]
    eos = tok2id["<EOS>"]
    unk = tok2id["<UNK>"]
    X_rows: list[list[int]] = []
    y_rows: list[int] = []
    for s in sents:
        ids = [tok2id.get(t, unk) for t in s] + [eos]
        context = [bos] * ctx
        for target in ids:
            X_rows.append(list(context))
            y_rows.append(target)
            context = context[1:] + [target]
    X = np.array(X_rows, dtype=np.int64)
    y = np.array(y_rows, dtype=np.int64)
    return vocab, X, y


def _lm_save_npz(model: dict[str, Any]) -> None:
    if np is None:
        return
    _LM_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        _LM_PATH,
        version=np.array([model["version"]], dtype=np.int64),
        context_size=np.array([model["context_size"]], dtype=np.int64),
        vocab=np.array(model["vocab"], dtype=object),
        E=model["E"],
        W1=model["W1"],
        b1=model["b1"],
        W2=model["W2"],
        b2=model["b2"],
    )


def _lm_train() -> dict[str, Any]:
    if np is None:
        raise RuntimeError("numpy unavailable for fflm training")
    ctx = 6
    vocab, X, y = _lm_build_dataset(ctx=ctx)
    v = len(vocab)
    emb = 28
    hid = 64

    rng = np.random.default_rng(1337)
    E = rng.normal(0.0, 0.08, size=(v, emb)).astype(np.float32)
    W1 = rng.normal(0.0, 0.05, size=(ctx * emb, hid)).astype(np.float32)
    b1 = np.zeros((hid,), dtype=np.float32)
    W2 = rng.normal(0.0, 0.05, size=(hid, v)).astype(np.float32)
    b2 = np.zeros((v,), dtype=np.float32)

    n = X.shape[0]
    bs = 24
    lr = 0.08
    wd = 1e-5

    for _epoch in range(85):
        idx = rng.permutation(n)
        Xs = X[idx]
        ys = y[idx]
        for i in range(0, n, bs):
            xb = Xs[i : i + bs]
            yb = ys[i : i + bs]
            bsz = xb.shape[0]

            embb = E[xb]  # [B,C,D]
            xflat = embb.reshape(bsz, -1)  # [B,C*D]
            hpre = xflat @ W1 + b1
            h = np.tanh(hpre)
            logits = h @ W2 + b2
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)

            dlog = probs
            dlog[np.arange(bsz), yb] -= 1.0
            dlog /= max(1, bsz)

            dW2 = h.T @ dlog + wd * W2
            db2 = dlog.sum(axis=0)
            dh = dlog @ W2.T
            dhpre = dh * (1.0 - h * h)
            dW1 = xflat.T @ dhpre + wd * W1
            db1 = dhpre.sum(axis=0)
            dx = dhpre @ W1.T
            demb = dx.reshape(bsz, ctx, emb)

            dE = np.zeros_like(E)
            np.add.at(dE, xb, demb)
            dE += wd * E

            E -= lr * dE
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2
        lr *= 0.985

    model = {
        "version": 3,
        "context_size": ctx,
        "vocab": vocab,
        "E": E,
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
    }
    _lm_save_npz(model)
    return model


def _lm_load() -> dict[str, Any]:
    global _LM_CACHE
    if _LM_CACHE is not None:
        return _LM_CACHE
    if np is None:
        # Fallback mode if numpy is unavailable in this runtime.
        _LM_CACHE = {"version": 0}
        return _LM_CACHE
    try:
        npz = np.load(_LM_PATH, allow_pickle=True)
        version = int(npz["version"][0])
        if version == 3:
            _LM_CACHE = {
                "version": version,
                "context_size": int(npz["context_size"][0]),
                "vocab": [str(x) for x in npz["vocab"].tolist()],
                "E": npz["E"].astype(np.float32),
                "W1": npz["W1"].astype(np.float32),
                "b1": npz["b1"].astype(np.float32),
                "W2": npz["W2"].astype(np.float32),
                "b2": npz["b2"].astype(np.float32),
            }
            return _LM_CACHE
    except Exception:
        pass

    lm = _lm_train()
    _LM_CACHE = lm
    return lm


def _lm_detokenize(tokens: list[str]) -> str:
    if not tokens:
        return "Pyx is ready. Ask your question and I will answer clearly."
    s = ""
    for t in tokens:
        if t in ".,!?;:":
            s = s.rstrip() + t + " "
        else:
            s += t + " "
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s*([.!?])\s*", r"\1 ", s).strip()
    s = re.sub(r"([.!?]){2,}", r"\1", s)
    if s:
        s = s[0].upper() + s[1:]
    if s and s[-1] not in ".!?":
        s += "."
    return s[:2400]


def _lm_is_readable(text: str) -> bool:
    words = re.findall(r"[a-zA-Z]+", text or "")
    if len(words) < 8:
        return False
    punct = len(re.findall(r"[,.!?;:]", text or ""))
    if punct > max(10, len(words) // 2):
        return False
    long_words = [w for w in words if len(w) >= 4]
    if len(long_words) < 4:
        return False
    # Reject outputs dominated by a few repeated tokens.
    low = [w.lower() for w in words]
    uniq = len(set(low)) / max(1, len(low))
    if uniq < 0.35:
        return False
    return True


def _lm_generate_text(last_user: str, messages: list[dict[str, str]]) -> str:
    lm = _lm_load()
    if lm.get("version") != 3 or np is None:
        # Safety fallback if numpy model is unavailable in this runtime.
        return _compose_body(last_user, messages, _detect_topics(last_user))

    vocab: list[str] = lm["vocab"]
    tok2id = {t: i for i, t in enumerate(vocab)}
    E = lm["E"]
    W1 = lm["W1"]
    b1 = lm["b1"]
    W2 = lm["W2"]
    b2 = lm["b2"]
    ctx = int(lm["context_size"])
    v = len(vocab)
    bos = tok2id.get("<BOS>", 1)
    eos = tok2id.get("<EOS>", 2)
    unk = tok2id.get("<UNK>", 3)

    prompt_tokens: list[str] = []
    for m in messages[-6:]:
        if m.get("role") == "user":
            prompt_tokens.extend(_lm_tokenize(m.get("content", ""))[:24])
    if not prompt_tokens:
        prompt_tokens = _lm_tokenize(last_user)
    prompt_ids = [tok2id.get(t, unk) for t in prompt_tokens]
    context = [bos] * ctx
    for tid in prompt_ids[-ctx:]:
        context = context[1:] + [tid]

    topic_bias = np.zeros((v,), dtype=np.float32)
    for t in _lm_tokenize(last_user):
        idx = tok2id.get(t)
        if idx is not None:
            topic_bias[idx] += 0.55

    out_ids: list[int] = []
    recent: list[int] = []
    rnd = random.Random()
    max_tokens = 140
    for i in range(max_tokens):
        x = E[np.array(context, dtype=np.int64)].reshape(1, -1)
        h = np.tanh(x @ W1 + b1)
        logits = (h @ W2 + b2).reshape(-1) + topic_bias
        for rid in recent[-30:]:
            logits[rid] -= 0.22
        temp = 0.78
        logits = logits / temp
        top_k = 36
        top_idx = np.argpartition(logits, -top_k)[-top_k:]
        top_logits = logits[top_idx]
        top_logits = top_logits - np.max(top_logits)
        probs = np.exp(top_logits)
        probs = probs / (np.sum(probs) + 1e-9)
        r = rnd.random()
        c = 0.0
        chosen = int(top_idx[int(np.argmax(probs))])
        for j, p in enumerate(probs):
            c += float(p)
            if r <= c:
                chosen = int(top_idx[j])
                break

        if chosen == eos:
            if i < 26:
                continue
            break
        if chosen in (bos,):
            continue
        out_ids.append(chosen)
        recent.append(chosen)
        context = context[1:] + [chosen]
        if len(out_ids) >= 44 and vocab[chosen] in (".", "!", "?"):
            break

    out_tokens = [vocab[i] for i in out_ids if i not in (eos, bos, unk)]
    text = _lm_detokenize(out_tokens)
    # If sampled text is malformed, fall back to the ranked sentence generator.
    if not _lm_is_readable(text):
        text = _compose_body(last_user, messages, _detect_topics(last_user))
    if "what would you like to" in text.lower():
        return text
    return text + " What would you like to do next?"


def _prompt_fingerprint(user: str) -> tuple[str, str]:
    """(8-char hex suffix, TitleCase slug for identifiers)."""
    h = hashlib.sha256((user or "").encode("utf-8", errors="replace")).hexdigest()[:8]
    words = re.findall(r"[a-z0-9]+", (user or "").lower())
    stop = _STOP | frozenset(
        "write generate create build make me please want need help code python js ts javascript typescript".split()
    )
    picked = [w for w in words if w not in stop and len(w) > 2][:4]
    if not picked:
        picked = ["pyx", "task"]
    slug = "".join(w.capitalize() for w in picked)[:48]
    if not slug[0].isalpha():
        slug = "X" + slug
    return h, slug


def _synth_python_module(user: str) -> str:
    h, slug = _prompt_fingerprint(user)
    safe_one_line = (user or "").replace("\n", " ").strip()[:180]
    lines: list[str] = [
        "#!/usr/bin/env python3",
        '"""',
        f"Auto-synthesized module (Pyx 1.3 website preview).",
        f"Fingerprint: {h} — not audited for production.",
        f"Origin prompt (truncated): {safe_one_line!r}",
        '"""',
        "from __future__ import annotations",
        "",
        "import argparse",
        "import json",
        "import logging",
        "import sys",
        "import time",
        "from dataclasses import asdict, dataclass, field",
        "from pathlib import Path",
        "from typing import Any, Callable, Iterable, Iterator, Mapping, Protocol, Sequence",
        "",
        "LOG = logging.getLogger(__name__)",
        "",
        "",
        "@dataclass(slots=True)",
        f"class {slug}Config:",
        '    """Runtime knobs — tweak for your deployment."""',
        "    app_name: str = field(default_factory=lambda: 'pyx-preview-app')",
        "    data_dir: Path = field(default_factory=lambda: Path('.pyx_preview_data'))",
        "    max_retries: int = 3",
        "    backoff_seconds: float = 0.35",
        "    strict_json: bool = True",
        "",
        "",
        "class SupportsValidate(Protocol):",
        "    def validate(self, payload: Mapping[str, Any]) -> tuple[bool, list[str]]: ...",
        "",
        "",
        f"class {slug}Validator:",
        '    """Light structural validation (extend with your schema)."""',
        "",
        "    REQUIRED_KEYS: tuple[str, ...] = ('op', 'payload')",
        "",
        "    def validate(self, payload: Mapping[str, Any]) -> tuple[bool, list[str]]:",
        "        errs: list[str] = []",
        "        for k in self.REQUIRED_KEYS:",
        "            if k not in payload:",
        "                errs.append(f'missing:{k}')",
        "        op = payload.get('op')",
        "        if op is not None and not isinstance(op, str):",
        "            errs.append('op_must_be_str')",
        "        return (len(errs) == 0, errs)",
        "",
        "",
        f"class {slug}Repository:",
        '    """Tiny JSONL event log — swap for SQLite / HTTP as needed."""',
        "",
        "    def __init__(self, root: Path) -> None:",
        "        self._root = root",
        "        self._root.mkdir(parents=True, exist_ok=True)",
        "        self._path = self._root / 'events.jsonl'",
        "",
        "    def append(self, record: Mapping[str, Any]) -> None:",
        "        line = json.dumps(record, ensure_ascii=False, separators=(',', ':'))",
        "        with self._path.open('a', encoding='utf-8') as fh:",
        "            fh.write(line + '\\n')",
        "",
        "    def tail(self, n: int = 50) -> list[dict[str, Any]]:",
        "        if not self._path.is_file():",
        "            return []",
        "        rows: list[str] = self._path.read_text(encoding='utf-8', errors='replace').splitlines()",
        "        out: list[dict[str, Any]] = []",
        "        for raw in rows[-n:]:",
        "            try:",
        "                out.append(json.loads(raw))",
        "            except json.JSONDecodeError:",
        "                continue",
        "        return out",
        "",
        "",
        f"class {slug}Pipeline:",
        '    """Composable transform stages — logic is expanded procedurally below."""',
        "",
        "    def __init__(self, cfg: {slug}Config) -> None:".replace("{slug}", slug),
        "        self._cfg = cfg",
        "        self._validators: list[SupportsValidate] = [{slug}Validator()]".replace(
            "{slug}", slug
        ),
        "",
        "    def _run_validators(self, payload: Mapping[str, Any]) -> None:",
        "        for v in self._validators:",
        "            ok, errs = v.validate(payload)",
        "            if not ok:",
        "                raise ValueError('validation_failed:' + ','.join(errs))",
    ]

    # Procedurally generated stage functions (unique per fingerprint)
    seed = int(h[:6], 16)
    n_stages = 18 + (seed % 14)
    for i in range(n_stages):
        bias = (seed >> (i % 8)) & 0xFF
        lines.extend(
            [
                "",
                f"    def stage_{i:02d}_{h}(self, data: dict[str, Any]) -> dict[str, Any]:",
                f'        """Transform stage {i} — mixed bias {bias}."""',
                "        out = dict(data)",
                f"        out['stage_{i:02d}'] = {{'bias': {bias}, 'ts': time.time()}}",
                "        meta = out.setdefault('_meta', {})",
                f"        trace = meta.setdefault('trace', [])",
                f"        trace.append('stage_{i:02d}')",
                "        return out",
            ]
        )

    lines.extend(
        [
            "",
            "    def run(self, initial: dict[str, Any]) -> dict[str, Any]:",
            "        cur = dict(initial)",
            f"        for i in range({n_stages}):",
            f"            fn = getattr(self, f'stage_{{i:02d}}_{h}')",
            "            cur = fn(cur)",
            "        return cur",
            "",
            "",
            f"class {slug}Service:",
            '    """Facade: validation + pipeline + persistence."""',
            "",
            "    def __init__(self, cfg: {slug}Config) -> None:".replace("{slug}", slug),
            "        self._cfg = cfg",
            "        self._repo = {slug}Repository(cfg.data_dir)".replace("{slug}", slug),
            "        self._pipe = {slug}Pipeline(cfg)".replace("{slug}", slug),
            "",
            "    def handle(self, envelope: Mapping[str, Any]) -> dict[str, Any]:",
            "        ok, errs = {slug}Validator().validate(envelope)".replace("{slug}", slug),
            "        if not ok:",
            "            raise ValueError('invalid_envelope:' + ','.join(errs))",
            "        op = str(envelope['op'])",
            "        payload = envelope.get('payload')",
            "        if not isinstance(payload, dict):",
            "            raise TypeError('payload_must_be_dict')",
            "        blob = {'op': op, 'payload': payload, 'received_at': time.time()}",
            "        result = self._pipe.run(blob)",
            "        self._repo.append({'kind': 'result', 'body': result})",
            "        return result",
            "",
            "    def replay(self, n: int = 20) -> list[dict[str, Any]]:",
            "        return self._repo.tail(n)",
            "",
            "",
            "def _configure_logging(verbose: bool) -> None:",
            "    level = logging.DEBUG if verbose else logging.INFO",
            "    logging.basicConfig(",
            "        level=level,",
            "        format='%(asctime)s %(levelname)s %(name)s — %(message)s',",
            "    )",
            "",
            "",
            "def _demo_payload() -> dict[str, Any]:",
            "    return {",
            "        'op': 'demo.echo',",
            "        'payload': {",
            f"            'note': 'synthetic run {h}',",
            "            'values': list(range(8)),",
            "        },",
            "    }",
            "",
            "",
            "def main(argv: Sequence[str] | None = None) -> int:",
            "    p = argparse.ArgumentParser(description='Synthesized Pyx preview CLI')",
            "    p.add_argument('--verbose', action='store_true')",
            "    p.add_argument('--replay', type=int, default=0, help='Print last N log rows')",
            "    args = p.parse_args(list(argv) if argv is not None else None)",
            "    _configure_logging(args.verbose)",
            "    cfg = {slug}Config()".replace("{slug}", slug),
            "    svc = {slug}Service(cfg)".replace("{slug}", slug),
            "    if args.replay > 0:",
            "        print(json.dumps(svc.replay(args.replay), indent=2, ensure_ascii=False))",
            "        return 0",
            "    try:",
            "        out = svc.handle(_demo_payload())",
            "        print(json.dumps(out, indent=2, ensure_ascii=False))",
            "        return 0",
            "    except Exception as exc:  # pragma: no cover — demo surface",
            "        LOG.exception('handler_failed')",
            "        print(f'error:{exc}', file=sys.stderr)",
            "        return 1",
            "",
            "",
            'if __name__ == "__main__":',
            "    raise SystemExit(main())",
            "",
        ]
    )

    return "\n".join(lines)


def _synth_javascript_module(user: str) -> str:
    h, slug = _prompt_fingerprint(user)
    safe = (user or "").replace("`", "'").replace("\n", " ").strip()[:160]
    lines = [
        "/**",
        " * Auto-synthesized ESM module (Pyx 1.3 website preview).",
        f" * Fingerprint: {h}",
        f" * Prompt (truncated): {safe}",
        " */",
        "",
        "const LOG_PREFIX = '[pyx-preview]';",
        "",
        f"export class {slug}Config {{",
        "  constructor() {",
        "    this.maxRetries = 3;",
        "    this.backoffMs = 350;",
        f"    this.appName = 'pyx-preview-{h}';",
        "  }",
        "}",
        "",
        f"export class {slug}Validator {{",
        "  validate(envelope) {",
        "    const errs = [];",
        "    if (!envelope || typeof envelope !== 'object') errs.push('envelope_object');",
        "    if (!('op' in envelope)) errs.push('missing_op');",
        "    if (!('payload' in envelope)) errs.push('missing_payload');",
        "    return { ok: errs.length === 0, errs };",
        "  }",
        "}",
        "",
        f"export class {slug}Pipeline {{",
        "  constructor(cfg) {",
        "    this.cfg = cfg;",
        "    this._buildStages();",
        "  }",
        "",
        "  _buildStages() {",
        f"    const h = '{h}';",
        f"    this._stages = [];",
        f"    for (let i = 0; i < 26; i++) {{",
        "      this._stages.push((data) => ({",
        "        ...data,",
        "        [`stage_${String(i).padStart(2, '0')}`]: { bias: (i * 17 + h.charCodeAt(i % 8)) % 255, at: Date.now() },",
        "      }));",
        "    }",
        "  }",
        "",
        "  run(initial) {",
        "    return this._stages.reduce((acc, fn) => fn(acc), { ...initial });",
        "  }",
        "}",
        "",
        f"export class {slug}Service {{",
        "  constructor(cfg) {",
        "    this.cfg = cfg;",
        f"    this.validator = new {slug}Validator();",
        f"    this.pipeline = new {slug}Pipeline(cfg);",
        "    this._log = [];",
        "  }",
        "",
        "  handle(envelope) {",
        "    const { ok, errs } = this.validator.validate(envelope);",
        "    if (!ok) throw new Error('validation_failed:' + errs.join(','));",
        "    const merged = { ...envelope, receivedAt: Date.now() };",
        "    const result = this.pipeline.run(merged);",
        "    this._log.push({ kind: 'result', body: result });",
        "    return result;",
        "  }",
        "",
        "  tail(n = 20) {",
        "    return this._log.slice(-n);",
        "  }",
        "}",
        "",
        "export function demoEnvelope() {",
        "  return {",
        "    op: 'demo.echo',",
        "    payload: { values: Array.from({ length: 8 }, (_, i) => i), note: 'synthetic' },",
        "  };",
        "}",
        "",
        "export async function main() {",
        "  const cfg = new {slug}Config();".replace("{slug}", slug),
        f"  const svc = new {slug}Service(cfg);",
        "  const out = svc.handle(demoEnvelope());",
        "  console.log(JSON.stringify(out, null, 2));",
        "}",
        "",
        "// Example: node --experimental-vm-modules yourfile.mjs (if using top-level await elsewhere)",
    ]
    return "\n".join(lines)


def _pick_lang(user: str) -> str:
    t = (user or "").lower()
    if "typescript" in t or ".ts" in t or " ts " in t:
        return "ts"
    if "javascript" in t or "node" in t or ".js" in t or " js " in t:
        return "js"
    return "py"


def _compose_code_reply(user: str, messages: list[dict[str, str]]) -> str:
    lang = _pick_lang(user)
    if lang == "py":
        code = _synth_python_module(user)
        fence = "python"
    else:
        code = _synth_javascript_module(user)
        fence = "typescript" if lang == "ts" else "javascript"

    hist_note = ""
    for m in messages[:-1][-2:]:
        if m.get("role") == "user":
            frag = (m.get("content") or "").replace("\n", " ").strip()[:120]
            if frag:
                hist_note += f"- Earlier: {frag}\n"

    intro = textwrap.dedent(
        f"""\
        Here is **long, synthesized source** from the Pyx 1.3 website preview (procedural generator — no LLM weights). Names and stage count are derived from a hash of your prompt, so each request differs slightly.

        {hist_note if hist_note else ""}Adapt types, validation, and I/O to your real project; this is a starting scaffold, not a security-reviewed library.
        """
    ).strip()

    return f"{intro}\n\n```{fence}\n{code}\n```\n\n*Preview limitation:* logic inside stages is generic; swap in your domain rules.*"


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


def build_preview_reply(
    messages: list[dict[str, str]],
    *,
    use_web: bool = False,
    use_web_auto: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Returns (reply_text, meta)."""
    last_user = messages[-1]["content"]
    topics = _detect_topics(last_user)

    do_web = use_web or (use_web_auto and _web_auto_trigger(last_user))
    meta: dict[str, Any] = {
        "engine": "pyx-1.3-preview",
        "model": "natural-language+corpus-rank+optional-web",
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

    # In-chat App: Pyxel grid (client auto-runs pixel_art from this hint)
    if _wants_pyxel_in_chat(last_user):
        subj = _extract_pyxel_subject(last_user) or "custom prompt"
        safe = subj.replace("*", "\\*").replace("<", "")
        meta["in_chat_app"] = {
            "kind": "pyxel_grid",
            "prompt": subj,
            "auto_run": True,
            "label": "Pyxel",
        }
        meta["model"] = "in-chat-app-pyxel+optional-web"
        short = (
            f"**In-chat App · Pyxel grid** — launching the 10×10 pixel editor for: *{safe}*.\n\n"
            "The grid appears below once pixels return from the API (same as Pyx 1.0 Pyxel)."
        )
        reply = (web_block + short).strip()
        return reply, meta

    if _wants_long_code(last_user):
        meta["model"] = "procedural-long-code+optional-web"
        code_part = _compose_code_reply(last_user, messages)
        reply = (web_block + code_part).strip()
        return reply, meta

    meta["model"] = "pyx13-made-llm-v2+optional-web"
    body = _lm_generate_text(last_user, messages)
    reply = (web_block + body).strip()
    reply = re.sub(r"[ \t]+", " ", reply)
    return reply, meta
