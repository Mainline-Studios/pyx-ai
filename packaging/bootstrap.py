"""Pyx 1.5 first-run model bootstrap.

Everything in this module is local-only: it talks to Ollama's HTTP API on
``http://127.0.0.1:11434`` (no OpenAI-compat layer) because the native API
emits streaming pull progress as NDJSON — perfect for a progress page.

The launcher imports this, exposes routes on the same Flask app, and the
setup page (``public/pyx-setup.html``) drives it from the browser.

No hard dependency on ``ollama`` on PATH: if it's installed we can cold-start
``ollama serve`` ourselves; if not, the status endpoint reports the correct
download URL for the user's platform.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, Iterator, List, Optional, Tuple


# Laptop-friendly defaults — smaller than the cloud mix so first-run downloads
# in minutes, not hours. Users can override via env vars; docs list alternatives.
DEFAULT_MODELS: Dict[str, str] = {
    "PYX_TALK_MODEL_FAST": "llama3.2:3b-instruct",
    "PYX_TALK_MODEL_SMART": "llama3.1:8b-instruct",
    "PYX_TALK_MODEL_THINKING": "llama3.1:8b-instruct",
    "PYX_CODE_MODEL": "gpt-oss:20b",
    "PYX_PIXEL_MODEL": "gpt-oss:20b",
}

OLLAMA_BASE = "http://127.0.0.1:11434"
PULL_TIMEOUT = 60 * 60  # NDJSON stream timeout (1h cap)


# Curated catalog the main screen renders as cards. Approx sizes are Q4 quants
# Ollama ships by default (so Ollama's own numbers are the source of truth —
# we only show them to help the user budget disk).
CATALOG: List[Dict[str, object]] = [
    # ---- Llama (Meta) ----
    {
        "id": "llama3.2:1b",
        "name": "Llama 3.2 1B",
        "family": "llama",
        "family_label": "Llama",
        "size_gb": 1.3,
        "role": "test",
        "tags": ["test", "tiny"],
        "blurb": "Tiny Llama 3.2 — instant replies, great for trying Pyx on a laptop CPU.",
    },
    {
        "id": "llama3.2:3b-instruct",
        "name": "Llama 3.2 3B Instruct",
        "family": "llama",
        "family_label": "Llama",
        "size_gb": 2.0,
        "role": "fast",
        "tags": ["default", "talk-fast"],
        "blurb": "Default for Pyx Talk fast mode. Fits easily in 8 GB RAM.",
    },
    {
        "id": "llama3.1:8b-instruct",
        "name": "Llama 3.1 8B Instruct",
        "family": "llama",
        "family_label": "Llama",
        "size_gb": 4.7,
        "role": "smart",
        "tags": ["default", "talk-smart", "talk-thinking"],
        "blurb": "Default for Pyx Talk smart + reasoning. Best balance on consumer GPUs.",
    },
    {
        "id": "llama3.3:70b-instruct",
        "name": "Llama 3.3 70B Instruct",
        "family": "llama",
        "family_label": "Llama",
        "size_gb": 40.0,
        "role": "flagship",
        "tags": ["optional", "heavy"],
        "blurb": "Meta's flagship 70B. Needs 48 GB VRAM (Q4) or heavy CPU offload.",
    },
    # ---- GPT-OSS (OpenAI, 2025) ----
    {
        "id": "gpt-oss:20b",
        "name": "GPT-OSS 20B",
        "family": "gpt-oss",
        "family_label": "GPT-OSS",
        "size_gb": 13.0,
        "role": "code",
        "tags": ["default", "code", "pixel"],
        "blurb": "Default for Pyx Code + Pyxel. Excellent at code completion and structured output.",
    },
    {
        "id": "gpt-oss:120b",
        "name": "GPT-OSS 120B",
        "family": "gpt-oss",
        "family_label": "GPT-OSS",
        "size_gb": 65.0,
        "role": "flagship",
        "tags": ["optional", "heavy"],
        "blurb": "OpenAI's flagship open-weight model. 80 GB+ VRAM or big-RAM CPU offload.",
    },
]


# ---------------------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------------------

def ollama_download_url() -> str:
    sys_name = platform.system().lower()
    if sys_name == "darwin":
        return "https://ollama.com/download/mac"
    if sys_name == "windows":
        return "https://ollama.com/download/windows"
    return "https://ollama.com/download/linux"


def ollama_binary() -> Optional[str]:
    """Path to a usable ``ollama`` CLI, or None."""
    path = shutil.which("ollama")
    if path:
        return path
    # macOS: `Ollama.app` bundles a CLI at a predictable location.
    if platform.system() == "Darwin":
        for candidate in (
            "/Applications/Ollama.app/Contents/Resources/ollama",
            os.path.expanduser("~/Applications/Ollama.app/Contents/Resources/ollama"),
        ):
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
    return None


def ollama_serve_background() -> bool:
    """Start ``ollama serve`` detached (best-effort). Returns True if the
    HTTP port became reachable within ~15s."""
    binary = ollama_binary()
    if not binary:
        return False
    try:
        kwargs = {}
        if platform.system() == "Windows":
            # Avoid a popup console window on Windows.
            DETACHED_PROCESS = 0x00000008
            kwargs["creationflags"] = DETACHED_PROCESS
        else:
            kwargs["start_new_session"] = True
        subprocess.Popen(
            [binary, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            **kwargs,
        )
    except Exception:
        return False

    deadline = time.time() + 15.0
    while time.time() < deadline:
        if ollama_is_up():
            return True
        time.sleep(0.4)
    return False


# ---------------------------------------------------------------------------
# Ollama HTTP API
# ---------------------------------------------------------------------------

def _get(path: str, timeout: float = 3.0) -> Optional[dict]:
    try:
        req = urllib.request.Request(OLLAMA_BASE + path, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace") or "{}")
    except Exception:
        return None


def ollama_is_up() -> bool:
    return _get("/api/tags", timeout=1.5) is not None


def installed_model_names() -> List[str]:
    data = _get("/api/tags") or {}
    names: List[str] = []
    for item in data.get("models", []) or []:
        n = item.get("name")
        if isinstance(n, str) and n:
            names.append(n)
    return names


def required_models() -> List[str]:
    """De-duplicated list of model names Pyx 1.5 needs right now (env-aware)."""
    seen: Dict[str, bool] = {}
    for env_key, default in DEFAULT_MODELS.items():
        name = (os.environ.get(env_key) or "").strip() or default
        if name and name not in seen:
            seen[name] = True
    return list(seen.keys())


def pull_model(name: str) -> Iterator[Dict[str, object]]:
    """Stream NDJSON progress events from ``POST /api/pull``.

    Yields dicts like:
        {"status": "pulling manifest"}
        {"status": "downloading", "digest": "sha256:…", "total": 1234, "completed": 100}
        {"status": "success"}
    Plus local "error" events if the stream fails.
    """
    body = json.dumps({"model": name, "stream": True}).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_BASE + "/api/pull",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=PULL_TIMEOUT) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    yield {"status": "note", "text": line[:200]}
    except urllib.error.HTTPError as e:  # pragma: no cover
        detail = e.read().decode("utf-8", errors="replace")[:500] if e.fp else ""
        yield {"status": "error", "code": e.code, "detail": detail}
    except urllib.error.URLError as e:  # pragma: no cover
        yield {"status": "error", "detail": str(e.reason)}
    except Exception as e:  # pragma: no cover
        yield {"status": "error", "detail": str(e)}


# ---------------------------------------------------------------------------
# Public status snapshot used by the setup page
# ---------------------------------------------------------------------------

def snapshot() -> Dict[str, object]:
    binary = ollama_binary()
    running = ollama_is_up()
    installed_names = set(installed_model_names()) if running else set()
    wanted = required_models()
    missing = [m for m in wanted if m not in installed_names]
    all_ready = running and not missing

    # Catalog enriched with per-model install state, so one fetch powers the UI.
    catalog_out: List[Dict[str, object]] = []
    catalog_ids = {item["id"] for item in CATALOG}
    for item in CATALOG:
        mid = item["id"]  # type: ignore[index]
        catalog_out.append({
            **item,
            "installed": mid in installed_names,
            "required": mid in wanted,
        })
    # Include any model the user already pulled that we don't know about.
    for name in sorted(installed_names):
        if name not in catalog_ids:
            catalog_out.append({
                "id": name,
                "name": name,
                "family": "other",
                "family_label": "Other",
                "size_gb": 0,
                "role": "custom",
                "tags": ["custom"],
                "blurb": "Manually installed model.",
                "installed": True,
                "required": name in wanted,
            })

    return {
        "ollama": {
            "installed": bool(binary),
            "binary": binary,
            "running": running,
            "base_url": OLLAMA_BASE,
            "download_url": ollama_download_url(),
            "platform": platform.system().lower(),
        },
        "models": {
            "required": wanted,
            "installed": sorted(installed_names),
            "missing": missing,
            "catalog": catalog_out,
            "defaults": dict(DEFAULT_MODELS),
        },
        "ready": bool(all_ready),
    }


def human_gb(n: int) -> str:
    if not isinstance(n, (int, float)) or n <= 0:
        return ""
    v = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if v < 1024 or unit == "TB":
            return f"{v:.1f} {unit}"
        v /= 1024
    return f"{v:.1f} TB"
