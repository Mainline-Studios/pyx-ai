"""Desktop GGUF path: Llama + GPT-OSS files on disk, optional HTTP download, llama-server.

Pyx 1.5 defaults to this engine unless ``PYX_USE_OLLAMA=1`` (legacy Ollama flow).

- Model files live under ``models_dir()`` (override with ``PYX_MODELS_DIR``).
- Filenames come from ``gguf_manifest.json`` next to this module (edit ``download_url``
  when you host weights; users can also copy ``.gguf`` files into the folder).
- Inference uses the ``llama-server`` binary from ``PYX_LLAMA_SERVER`` or ``PATH``.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterator, List, Optional


def use_ollama_engine() -> bool:
    return os.environ.get("PYX_USE_OLLAMA", "").strip().lower() in ("1", "true", "yes", "on")


def manifest_path() -> Path:
    override = os.environ.get("PYX_GGUF_MANIFEST", "").strip()
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parent / "gguf_manifest.json"


def load_manifest() -> Dict[str, Any]:
    p = manifest_path()
    if not p.is_file():
        return {"talk": {"filename": "pyx-llama.gguf", "label": "Llama", "download_url": ""}, "code": {"filename": "pyx-gpt-oss.gguf", "label": "GPT-OSS", "download_url": ""}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def models_dir() -> Path:
    d = os.environ.get("PYX_MODELS_DIR", "").strip()
    if d:
        return Path(d).expanduser()
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Application Support" / "Pyx" / "models"
    if platform.system() == "Windows":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "Pyx" / "models"
    return Path.home() / ".local" / "share" / "pyx" / "models"


def ensure_models_dir() -> Path:
    d = models_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def slot_path(slot: str) -> Optional[Path]:
    m = load_manifest()
    entry = m.get(slot)
    if not isinstance(entry, dict):
        return None
    fn = (entry.get("filename") or "").strip()
    if not fn:
        return None
    return ensure_models_dir() / fn


def find_llama_server_bin() -> Optional[str]:
    for key in ("PYX_LLAMA_SERVER", "LLAMA_SERVER"):
        v = os.environ.get(key, "").strip()
        if v and Path(v).expanduser().is_file():
            return str(Path(v).expanduser())
    w = shutil.which("llama-server")
    return w


def _llama_http_ok(port: int) -> bool:
    for path in ("/v1/models", "/health"):
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method="GET")
            with urllib.request.urlopen(req, timeout=0.6) as r:
                if getattr(r, "status", 200) in (200, 404):  # some builds 404 /health
                    return True
        except Exception:
            continue
    return False


def _start_llama_process(binpath: str, gguf: Path, port: int) -> bool:
    if not Path(gguf).is_file():
        return False
    args = [
        binpath,
        "-m",
        str(gguf),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    ngl = os.environ.get("PYX_LLAMA_NGL", "").strip()
    if ngl.isdigit():
        args.extend(["-ngl", ngl])
    ctx = os.environ.get("PYX_LLAMA_CTX", "8192").strip()
    if ctx.isdigit():
        args.extend(["-c", ctx])
    try:
        kwargs: Dict[str, Any] = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "stdin": subprocess.DEVNULL,
        }
        if platform.system() == "Windows":
            kwargs["creationflags"] = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
        else:
            kwargs["start_new_session"] = True
        subprocess.Popen(args, **kwargs)
    except Exception:
        return False
    deadline = time.time() + 30.0
    while time.time() < deadline:
        if _llama_http_ok(port):
            return True
        time.sleep(0.35)
    return False


_talk_port = int(os.environ.get("PYX_LLAMA_TALK_PORT", "11441"))
_code_port = int(os.environ.get("PYX_LLAMA_CODE_PORT", "11442"))


def ensure_llama_servers(_base: Path) -> None:
    """Start one or two ``llama-server`` processes when GGUF files and binary exist."""
    if use_ollama_engine():
        return
    binpath = find_llama_server_bin()
    if not binpath:
        return
    talk_p = slot_path("talk")
    code_p = slot_path("code")
    if not talk_p or not talk_p.is_file():
        return
    single = not code_p or not code_p.is_file() or code_p.resolve() == talk_p.resolve()
    if single:
        os.environ["PYX_LLAMA_SINGLE"] = "1"
        if _llama_http_ok(_talk_port):
            return
        _start_llama_process(binpath, talk_p, _talk_port)
        return
    os.environ.pop("PYX_LLAMA_SINGLE", None)
    if _llama_http_ok(_talk_port) and _llama_http_ok(_code_port):
        return
    _start_llama_process(binpath, talk_p, _talk_port)
    _start_llama_process(binpath, code_p, _code_port)


def apply_default_llm_env() -> None:
    """Point Pyx at local OpenAI-compatible endpoints (llama-server)."""
    if use_ollama_engine():
        return
    talk_url = f"http://127.0.0.1:{_talk_port}/v1/chat/completions"
    code_url = f"http://127.0.0.1:{_code_port}/v1/chat/completions"
    if os.environ.get("PYX_LLAMA_SINGLE") == "1":
        code_url = talk_url
    os.environ.setdefault("PYX_TALK_LLM_URL", talk_url)
    os.environ.setdefault("PYX_CODE_LLM_URL", code_url)
    os.environ.setdefault("PYX_PIXEL_LLM_URL", code_url)
    mid = (os.environ.get("PYX_LOCAL_MODEL_ID") or "gpt-3.5-turbo").strip()
    os.environ.setdefault("PYX_TALK_MODEL_FAST", mid)
    os.environ.setdefault("PYX_TALK_MODEL_SMART", mid)
    os.environ.setdefault("PYX_TALK_MODEL_THINKING", mid)
    os.environ.setdefault("PYX_CODE_MODEL", mid)
    os.environ.setdefault("PYX_PIXEL_MODEL", mid)


def slot_status() -> List[Dict[str, Any]]:
    m = load_manifest()
    out: List[Dict[str, Any]] = []
    for key in ("talk", "code"):
        entry = m.get(key)
        if not isinstance(entry, dict):
            continue
        fn = (entry.get("filename") or "").strip()
        path = ensure_models_dir() / fn if fn else None
        exists = bool(path and path.is_file())
        size = path.stat().st_size if exists and path else 0
        out.append(
            {
                "slot": key,
                "label": entry.get("label") or key,
                "filename": fn,
                "path": str(path) if path else "",
                "exists": exists,
                "bytes": size,
                "download_url": (entry.get("download_url") or "").strip(),
            }
        )
    return out


def gguf_ready() -> bool:
    if use_ollama_engine():
        return False
    if not find_llama_server_bin():
        return False
    talk_p = slot_path("talk")
    if not talk_p or not talk_p.is_file():
        return False
    code_p = slot_path("code")
    single = not code_p or not code_p.is_file() or code_p.resolve() == talk_p.resolve()
    if not single and (not code_p or not code_p.is_file()):
        return False
    if single:
        return _llama_http_ok(_talk_port)
    return _llama_http_ok(_talk_port) and _llama_http_ok(_code_port)


def gguf_snapshot(_base: Path) -> Dict[str, Any]:
    binp = find_llama_server_bin()
    d = ensure_models_dir()
    slots = slot_status()
    single = os.environ.get("PYX_LLAMA_SINGLE") == "1"
    talk_up = _llama_http_ok(_talk_port)
    servers = {
        "talk_port": _talk_port,
        "code_port": _code_port,
        "talk_up": talk_up,
        "code_up": talk_up if single else _llama_http_ok(_code_port),
        "binary": binp,
        "single_model": single,
    }
    return {
        "models_dir": str(d),
        "manifest": str(manifest_path()),
        "slots": slots,
        "servers": servers,
        "ready": gguf_ready(),
    }


def download_slot(slot: str, url: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    """Stream progress events compatible with the setup page SSE handler."""
    m = load_manifest()
    entry = m.get(slot)
    if not isinstance(entry, dict):
        yield {"status": "error", "detail": "unknown slot"}
        return
    fn = (entry.get("filename") or "").strip()
    dl = (url or (entry.get("download_url") or "").strip()).strip()
    if not fn:
        yield {"status": "error", "detail": "missing filename in manifest"}
        return
    if not dl:
        yield {"status": "error", "detail": "no download_url — copy the .gguf into " + str(models_dir())}
        return
    dest = ensure_models_dir() / fn
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(dl, method="GET", headers={"User-Agent": "Pyx-1.5-desktop"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            block = 1024 * 256
            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(block)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    yield {"status": "downloading", "total": total, "completed": done, "slot": slot}
        tmp.replace(dest)
        yield {"status": "success", "path": str(dest)}
    except urllib.error.HTTPError as e:
        yield {"status": "error", "detail": f"HTTP {e.code}", "slot": slot}
    except Exception as e:
        yield {"status": "error", "detail": str(e), "slot": slot}
        try:
            if tmp.is_file():
                tmp.unlink()
        except Exception:
            pass
