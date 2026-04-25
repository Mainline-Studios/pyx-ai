"""Pyx 1.5 desktop launcher.

Runs the same Flask app that ships on Cloud Run, serves the static ``public/``
UI on the same origin, opens the default browser, and stays in the foreground
so Ctrl-C / closing the console window stops the app.

Used as the PyInstaller entry point — everything inside ``pyx-ai/`` can reach
this file via a sibling import. The bundled binary ships Python + deps, so the
user only needs to install Ollama (or any OpenAI-compatible local server) and
pull weights. See PYX_1_5_LOCAL.md for links.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path


def _base_dir() -> Path:
    """Repo root during dev; bundle root inside PyInstaller onefile/onedir."""
    if getattr(sys, "frozen", False):
        # PyInstaller sets _MEIPASS to the temp unpack dir (onefile) or the bundle dir.
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent


def _prep_env(base: Path) -> None:
    """Default to local Ollama with a sensible Pyx 1.5 model mix if caller didn't set envs."""
    os.environ.setdefault(
        "PYX_TALK_LLM_URL", "http://127.0.0.1:11434/v1/chat/completions"
    )
    os.environ.setdefault("PYX_TALK_MODEL_FAST", "llama3.1:8b-instruct")
    os.environ.setdefault("PYX_TALK_MODEL_SMART", "llama3.3:70b-instruct")
    os.environ.setdefault("PYX_TALK_MODEL_THINKING", "llama3.3:70b-instruct")
    os.environ.setdefault("PYX_CODE_MODEL", "gpt-oss:20b")
    os.environ.setdefault("PYX_PIXEL_MODEL", "gpt-oss:20b")
    os.environ.setdefault("PYX_TALK_TIMEOUT", "600")
    os.environ.pop("PYX_TALK_LLM_KEY", None)  # local mode — no Groq key

    # Make sure imports find the bundled modules.
    sys.path.insert(0, str(base))


def _pick_port(preferred: int = 8765) -> int:
    s = socket.socket()
    try:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
    finally:
        s.close()


def _register_static(app, public_dir: Path) -> None:
    """Serve the shipped ``public/`` folder on the same origin."""
    from flask import send_from_directory

    public_dir = public_dir.resolve()

    @app.route("/pyx.html")
    @app.route("/app")
    def _home():
        return send_from_directory(str(public_dir), "pyx-talk.html")

    @app.route("/static-assets/<path:fname>")
    def _named_static(fname):
        return send_from_directory(str(public_dir), fname)

    # Catch-all: only serve files that exist; otherwise let Flask 404 so the
    # real API routes win on conflicts.
    @app.route("/<path:fname>")
    def _public_passthrough(fname):
        target = public_dir / fname
        if target.is_file():
            return send_from_directory(str(public_dir), fname)
        return ("", 404)


def main() -> int:
    base = _base_dir()
    _prep_env(base)

    try:
        import app as pyx_app  # noqa: E402
    except Exception as e:  # pragma: no cover
        print(f"[pyx] failed to import Flask app: {e}", file=sys.stderr)
        return 2

    public_dir = base / "public"
    if public_dir.is_dir():
        _register_static(pyx_app.app, public_dir)
    else:
        print(f"[pyx] warning: public/ not found at {public_dir} — UI routes disabled")

    port = _pick_port(int(os.environ.get("PORT", "8765")))
    url = f"http://127.0.0.1:{port}/pyx-talk.html"

    def _open_when_ready() -> None:
        deadline = time.time() + 8.0
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                    break
            except OSError:
                time.sleep(0.15)
        webbrowser.open(url)

    threading.Thread(target=_open_when_ready, daemon=True).start()

    print("Pyx 1.5 desktop")
    print(f"  URL   : {url}")
    print(f"  Bundle: {base}")
    print("  Stop  : Ctrl+C / close this window")

    try:
        from werkzeug.serving import make_server

        srv = make_server("127.0.0.1", port, pyx_app.app, threaded=True)
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[pyx] shutting down")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
