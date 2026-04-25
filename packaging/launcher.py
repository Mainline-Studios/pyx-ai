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
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


def _base_dir() -> Path:
    """Repo root during dev; bundle root inside PyInstaller onefile/onedir."""
    if getattr(sys, "frozen", False):
        # PyInstaller sets _MEIPASS to the temp unpack dir (onefile) or the bundle dir.
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent


def _prep_env(base: Path) -> None:
    """Default to local Ollama with a laptop-friendly Pyx 1.5 model mix."""
    os.environ.setdefault(
        "PYX_TALK_LLM_URL", "http://127.0.0.1:11434/v1/chat/completions"
    )
    # Smaller defaults than the cloud mix so the first-run download is a few GB,
    # not tens of GB. Users can bump these later with env vars (see docs).
    os.environ.setdefault("PYX_TALK_MODEL_FAST", "llama3.2:3b-instruct")
    os.environ.setdefault("PYX_TALK_MODEL_SMART", "llama3.1:8b-instruct")
    os.environ.setdefault("PYX_TALK_MODEL_THINKING", "llama3.1:8b-instruct")
    os.environ.setdefault("PYX_CODE_MODEL", "gpt-oss:20b")
    os.environ.setdefault("PYX_PIXEL_MODEL", "gpt-oss:20b")
    os.environ.setdefault("PYX_TALK_TIMEOUT", "600")
    os.environ.pop("PYX_TALK_LLM_KEY", None)  # local mode — no Groq key

    # Make sure imports find the bundled modules.
    sys.path.insert(0, str(base))


def _wait_for_health(port: int, timeout: float = 20.0) -> bool:
    """Return True once ``GET /health`` returns HTTP 200 (server is accepting work)."""
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=0.75) as resp:
                if getattr(resp, "status", 200) == 200:
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            time.sleep(0.12)
    return False


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


def _register_bootstrap(app) -> None:
    """First-run setup routes: detect Ollama, auto-start, pull missing models with progress."""
    import json
    from flask import Response, jsonify, request, stream_with_context

    try:
        from packaging import bootstrap  # type: ignore
    except Exception:
        import bootstrap  # type: ignore  # works when launcher/ dir is flattened by PyInstaller

    @app.route("/bootstrap/status", methods=["GET"])
    def _bs_status():
        return jsonify(bootstrap.snapshot())

    @app.route("/bootstrap/start", methods=["POST"])
    def _bs_start():
        started = False
        if not bootstrap.ollama_is_up():
            started = bootstrap.ollama_serve_background()
        return jsonify({"started": started, "running": bootstrap.ollama_is_up()})

    @app.route("/bootstrap/pull", methods=["GET"])
    def _bs_pull():
        model = (request.args.get("model") or "").strip()
        if not model:
            return jsonify({"error": "missing ?model"}), 400
        if not bootstrap.ollama_is_up():
            bootstrap.ollama_serve_background()

        def sse(obj):
            return "data: " + json.dumps(obj) + "\n\n"

        def gen():
            ok = False
            try:
                for evt in bootstrap.pull_model(model):
                    yield sse(evt)
                    if evt.get("status") == "success":
                        ok = True
                    if evt.get("status") == "error":
                        ok = False
            except GeneratorExit:
                return
            except Exception as e:  # pragma: no cover
                yield sse({"status": "error", "detail": str(e)})
            yield sse({"done": True, "ok": ok, "model": model})

        return Response(
            stream_with_context(gen()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )


def main() -> int:
    base = _base_dir()
    _prep_env(base)

    try:
        import app as pyx_app  # noqa: E402
    except Exception as e:  # pragma: no cover
        print(f"[pyx] failed to import Flask app: {e}", file=sys.stderr)
        return 2

    _pub_override = os.environ.get("PYX_PUBLIC_DIR", "").strip()
    public_dir = Path(_pub_override).expanduser() if _pub_override else (base / "public")
    if public_dir.is_dir():
        _register_static(pyx_app.app, public_dir)
    else:
        print(
            f"[pyx] error: public/ not found at {public_dir} — cannot serve the local UI.\n"
            f"  If you moved files, set PYX_PUBLIC_DIR to the folder that contains pyx-talk.html.",
            file=sys.stderr,
        )
        return 3

    _register_bootstrap(pyx_app.app)

    # Decide first-load URL: if Ollama isn't running or any required model is
    # missing, go to /pyx-setup.html so the user can download in-browser.
    try:
        from packaging import bootstrap  # type: ignore
    except Exception:
        import bootstrap  # type: ignore
    # Best-effort cold start so the setup page can show "Running" instantly.
    if not bootstrap.ollama_is_up():
        bootstrap.ollama_serve_background()

    snap = bootstrap.snapshot()
    force_setup = os.environ.get("PYX_FORCE_SETUP", "").strip() in ("1", "true", "yes")
    first_page = "pyx-setup.html" if (force_setup or not snap.get("ready")) else "pyx-talk.html"

    port = _pick_port(int(os.environ.get("PORT", "8765")))
    url = f"http://127.0.0.1:{port}/{first_page}"

    def _open_when_ready() -> None:
        if _wait_for_health(port):
            webbrowser.open_new(url)
        else:
            print(
                f"[pyx] server did not become ready in time.\n"
                f"  Open manually: {url}",
                file=sys.stderr,
            )

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
