"""Pyx 1.5 desktop launcher.

Runs the same Flask app that ships on Cloud Run, serves the static ``public/``
UI on the same origin, and opens it in a **native app window** (pywebview —
WKWebView on macOS, WebView2 on Windows), not the default browser.

Set ``PYX_USE_BROWSER=1`` to restore the old browser-only behavior (e.g. remote
debugging). The local server runs in a background thread while the GUI owns
the main thread.

Used as the PyInstaller entry point — see PYX_1_5_LOCAL.md. Default desktop
engine is **GGUF + llama-server** (Llama / GPT-OSS files). Set ``PYX_USE_OLLAMA=1``
for the legacy Ollama flow.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def _base_dir() -> Path:
    """Repo root during dev; bundle root inside PyInstaller onefile/onedir."""
    if getattr(sys, "frozen", False):
        # PyInstaller sets _MEIPASS to the temp unpack dir (onefile) or the bundle dir.
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent


def _prep_env(base: Path) -> None:
    """Shared env: path, timeouts, no cloud key."""
    sys.path.insert(0, str(base))
    os.environ.setdefault("PYX_TALK_TIMEOUT", "600")
    os.environ.pop("PYX_TALK_LLM_KEY", None)


def _prep_ollama_defaults() -> None:
    """Legacy engine: local Ollama OpenAI-compat + registry model names."""
    os.environ.setdefault(
        "PYX_TALK_LLM_URL", "http://127.0.0.1:11434/v1/chat/completions"
    )
    os.environ.setdefault("PYX_TALK_MODEL_FAST", "llama2:7b")
    os.environ.setdefault("PYX_TALK_MODEL_SMART", "llama4:scout")
    os.environ.setdefault("PYX_TALK_MODEL_THINKING", "llama4:scout")
    os.environ.setdefault("PYX_CODE_MODEL", "gpt-oss:20b")
    os.environ.setdefault("PYX_PIXEL_MODEL", "gpt-oss:20b")


def _wait_for_health(port: int, timeout: float = 25.0) -> bool:
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
        return send_from_directory(str(public_dir), "pyx-launcher.html")

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
        try:
            from packaging import gguf_engine  # type: ignore
        except Exception:
            import gguf_engine  # type: ignore
        if gguf_engine.use_ollama_engine():
            started = False
            if not bootstrap.ollama_is_up():
                started = bootstrap.ollama_serve_background()
            return jsonify({"started": started, "running": bootstrap.ollama_is_up(), "engine": "ollama"})
        gguf_engine.ensure_llama_servers(_base_dir())
        gguf_engine.apply_default_llm_env()
        return jsonify(
            {
                "started": True,
                "running": gguf_engine.gguf_ready(),
                "engine": "gguf",
            }
        )

    @app.route("/bootstrap/gguf-pull", methods=["GET"])
    def _bs_gguf_pull():
        try:
            from packaging import gguf_engine  # type: ignore
        except Exception:
            import gguf_engine  # type: ignore
        slot = (request.args.get("slot") or "").strip()
        if slot not in ("talk", "code"):
            return jsonify({"error": "slot must be talk or code"}), 400
        url = (request.args.get("url") or "").strip() or None

        def sse(obj):
            return "data: " + json.dumps(obj) + "\n\n"

        def gen():
            ok = False
            try:
                for evt in gguf_engine.download_slot(slot, url):
                    yield sse(evt)
                    if evt.get("status") == "success":
                        ok = True
                    if evt.get("status") == "error":
                        ok = False
            except GeneratorExit:
                return
            except Exception as e:  # pragma: no cover
                yield sse({"status": "error", "detail": str(e)})
            yield sse({"done": True, "ok": ok, "slot": slot})

        return Response(
            stream_with_context(gen()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.route("/bootstrap/pull", methods=["GET"])
    def _bs_pull():
        try:
            from packaging import gguf_engine  # type: ignore
        except Exception:
            import gguf_engine  # type: ignore
        if not gguf_engine.use_ollama_engine():
            return jsonify({"error": "Ollama pull disabled — use /bootstrap/gguf-pull?slot=talk|code"}), 400
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


def _want_native_window() -> bool:
    """Native pywebview window unless user opts into a normal browser tab."""
    v = os.environ.get("PYX_USE_BROWSER", "").strip().lower()
    return v not in ("1", "true", "yes", "on")


def _app_icon_path(base: Path) -> str | None:
    """PNG for pywebview (macOS Cocoa); .app Dock icon comes from CFBundleIconFile."""
    for rel in ("public/brand/pyx-app-icon.png", "brand/pyx-app-icon.png"):
        p = base / rel
        if p.is_file():
            return str(p)
    return None


def _run_with_webview(url: str, port: int, base: Path) -> int:
    import webview

    import app as pyx_app

    srv_holder: list = []

    def _serve():
        from werkzeug.serving import make_server

        s = make_server("127.0.0.1", port, pyx_app.app, threaded=True)
        srv_holder.append(s)
        s.serve_forever()

    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    # Wait until Werkzeug thread has bound and Flask answers /health.
    if not _wait_for_health(port):
        print(
            f"[pyx] server did not become ready in time.\n"
            f"  Open manually: {url}",
            file=sys.stderr,
        )
        return 4

    w = int(os.environ.get("PYX_WINDOW_WIDTH", "1280"))
    h = int(os.environ.get("PYX_WINDOW_HEIGHT", "840"))

    print("Pyx 1.5 desktop (native window)")
    print(f"  URL   : {url}")
    print(f"  Bundle: {base}")
    print("  Stop  : close the Pyx window or press Ctrl+C in this console")

    icon_path = _app_icon_path(base)
    webview.create_window(
        "PYX.",
        url,
        width=w,
        height=h,
        min_size=(720, 480),
        text_select=True,
    )
    try:
        if icon_path:
            webview.start(icon=icon_path)
        else:
            webview.start()
    except KeyboardInterrupt:
        print("\n[pyx] shutting down")
    return 0


def _run_with_browser_tab(url: str, port: int) -> int:
    import webbrowser

    import app as pyx_app  # noqa: E402

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

    print("Pyx 1.5 desktop (system browser — PYX_USE_BROWSER=1)")
    print(f"  URL   : {url}")
    print("  Stop  : Ctrl+C / close this window")

    try:
        from werkzeug.serving import make_server

        srv = make_server("127.0.0.1", port, pyx_app.app, threaded=True)
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[pyx] shutting down")
        return 0
    return 0


def main() -> int:
    base = _base_dir()
    _prep_env(base)

    try:
        from packaging import gguf_engine  # type: ignore
    except Exception:
        import gguf_engine  # type: ignore

    if gguf_engine.use_ollama_engine():
        _prep_ollama_defaults()
    else:
        gguf_engine.ensure_models_dir()
        gguf_engine.ensure_llama_servers(base)
        gguf_engine.apply_default_llm_env()

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

    try:
        from packaging import bootstrap  # type: ignore
    except Exception:
        import bootstrap  # type: ignore
    if gguf_engine.use_ollama_engine():
        if not bootstrap.ollama_is_up():
            bootstrap.ollama_serve_background()

    snap = bootstrap.snapshot()
    force_setup = os.environ.get("PYX_FORCE_SETUP", "").strip() in ("1", "true", "yes")
    first_page = "pyx-setup.html" if (force_setup or not snap.get("ready")) else "pyx-launcher.html"

    port = _pick_port(int(os.environ.get("PORT", "8765")))
    url = f"http://127.0.0.1:{port}/{first_page}"

    if _want_native_window():
        try:
            return _run_with_webview(url, port, base)
        except ImportError:
            print(
                "[pyx] pywebview is not installed — falling back to system browser.\n"
                "  Install desktop deps: pip install -r packaging/requirements-desktop.txt",
                file=sys.stderr,
            )
        except Exception as e:  # pragma: no cover
            print(f"[pyx] native window failed ({e}); falling back to system browser.", file=sys.stderr)

    return _run_with_browser_tab(url, port)


if __name__ == "__main__":
    sys.exit(main())
