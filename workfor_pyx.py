"""Run public/*.php on Cloud Run via php-cli bridge (shared data/workforpyx storage)."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from flask import Request, Response, request
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / "public"
BRIDGE = PUBLIC / "_php_bridge.php"


def _run_php(script_name: str, req: Request) -> Response:
    script = PUBLIC / script_name
    if not script.is_file():
        return Response("Not found", status=404)

    env = os.environ.copy()
    env["PYX_PHP_BRIDGE"] = "1"
    env["REQUEST_METHOD"] = req.method
    env["QUERY_STRING"] = req.query_string.decode("latin-1", errors="replace")
    env["REQUEST_URI"] = req.path + (
        ("?" + env["QUERY_STRING"]) if env["QUERY_STRING"] else ""
    )

    temp_files: list[str] = []
    try:
        if req.method == "POST":
            if req.content_type and "multipart/form-data" in req.content_type:
                post_data = {}
                files_spec = {}
                for key in req.form:
                    post_data[key] = req.form.get(key)
                for key in req.files:
                    f = req.files[key]
                    if not f or not f.filename:
                        continue
                    suffix = Path(secure_filename(f.filename)).suffix
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                    f.save(tmp.name)
                    tmp.close()
                    temp_files.append(tmp.name)
                    files_spec[key] = {
                        "path": tmp.name,
                        "name": f.filename,
                        "type": f.content_type or "application/octet-stream",
                        "size": os.path.getsize(tmp.name),
                    }
                env["PYX_POST_JSON"] = json.dumps(post_data)
                env["PYX_FILES_JSON"] = json.dumps(files_spec)
            else:
                env["PYX_POST_JSON"] = json.dumps(req.form.to_dict(flat=True))
                env.pop("PYX_FILES_JSON", None)
        else:
            env.pop("PYX_POST_JSON", None)
            env.pop("PYX_FILES_JSON", None)

        cmd = [
            "php",
            "-d",
            f"auto_prepend_file={BRIDGE}",
            str(script),
        ]
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            timeout=60,
            cwd=str(ROOT),
        )
        body = proc.stdout.decode("utf-8", errors="replace")
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace")[:800]
            return Response(
                f"<pre>PHP error\n{err}</pre>",
                status=500,
                mimetype="text/html",
            )
        return Response(body, mimetype="text/html; charset=utf-8")
    finally:
        for p in temp_files:
            try:
                os.unlink(p)
            except OSError:
                pass


def register_workforpyx_routes(app) -> None:
    @app.route("/workforpyx.php", methods=["GET", "POST", "OPTIONS"])
    def workforpyx_php():
        if request.method == "OPTIONS":
            return "", 204
        return _run_php("workforpyx.php", request)

    @app.route("/dev-workshop.php", methods=["GET", "POST", "OPTIONS"])
    def dev_workshop_php():
        if request.method == "OPTIONS":
            return "", 204
        return _run_php("dev-workshop.php", request)

    @app.route("/workforpyx_lib.php", methods=["GET"])
    def block_workforpyx_lib():
        return Response("Not found", status=404)
