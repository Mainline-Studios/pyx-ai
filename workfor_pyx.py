"""Run public/*.php on Cloud Run via php-cli bridge (shared data/workforpyx storage)."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from flask import Request, Response, jsonify, redirect, request, send_from_directory
from urllib.parse import quote
from werkzeug.utils import secure_filename

from workforpyx_mail import send_application_decision
from workforpyx_storage import DATA_DIR, find_application
from workforpyx_traffic import (
    IMAGE_CACHE_DIR,
    add_training_sample,
    analyze_features,
    analyze_image,
    create_captcha_challenge,
    delete_sample,
    list_samples,
    publish_images_for_training,
    record_emit,
    sample_stats,
    submit_captcha,
    traffic_capabilities,
)

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / "public"
BRIDGE = PUBLIC / "_php_bridge.php"
RESUME_DIR = DATA_DIR / "resumes"

_RESUME_MIME = {
    "pdf": "application/pdf",
    "doc": "application/msword",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "txt": "text/plain; charset=utf-8",
    "rtf": "application/rtf",
}


def _serve_resume_download(app_id: str) -> Response:
    app_id = (app_id or "").strip()
    if not app_id:
        return Response("Missing application id.", status=400)
    record = find_application(app_id)
    if not record or not record.get("resume_stored"):
        return Response("Resume not found.", status=404)
    stored = Path(record["resume_stored"]).name
    path = RESUME_DIR / stored
    if not path.is_file():
        return Response("Resume file missing on server.", status=404)
    ext = path.suffix.lower().lstrip(".")
    mime = _RESUME_MIME.get(ext, "application/octet-stream")
    download_name = secure_filename(
        record.get("resume_original") or stored or "resume.pdf"
    )
    if not download_name:
        download_name = stored
    return send_from_directory(
        RESUME_DIR,
        stored,
        mimetype=mime,
        as_attachment=True,
        download_name=download_name,
    )


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
                if os.path.isfile(p):
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
        if request.method == "GET" and request.args.get("action") == "resume":
            return _serve_resume_download((request.args.get("id") or "").strip())
        if request.method == "POST" and request.form.get("action") == "decision":
            app_id = (request.form.get("id") or "").strip()
            status = (request.form.get("status") or "").strip()
            note = (request.form.get("decision_note") or "").strip()
            result = send_application_decision(app_id, status, note)
            if result.get("ok"):
                msg = f"ok:Email sent to {result.get('email')} ({status})."
            else:
                msg = "err:" + (result.get("error") or "Could not send email.")
            target = f"/dev-workshop.php?id={quote(app_id)}&flash={quote(msg)}"
            return redirect(target)
        return _run_php("dev-workshop.php", request)

    @app.route("/workforpyx_lib.php", methods=["GET"])
    def block_workforpyx_lib():
        return Response("Not found", status=404)

    def _traffic_json(handler):
        if request.method == "OPTIONS":
            return "", 204
        try:
            out = handler()
            if isinstance(out, tuple):
                return out
            return out
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/dev-workshop/traffic/samples", methods=["GET", "OPTIONS"])
    def traffic_list_samples():
        def run():
            samples = list_samples()
            return jsonify(
                {
                    "ok": True,
                    "samples": samples,
                    "stats": sample_stats(samples),
                }
            )

        return _traffic_json(run)

    @app.route("/api/dev-workshop/traffic/search-images", methods=["POST", "OPTIONS"])
    def traffic_search_images():
        def run():
            data = request.get_json(silent=True) or {}
            query = str(data.get("query") or "").strip()
            try:
                max_n = int(data.get("max") or 50)
            except (TypeError, ValueError):
                max_n = 50
            max_n = max(1, min(max_n, 50))
            images, err = publish_images_for_training(query, max_results=max_n)
            if err:
                return jsonify({"ok": False, "error": err}), 400
            return jsonify({"ok": True, "query": query, "images": images, "count": len(images)})

        return _traffic_json(run)

    @app.route("/api/dev-workshop/traffic/img/<path:filename>", methods=["GET", "OPTIONS"])
    def traffic_public_image(filename: str):
        if request.method == "OPTIONS":
            return "", 204
        safe = secure_filename(filename)
        if not safe or safe != filename.replace("\\", "/").split("/")[-1]:
            return Response("Not found", status=404)
        path = IMAGE_CACHE_DIR / safe
        if not path.is_file():
            return Response("Not found", status=404)
        return send_from_directory(IMAGE_CACHE_DIR, safe, max_age=86400)

    @app.route(
        "/api/dev-workshop/traffic/samples/<sample_id>",
        methods=["DELETE", "OPTIONS"],
    )
    def traffic_delete_sample(sample_id: str):
        def run():
            ok = delete_sample(sample_id)
            return jsonify({"ok": ok}), (200 if ok else 404)

        return _traffic_json(run)

    @app.route("/api/dev-workshop/traffic/train", methods=["POST", "OPTIONS"])
    def traffic_train():
        def run():
            data = request.get_json(silent=True) or {}
            sample = add_training_sample(
                str(data.get("image_url") or ""),
                str(data.get("color") or ""),
                data.get("features") or [],
                dev=str(data.get("dev") or "dev"),
            )
            return jsonify({"ok": True, "sample": sample})

        return _traffic_json(run)

    @app.route("/api/dev-workshop/traffic/capabilities", methods=["GET", "OPTIONS"])
    def traffic_capabilities_route():
        def run():
            return jsonify({"ok": True, **traffic_capabilities()})

        return _traffic_json(run)

    def _traffic_analyze_body(data: dict):
        mode = str(data.get("mode") or "image")
        source = data.get("source")
        frame_id = data.get("frame_id")
        features = data.get("features")
        if features is not None:
            return analyze_features(
                features,
                mode=mode,
                source=str(source) if source else None,
                frame_id=str(frame_id) if frame_id else None,
                image_url=data.get("image_url"),
            )
        return analyze_image(
            image_url=data.get("image_url"),
            features=None,
            mode=mode,
            source=str(source) if source else None,
            frame_id=str(frame_id) if frame_id else None,
        )

    @app.route("/api/dev-workshop/traffic/analyze", methods=["POST", "OPTIONS"])
    def traffic_analyze():
        def run():
            data = request.get_json(silent=True) or {}
            out = _traffic_analyze_body(data)
            status = 200 if out.get("ok") else 400
            return jsonify(out), status

        return _traffic_json(run)

    @app.route("/api/dev-workshop/traffic/frame", methods=["POST", "OPTIONS"])
    def traffic_analyze_frame():
        """Live video: one frame’s feature vector (same classifier as still images)."""

        def run():
            data = request.get_json(silent=True) or {}
            if not data.get("mode"):
                data = {**data, "mode": "frame"}
            out = _traffic_analyze_body(data)
            status = 200 if out.get("ok") else 400
            return jsonify(out), status

        return _traffic_json(run)

    @app.route("/api/dev-workshop/traffic/emit", methods=["POST", "OPTIONS"])
    def traffic_emit():
        def run():
            data = request.get_json(silent=True) or {}
            color = str(data.get("color") or "unknown")
            hex_val = str(data.get("hex") or "")
            event = record_emit(
                color,
                hex_val,
                source=str(data.get("source") or "workshop"),
                mode=data.get("mode"),
                frame_id=data.get("frame_id"),
            )
            return jsonify({"ok": True, "event": event, "color": color, "hex": hex_val})

        return _traffic_json(run)

    @app.route("/api/dev-workshop/traffic/captcha/challenge", methods=["GET", "OPTIONS"])
    def traffic_captcha_challenge():
        def run():
            hint = request.args.get("hint")
            challenge = create_captcha_challenge(hint=hint)
            return jsonify({"ok": True, **challenge})

        return _traffic_json(run)

    @app.route("/api/dev-workshop/traffic/captcha/submit", methods=["POST", "OPTIONS"])
    def traffic_captcha_submit():
        def run():
            data = request.get_json(silent=True) or {}
            out = submit_captcha(
                str(data.get("challenge_id") or ""),
                str(data.get("color") or ""),
                data.get("features"),
            )
            status = 200 if out.get("ok") else 400
            return jsonify(out), status

        return _traffic_json(run)
