"""
Pyx HTTP service — score text over the network.

Run: python3 pyx_server.py [--port 8765] [--host 0.0.0.0]

  POST /score   Body: {"text": "..."}   → {"score": float, "bad": bool, "censored": "..."}
  GET  /health  → 200 OK
"""

import json
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler

from Pyx_ai_moderator import PyxAI, BAN_LINE, censor_letters


# One Pyx instance shared across requests (loads once at startup)
pyx = PyxAI()


class PyxHandler(BaseHTTPRequestHandler):
    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, message: str, status: int = 400):
        self._send_json({"error": message}, status=status)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/health"):
            self._send_json({"status": "ok", "service": "pyx"})
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path != "/score":
            self.send_response(404)
            self.end_headers()
            return
        content_length = self.headers.get("Content-Length")
        if not content_length:
            self._send_error_json("Missing Content-Length", 411)
            return
        try:
            length = int(content_length)
        except ValueError:
            self._send_error_json("Invalid Content-Length", 400)
            return
        if length > 1_000_000:  # 1 MB max
            self._send_error_json("Body too large", 413)
            return
        body = self.rfile.read(length)
        try:
            data = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self._send_error_json(f"Invalid JSON or encoding: {e}", 400)
            return
        text = data.get("text")
        if text is None:
            self._send_error_json('Missing "text" in body', 400)
            return
        if not isinstance(text, str):
            self._send_error_json('"text" must be a string', 400)
            return
        score = pyx.score(text)
        bad = score >= BAN_LINE
        censored = censor_letters(text) if bad else text
        self._send_json({
            "score": round(score, 4),
            "bad": bad,
            "censored": censored,
        })

    def log_message(self, format, *args):
        # Optional: use logging instead of print
        print(args[0] if args else "")


def main():
    parser = argparse.ArgumentParser(description="Pyx HTTP service")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="Port (default: 8765)")
    args = parser.parse_args()
    server = HTTPServer((args.host, args.port), PyxHandler)
    print(f"Pyx HTTP service at http://{args.host}:{args.port}")
    print("  POST /score  body: {\"text\": \"...\"}")
    print("  GET  /health")
    server.serve_forever()


if __name__ == "__main__":
    main()
