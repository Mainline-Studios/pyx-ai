"""Server-side trainer/staff gate: Firebase ID-token verify + signed session cookie.

Protects pages that must NOT be publicly downloadable (the Firebase trainer and
the Dev Workshop). The browser logs in with Firebase email-link sign-in on
/pyx-trainer-auth.html, then POSTs its ID token to /api/session/login. We verify
the token, check the email allowlist, and set an HttpOnly signed cookie.
Protected routes require that cookie before returning any HTML.

The session cookie is a compact HMAC-signed token (no external dependency). Set
PYX_SESSION_SECRET in the environment for a stable, private signing key across
instances/restarts; a project-derived fallback keeps the gate functional if it
is unset.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Optional

# Accounts allowed into trainer/staff tools.
ALLOWED_EMAILS = {
    "boehmlaird@gmail.com",
    "bdawgsaweaome@icloud.com",
}

SESSION_COOKIE = "pyx_session"
SESSION_TTL_SECONDS = int(os.environ.get("PYX_SESSION_TTL", "43200"))  # 12 hours
_FIREBASE_PROJECT_ID = os.environ.get(
    "FIREBASE_PROJECT_ID", os.environ.get("GOOGLE_CLOUD_PROJECT", "pyx-ai")
)


def is_allowed(email: Optional[str]) -> bool:
    return bool(email) and email.strip().lower() in ALLOWED_EMAILS


def _secret() -> bytes:
    s = (os.environ.get("PYX_SESSION_SECRET") or "").strip()
    if not s:
        # Functional fallback; set PYX_SESSION_SECRET in production for a stable,
        # private key shared across Cloud Run instances.
        s = "pyx-trainer-session::" + _FIREBASE_PROJECT_ID + "::fallback"
    return s.encode("utf-8")


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64d(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + pad)


def _sign(body: str) -> str:
    return _b64e(hmac.new(_secret(), body.encode("ascii"), hashlib.sha256).digest())


def make_session(email: str) -> str:
    payload = {
        "email": email.strip().lower(),
        "exp": int(time.time()) + SESSION_TTL_SECONDS,
    }
    body = _b64e(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    return body + "." + _sign(body)


def read_session(token: Optional[str]) -> Optional[str]:
    """Return the allowed email from a valid, unexpired session token, else None."""
    if not token or "." not in token:
        return None
    body, _, sig = token.partition(".")
    if not hmac.compare_digest(sig, _sign(body)):
        return None
    try:
        payload = json.loads(_b64d(body))
    except Exception:
        return None
    if int(payload.get("exp", 0)) < int(time.time()):
        return None
    email = payload.get("email")
    return email if is_allowed(email) else None


def verify_firebase_id_token(id_token: str) -> Optional[str]:
    """Verify a Firebase ID token; return its email if valid, else None."""
    if not id_token:
        return None
    try:
        import firebase_admin
        from firebase_admin import auth as fb_auth, credentials
    except ImportError:
        return None
    try:
        if not firebase_admin._apps:
            try:
                cred = credentials.ApplicationDefault()
                firebase_admin.initialize_app(cred, {"projectId": _FIREBASE_PROJECT_ID})
            except Exception:
                firebase_admin.initialize_app(options={"projectId": _FIREBASE_PROJECT_ID})
        decoded = fb_auth.verify_id_token(id_token)
        return decoded.get("email")
    except Exception:
        return None


def cookie_email(request) -> Optional[str]:
    """Allowed email for the current request's session cookie, or None."""
    return read_session(request.cookies.get(SESSION_COOKIE))
