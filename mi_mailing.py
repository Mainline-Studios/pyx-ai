"""Mainline Intelligence mailing list — subscribe + welcome email."""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from workforpyx_mail import send_smtp, _env, _load_env_file, _escape_html

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "mi_mailing"
SUBSCRIBERS_PATH = DATA_DIR / "subscribers.json"
_LOCK = threading.Lock()

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def mailing_smtp_config() -> dict[str, Any]:
    """SMTP for MI list mail. Prefer MI_* vars; fall back to shared Pyx SMTP."""
    _load_env_file()
    from_email = _env(
        "MI_MAILING_FROM",
        "MI_MAILING_SMTP_FROM",
        default="no-reply@pixelplaceofficial.com",
    )
    user = _env(
        "MI_MAILING_SMTP_USER",
        "PYX_APPLICATION_SMTP_USER",
        "EMAIL_VERIFICATION_SMTP_USER",
        default=from_email,
    )
    password = _env(
        "MI_MAILING_SMTP_PASS",
        "PYX_APPLICATION_SMTP_PASS",
        "EMAIL_VERIFICATION_SMTP_PASS",
        "EMAIL_VERIFICATION_FROM_APP_PASSWORD",
    ).replace(" ", "")
    return {
        "from_email": from_email,
        "from_name": _env(
            "MI_MAILING_FROM_NAME",
            default="Mainline Intelligence",
        ),
        "reply_to": _env(
            "MI_MAILING_REPLY_TO",
            default="support@pixelplaceofficial.com",
        ),
        "user": user,
        "password": password,
        "host": _env(
            "MI_MAILING_SMTP_HOST",
            "PYX_APPLICATION_SMTP_HOST",
            "EMAIL_VERIFICATION_SMTP_HOST",
            default="smtp.gmail.com",
        ),
        "port": int(
            _env(
                "MI_MAILING_SMTP_PORT",
                "PYX_APPLICATION_SMTP_PORT",
                "EMAIL_VERIFICATION_SMTP_PORT",
                default="465",
            )
        ),
        "public_url": _env(
            "PYX_APP_PUBLIC_URL",
            "APP_PUBLIC_URL",
            default="https://pyx-ai.web.app",
        ).rstrip("/"),
        "notify_to": _env(
            "MI_MAILING_NOTIFY_TO",
            default="support@pixelplaceofficial.com",
        ),
    }


def normalize_email(raw: str) -> str:
    return (raw or "").strip().lower()


def valid_email(email: str) -> bool:
    return bool(email) and len(email) <= 320 and bool(_EMAIL_RE.match(email))


def _ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def _load_subscribers() -> list[dict[str, Any]]:
    _ensure_data_dir()
    if not SUBSCRIBERS_PATH.is_file():
        return []
    raw = SUBSCRIBERS_PATH.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    data = json.loads(raw)
    return data if isinstance(data, list) else []


def _save_subscribers(rows: list[dict[str, Any]]) -> None:
    _ensure_data_dir()
    SUBSCRIBERS_PATH.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _store_firestore(email: str, source: str, now: str) -> bool:
    try:
        from firestore_sync import init_firestore

        db = init_firestore()
        if not db:
            return False
        db.collection("mi_mailing_list").document(email).set(
            {
                "email": email,
                "source": source,
                "subscribed_at": now,
                "list": "mainline_intelligence",
            },
            merge=True,
        )
        return True
    except Exception:
        return False


def store_subscriber(email: str, source: str = "mi_site") -> dict[str, Any]:
    """Persist subscriber. Returns {ok, already, email}."""
    email = normalize_email(email)
    if not valid_email(email):
        return {"ok": False, "error": "need a real email address"}
    now = datetime.now(timezone.utc).isoformat()
    already = False
    with _LOCK:
        rows = _load_subscribers()
        for row in rows:
            if normalize_email(str(row.get("email") or "")) == email:
                already = True
                row["source"] = source or row.get("source") or "mi_site"
                row["updated_at"] = now
                break
        if not already:
            rows.append(
                {
                    "email": email,
                    "source": source or "mi_site",
                    "subscribed_at": now,
                }
            )
        _save_subscribers(rows)
    _store_firestore(email, source or "mi_site", now)
    return {"ok": True, "already": already, "email": email}


def _welcome_messages(email: str, cfg: dict[str, Any]) -> tuple[str, str, str]:
    site = f"{cfg['public_url']}/mainlineintelligence"
    subject = "you're on the mainline intelligence list"
    text = (
        "hey — you're on the mainline intelligence mailing list.\n\n"
        "we'll ping you about mi moderator, marii, mci, and the rest of the new wave.\n"
        "no ads. we pinky-swear. (we've already made ads our villain origin story.)\n\n"
        f"home base: {site}\n"
        "questions? reply lands at support@pixelplaceofficial.com\n"
    )
    safe_email = _escape_html(email)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>mainline intelligence</title></head>
<body style="margin:0;padding:0;background:#e8f7f4;font-family:system-ui,-apple-system,sans-serif;color:#1a4a52;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="padding:32px 12px;">
    <tr><td align="center">
      <table role="presentation" width="520" style="max-width:520px;width:100%;background:#fff;border-radius:16px;padding:28px;border:1px solid rgba(46,196,182,0.28);">
        <tr><td style="font-size:1.35rem;font-weight:700;letter-spacing:-0.03em;color:#0d7377;">
          mainline intelligence
        </td></tr>
        <tr><td style="padding-top:14px;line-height:1.55;color:#4a8a92;font-size:1rem;">
          hey {safe_email} — you're on the list.
        </td></tr>
        <tr><td style="padding-top:10px;line-height:1.55;color:#4a8a92;font-size:0.98rem;">
          we'll ping you about mi moderator, marii, mci, and the rest of the new wave of pyx.
        </td></tr>
        <tr><td style="padding-top:10px;line-height:1.55;color:#0d7377;font-size:0.95rem;font-weight:600;">
          no ads. we pinky-swear. (we've already made ads our villain origin story.)
        </td></tr>
        <tr><td style="padding-top:22px;">
          <a href="{_escape_html(site)}" style="color:#2ec4b6;text-decoration:none;border-bottom:1px solid rgba(46,196,182,0.5);">open mainline intelligence →</a>
        </td></tr>
        <tr><td style="padding-top:22px;font-size:0.8rem;color:#7aadb4;">
          sent from no-reply@pixelplaceofficial.com · replies go to support
        </td></tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""
    return subject, html, text


def _notify_staff(email: str, already: bool, cfg: dict[str, Any]) -> None:
    to = (cfg.get("notify_to") or "").strip()
    if not to or "@" not in to:
        return
    flag = "re-joined" if already else "new"
    subject = f"[mi list] {flag}: {email}"
    text = f"{flag} subscriber: {email}\nlist: mainline intelligence\n"
    html = f"<p><strong>{_escape_html(flag)}</strong> subscriber: {_escape_html(email)}</p>"
    try:
        send_smtp(to, subject, html, text, cfg)
    except Exception:
        pass


def subscribe(email: str, source: str = "mi_site") -> dict[str, Any]:
    """Store subscriber and send welcome from no-reply@…"""
    stored = store_subscriber(email, source=source)
    if not stored.get("ok"):
        return stored

    cfg = mailing_smtp_config()
    subject, html, text = _welcome_messages(stored["email"], cfg)
    mailed = False
    mail_error = None
    try:
        send_smtp(stored["email"], subject, html, text, cfg)
        mailed = True
    except Exception as e:
        mail_error = str(e)[:300]

    _notify_staff(stored["email"], bool(stored.get("already")), cfg)

    out: dict[str, Any] = {
        "ok": True,
        "email": stored["email"],
        "already": bool(stored.get("already")),
        "mailed": mailed,
    }
    if mail_error:
        # Still subscribed even if SMTP isn't configured yet.
        out["mail_warning"] = mail_error
    return out
