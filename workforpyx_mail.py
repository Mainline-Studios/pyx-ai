"""SMTP decision emails for Work with Pyx (hired / rejected)."""

from __future__ import annotations

import os
import smtplib
import ssl
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from workforpyx_storage import find_application, update_application

ROOT = Path(__file__).resolve().parent


def _load_env_file() -> None:
    """Load optional repo-root .env (gitignored) for local / manual deploy."""
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def _env(*keys: str, default: str = "") -> str:
    for key in keys:
        val = os.environ.get(key)
        if val and str(val).strip():
            return str(val).strip()
    return default


def smtp_config() -> dict[str, Any]:
    _load_env_file()
    from_email = _env(
        "PYX_APPLICATION_SMTP_FROM",
        "PYX_APPLICATION_FROM",
        "EMAIL_VERIFICATION_FROM",
        default="boehmlaird@gmail.com",
    )
    user = _env(
        "PYX_APPLICATION_SMTP_USER",
        "EMAIL_VERIFICATION_SMTP_USER",
        default=from_email,
    )
    password = _env(
        "PYX_APPLICATION_SMTP_PASS",
        "EMAIL_VERIFICATION_SMTP_PASS",
        "EMAIL_VERIFICATION_FROM_APP_PASSWORD",
    ).replace(" ", "")
    return {
        "from_email": from_email,
        "from_name": _env(
            "PYX_APPLICATION_FROM_NAME",
            "EMAIL_VERIFICATION_FROM_NAME",
            default="Pyx AI",
        ),
        "reply_to": _env(
            "PYX_APPLICATION_REPLY_TO",
            "EMAIL_VERIFICATION_REPLY_TO",
            default="",
        ),
        "user": user,
        "password": password,
        "host": _env(
            "PYX_APPLICATION_SMTP_HOST",
            "EMAIL_VERIFICATION_SMTP_HOST",
            default="smtp.gmail.com",
        ),
        "port": int(
            _env(
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
    }


def _html_email(
    *,
    title: str,
    lead: str,
    body_html: str,
    cta_label: str,
    cta_url: str,
    cfg: dict[str, Any],
) -> str:
    logo = f"{cfg['public_url']}/brand/pyx-app-icon.png"
    year = datetime.now(timezone.utc).year
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
</head>
<body style="margin:0;padding:0;background:#0f172a;font-family:'Segoe UI',system-ui,-apple-system,sans-serif;color:#e2e8f0;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:linear-gradient(160deg,#0f172a 0%,#0c4a6e 50%,#1e1b4b 100%);padding:32px 12px;">
    <tr>
      <td align="center">
        <table role="presentation" width="600" cellpadding="0" cellspacing="0" style="max-width:600px;width:100%;background:#1e293b;border-radius:16px;overflow:hidden;border:1px solid #334155;">
          <tr>
            <td style="padding:28px 28px 20px;text-align:center;background:linear-gradient(135deg,rgba(99,102,241,0.35),rgba(14,165,233,0.25));">
              <img src="{logo}" alt="Pyx AI" width="72" height="72" style="display:block;margin:0 auto 14px;border-radius:16px;">
              <p style="margin:0;font-size:13px;letter-spacing:0.12em;text-transform:uppercase;color:#a5b4fc;font-weight:700;">Pyx AI · Work with Pyx</p>
              <h1 style="margin:12px 0 0;font-size:24px;line-height:1.3;color:#f8fafc;font-weight:800;">{title}</h1>
            </td>
          </tr>
          <tr>
            <td style="padding:28px;line-height:1.65;font-size:16px;color:#cbd5e1;">
              <p style="margin:0 0 16px;">{lead}</p>
              {body_html}
              <p style="margin:24px 0 0;text-align:center;">
                <a href="{cta_url}" style="display:inline-block;padding:14px 28px;border-radius:999px;background:linear-gradient(90deg,#6366f1,#0ea5e9);color:#ffffff;text-decoration:none;font-weight:700;font-size:16px;">{cta_label}</a>
              </p>
            </td>
          </tr>
          <tr>
            <td style="padding:20px 28px;background:#0f172a;border-top:1px solid #334155;font-size:13px;line-height:1.5;color:#94a3b8;text-align:center;">
              <p style="margin:0 0 8px;"><strong style="color:#e2e8f0;">Didn&rsquo;t apply?</strong> If you didn&rsquo;t submit an application on Pyx Studio, you can safely ignore this email.</p>
              <p style="margin:0;">&copy; {year} Pyx AI · <a href="{cfg['public_url']}" style="color:#818cf8;">{cfg['public_url'].replace('https://', '')}</a></p>
              <p style="margin:10px 0 0;font-size:11px;">Automated message from {cfg['from_name']}.</p>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""


def _message_for_decision(
    app: dict[str, Any], status: str, note: str, cfg: dict[str, Any]
) -> tuple[str, str, str]:
    name = (app.get("name") or "there").strip()
    track = (app.get("track_label") or app.get("track") or "Pyx").strip()
    app_id = app.get("id") or ""
    public = cfg["public_url"]

    if status == "hired":
        subject = "Your Pyx application — welcome aboard!"
        lead = f"Hi {name},"
        extra = (
            f"<p style=\"margin:0 0 12px;\">We reviewed your application for <strong>{track}</strong> "
            f"and would like to move forward with you at Pyx.</p>"
            f"<p style=\"margin:0 0 12px;\">Reference: <code style=\"background:#0f172a;padding:2px 6px;border-radius:4px;\">{app_id}</code></p>"
        )
        if note:
            extra += f"<p style=\"margin:0;padding:14px;border-radius:10px;background:rgba(52,211,153,0.12);border:1px solid rgba(52,211,153,0.35);\"><strong>Next steps:</strong><br>{_escape_html(note)}</p>"
        else:
            extra += (
                "<p style=\"margin:0;\">Someone from the team will reach out soon with next steps. "
                "We're excited to have you help build Pyx Studio!</p>"
            )
        body_html = extra
        text = (
            f"Hi {name},\n\n"
            f"Your Pyx application ({track}) was accepted. Reference: {app_id}\n\n"
            f"{note + chr(10) + chr(10) if note else ''}"
            f"Visit {public}\n\n"
            "If you didn't submit an application to Pyx, you can ignore this email.\n"
        )
        return subject, _html_email(
            title="You're on the team!",
            lead=lead,
            body_html=body_html,
            cta_label="Open Pyx Studio",
            cta_url=public,
            cfg=cfg,
        ), text

    subject = "Update on your Pyx application"
    lead = f"Hi {name},"
    extra = (
        f"<p style=\"margin:0 0 12px;\">Thank you for applying to work with Pyx on <strong>{track}</strong>. "
        f"After reviewing your materials, we won&rsquo;t be moving forward with this application at this time.</p>"
        f"<p style=\"margin:0 0 12px;\">Reference: <code style=\"background:#0f172a;padding:2px 6px;border-radius:4px;\">{app_id}</code></p>"
    )
    if note:
        extra += f"<p style=\"margin:0;\">{_escape_html(note)}</p>"
    else:
        extra += (
            "<p style=\"margin:0;\">We appreciate the time you put into your application and encourage you "
            "to stay in touch as Pyx grows.</p>"
        )
    body_html = extra
    text = (
        f"Hi {name},\n\n"
        f"Thank you for applying to Pyx ({track}). We won't be moving forward at this time. "
        f"Reference: {app_id}\n\n"
        f"{note + chr(10) + chr(10) if note else ''}"
        f"{public}\n\n"
        "If you didn't submit an application to Pyx, you can ignore this email.\n"
    )
    return subject, _html_email(
        title="Application update",
        lead=lead,
        body_html=body_html,
        cta_label="Visit Pyx Studio",
        cta_url=public,
        cfg=cfg,
    ), text


def _escape_html(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def send_smtp(to: str, subject: str, html: str, text: str, cfg: dict[str, Any]) -> None:
    if not cfg["password"] or not cfg["user"]:
        raise RuntimeError(
            "Email not configured. Set PYX_APPLICATION_SMTP_PASS (or EMAIL_VERIFICATION_SMTP_PASS) "
            "on Cloud Run or in a gitignored .env file."
        )
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = (
        f"{cfg['from_name']} <{cfg['from_email']}>"
        if cfg["from_name"]
        else cfg["from_email"]
    )
    msg["To"] = to
    if cfg["reply_to"]:
        msg["Reply-To"] = cfg["reply_to"]
    msg.attach(MIMEText(text, "plain", "utf-8"))
    msg.attach(MIMEText(html, "html", "utf-8"))

    context = ssl.create_default_context()
    port = cfg["port"]
    if port == 465:
        with smtplib.SMTP_SSL(cfg["host"], port, context=context) as smtp:
            smtp.login(cfg["user"], cfg["password"])
            smtp.sendmail(cfg["from_email"], [to], msg.as_string())
    else:
        with smtplib.SMTP(cfg["host"], port) as smtp:
            smtp.starttls(context=context)
            smtp.login(cfg["user"], cfg["password"])
            smtp.sendmail(cfg["from_email"], [to], msg.as_string())


def send_application_decision(
    app_id: str, status: str, note: str = ""
) -> dict[str, Any]:
    status = (status or "").strip().lower()
    if status not in ("hired", "rejected"):
        return {"ok": False, "error": "Status must be hired or rejected."}

    app = find_application(app_id)
    if not app:
        return {"ok": False, "error": "Application not found."}

    email = (app.get("email") or "").strip()
    if not email or "@" not in email:
        return {"ok": False, "error": "Applicant has no valid email."}

    cfg = smtp_config()
    subject, html, text = _message_for_decision(app, status, note.strip(), cfg)
    try:
        send_smtp(email, subject, html, text, cfg)
    except Exception as e:
        return {"ok": False, "error": str(e)[:300]}

    now = datetime.now(timezone.utc).isoformat()
    update_application(
        app_id,
        {
            "status": status,
            "decision_at": now,
            "decision_note": note.strip(),
            "decision_emailed_at": now,
        },
    )
    return {"ok": True, "email": email, "status": status}
