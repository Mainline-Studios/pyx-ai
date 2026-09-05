/**
 * MI mailing list — Cloudflare Worker + Resend
 * POST /subscribe  { "email": "...", "source": "mi_site" }
 */
const FROM = "Mainline Intelligence <no-reply@pixelplaceofficial.com>";
const REPLY_TO = "support@pixelplaceofficial.com";
const NOTIFY_TO = "support@pixelplaceofficial.com";
const ALLOWED_ORIGINS = new Set([
  "https://pyx-ai.web.app",
  "https://pyx-ai.firebaseapp.com",
  "http://localhost:5000",
  "http://127.0.0.1:5000",
  "http://localhost:8080",
  "http://127.0.0.1:8080",
]);

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

function corsHeaders(origin) {
  const allow =
    origin && ALLOWED_ORIGINS.has(origin) ? origin : "https://pyx-ai.web.app";
  return {
    "Access-Control-Allow-Origin": allow,
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Max-Age": "86400",
    Vary: "Origin",
  };
}

function json(body, status, origin) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      ...corsHeaders(origin),
    },
  });
}

function normalizeEmail(raw) {
  return String(raw || "")
    .trim()
    .toLowerCase();
}

function validEmail(email) {
  return Boolean(email) && email.length <= 320 && EMAIL_RE.test(email);
}

function welcomeHtml(email) {
  const safe = email
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
  return `<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>mainline intelligence</title></head>
<body style="margin:0;padding:0;background:#e8f7f4;font-family:system-ui,-apple-system,sans-serif;color:#1a4a52;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="padding:32px 12px;">
    <tr><td align="center">
      <table role="presentation" width="520" style="max-width:520px;width:100%;background:#fff;border-radius:16px;padding:28px;border:1px solid rgba(46,196,182,0.28);">
        <tr><td style="font-size:1.35rem;font-weight:700;letter-spacing:-0.03em;color:#0d7377;">mainline intelligence</td></tr>
        <tr><td style="padding-top:14px;line-height:1.55;color:#4a8a92;">hey ${safe} — you're on the list.</td></tr>
        <tr><td style="padding-top:10px;line-height:1.55;color:#4a8a92;">we'll ping you about mi moderator, marii, mci, and the rest of the new wave of pyx.</td></tr>
        <tr><td style="padding-top:10px;line-height:1.55;color:#0d7377;font-weight:600;">no ads. we pinky-swear. (we've already done enough crimes against your attention span.)</td></tr>
        <tr><td style="padding-top:22px;"><a href="https://pyx-ai.web.app/mainlineintelligence" style="color:#2ec4b6;">open mainline intelligence →</a></td></tr>
        <tr><td style="padding-top:22px;font-size:0.8rem;color:#7aadb4;">sent from no-reply@pixelplaceofficial.com · replies go to support</td></tr>
      </table>
    </td></tr>
  </table>
</body></html>`;
}

async function sendResend(env, { to, subject, html, text, replyTo }) {
  const res = await fetch("https://api.resend.com/emails", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.RESEND_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      from: FROM,
      to: [to],
      subject,
      html,
      text,
      reply_to: replyTo || REPLY_TO,
    }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg =
      (data && data.message) ||
      (data && data.error) ||
      `resend ${res.status}`;
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return data;
}

export default {
  async fetch(request, env) {
    const origin = request.headers.get("Origin") || "";

    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders(origin) });
    }

    const url = new URL(request.url);
    if (request.method === "GET" && (url.pathname === "/" || url.pathname === "/health")) {
      return json({ ok: true, service: "mi-mailing" }, 200, origin);
    }

    if (request.method !== "POST" || url.pathname !== "/subscribe") {
      return json({ ok: false, error: "not found" }, 404, origin);
    }

    if (!env.RESEND_API_KEY) {
      return json(
        { ok: false, error: "resend not configured (missing RESEND_API_KEY)" },
        503,
        origin
      );
    }

    let body;
    try {
      body = await request.json();
    } catch {
      return json({ ok: false, error: "invalid json" }, 400, origin);
    }

    const email = normalizeEmail(body && body.email);
    const source = String((body && body.source) || "mi_site").slice(0, 80);
    if (!validEmail(email)) {
      return json({ ok: false, error: "need a real email address" }, 400, origin);
    }

    const now = new Date().toISOString();
    let already = false;
    if (env.SUBSCRIBERS) {
      const prev = await env.SUBSCRIBERS.get(email);
      already = Boolean(prev);
      await env.SUBSCRIBERS.put(
        email,
        JSON.stringify({
          email,
          source,
          subscribed_at: already && prev ? JSON.parse(prev).subscribed_at || now : now,
          updated_at: now,
        })
      );
    }

    let mailed = false;
    let mail_warning;
    try {
      await sendResend(env, {
        to: email,
        subject: "you're on the mainline intelligence list",
        html: welcomeHtml(email),
        text:
          "hey — you're on the mainline intelligence mailing list.\n\n" +
          "we'll ping you about mi moderator, marii, mci, and the rest of the new wave.\n" +
          "no ads. we pinky-swear.\n\n" +
          "https://pyx-ai.web.app/mainlineintelligence\n",
      });
      mailed = true;
    } catch (e) {
      mail_warning = String(e && e.message ? e.message : e).slice(0, 300);
    }

    // Staff ping (best-effort)
    try {
      await sendResend(env, {
        to: NOTIFY_TO,
        subject: `[mi list] ${already ? "re-joined" : "new"}: ${email}`,
        html: `<p><strong>${already ? "re-joined" : "new"}</strong> subscriber: ${email}</p>`,
        text: `${already ? "re-joined" : "new"} subscriber: ${email}\n`,
      });
    } catch {
      /* ignore */
    }

    const out = { ok: true, email, already, mailed };
    if (mail_warning) out.mail_warning = mail_warning;
    // Still ok if stored but mail failed (e.g. domain not verified yet)
    if (!mailed && !env.SUBSCRIBERS) {
      return json(
        { ok: false, error: mail_warning || "mail failed", emailed: false },
        502,
        origin
      );
    }
    return json(out, 200, origin);
  },
};
