/**
 * MI mailing list — Cloudflare Worker + Resend
 * POST /subscribe  { "email": "...", "source": "mi_site" }
 *
 * Secrets: RESEND_API_KEY
 * Optional vars:
 *   RESEND_FROM — default Mainline Intelligence <no-reply@pixelplaceofficial.com>
 *   NEWSLETTER_PDF_URL — PDF attached for new subscribers
 */
const DEFAULT_FROM = "Mainline Intelligence <no-reply@pixelplaceofficial.com>";
const REPLY_TO = "support@pixelplaceofficial.com";
const NOTIFY_TO = "support@pixelplaceofficial.com";
const DEFAULT_NEWSLETTER_PDF =
  "https://pyx-ai.web.app/mainlineintelligence/newsletters/mi-newsletter-001.pdf";
const NEWSLETTER_FILENAME = "mi-newsletter-001.pdf";

const ALLOWED_ORIGINS = new Set([
  "https://pyx-ai.web.app",
  "https://pyx-ai.firebaseapp.com",
  "http://localhost:5000",
  "http://127.0.0.1:5000",
  "http://localhost:8080",
  "http://127.0.0.1:8080",
]);

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

function fromAddress(env) {
  return (env.RESEND_FROM && String(env.RESEND_FROM).trim()) || DEFAULT_FROM;
}

function newsletterUrl(env) {
  return (
    (env.NEWSLETTER_PDF_URL && String(env.NEWSLETTER_PDF_URL).trim()) ||
    DEFAULT_NEWSLETTER_PDF
  );
}

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

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function welcomeHtml(email, { withNewsletter }) {
  const safe = escapeHtml(email);
  const pdfNote = withNewsletter
    ? `<tr><td style="padding-top:14px;line-height:1.55;color:#4a8a92;font-size:0.95rem;">
        issue <strong style="color:#0d7377">001</strong> is attached as a pdf — welcome + the latest on moderator, marii, and mci.
      </td></tr>`
    : "";
  return `<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>mainline intelligence</title></head>
<body style="margin:0;padding:0;background:#e8f7f4;font-family:system-ui,-apple-system,sans-serif;color:#1a4a52;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="padding:32px 12px;">
    <tr><td align="center">
      <table role="presentation" width="520" style="max-width:520px;width:100%;background:#fff;border-radius:16px;padding:28px;border:1px solid rgba(46,196,182,0.28);">
        <tr><td style="font-size:0.8rem;letter-spacing:0.04em;color:#7aadb4;">welcome + latest</td></tr>
        <tr><td style="padding-top:6px;font-size:1.4rem;font-weight:700;letter-spacing:-0.03em;color:#0d7377;">mainline intelligence</td></tr>
        <tr><td style="padding-top:14px;line-height:1.55;color:#4a8a92;font-size:1rem;">
          hey ${safe} — you're on the list.
        </td></tr>
        <tr><td style="padding-top:10px;line-height:1.55;color:#4a8a92;font-size:0.98rem;">
          here's what's happening in the new wave of pyx:
        </td></tr>
        <tr><td style="padding-top:14px;line-height:1.55;color:#1a4a52;font-size:0.95rem;">
          <strong style="color:#0d7377;">mi moderator — live</strong><br/>
          check chat/game text via <code style="color:#2ec4b6;">/moderator/check/&lt;text&gt;</code>.
          you get <code style="color:#2ec4b6;">{"appropriate":false,"score":"700"}</code>. threshold defaults to 700.
        </td></tr>
        <tr><td style="padding-top:12px;line-height:1.55;color:#1a4a52;font-size:0.95rem;">
          <strong style="color:#0d7377;">marii — coming soon</strong><br/>
          mainline artificial realtime instant intelligence. instant responses. real data. all free.
        </td></tr>
        <tr><td style="padding-top:12px;line-height:1.55;color:#1a4a52;font-size:0.95rem;">
          <strong style="color:#0d7377;">mci — coming soon</strong><br/>
          mainline conversational intelligence — the industrial evolution of pyx for teams that ship.
        </td></tr>
        <tr><td style="padding-top:12px;line-height:1.55;color:#1a4a52;font-size:0.95rem;">
          <strong style="color:#0d7377;">betas — pyx assistant (first public beta)</strong><br/>
          powered by <strong>marii</strong>. extremely early and still being improved.
          try it: <a href="https://pyx-ai.web.app/betas/" style="color:#2ec4b6;">pyx-ai.web.app/betas</a>
        </td></tr>
        ${pdfNote}
        <tr><td style="padding-top:14px;line-height:1.55;color:#0d7377;font-size:0.95rem;font-weight:600;">
          no ads. we pinky-swear. (we've already done enough crimes against your attention span.)
        </td></tr>
        <tr><td style="padding-top:22px;">
          <a href="https://pyx-ai.web.app/mainlineintelligence" style="color:#2ec4b6;text-decoration:none;border-bottom:1px solid rgba(46,196,182,0.5);">open mainline intelligence →</a>
        </td></tr>
        <tr><td style="padding-top:22px;font-size:0.8rem;color:#7aadb4;">
          sent from no-reply@pixelplaceofficial.com · replies go to support
        </td></tr>
      </table>
    </td></tr>
  </table>
</body></html>`;
}

function welcomeText(email, { withNewsletter }) {
  return (
    `hey ${email} — you're on the mainline intelligence list.\n\n` +
    `latest:\n` +
    `- mi moderator is live (/moderator/check/<text>)\n` +
    `- marii coming soon — realtime / instant intelligence\n` +
    `- mci coming soon — industrial conversational pyx\n` +
    `- betas: pyx assistant (first public beta, powered by marii) — extremely early\n` +
    `  https://pyx-ai.web.app/betas/\n\n` +
    (withNewsletter
      ? `issue 001 pdf is attached (welcome + news).\n\n`
      : "") +
    `no ads. we pinky-swear.\n` +
    `https://pyx-ai.web.app/mainlineintelligence\n`
  );
}

async function fetchNewsletterPdfBase64(env) {
  const url = newsletterUrl(env);
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`newsletter pdf fetch failed (${res.status})`);
  }
  const buf = await res.arrayBuffer();
  const bytes = new Uint8Array(buf);
  let binary = "";
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

async function sendResend(env, { to, subject, html, text, replyTo, attachments }) {
  const payload = {
    from: fromAddress(env),
    to: [to],
    subject,
    html,
    text,
    reply_to: replyTo || REPLY_TO,
  };
  if (attachments && attachments.length) {
    payload.attachments = attachments;
  }
  const res = await fetch("https://api.resend.com/emails", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.RESEND_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
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
      return json(
        {
          ok: true,
          service: "mi-mailing",
          newsletter: NEWSLETTER_FILENAME,
          from: fromAddress(env),
        },
        200,
        origin
      );
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
          newsletter_001: true,
        })
      );
    }

    // New subscribers get welcome + latest news + issue 001 PDF.
    // Re-joins get a short note without re-attaching the PDF.
    const withNewsletter = !already;
    let mailed = false;
    let mail_warning;
    let pdf_attached = false;
    try {
      const attachments = [];
      if (withNewsletter) {
        try {
          const content = await fetchNewsletterPdfBase64(env);
          attachments.push({
            filename: NEWSLETTER_FILENAME,
            content,
            content_type: "application/pdf",
          });
          pdf_attached = true;
        } catch (pdfErr) {
          mail_warning =
            "welcome sent without pdf: " +
            String(pdfErr && pdfErr.message ? pdfErr.message : pdfErr).slice(0, 200);
        }
      }

      await sendResend(env, {
        to: email,
        subject: withNewsletter
          ? "welcome to mainline intelligence — issue 001 inside"
          : "you're still on the mainline intelligence list",
        html: welcomeHtml(email, { withNewsletter }),
        text: welcomeText(email, { withNewsletter }),
        attachments,
      });
      mailed = true;
    } catch (e) {
      mail_warning = String(e && e.message ? e.message : e).slice(0, 300);
    }

    try {
      await sendResend(env, {
        to: NOTIFY_TO,
        subject: `[mi list] ${already ? "re-joined" : "new"}: ${email}`,
        html: `<p><strong>${already ? "re-joined" : "new"}</strong> subscriber: ${escapeHtml(
          email
        )}${pdf_attached ? " · newsletter 001 attached" : ""}</p>`,
        text: `${already ? "re-joined" : "new"} subscriber: ${email}\n`,
      });
    } catch {
      /* ignore */
    }

    const out = {
      ok: true,
      email,
      already,
      mailed,
      pdf_attached,
      newsletter: withNewsletter ? NEWSLETTER_FILENAME : null,
    };
    if (mail_warning) out.mail_warning = mail_warning;
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
