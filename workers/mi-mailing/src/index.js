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
  "https://pyx-ai.web.app/mainlineintelligence/newsletters/mi-newsletter-003.pdf";
const NEWSLETTER_FILENAME = "mi-newsletter-003.pdf";
const NEWSLETTER_ISSUE = "003";
const NEWSLETTER_KV_FLAG = "newsletter_003";
const NEWSLETTER_WEB =
  "https://pyx-ai.web.app/mainlineintelligence/newsletters/mi-newsletter-003.html";
const NEWSLETTER_SUBJECT =
  "mainline intelligence — issue 003 (marii ask + pyx assistant)";

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
    "Access-Control-Allow-Headers": "Content-Type, X-Broadcast-Secret",
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
        issue <strong style="color:#0d7377">${NEWSLETTER_ISSUE}</strong> is attached as a pdf — marii ask now runs on the same local pipeline as pyx assistant.
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
          <strong style="color:#0d7377;">marii ask — same engine as pyx assistant</strong><br/>
          <code style="color:#2ec4b6;">/api/marii/ask/&lt;request&gt;</code> now runs slu, math, live sports, weather, kb, and wiki in the browser.
          try <a href="https://pyx-ai.web.app/api/marii/ask/how%20is%20judge" style="color:#2ec4b6;">/api/marii/ask/how is judge</a>.
          no cloud run. no cloud llm.
        </td></tr>
        <tr><td style="padding-top:12px;line-height:1.55;color:#1a4a52;font-size:0.95rem;">
          <strong style="color:#0d7377;">mi moderator — live</strong><br/>
          check chat/game text via <code style="color:#2ec4b6;">/moderator/check/&lt;text&gt;</code>.
          you get <code style="color:#2ec4b6;">{"appropriate":false,"score":"700"}</code>. threshold defaults to 700.
        </td></tr>
        <tr><td style="padding-top:12px;line-height:1.55;color:#1a4a52;font-size:0.95rem;">
          <strong style="color:#0d7377;">announcer + game-only marii</strong><br/>
          live mlb booth, neural/kokoro voices, algorithmic projections from recent finals.
        </td></tr>
        <tr><td style="padding-top:12px;line-height:1.55;color:#1a4a52;font-size:0.95rem;">
          <strong style="color:#0d7377;">mci — coming soon</strong><br/>
          mainline conversational intelligence — the industrial evolution of pyx for teams that ship.
        </td></tr>
        <tr><td style="padding-top:12px;line-height:1.55;color:#1a4a52;font-size:0.95rem;">
          <strong style="color:#0d7377;">betas — pyx assistant + announcer</strong><br/>
          powered by <strong>marii</strong>. extremely early and still being improved.
          try pyx assistant: <a href="https://pyx-ai.web.app/betas/pyxassistant" style="color:#2ec4b6;">pyx-ai.web.app/betas/pyxassistant</a>
          · <a href="https://pyx-ai.web.app/mainlineintelligence/pyx-assistant" style="color:#2ec4b6;">what it is</a>
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
    `- marii ask now runs on the pyx assistant pipeline (no cloud run / no cloud llm)\n` +
    `  https://pyx-ai.web.app/api/marii/ask/how%20is%20judge\n` +
    `- mi moderator is live (/moderator/check/<text>)\n` +
    `- announcer + game-only marii — mlb booth + projections from recent finals\n` +
    `- mci coming soon — industrial conversational pyx\n` +
    `- betas: pyx assistant + announcer — extremely early\n` +
    `  https://pyx-ai.web.app/betas/pyxassistant\n` +
    `  about: https://pyx-ai.web.app/mainlineintelligence/pyx-assistant\n\n` +
    (withNewsletter
      ? `issue ${NEWSLETTER_ISSUE} pdf is attached.\n\n`
      : "") +
    `no ads. we pinky-swear.\n` +
    `https://pyx-ai.web.app/mainlineintelligence\n`
  );
}

function issueHtml() {
  return `<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>mi newsletter ${NEWSLETTER_ISSUE}</title></head>
<body style="margin:0;padding:0;background:#e8f7f4;font-family:system-ui,-apple-system,sans-serif;color:#1a4a52;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="padding:32px 12px;">
    <tr><td align="center">
      <table role="presentation" width="520" style="max-width:520px;width:100%;background:#fff;border-radius:16px;padding:28px;border:1px solid rgba(46,196,182,0.28);">
        <tr><td style="font-size:0.8rem;letter-spacing:0.04em;color:#7aadb4;">newsletter · issue ${NEWSLETTER_ISSUE}</td></tr>
        <tr><td style="padding-top:6px;font-size:1.4rem;font-weight:700;letter-spacing:-0.03em;color:#0d7377;">mainline intelligence</td></tr>
        <tr><td style="padding-top:14px;line-height:1.55;color:#4a8a92;font-size:0.98rem;">
          marii ask now runs on the same local pipeline as pyx assistant. no cloud run. no cloud llm.
        </td></tr>
        <tr><td style="padding-top:14px;line-height:1.55;color:#1a4a52;font-size:0.95rem;">
          <strong style="color:#0d7377;">marii ask</strong><br/>
          <code style="color:#2ec4b6;">/api/marii/ask/&lt;request&gt;</code> answers from slu, math, live sports, weather, kb, and wiki.
          try <a href="https://pyx-ai.web.app/api/marii/ask/how%20is%20judge" style="color:#2ec4b6;">how is judge</a>
          or the box on <a href="https://pyx-ai.web.app/mainlineintelligence#marii-ask" style="color:#2ec4b6;">mi home</a>.
        </td></tr>
        <tr><td style="padding-top:12px;line-height:1.55;color:#1a4a52;font-size:0.95rem;">
          <strong style="color:#0d7377;">hosting-only</strong><br/>
          a tiny static page runs the assistant in the browser. if cloud run is down, ask still answers.
        </td></tr>
        <tr><td style="padding-top:12px;line-height:1.55;color:#1a4a52;font-size:0.95rem;">
          <strong style="color:#0d7377;">announcer + moderator</strong><br/>
          booth is still live. chat/game check is still live. mci is still coming.
        </td></tr>
        <tr><td style="padding-top:16px;line-height:1.55;color:#4a8a92;font-size:0.92rem;">
          read / download: <a href="${NEWSLETTER_WEB}" style="color:#2ec4b6;">issue ${NEWSLETTER_ISSUE} on the web</a>
          · pdf attached.
        </td></tr>
        <tr><td style="padding-top:14px;line-height:1.55;color:#0d7377;font-size:0.95rem;font-weight:600;">
          no ads. we pinky-swear.
        </td></tr>
        <tr><td style="padding-top:22px;font-size:0.8rem;color:#7aadb4;">
          sent from no-reply@pixelplaceofficial.com · replies go to support
        </td></tr>
      </table>
    </td></tr>
  </table>
</body></html>`;
}

function issueText() {
  return (
    `mainline intelligence — newsletter issue ${NEWSLETTER_ISSUE}\n\n` +
    `marii ask now runs on the same local pipeline as pyx assistant.\n` +
    `https://pyx-ai.web.app/api/marii/ask/how%20is%20judge\n\n` +
    `hosting-only — no cloud run, no cloud llm.\n` +
    `try the box: https://pyx-ai.web.app/mainlineintelligence#marii-ask\n\n` +
    `announcer + mi moderator still live. mci still coming.\n\n` +
    `web: ${NEWSLETTER_WEB}\n` +
    `pdf attached.\n\n` +
    `no ads. we pinky-swear.\n`
  );
}

async function listSubscriberEmails(env) {
  if (!env.SUBSCRIBERS) return [];
  const emails = [];
  let cursor;
  do {
    const page = await env.SUBSCRIBERS.list({ cursor, limit: 1000 });
    for (const key of page.keys || []) {
      if (key && key.name && key.name.includes("@")) emails.push(key.name);
    }
    cursor = page.list_complete ? undefined : page.cursor;
  } while (cursor);
  return emails;
}

async function subscriberAlreadyGotIssue(env, email) {
  if (!env.SUBSCRIBERS) return false;
  const prev = await env.SUBSCRIBERS.get(email);
  if (!prev) return false;
  try {
    const row = JSON.parse(prev);
    return Boolean(row && row[NEWSLETTER_KV_FLAG]);
  } catch {
    return false;
  }
}

async function markSubscriberIssue(env, email) {
  if (!env.SUBSCRIBERS) return;
  const prev = await env.SUBSCRIBERS.get(email);
  let row = {};
  try {
    row = prev ? JSON.parse(prev) : {};
  } catch {
    row = { email };
  }
  row.email = email;
  row[NEWSLETTER_KV_FLAG] = true;
  row.updated_at = new Date().toISOString();
  await env.SUBSCRIBERS.put(email, JSON.stringify(row));
}

async function broadcastLatest(env) {
  const emails = await listSubscriberEmails(env);
  let pdfContent = null;
  try {
    pdfContent = await fetchNewsletterPdfBase64(env);
  } catch (e) {
    /* optional */
  }
  const attachments = pdfContent
    ? [
        {
          filename: NEWSLETTER_FILENAME,
          content: pdfContent,
          content_type: "application/pdf",
        },
      ]
    : [];
  const results = [];
  let skipped = 0;
  for (const email of emails) {
    try {
      if (await subscriberAlreadyGotIssue(env, email)) {
        skipped += 1;
        results.push({ email, ok: true, skipped: true });
        continue;
      }
      await sendResend(env, {
        to: email,
        subject: NEWSLETTER_SUBJECT,
        html: issueHtml(),
        text: issueText(),
        attachments,
      });
      await markSubscriberIssue(env, email);
      results.push({ email, ok: true });
    } catch (e) {
      results.push({
        email,
        ok: false,
        error: String(e && e.message ? e.message : e).slice(0, 200),
      });
    }
  }
  return {
    sent: results.filter((r) => r.ok && !r.skipped).length,
    skipped,
    total: emails.length,
    results,
  };
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
          issue: NEWSLETTER_ISSUE,
          from: fromAddress(env),
        },
        200,
        origin
      );
    }

    if (request.method === "POST" && url.pathname === "/broadcast") {
      if (!env.RESEND_API_KEY) {
        return json({ ok: false, error: "resend not configured" }, 503, origin);
      }
      const secret = request.headers.get("X-Broadcast-Secret") || "";
      if (!env.BROADCAST_SECRET || secret !== env.BROADCAST_SECRET) {
        return json({ ok: false, error: "unauthorized" }, 401, origin);
      }
      let body = {};
      try {
        body = await request.json();
      } catch {
        body = {};
      }
      if (body && body.issue && String(body.issue) !== NEWSLETTER_ISSUE) {
        return json({ ok: false, error: "unsupported issue" }, 400, origin);
      }
      try {
        const out = await broadcastLatest(env);
        try {
          await sendResend(env, {
            to: NOTIFY_TO,
            subject: `[mi list] broadcast issue ${NEWSLETTER_ISSUE}: ${out.sent}/${out.total}`,
            html: `<p>broadcast issue <strong>${NEWSLETTER_ISSUE}</strong>: ${out.sent}/${out.total}</p>`,
            text: `broadcast issue ${NEWSLETTER_ISSUE}: ${out.sent}/${out.total}\n`,
          });
        } catch {
          /* ignore */
        }
        return json({ ok: true, issue: NEWSLETTER_ISSUE, ...out }, 200, origin);
      } catch (e) {
        return json(
          { ok: false, error: String(e && e.message ? e.message : e).slice(0, 300) },
          502,
          origin
        );
      }
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
    let prevRow;
    if (env.SUBSCRIBERS) {
      const prev = await env.SUBSCRIBERS.get(email);
      already = Boolean(prev);
      prevRow = prev;
    }
    // New subscribers get welcome + latest news + issue PDF.
    // Re-joins get a short note without re-attaching the PDF.
    const withNewsletter = !already;
    if (env.SUBSCRIBERS) {
      let subscribedAt = now;
      let alreadyGotLatest = false;
      if (already && prevRow) {
        try {
          const parsed = JSON.parse(prevRow);
          subscribedAt = parsed.subscribed_at || now;
          alreadyGotLatest = Boolean(parsed[NEWSLETTER_KV_FLAG]);
        } catch {
          subscribedAt = now;
        }
      }
      await env.SUBSCRIBERS.put(
        email,
        JSON.stringify({
          email,
          source,
          subscribed_at: subscribedAt,
          updated_at: now,
          newsletter_001: true,
          newsletter_002: true,
          [NEWSLETTER_KV_FLAG]: withNewsletter || alreadyGotLatest || undefined,
        })
      );
    }
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
          ? `welcome to mainline intelligence — issue ${NEWSLETTER_ISSUE} inside`
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
        )}${pdf_attached ? ` · newsletter ${NEWSLETTER_ISSUE} attached` : ""}</p>`,
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
