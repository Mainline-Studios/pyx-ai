/**
 * MARII ask — Cloudflare Worker fallback when Cloud Run billing is off.
 * POST /ask  { "text": "...", "mode?": "fast", "use_web?": true }
 *
 * Secrets: GROQ_API_KEY (or PYX_TALK_LLM_KEY)
 */
const GROQ_URL = "https://api.groq.com/openai/v1/chat/completions";
const DEFAULT_MODEL = "openai/gpt-oss-20b";

const ALLOWED_ORIGINS = new Set([
  "https://pyx-ai.web.app",
  "https://pyx-ai.firebaseapp.com",
  "http://localhost:5000",
  "http://127.0.0.1:5000",
  "http://localhost:8080",
  "http://127.0.0.1:8080",
]);

const SYSTEM = (
  "You are MARII — Mainline Artificial Realtime Instant Intelligence. " +
  "Be concise, clear, and helpful. Prefer short answers that feel instant. " +
  "You power Pyx Assistant’s optional cloud boost. Stay safe for general audiences."
);

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

async function groqAsk(env, text) {
  const key = (env.GROQ_API_KEY || env.PYX_TALK_LLM_KEY || "").trim();
  if (!key) {
    const err = new Error("GROQ_API_KEY not configured");
    err.status = 503;
    throw err;
  }
  const model = (env.MARII_MODEL || "").trim() || DEFAULT_MODEL;
  const res = await fetch(GROQ_URL, {
    method: "POST",
    headers: {
      Authorization: "Bearer " + key,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model,
      max_tokens: 384,
      temperature: 0.55,
      messages: [
        { role: "system", content: SYSTEM },
        { role: "user", content: text },
      ],
    }),
  });
  if (!res.ok) {
    const detail = (await res.text()).slice(0, 400);
    const err = new Error(detail || "LLM failed");
    err.status = 502;
    throw err;
  }
  const data = await res.json();
  const answer =
    data &&
    data.choices &&
    data.choices[0] &&
    data.choices[0].message &&
    data.choices[0].message.content;
  if (!answer || typeof answer !== "string") {
    const err = new Error("empty LLM content");
    err.status = 502;
    throw err;
  }
  return { answer: answer.trim(), model };
}

export default {
  async fetch(request, env) {
    const origin = request.headers.get("Origin") || "";
    const url = new URL(request.url);

    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders(origin) });
    }

    if (url.pathname === "/health" || url.pathname === "/") {
      return json({ ok: true, service: "marii-ask" }, 200, origin);
    }

    if (request.method !== "POST" || (url.pathname !== "/ask" && url.pathname !== "/api/marii/ask")) {
      return json({ error: "Not found" }, 404, origin);
    }

    const started = Date.now();
    let body;
    try {
      body = await request.json();
    } catch {
      return json({ error: "Invalid JSON" }, 400, origin);
    }
    const text = typeof body.text === "string" ? body.text.trim() : "";
    if (!text) return json({ error: "Missing text" }, 400, origin);
    if (text.length > 4000) return json({ error: "Text too long" }, 413, origin);

    try {
      const { answer, model } = await groqAsk(env, text);
      return json(
        {
          answer,
          source: "marii",
          latency_ms: Date.now() - started,
          model,
        },
        200,
        origin
      );
    } catch (e) {
      const status = e && e.status ? e.status : 500;
      return json({ error: String(e.message || e) }, status, origin);
    }
  },
};
