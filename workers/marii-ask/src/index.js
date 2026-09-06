/**
 * MARII ask — Cloudflare Worker (Workers AI, NOT Groq).
 * POST /ask  { "text": "...", "mode?": "fast", "use_web?": true }
 *
 * Binding: AI (Workers AI). No Groq keys or api.groq.com calls.
 */
const DEFAULT_MODEL = "@cf/meta/llama-3.1-8b-instruct-fp8";

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
  "You power Pyx Assistant’s optional cloud boost and Announcer ask. " +
  "Stay safe for general audiences. For sports projections, give light booth color only — not betting advice."
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

function extractAnswer(result) {
  if (!result) return "";
  if (typeof result === "string") return result.trim();
  if (typeof result.response === "string") return result.response.trim();
  if (typeof result.result === "string") return result.result.trim();
  if (
    result.choices &&
    result.choices[0] &&
    result.choices[0].message &&
    typeof result.choices[0].message.content === "string"
  ) {
    return result.choices[0].message.content.trim();
  }
  return "";
}

async function workersAiAsk(env, text) {
  if (!env.AI || typeof env.AI.run !== "function") {
    const err = new Error("Workers AI binding missing");
    err.status = 503;
    throw err;
  }
  const model = (env.MARII_MODEL || "").trim() || DEFAULT_MODEL;
  const result = await env.AI.run(model, {
    messages: [
      { role: "system", content: SYSTEM },
      { role: "user", content: text },
    ],
    max_tokens: 512,
    temperature: 0.55,
  });
  const answer = extractAnswer(result);
  if (!answer) {
    const err = new Error("empty Workers AI content");
    err.status = 502;
    throw err;
  }
  return { answer, model };
}

export default {
  async fetch(request, env) {
    const origin = request.headers.get("Origin") || "";
    const url = new URL(request.url);

    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders(origin) });
    }

    if (url.pathname === "/health" || url.pathname === "/") {
      return json(
        {
          ok: true,
          service: "marii-ask",
          backend: "workers-ai",
          groq: false,
        },
        200,
        origin
      );
    }

    if (
      request.method !== "POST" ||
      (url.pathname !== "/ask" && url.pathname !== "/api/marii/ask")
    ) {
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
      const { answer, model } = await workersAiAsk(env, text);
      return json(
        {
          answer,
          source: "marii",
          backend: "workers-ai",
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
