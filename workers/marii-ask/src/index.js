/**
 * MARII ask Worker — no LLM / no Groq / no Workers AI.
 * MARII answers are produced client-side from live data + local packs.
 *
 * POST /ask stays for compatibility and returns a clear non-AI response.
 */
const ALLOWED_ORIGINS = new Set([
  "https://pyx-ai.web.app",
  "https://pyx-ai.firebaseapp.com",
  "http://localhost:5000",
  "http://127.0.0.1:5000",
  "http://localhost:8080",
  "http://127.0.0.1:8080",
]);

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
          backend: "none",
          ai: false,
          groq: false,
          note: "MARII is local-only; this Worker does not run an LLM.",
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

    return json(
      {
        error: "MARII does not use cloud AI",
        hint: "Answers come from local KB / live feeds in the client. No Groq, no Workers AI.",
        source: "marii",
        backend: "none",
        ai: false,
      },
      501,
      origin
    );
  },
};
