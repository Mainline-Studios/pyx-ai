/**
 * MARII ask Worker — no LLM / no Groq / no Workers AI.
 *
 * GET /api/marii/ask/<request> (and /marii/ask/<request>, /ask/<request>)
 * returns the request plus a short local reply.
 * POST /ask and POST /api/marii/ask stay 501 (no cloud LLM).
 */
const ALLOWED_ORIGINS = new Set([
  "https://pyx-ai.web.app",
  "https://pyx-ai.firebaseapp.com",
  "http://localhost:5000",
  "http://127.0.0.1:5000",
  "http://localhost:8080",
  "http://127.0.0.1:8080",
]);

const ASK_MAX = 4000;

function corsHeaders(origin) {
  const allow =
    origin && ALLOWED_ORIGINS.has(origin) ? origin : "https://pyx-ai.web.app";
  return {
    "Access-Control-Allow-Origin": allow,
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
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

function localReply(text) {
  const q = String(text || "").trim().replace(/\s+/g, " ");
  if (!q) return "";
  const low = q.toLowerCase();
  if (low === "hi" || low === "hello" || low === "hey" || low === "yo") {
    return "hi — marii ask is live. drop a request on /api/marii/ask/<request>.";
  }
  if (/\bwhat(?:'s| is) marii\b|\bwho is marii\b/i.test(q)) {
    return "MARII is Mainline Artificial Realtime Instant Intelligence — instant when it matters, realtime when the world moves. No cloud LLM on this path.";
  }
  if (/\bwhat(?:'s| is) (?:mi|mainline intelligence)\b/i.test(q)) {
    return "Mainline Intelligence (MI) is the umbrella for moderator, MARII, MCI, and the rest of the new Pyx wave.";
  }
  if (/\bwhat(?:'s| is) mci\b/i.test(q)) {
    return "MCI is Mainline Conversational Intelligence — coming soon. Studio-grade dialogue for teams that ship.";
  }
  if (/\bwho are you\b|\bwhat are you\b/i.test(q)) {
    return "I'm the MARII ask API. Send GET /api/marii/ask/<request> and I'll return your request plus a short local reply.";
  }
  return q;
}

function askPayload(text) {
  const q = String(text || "").trim().replace(/\s+/g, " ");
  return {
    ok: true,
    request: q,
    reply: localReply(q),
    echo: q,
    source: "marii",
    backend: "local",
    ai: false,
  };
}

function askFromText(raw, origin) {
  let decoded = "";
  try {
    decoded = decodeURIComponent(String(raw || "")).trim();
  } catch (e) {
    decoded = String(raw || "").trim();
  }
  if (!decoded) {
    return json({ error: "Missing request in path", ok: false }, 400, origin);
  }
  if (decoded.length > ASK_MAX) {
    return json({ error: "Request too long", ok: false }, 413, origin);
  }
  return json(askPayload(decoded), 200, origin);
}

function pathRequest(pathname) {
  const prefixes = ["/api/marii/ask/", "/marii/ask/", "/ask/"];
  for (const prefix of prefixes) {
    if (pathname.startsWith(prefix)) {
      return pathname.slice(prefix.length);
    }
  }
  return null;
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
          backend: "local",
          ai: false,
          groq: false,
          note: "GET /api/marii/ask/<request> returns the request plus a local reply. No LLM.",
        },
        200,
        origin
      );
    }

    if (request.method === "GET") {
      const fromPath = pathRequest(url.pathname);
      if (fromPath != null) {
        return askFromText(fromPath, origin);
      }
      if (
        url.pathname === "/api/marii/ask" ||
        url.pathname === "/marii/ask" ||
        url.pathname === "/ask"
      ) {
        const raw = url.searchParams.get("request") || url.searchParams.get("q") || url.searchParams.get("text");
        if (raw == null) {
          return json({ error: 'Missing "request"', ok: false }, 400, origin);
        }
        return askFromText(raw, origin);
      }
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
        hint: "Use GET /api/marii/ask/<request>. No Groq, no Workers AI.",
        source: "marii",
        backend: "none",
        ai: false,
      },
      501,
      origin
    );
  },
};
