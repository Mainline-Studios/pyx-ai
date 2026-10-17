/**
 * MARII ask Worker — same pipeline as Pyx Assistant.
 * GET returns JSON. No Groq, no Workers AI, no Cloud Run.
 */
import * as i18nMod from "../../../public/betas/pyxassistant/pyx-assistant-i18n.js";
import * as sluMod from "../../../public/betas/pyxassistant/pyx-assistant-slu.js";
import * as mathMod from "../../../public/betas/pyxassistant/pyx-assistant-math.js";
import * as kbMod from "../../../public/betas/pyxassistant/pyx-assistant-kb.js";
import * as sportsMod from "../../../public/betas/pyxassistant/pyx-assistant-sports.js";
import * as weatherMod from "../../../public/betas/pyxassistant/pyx-assistant-weather.js";
import * as wikiRephraseMod from "../../../public/betas/pyxassistant/pyx-assistant-wiki-rephrase.js";
import * as wikiMod from "../../../public/betas/pyxassistant/pyx-assistant-wiki.js";
import * as askMod from "../../../public/js/marii-ask.js";

function bind(mod) {
  if (!mod) return mod;
  if (mod.default) return mod.default;
  return mod;
}

const i18n = bind(i18nMod);
const slu = bind(sluMod);
const math = bind(mathMod);
const kb = bind(kbMod);
const sports = bind(sportsMod);
const weather = bind(weatherMod);
const wikiRephrase = bind(wikiRephraseMod);
const wiki = bind(wikiMod);
const ask = bind(askMod);

globalThis.PyxAssistantI18n = i18n;
globalThis.PyxAssistantSLU = slu;
globalThis.PyxAssistantMath = math;
globalThis.PyxAssistantKB = kb;
globalThis.PyxAssistantSports = sports;
globalThis.PyxAssistantWeather = weather;
globalThis.PyxAssistantWikiRephrase = wikiRephrase;
globalThis.PyxAssistantWiki = wiki;
globalThis.MARII_ASK_KB_URL =
  "https://pyx-ai.web.app/betas/pyxassistant/kb/pyx-assistant-kb.json?v=4";

const ASK_MAX = 4000;
const SKIP_BARE = new Set(["", "health", "favicon.ico", "robots.txt"]);

function corsHeaders() {
  return {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Max-Age": "86400",
  };
}

function json(body, status) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store",
      ...corsHeaders(),
    },
  });
}

function decodeRequest(raw) {
  let s = String(raw == null ? "" : raw);
  try {
    s = decodeURIComponent(s.replace(/\+/g, " "));
  } catch (e) {}
  return s.trim().replace(/\s+/g, " ");
}

function pathRequest(pathname) {
  const path = String(pathname || "");
  const prefixes = ["/api/marii/ask/", "/marii/ask/", "/ask/"];
  for (const prefix of prefixes) {
    if (path.startsWith(prefix)) return path.slice(prefix.length);
  }
  if (path === "/" || path === "/health") return null;
  if (path === "/api/marii/ask" || path === "/marii/ask" || path === "/ask") {
    return "";
  }
  if (path.charAt(0) === "/") {
    const bare = path.slice(1);
    if (!SKIP_BARE.has(bare) && bare.indexOf(".") === -1) return bare;
  }
  return null;
}

export default {
  async fetch(request) {
    const url = new URL(request.url);

    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders() });
    }

    if (request.method === "GET" && (url.pathname === "/" || url.pathname === "/health")) {
      return json(
        {
          ok: true,
          service: "marii-ask",
          backend: "pyx-assistant",
          ai: false,
          groq: false,
          curl: 'curl -sS "https://marii-ask.mainline-mi.workers.dev/how%20is%20judge"',
          engines: {
            slu: typeof slu.classify === "function",
            math: typeof math.looksMath === "function",
            sports: typeof sports.answer === "function",
            wiki: typeof wiki.looksWikiWorthy === "function",
            ask: typeof ask.ask === "function",
          },
        },
        200
      );
    }

    if (request.method !== "GET") {
      return json(
        {
          ok: false,
          error: "MARII does not use cloud AI",
          hint: 'Use GET. Example: curl -sS "https://marii-ask.mainline-mi.workers.dev/how%20is%20judge"',
          ai: false,
        },
        501
      );
    }

    let raw = pathRequest(url.pathname);
    if (raw == null) {
      return json({ ok: false, error: "not found" }, 404);
    }
    if (!raw) {
      raw =
        url.searchParams.get("request") ||
        url.searchParams.get("q") ||
        url.searchParams.get("text") ||
        "";
    }
    const q = decodeRequest(raw);
    if (!q) {
      return json({ ok: false, error: "Missing request in path" }, 400);
    }
    if (q.length > ASK_MAX) {
      return json({ ok: false, error: "Request too long" }, 413);
    }

    try {
      const data = await ask.ask(q);
      return json(data, data && data.ok === false ? 400 : 200);
    } catch (err) {
      return json(
        {
          ok: false,
          error: String(err && err.message ? err.message : err).slice(0, 300),
          backend: "pyx-assistant",
          ai: false,
        },
        502
      );
    }
  },
};
