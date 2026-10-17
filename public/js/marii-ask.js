/**
 * MARII ask — local, no Cloud Run / no LLM.
 * Used by the MI home try-it and /api/marii/ask/<request> (Hosting page).
 */
(function (root) {
  "use strict";

  var ASK_MAX = 4000;

  function normalize(text) {
    return String(text || "").trim().replace(/\s+/g, " ");
  }

  function decodeRequest(raw) {
    var s = String(raw == null ? "" : raw);
    try {
      s = decodeURIComponent(s.replace(/\+/g, " "));
    } catch (e) {}
    return normalize(s);
  }

  function reply(text) {
    var q = normalize(text);
    if (!q) return "";
    var low = q.toLowerCase();
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

  function ask(text) {
    var q = normalize(text);
    if (!q) {
      return { ok: false, error: "Missing request in path" };
    }
    if (q.length > ASK_MAX) {
      return { ok: false, error: "Request too long" };
    }
    return {
      ok: true,
      request: q,
      reply: reply(q),
      echo: q,
      source: "marii",
      backend: "local",
      ai: false,
    };
  }

  function requestFromLocation(loc) {
    loc = loc || (typeof location !== "undefined" ? location : null);
    if (!loc) return "";
    var path = String(loc.pathname || "");
    var prefixes = ["/api/marii/ask/", "/marii/ask/", "/ask/"];
    var i;
    for (i = 0; i < prefixes.length; i++) {
      if (path.indexOf(prefixes[i]) === 0) {
        return decodeRequest(path.slice(prefixes[i].length));
      }
    }
    if (
      path === "/api/marii/ask" ||
      path === "/marii/ask" ||
      path === "/ask" ||
      path === "/marii-ask.html"
    ) {
      var params = new URLSearchParams(String(loc.search || "").replace(/^\?/, ""));
      return decodeRequest(params.get("request") || params.get("q") || params.get("text") || "");
    }
    return "";
  }

  var api = {
    ask: ask,
    reply: reply,
    decodeRequest: decodeRequest,
    requestFromLocation: requestFromLocation,
    ASK_MAX: ASK_MAX,
  };

  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.MariiAsk = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
