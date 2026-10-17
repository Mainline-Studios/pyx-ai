/**
 * MARII ask — same local pipeline as Pyx Assistant (SLU, math, sports,
 * weather, notebook, wiki). No Cloud Run / no cloud LLM.
 */
(function (root) {
  "use strict";

  var ASK_MAX = 4000;
  var kbLoad = null;

  function kbUrl() {
    if (root.MARII_ASK_KB_URL) return String(root.MARII_ASK_KB_URL);
    try {
      if (typeof location !== "undefined" && location.origin && location.origin !== "null") {
        return location.origin + "/betas/pyxassistant/kb/pyx-assistant-kb.json?v=4";
      }
    } catch (e) {}
    return "https://pyx-ai.web.app/betas/pyxassistant/kb/pyx-assistant-kb.json?v=4";
  }

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

  function engines() {
    return {
      slu: root.PyxAssistantSLU,
      i18n: root.PyxAssistantI18n,
      math: root.PyxAssistantMath,
      kb: root.PyxAssistantKB,
      sports: root.PyxAssistantSports,
      weather: root.PyxAssistantWeather,
      wiki: root.PyxAssistantWiki,
    };
  }

  function hasAssistant(e) {
    return !!(e && e.slu && (e.math || e.sports || e.kb));
  }

  function tFn(e) {
    if (e.i18n && typeof e.i18n.t === "function") {
      return function (lang, key) {
        return e.i18n.t(lang, key);
      };
    }
    return null;
  }

  function stubReply(text) {
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

  function loadKbFromRequire(kb) {
    try {
      var data = require("../betas/pyxassistant/kb/pyx-assistant-kb.json");
      kb.load(data);
    } catch (err) {}
  }

  function ensureKb(e) {
    if (!e.kb || e.kb.size) return Promise.resolve();
    if (typeof fetch === "undefined") {
      loadKbFromRequire(e.kb);
      return Promise.resolve();
    }
    if (kbLoad) return kbLoad;
    kbLoad = fetch(kbUrl(), { cache: "no-store" })
      .then(function (res) {
        if (!res.ok) throw new Error("kb " + res.status);
        return res.json();
      })
      .then(function (data) {
        e.kb.load(data);
      })
      .catch(function () {
        kbLoad = null;
      });
    return kbLoad;
  }

  async function assistantReply(text, e) {
    var q = e.slu.normalize ? e.slu.normalize(text) : normalize(text);
    if (e.sports && e.sports.clearBoard) e.sports.clearBoard();

    if (e.math && e.math.looksMath(q)) {
      var solved = e.math.answer(q);
      if (solved) return { reply: solved.reply, source: "local" };
      return {
        reply: "I couldn’t parse that as math. Try 15% of 80, 9 times 8, or 32 F to C.",
        source: "local",
      };
    }

    var understood = e.slu.classify(q);
    var resolved = e.slu.resolve(understood, { lang: "en", t: tFn(e) });

    if (understood.intent === "greet" && resolved.reply) {
      return { reply: resolved.reply, source: "local" };
    }
    if (resolved.special && e.kb) {
      await ensureKb(e);
      return { reply: e.kb.expandSpecial(resolved.special), source: "local" };
    }
    if (resolved.reply) {
      return { reply: resolved.reply, source: "local" };
    }

    if (e.weather && (resolved.useWeb || understood.intent === "weather")) {
      try {
        var weatherReply = await e.weather.answer(q);
        if (weatherReply) return { reply: weatherReply, source: "weather" };
      } catch (err) {
        return {
          reply: "I couldn’t reach live weather just now. Try again in a second. =)",
          source: "weather",
        };
      }
    }

    if (e.sports && (understood.intent === "sports" || e.sports.looksSports(q))) {
      try {
        var sportReply = await e.sports.answer(q);
        if (sportReply) return { reply: sportReply, source: "sports" };
      } catch (err) {
        return {
          reply: "I couldn’t reach live sports just now. Try me again in a second. =)",
          source: "sports",
        };
      }
    }

    await ensureKb(e);
    if (e.kb) {
      var found = e.kb.probe ? e.kb.probe(q, 0.62) : null;
      var hit = found && found.hit ? found : null;
      if (hit && hit.reply) return { reply: hit.reply, source: "local" };
      if (e.wiki && e.wiki.looksWikiWorthy && e.wiki.looksWikiWorthy(q)) {
        try {
          var wikiHit = await e.wiki.answer(q);
          if (wikiHit && wikiHit.reply) return { reply: wikiHit.reply, source: "wiki" };
        } catch (wikiErr) {}
      }
      var reason = found && found.reason ? found.reason : !e.kb.size ? "empty-pack" : "no-match";
      var missText = e.kb.honestFallback
        ? e.kb.honestFallback(q, reason)
        : e.kb.warmFallback
          ? e.kb.warmFallback(q)
          : stubReply(q);
      return {
        reply: missText,
        source: e.kb.missSource ? e.kb.missSource(reason) : "kb-miss",
      };
    }

    return { reply: stubReply(q), source: "local" };
  }

  function wrap(q, result, backend) {
    return {
      ok: true,
      request: q,
      reply: result && result.reply != null ? result.reply : "",
      echo: q,
      source: (result && result.source) || "marii",
      backend: backend,
      ai: false,
    };
  }

  function askSync(text) {
    var q = normalize(text);
    if (!q) return { ok: false, error: "Missing request in path" };
    if (q.length > ASK_MAX) return { ok: false, error: "Request too long" };
    var e = engines();
    if (hasAssistant(e)) {
      return { ok: false, error: "Use MariiAsk.ask() — Pyx Assistant path is async" };
    }
    return wrap(q, { reply: stubReply(q), source: "marii" }, "local");
  }

  function ask(text) {
    var q = normalize(text);
    if (!q) {
      return Promise.resolve({ ok: false, error: "Missing request in path" });
    }
    if (q.length > ASK_MAX) {
      return Promise.resolve({ ok: false, error: "Request too long" });
    }
    var e = engines();
    if (!hasAssistant(e)) {
      return Promise.resolve(wrap(q, { reply: stubReply(q), source: "marii" }, "local"));
    }
    return assistantReply(q, e).then(function (result) {
      return wrap(q, result, "pyx-assistant");
    });
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
    askSync: askSync,
    reply: stubReply,
    decodeRequest: decodeRequest,
    requestFromLocation: requestFromLocation,
    ASK_MAX: ASK_MAX,
    ready: function () {
      return ensureKb(engines());
    },
  };

  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.MariiAsk = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
