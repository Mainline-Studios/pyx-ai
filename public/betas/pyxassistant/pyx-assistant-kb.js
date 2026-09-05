/**
 * Pyx Assistant — local retrieval over the knowledge base.
 */
(function (root) {
  "use strict";

  var STOP = {
    a: 1, an: 1, the: 1, and: 1, or: 1, of: 1, to: 1, in: 1, on: 1, for: 1, is: 1,
    it: 1, me: 1, my: 1, you: 1, your: 1, we: 1, do: 1, did: 1, does: 1, be: 1,
    are: 1, was: 1, were: 1, with: 1, at: 1, as: 1, that: 1, this: 1, from: 1,
    just: 1, can: 1, please: 1, about: 1, tell: 1, what: 1, whats: 1,
  };

  var state = {
    records: [],
    index: {},
    used: { joke: [], fact: [], riddle: [], quote: [], trivia: [], compliment: [] },
    lastReply: "",
  };

  function tokens(text) {
    return String(text || "")
      .toLowerCase()
      .replace(/['’]/g, "")
      .replace(/[^a-z0-9\s]/g, " ")
      .split(/\s+/)
      .filter(function (w) {
        return w.length > 1 && !STOP[w];
      });
  }

  function addIndex(rec) {
    var bag = tokens(rec.q).concat(rec.tags || []);
    var seen = {};
    bag.forEach(function (w) {
      if (seen[w]) return;
      seen[w] = 1;
      if (!state.index[w]) state.index[w] = [];
      state.index[w].push(rec);
    });
  }

  function load(data) {
    state.records = (data && data.records) || [];
    state.index = {};
    state.records.forEach(addIndex);
    return state.records.length;
  }

  function pickUnused(kind, fallbackKind) {
    var pool = state.records.filter(function (r) {
      return r.kind === kind && r.a.indexOf("__") !== 0;
    });
    if (!pool.length && fallbackKind) {
      pool = state.records.filter(function (r) {
        return r.kind === fallbackKind && r.a.indexOf("__") !== 0;
      });
    }
    if (!pool.length) return null;
    var used = state.used[kind] || [];
    var fresh = pool.filter(function (r) {
      return used.indexOf(r.id) === -1;
    });
    if (!fresh.length) {
      state.used[kind] = [];
      fresh = pool;
    }
    var rec = fresh[Math.floor(Math.random() * fresh.length)];
    state.used[kind] = (state.used[kind] || []).concat([rec.id]).slice(-80);
    return rec;
  }

  function expandSpecial(text) {
    if (text === "__JOKE__") {
      var j = pickUnused("joke");
      return j ? j.a + " =)" : "I seem to have misplaced my joke drawer.";
    }
    if (text === "__FACT__") {
      var f = pickUnused("fact");
      return f ? f.a : "Fun fact: you’re talking to an orb.";
    }
    if (text === "__RIDDLE__") {
      var r = pickUnused("riddle");
      return r ? r.q + " (Ask “what’s the answer?” if you want it.)" : "Why did the riddle hide? It didn’t want to be solved yet.";
    }
    if (text === "__QUOTE__") {
      var q = pickUnused("quote");
      return q ? q.a : "“Be kind.” — basically everyone worth quoting.";
    }
    if (text === "__TRIVIA__") {
      var t = pickUnused("trivia");
      return t ? t.q + " (I can answer if you ask.)" : "Quiz later — snack now?";
    }
    if (text === "__COMPLIMENT__") {
      var c = pickUnused("compliment");
      return c ? c.a + " =)" : "You’re doing better than the voice in your head admits.";
    }
    if (text === "__REPEAT__") {
      return state.lastReply || "I haven’t said anything worth repeating yet.";
    }
    return text;
  }

  function score(query, rec) {
    var qt = tokens(query);
    if (!qt.length) return 0;
    var hay = tokens(rec.q).concat(rec.tags || []);
    var set = {};
    hay.forEach(function (w) {
      set[w] = (set[w] || 0) + 1;
    });
    var hit = 0;
    var extra = 0;
    qt.forEach(function (w) {
      if (set[w]) {
        hit += 1;
        extra += 1 / set[w];
      }
    });
    if (!hit) return 0;
    var cover = hit / qt.length;
    var bonus = rec.q.toLowerCase() === String(query || "").toLowerCase() ? 2 : 0;
    var s = cover * 3 + extra + bonus + Math.min(hit, 4) * 0.15;
    if (hit === 1 && qt.length >= 3 && bonus < 1) s *= 0.25;
    return s;
  }

  function retrieve(query, minScore) {
    minScore = minScore == null ? 0.55 : minScore;
    var qt = tokens(query);
    var candidates = [];
    var seen = {};
    qt.forEach(function (w) {
      (state.index[w] || []).forEach(function (rec) {
        if (seen[rec.id]) return;
        seen[rec.id] = 1;
        candidates.push(rec);
      });
    });
    var best = null;
    var bestScore = 0;
    candidates.forEach(function (rec) {
      var s = score(query, rec);
      if (s > bestScore) {
        bestScore = s;
        best = rec;
      }
    });
    if (!best || bestScore < minScore) return null;
    return { rec: best, score: bestScore, reply: expandSpecial(best.a) };
  }

  function warmFallback(query) {
    var q = String(query || "").toLowerCase();
    if (/\b(joke|laugh|funny)\b/.test(q)) return expandSpecial("__JOKE__");
    if (/\b(fact|interesting|trivia)\b/.test(q)) return expandSpecial("__FACT__");
    if (/\b(riddle)\b/.test(q)) return expandSpecial("__RIDDLE__");
    if (/\b(quote|inspire|inspiration)\b/.test(q)) return expandSpecial("__QUOTE__");
    if (/\b(compliment|encourage|nice)\b/.test(q)) return expandSpecial("__COMPLIMENT__");
    if (/\b(weather|forecast|rain|temperature outside)\b/.test(q)) {
      return "I don’t have live weather in this beta — peek outside or a weather app. I can still convert °F and °C if you want. =)";
    }
    if (/\b(poem|haiku|story|song)\b/.test(q)) {
      return "I’m not a cloud writer anymore, but here’s a tiny one: pastel light / an orb that waits to listen / you, saying hello.";
    }
    var bits = tokens(query).slice(0, 4).join(", ");
    return (
      "I don’t have a perfect page for that, but I’m still here. " +
      (bits ? "I heard “" + bits + ".” " : "") +
      "I can joke, do math, convert units, share facts, riddles, quotes, or how-tos — or we can just talk. =)"
    );
  }

  function remember(reply) {
    state.lastReply = reply || state.lastReply;
  }

  var api = {
    load: load,
    retrieve: retrieve,
    expandSpecial: expandSpecial,
    pickUnused: pickUnused,
    warmFallback: warmFallback,
    remember: remember,
    tokens: tokens,
    get size() {
      return state.records.length;
    },
    _state: state,
  };

  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.PyxAssistantKB = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
