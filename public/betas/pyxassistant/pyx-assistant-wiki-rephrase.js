/**
 * Tiny local wiki rephraser.
 *
 * Not a 1B LLM (we cannot train or ship one on Hosting). This is a word-bigram
 * model + phrase table fitted on the assistant notebook (definition voice)
 * and, at ask time, the Wikipedia lead itself. It only compresses text already
 * in the extract — it does not invent facts.
 */
(function (root) {
  "use strict";

  var MAX_CHARS = 220;
  var MIN_CHARS = 36;

  var PHRASES = [
    ["due to the fact that", "because"],
    ["is considered to be", "is"],
    ["is defined as", "is"],
    ["is a variety of", "is a kind of"],
    ["is a type of", "is a"],
    ["is a form of", "is"],
    ["is the process of", "is"],
    ["is the practice of", "is"],
    ["is the study of", "is"],
    ["refers to", "is"],
    ["known as a", "a"],
    ["chiefly known for", "known for"],
    ["best known for", "known for"],
    ["most commonly", "usually"],
    ["in order to", "to"],
    ["a number of", "some"],
    ["as well as", "and"],
    ["in which", "where"],
    ["that is used to", "used to"],
    ["which is used to", "used to"],
    ["which expands and puffs up", "that puffs up"],
    ["which expands and", "that"],
    ["corn kernel", "corn"],
    ["can be described as", "is"],
  ];

  // Notebook definition voice — the "other stuff before."
  var SEED = [
    "Plants use light to turn carbon dioxide and water into sugar and oxygen.",
    "A pull between masses. Earth hugging you so you don't float off during breakfast.",
    "A step-by-step recipe a computer or a person can follow.",
    "A reusable bundle of steps. Give it inputs, get an output.",
    "A way for programs to talk to each other with agreed-upon requests.",
    "The skeleton of a web page — headings, paragraphs, buttons.",
    "The request/response language of the web.",
    "An address on the web. A path to a thing.",
    "How much data can move at once.",
    "One tiny colored dot.",
    "The basic building block of elements.",
    "Atoms bonded together. Water is a famous trio.",
    "The smallest living unit in most organisms.",
    "Populations changing over generations.",
    "The long-term pattern of weather.",
    "What's happening in the sky right now.",
    "Liquid becoming gas.",
    "Gas becoming liquid.",
    "Energy from ancient life, stored underground.",
    "Energy that refills on human timescales — sun, wind, falling water.",
    "A system where people have a say, usually by voting.",
    "Feeling with someone.",
    "Control over who gets to see you.",
    "MARII is Mainline Artificial Realtime Instant Intelligence.",
    "Mainline Intelligence is the umbrella for moderator, MARII, and MCI.",
    "Popcorn is a kind of corn that puffs up when heated.",
    "Ada Lovelace was an English mathematician known for work on Babbage's engine.",
  ];

  function stripDecor(s) {
    return String(s || "")
      .replace(/\s*\([^)]*\)/g, "")
      .replace(/\s*\[[^\]]*\]/g, "")
      .replace(/\b(listen|pronunciation)\b/gi, "")
      .replace(/\s{2,}/g, " ")
      .replace(/\s+([,.;:])/g, "$1")
      .trim();
  }

  function sentences(s) {
    var parts = String(s || "")
      .replace(/;/g, ".")
      .match(/[^.!?]+[.!?]+(?:\s|$)|[^.!?]+$/g);
    if (!parts) return s ? [s] : [];
    return parts
      .map(function (p) {
        return p.trim();
      })
      .filter(Boolean);
  }

  function applyPhrases(s) {
    var out = s;
    var i;
    for (i = 0; i < PHRASES.length; i++) {
      var pair = PHRASES[i];
      out = out.replace(new RegExp(pair[0].replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi"), pair[1]);
    }
    return out.replace(/\s{2,}/g, " ").trim();
  }

  function leadWithTitle(s, title) {
    var t = String(title || "").trim();
    if (!t || t.length < 2) return s;
    var first = sentences(s)[0] || s;
    var comma = first.indexOf(",");
    var isPos = first.search(/\b(is|are|was|were)\b/i);
    if (comma > 8 && isPos > comma && isPos < 90) {
      var rest = first.slice(isPos).trim();
      return t + " " + rest.charAt(0).toLowerCase() + rest.slice(1);
    }
    return s;
  }

  function firstSentence(s) {
    var parts = sentences(s);
    return parts[0] || s;
  }

  function clip(s, max) {
    max = max || MAX_CHARS;
    if (s.length <= max) return s;
    var cut = s.slice(0, max - 1);
    var sp = cut.lastIndexOf(" ");
    if (sp > max * 0.55) cut = cut.slice(0, sp);
    return cut.replace(/[,;:\s]+$/, "") + ".";
  }

  function tokens(s) {
    return ("<s> " + String(s || "").toLowerCase().replace(/[^a-z0-9'\s]/g, " ") + " </s>")
      .split(/\s+/)
      .filter(Boolean);
  }

  function emptyModel() {
    return { uni: {}, bi: {}, n: 0 };
  }

  function observe(model, text, times) {
    var reps = times || 1;
    var r;
    for (r = 0; r < reps; r++) {
      var toks = tokens(text);
      var i;
      for (i = 0; i < toks.length; i++) {
        model.uni[toks[i]] = (model.uni[toks[i]] || 0) + 1;
        model.n += 1;
        if (i) {
          var k = toks[i - 1] + "\t" + toks[i];
          model.bi[k] = (model.bi[k] || 0) + 1;
        }
      }
    }
    return model;
  }

  function train(texts) {
    var model = emptyModel();
    (texts || []).forEach(function (t) {
      observe(model, t, 1);
    });
    return model;
  }

  var baseModel = null;
  function pretrained() {
    if (!baseModel) baseModel = train(SEED);
    return baseModel;
  }

  function logprob(model, text) {
    var toks = tokens(text);
    if (toks.length < 2) return -99;
    var lp = 0;
    var i;
    for (i = 1; i < toks.length; i++) {
      var prev = toks[i - 1];
      var cur = toks[i];
      var b = model.bi[prev + "\t" + cur] || 0;
      var u = model.uni[prev] || 0;
      lp += Math.log((b + 0.35) / (u + 0.35 * 4000));
    }
    return lp / toks.length;
  }

  function mentionsTopic(text, title) {
    var t = String(title || "")
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, " ")
      .trim()
      .split(/\s+/)
      .filter(function (w) {
        return w.length > 2;
      });
    if (!t.length) return true;
    var low = String(text || "").toLowerCase();
    return t.some(function (w) {
      return low.indexOf(w) !== -1;
    });
  }

  function candidates(extract, title) {
    var clean = applyPhrases(stripDecor(extract)).replace(/;/g, ".");
    clean = clean.replace(/^for other uses,.+$/im, "").trim();
    var titled = applyPhrases(leadWithTitle(clean, title));
    var one = firstSentence(titled);
    var two = sentences(titled).slice(0, 2).join(" ");
    var out = [one, two !== one ? two : ""];
    if (one.length > MAX_CHARS) out.push(clip(one, MAX_CHARS));
    return out.filter(function (s, i, arr) {
      return s && s.length >= MIN_CHARS && arr.indexOf(s) === i;
    });
  }

  function scoreCandidate(model, text, title) {
    var s = logprob(model, text);
    var n = text.length;
    var nSent = sentences(text).length;
    if (n < 40) s -= 0.6;
    if (n > MAX_CHARS) s -= 1.4 + (n - MAX_CHARS) / 90;
    else if (n > 170) s -= 0.45;
    else if (n >= 48 && n <= 140) s += 0.45;
    if (nSent === 1 && n >= 40 && n <= 140) s += 0.55;
    if (nSent > 1) s -= 0.25;
    if (!mentionsTopic(text, title)) s -= 1.6;
    if (/\([^)]{12,}\)/.test(text)) s -= 0.5;
    return s;
  }

  function fitOnPage(extract) {
    var model = train(SEED);
    observe(model, extract, 3);
    return model;
  }

  function shorten(extract, title) {
    var raw = String(extract || "").replace(/\s+/g, " ").trim();
    if (!raw) return "";
    var model = fitOnPage(raw);
    var opts = candidates(raw, title);
    if (!opts.length) {
      var fallback = clip(applyPhrases(stripDecor(raw)), MAX_CHARS);
      return fallback;
    }
    var best = opts[0];
    var bestS = -Infinity;
    var i;
    for (i = 0; i < opts.length; i++) {
      var sc = scoreCandidate(model, opts[i], title);
      if (sc > bestS) {
        bestS = sc;
        best = opts[i];
      }
    }
    if (best.length > MAX_CHARS) best = clip(best, MAX_CHARS);
    return best;
  }

  var api = {
    shorten: shorten,
    stripDecor: stripDecor,
    applyPhrases: applyPhrases,
    candidates: candidates,
    train: train,
    MAX_CHARS: MAX_CHARS,
  };

  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.PyxAssistantWikiRephrase = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
