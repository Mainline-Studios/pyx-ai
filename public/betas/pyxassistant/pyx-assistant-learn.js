/**
 * Pyx Assistant — on-device preference model (online softmax / SGD)
 * plus a light profile of name, likes, and dislikes.
 */
(function (root) {
  "use strict";

  var CLASSES = ["joke", "math", "fact", "riddle", "quote", "trivia", "howto", "talk"];
  var BUCKETS = 48;
  var LEX = [
    "joke", "funny", "laugh", "math", "plus", "percent", "fact", "science",
    "riddle", "quote", "trivia", "how", "like", "love", "hate",
  ];
  var DIM = BUCKETS + LEX.length;
  var LR = 0.12;
  var L2 = 0.0008;

  var profile = {
    name: "",
    likes: [],
    dislikes: [],
    seen: 0,
    lastKind: "talk",
  };
  var weights = [];

  function zeroWeights() {
    weights = [];
    var k, d;
    for (k = 0; k < CLASSES.length; k++) {
      weights[k] = [];
      for (d = 0; d < DIM; d++) weights[k][d] = 0;
    }
  }
  zeroWeights();

  function hash(s) {
    var h = 2166136261;
    var i;
    for (i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return (h >>> 0) % BUCKETS;
  }

  function tokens(text) {
    return String(text || "")
      .toLowerCase()
      .replace(/['’]/g, "")
      .replace(/[^a-z0-9\s]/g, " ")
      .split(/\s+/)
      .filter(function (w) {
        return w.length > 1;
      });
  }

  function features(text) {
    var x = [];
    var i;
    for (i = 0; i < DIM; i++) x[i] = 0;
    var toks = tokens(text);
    toks.forEach(function (w) {
      x[hash(w)] += 1;
    });
    var i2;
    for (i2 = 0; i2 < toks.length - 1; i2++) {
      x[hash(toks[i2] + "_" + toks[i2 + 1])] += 0.7;
    }
    var low = String(text || "").toLowerCase();
    LEX.forEach(function (w, idx) {
      if (low.indexOf(w) !== -1) x[BUCKETS + idx] = 1;
    });
    var norm = 0;
    for (i = 0; i < DIM; i++) norm += x[i] * x[i];
    norm = Math.sqrt(norm) || 1;
    for (i = 0; i < DIM; i++) x[i] /= norm;
    return x;
  }

  function softmax(scores) {
    var m = scores[0];
    var i;
    for (i = 1; i < scores.length; i++) if (scores[i] > m) m = scores[i];
    var ex = [];
    var sum = 0;
    for (i = 0; i < scores.length; i++) {
      ex[i] = Math.exp(scores[i] - m);
      sum += ex[i];
    }
    for (i = 0; i < ex.length; i++) ex[i] /= sum || 1;
    return ex;
  }

  function scoresFor(x) {
    var s = [];
    var k, d;
    for (k = 0; k < CLASSES.length; k++) {
      var acc = 0;
      for (d = 0; d < DIM; d++) acc += weights[k][d] * x[d];
      s[k] = acc;
    }
    return s;
  }

  function predict(text) {
    var p = softmax(scoresFor(features(text)));
    var map = {};
    CLASSES.forEach(function (c, i) {
      map[c] = p[i];
    });
    return map;
  }

  function topKind(text) {
    var p = predict(text);
    var best = CLASSES[0];
    var i;
    for (i = 1; i < CLASSES.length; i++) {
      if (p[CLASSES[i]] > p[best]) best = CLASSES[i];
    }
    return best;
  }

  function observe(text, kind) {
    var y = CLASSES.indexOf(kind);
    if (y < 0) y = CLASSES.indexOf("talk");
    var x = features(text);
    var p = softmax(scoresFor(x));
    var k, d;
    for (k = 0; k < CLASSES.length; k++) {
      var err = (k === y ? 1 : 0) - p[k];
      for (d = 0; d < DIM; d++) {
        weights[k][d] += LR * err * x[d] - LR * L2 * weights[k][d];
      }
    }
    profile.seen += 1;
    profile.lastKind = CLASSES[y];
    return CLASSES[y];
  }

  function observeFeedback(text) {
    var low = String(text || "").toLowerCase();
    if (/\b(haha|lol|lmao|funny|nice|love that|another|more|again|good one)\b/.test(low)) {
      observe(text + " " + profile.lastKind, profile.lastKind);
      return "pos";
    }
    if (/\b(boring|not funny|meh|stop|whatever|don't like|hate that)\b/.test(low)) {
      observe(text, "talk");
      return "neg";
    }
    return null;
  }

  function kindFromIntent(intent, recKind) {
    if (recKind && CLASSES.indexOf(recKind) !== -1) return recKind;
    switch (intent) {
      case "joke":
        return "joke";
      case "calculator":
        return "math";
      case "fact":
        return "fact";
      case "riddle":
        return "riddle";
      case "quote":
        return "quote";
      case "trivia":
        return "trivia";
      case "help":
        return "howto";
      case "greet":
      case "farewell":
      case "identity":
      case "time":
      case "date":
      case "weather":
      case "theme":
      case "language":
      case "settings":
      case "clear":
      case "open_talk":
      case "open_studio":
      case "share":
      case "slack_send":
      case "discord_send":
      case "compliment":
      case "repeat":
      case "chat":
      case "empty":
      case "data":
      case "sports":
        return "talk";
      default: {
        var unused = intent;
        void unused;
        return "talk";
      }
    }
  }

  function priorsFor(text) {
    var p = predict(text);
    profile.likes.forEach(function (like) {
      var k = topKind(like);
      p[k] = Math.min(1, (p[k] || 0) + 0.14);
    });
    return p;
  }

  function addUnique(arr, item, cap) {
    var v = String(item || "").replace(/\s+/g, " ").trim().slice(0, 40);
    if (!v || v.length < 2) return arr;
    var low = v.toLowerCase();
    if (["you", "it", "that", "this", "pyx", "jokes"].indexOf(low) !== -1 && v.length < 5) return arr;
    var next = arr.filter(function (x) {
      return x.toLowerCase() !== low;
    });
    next.unshift(v);
    return next.slice(0, cap || 8);
  }

  function ingest(text) {
    var raw = String(text || "").trim();
    var out = { reply: null, kind: null };
    var nameM = raw.match(/\b(?:my name is|i am|i'm|im|call me)\s+([A-Za-z][A-Za-z'-]{1,20})\b/i);
    if (nameM) {
      var n = nameM[1];
        if (!/^(pyx|good|fine|ok|okay|here|back|just|bored|hungry|tired|ready|sorry|not|so|still|really|very|too|also|actually|home|lost|well)$/i.test(n)) {
        profile.name = n.charAt(0).toUpperCase() + n.slice(1);
        out.reply = "Got it, " + profile.name + ". I’ll remember that on this device. =)";
        out.kind = "talk";
        return out;
      }
    }
    var likeM = raw.match(/\bi (?:like|love|enjoy|adore)\s+(.{2,48}?)(?:[.!?]|please|$)/i);
    if (likeM) {
      profile.likes = addUnique(profile.likes, likeM[1], 8);
      out.reply = "I’ll keep " + likeM[1].trim() + " in mind. Want something in that lane? =)";
      out.kind = topKind(likeM[1]);
      return out;
    }
    var hateM = raw.match(/\bi (?:hate|dislike|can't stand|dont like|don't like)\s+(.{2,48}?)(?:[.!?]|$)/i);
    if (hateM) {
      profile.dislikes = addUnique(profile.dislikes, hateM[1], 6);
      out.reply = "Okay — I’ll steer away from " + hateM[1].trim() + ".";
      out.kind = "talk";
      return out;
    }
    var whoM = /\b(what(?:'s| is) my name|who am i|do you know me|what do i like)\b/i.test(raw);
    if (whoM) {
      out.reply = summary();
      out.kind = "talk";
      return out;
    }
    return out;
  }

  function summary() {
    var bits = [];
    if (profile.name) bits.push("you’re " + profile.name);
    if (profile.likes.length) bits.push("you like " + profile.likes.slice(0, 3).join(", "));
    if (profile.dislikes.length) bits.push("you’d rather skip " + profile.dislikes[0]);
    var fav = favorite();
    if (profile.seen >= 3) bits.push("you’ve been leaning " + fav);
    if (!bits.length) return "I’m still getting to know you. Tell me your name or what you like. =)";
    return "From what I’ve learned on-device: " + bits.join("; ") + ". =)";
  }

  function favorite() {
    var p = predict("tell me something I would enjoy");
    var best = CLASSES[0];
    CLASSES.forEach(function (c) {
      if (p[c] > p[best]) best = c;
    });
    return best;
  }

  var KIND_LABELS = {
    joke: "Jokes and funny stuff",
    math: "Math and numbers",
    fact: "Fun facts",
    riddle: "Riddles",
    quote: "Quotes",
    trivia: "Trivia",
    howto: "How-tos and explainers",
    talk: "Just chatting",
  };

  function amountKey(score, maxScore, seen, spread) {
    if (seen < 2 || spread < 0.03) return "unknown";
    var rel = score / (maxScore || 1);
    if (rel > 0.92 && score > 0.16) return "favorite";
    if (rel > 0.7) return "often";
    if (rel > 0.45) return "sometimes";
    return "rare";
  }

  function explain() {
    var p = predict("tell me something I would enjoy");
    var ranked = CLASSES.map(function (c) {
      return { id: c, label: KIND_LABELS[c] || c, score: p[c] };
    }).sort(function (a, b) {
      return b.score - a.score;
    });
    var maxP = ranked[0].score;
    var minP = ranked[ranked.length - 1].score;
    var spread = maxP - minP;
    var patterned = profile.seen >= 2 && spread >= 0.03;
    ranked.forEach(function (row) {
      row.amount = amountKey(row.score, maxP, profile.seen, spread);
      row.bar = patterned ? Math.max(8, Math.round((row.score / maxP) * 100)) : 0;
    });
    return {
      name: profile.name || "",
      likes: profile.likes.slice(),
      dislikes: profile.dislikes.slice(),
      seen: profile.seen,
      lastKind: profile.lastKind,
      lastLabel: KIND_LABELS[profile.lastKind] || KIND_LABELS.talk,
      favoriteId: favorite(),
      favoriteLabel: KIND_LABELS[favorite()] || KIND_LABELS.talk,
      tastes: ranked,
      patterned: patterned,
      known: !!(profile.name || profile.likes.length || profile.dislikes.length || profile.seen),
    };
  }

  function greeting(fallback) {
    if (profile.name && profile.likes[0]) {
      return "Hey " + profile.name + " — still into " + profile.likes[0] + "? I’m here. =)";
    }
    if (profile.name) return "Hi " + profile.name + ". What are we doing? =)";
    if (profile.seen >= 4) {
      return "Welcome back. You usually like " + favorite() + "s — want one, or something new? =)";
    }
    return fallback || "Hi, I’m Pyx. =)";
  }

  function flavor(reply) {
    var s = String(reply || "");
    if (profile.name && profile.seen > 2 && s.indexOf(profile.name) === -1 && s.length < 180 && Math.random() < 0.22) {
      s = profile.name + " — " + s;
    }
    return s;
  }

  function pack() {
    var flat = [];
    weights.forEach(function (row) {
      row.forEach(function (v) {
        flat.push(Math.round(v * 1000) / 1000);
      });
    });
    return {
      v: 1,
      n: profile.name,
      likes: profile.likes,
      dislikes: profile.dislikes,
      seen: profile.seen,
      lastKind: profile.lastKind,
      w: flat,
    };
  }

  function unpack(o) {
    if (!o || o.v !== 1) return;
    profile.name = o.n || "";
    profile.likes = Array.isArray(o.likes) ? o.likes.slice(0, 8) : [];
    profile.dislikes = Array.isArray(o.dislikes) ? o.dislikes.slice(0, 6) : [];
    profile.seen = o.seen || 0;
    profile.lastKind = o.lastKind || "talk";
    if (Array.isArray(o.w) && o.w.length === CLASSES.length * DIM) {
      zeroWeights();
      var i = 0;
      var k, d;
      for (k = 0; k < CLASSES.length; k++) {
        for (d = 0; d < DIM; d++) {
          weights[k][d] = o.w[i++] || 0;
        }
      }
    }
  }

  function reset() {
    profile = { name: "", likes: [], dislikes: [], seen: 0, lastKind: "talk" };
    zeroWeights();
  }

  var api = {
    CLASSES: CLASSES,
    features: features,
    predict: predict,
    priorsFor: priorsFor,
    topKind: topKind,
    observe: observe,
    observeFeedback: observeFeedback,
    kindFromIntent: kindFromIntent,
    ingest: ingest,
    summary: summary,
    favorite: favorite,
    greeting: greeting,
    explain: explain,
    KIND_LABELS: KIND_LABELS,
    flavor: flavor,
    pack: pack,
    unpack: unpack,
    reset: reset,
    get profile() {
      return profile;
    },
  };

  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.PyxAssistantLearn = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
