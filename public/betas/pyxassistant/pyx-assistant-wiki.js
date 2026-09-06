/**
 * Pyx Assistant — high-confidence Wikipedia lookup for KB misses.
 * Only answers when a page title clearly matches the asked topic.
 */
(function (root) {
  "use strict";

  var API = "https://en.wikipedia.org/w/api.php";
  var REST = "https://en.wikipedia.org/api/rest_v1/page/summary/";

  function norm(s) {
    return String(s || "")
      .toLowerCase()
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .replace(/['’]/g, "")
      .replace(/[^a-z0-9\s-]/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  function stripNoise(s) {
    return norm(s)
      .replace(
        /^(what|who|where|when|why|how|whats|whos|wheres|whens|tell me about|tell me|explain|define|describe|lookup|look up|search|wiki|wikipedia)\b/g,
        ""
      )
      .replace(/\b(is|are|was|were|the|a|an|of|for|about|please|mean|means|meaning)\b/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  function looksWikiWorthy(text) {
    var raw = String(text || "").trim();
    if (raw.length < 6 || raw.length > 140) return false;
    var low = raw.toLowerCase();
    if (
      /\b(joke|laugh|funny|riddle|poem|haiku|song|weather|forecast|score|scores|standings|calculate|math|hello|hi\b|hey\b|thanks|thank you|how are you|my name|forget me)\b/.test(
        low
      )
    ) {
      return false;
    }
    if (
      /\b(what(?:'s| is| are| was| were)|who(?:'s| is| was| were| are)|where(?:'s| is| was)|when(?:'s| was)|tell me about|explain|define|what does .+ mean)\b/i.test(
        raw
      )
    ) {
      return true;
    }
    // Short proper-noun style: "Ada Lovelace?" / "photosynthesis"
    var topic = stripNoise(raw);
    return topic.length >= 4 && topic.split(" ").length <= 6;
  }

  function extractTopic(text) {
    var raw = String(text || "").trim();
    var m =
      raw.match(/\b(?:what(?:'s| is| are| was| were)|who(?:'s| is| was| were| are)|where(?:'s| is| was)|when(?:'s| was))\s+(.+)$/i) ||
      raw.match(/\b(?:tell me about|explain|define|look up|lookup)\s+(.+)$/i) ||
      raw.match(/\bwhat does\s+(.+?)\s+mean\b/i);
    var topic = m ? m[1] : raw;
    topic = topic
      .replace(/[?.!,]+$/g, "")
      .replace(/\b(please|in simple terms|simply|briefly)\b/gi, "")
      .trim();
    return stripNoise(topic) || stripNoise(raw);
  }

  function titleMatchScore(topic, title) {
    var q = norm(topic);
    var t = norm(title);
    if (!q || q.length < 3 || !t) return 0;
    if (q === t) return 1;
    // plural/singular soft match
    if (q + "s" === t || q === t + "s" || q + "es" === t || q === t + "es") return 0.98;
    // exact phrase as whole title start
    if (t === q || t.indexOf(q + " ") === 0 || t.indexOf(q + ",") === 0) return 0.97;
    if (q.indexOf(t) === 0 && t.length >= 5 && (q.length === t.length || q[t.length] === " ")) return 0.95;
    // Every significant query token must appear in the title, and title can't be much longer fluff
    var qTok = q.split(" ").filter(function (w) {
      return w.length > 2;
    });
    var tTok = t.split(" ").filter(function (w) {
      return w.length > 2;
    });
    if (!qTok.length) return 0;
    var allIn = qTok.every(function (w) {
      return t.indexOf(w) !== -1;
    });
    if (!allIn) return 0;
    // Require near-complete title coverage — reject "X (disambiguation)" style extras unless query has them
    if (tTok.length - qTok.length > 1) return 0;
    if (qTok.length >= 2 && allIn) return 0.92;
    // Single-token queries: title must be that token (or plural), already handled above
    if (qTok.length === 1 && tTok.length === 1 && qTok[0] === tTok[0]) return 1;
    return 0;
  }

  function firstSentences(extract, maxN) {
    var s = String(extract || "")
      .replace(/\s+/g, " ")
      .trim();
    if (!s) return "";
    var parts = s.match(/[^.!?]+[.!?]+/g);
    if (!parts || !parts.length) {
      return s.length > 220 ? s.slice(0, 217).trim() + "…" : s;
    }
    return parts
      .slice(0, maxN || 2)
      .map(function (p) {
        return p.trim();
      })
      .join(" ")
      .trim();
  }

  async function searchTitles(topic) {
    var url =
      API +
      "?action=opensearch&search=" +
      encodeURIComponent(topic) +
      "&limit=5&namespace=0&format=json&origin=*";
    var res = await fetch(url);
    if (!res.ok) throw new Error("wiki search " + res.status);
    var data = await res.json();
    return (data && data[1]) || [];
  }

  async function summaryForTitle(title) {
    var url = REST + encodeURIComponent(title.replace(/ /g, "_")) + "?redirect=true";
    var res = await fetch(url, { headers: { Accept: "application/json" } });
    if (!res.ok) throw new Error("wiki summary " + res.status);
    return res.json();
  }

  /**
   * @returns {Promise<null|{reply:string,title:string,score:number}>}
   */
  async function answer(text) {
    if (!looksWikiWorthy(text)) return null;
    var topic = extractTopic(text);
    if (!topic || topic.length < 3) return null;

    var titles = await searchTitles(topic);
    if (!titles.length) return null;

    var best = null;
    var i;
    for (i = 0; i < titles.length; i++) {
      var score = titleMatchScore(topic, titles[i]);
      // Strict: only near-certain title connection
      if (score < 0.92) continue;
      if (!best || score > best.score) best = { title: titles[i], score: score };
      if (score >= 0.97) break;
    }
    if (!best) return null;

    var sum = await summaryForTitle(best.title);
    if (!sum || sum.type === "disambiguation" || sum.type === "notfound") return null;
    // Confirm the resolved page title still matches (redirects can drift)
    var resolvedTitle = (sum.title || best.title || "").trim();
    var resolvedScore = titleMatchScore(topic, resolvedTitle);
    if (resolvedScore < 0.92) return null;

    var blurb = firstSentences(sum.extract || sum.description || "", 2);
    if (!blurb || blurb.length < 40) return null;

    return {
      reply: "From Wikipedia — " + resolvedTitle + ": " + blurb,
      title: resolvedTitle,
      score: resolvedScore,
    };
  }

  var api = {
    looksWikiWorthy: looksWikiWorthy,
    extractTopic: extractTopic,
    titleMatchScore: titleMatchScore,
    firstSentences: firstSentences,
    answer: answer,
  };

  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.PyxAssistantWiki = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
