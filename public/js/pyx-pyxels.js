/**
 * Pyxels — custom Pyx Talk personas (Gemini Gems–style). Stored in localStorage.
 * Note: "Pyxel" (singular) is the pixel-art app; Pyxels are chat assistants.
 */
(function (global) {
  "use strict";

  var STORE_KEY = "pyx.pyxels.v1";
  var ACTIVE_KEY = "pyx.talk.activePyxelId";
  var DISMISS_BANNER_KEY = "pyx.pyxels.bannerDismissed";

  var PREMADE = [
    {
      id: "premade-brainstorm",
      premade: true,
      name: "Brainstormer",
      emoji: "💡",
      description:
        "Sparks ideas for projects, stories, and creative briefs. Asks clarifying questions before going deep.",
      instructions:
        "You are Brainstormer, a creative Pyx assistant. Help the user generate many distinct ideas before narrowing down. "
        "Use short bullet lists, numbered options, and one follow-up question per turn. Stay upbeat and practical.",
    },
    {
      id: "premade-coding",
      premade: true,
      name: "Coding partner",
      emoji: "⌨️",
      description:
        "Pair-programming mindset: plans first, then code snippets, then how to test. Great for homework and side projects.",
      instructions:
        "You are Coding partner, a patient programming tutor. Prefer small steps: clarify requirements, outline an approach, "
        "then show fenced code with language tags. Mention edge cases and how to run or test. Never write malware or exploits.",
    },
    {
      id: "premade-career",
      premade: true,
      name: "Career guide",
      emoji: "🧭",
      description:
        "Resumes, interviews, portfolios, and next-step planning — concrete and encouraging.",
      instructions:
        "You are Career guide, a supportive coach for students and early-career builders. Give actionable advice on resumes, "
        "interviews, portfolios, and learning paths. Use bullet lists and examples. Avoid guaranteeing job outcomes.",
    },
    {
      id: "premade-chess",
      premade: true,
      name: "Chess champ",
      emoji: "♟️",
      experiment: true,
      description:
        "Explains moves in plain language, suggests practice puzzles, and keeps games kid-friendly.",
      instructions:
        "You are Chess champ, a friendly chess coach. Explain tactics simply, suggest one improvement per reply, and use "
        "algebraic notation when helpful. No gambling or harsh trash talk.",
    },
    {
      id: "premade-tutor",
      premade: true,
      name: "Study buddy",
      emoji: "📚",
      description:
        "Socratic tutoring: hints first, full answers only when asked. Good for math, science, and essays.",
      instructions:
        "You are Study buddy, a Socratic tutor. Guide with questions and hints; do not dump full solutions unless the user "
        "explicitly asks. Check understanding. Keep language appropriate for school-age learners.",
    },
    {
      id: "premade-story",
      premade: true,
      name: "Story spinner",
      emoji: "✨",
      description:
        "Co-writes short fiction, RPG scenes, and game lore — vivid but family-friendly.",
      instructions:
        "You are Story spinner, a collaborative fiction partner. Write in vivid but family-friendly prose. Offer choices "
        "when the plot branches. Keep paragraphs short unless the user wants a long scene.",
    },
    {
      id: "premade-game",
      premade: true,
      name: "Game designer",
      emoji: "🎮",
      description:
        "Mechanics, loops, and Roblox-style feature ideas — scoped for indie and classroom projects.",
      instructions:
        "You are Game designer, helping brainstorm game mechanics, loops, and features for indie or classroom games. "
        "Suggest scope-friendly MVPs. Mention playtesting and moderation when user-generated content appears.",
    },
    {
      id: "premade-moderator",
      premade: true,
      name: "Moderation coach",
      emoji: "🛡️",
      description:
        "Helps teams design kid-safe chat rules and test phrases against a ban line mindset.",
      instructions:
        "You are Moderation coach for kid-safe online communities. Help write clear community rules, explain why phrases "
        "might be harmful in context, and suggest kind alternatives. Align with context-aware filtering (same word can be "
        "safe or not depending on phrase). Do not generate slurs or harassment examples unless analyzing why they are banned.",
    },
  ];

  function uid() {
    return "custom-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 8);
  }

  function readStore() {
    try {
      var raw = localStorage.getItem(STORE_KEY);
      if (!raw) return { custom: [] };
      var j = JSON.parse(raw);
      if (!j || typeof j !== "object") return { custom: [] };
      if (!Array.isArray(j.custom)) j.custom = [];
      return j;
    } catch (e) {
      return { custom: [] };
    }
  }

  function writeStore(data) {
    localStorage.setItem(STORE_KEY, JSON.stringify({ custom: data.custom || [] }));
  }

  function listPremade() {
    return PREMADE.slice();
  }

  function listCustom() {
    return readStore().custom.slice();
  }

  function listAll() {
    return listPremade().concat(listCustom());
  }

  function getById(id) {
    if (!id) return null;
    var i;
    for (i = 0; i < PREMADE.length; i++) {
      if (PREMADE[i].id === id) return Object.assign({}, PREMADE[i]);
    }
    var customs = listCustom();
    for (i = 0; i < customs.length; i++) {
      if (customs[i].id === id) return Object.assign({}, customs[i]);
    }
    return null;
  }

  function saveCustom(entry) {
    var data = readStore();
    var now = new Date().toISOString();
    var item = {
      id: entry.id || uid(),
      premade: false,
      name: (entry.name || "Untitled Pyxel").trim().slice(0, 80),
      emoji: (entry.emoji || "✦").trim().slice(0, 8) || "✦",
      description: (entry.description || "").trim().slice(0, 280),
      instructions: (entry.instructions || "").trim().slice(0, 4000),
      updatedAt: now,
    };
    if (!item.instructions) {
      return { ok: false, error: "Instructions are required." };
    }
    var found = false;
    data.custom = data.custom.map(function (c) {
      if (c.id === item.id) {
        found = true;
        item.createdAt = c.createdAt || now;
        return item;
      }
      return c;
    });
    if (!found) {
      item.createdAt = now;
      data.custom.push(item);
    }
    writeStore(data);
    return { ok: true, item: item };
  }

  function removeCustom(id) {
    var data = readStore();
    data.custom = data.custom.filter(function (c) {
      return c.id !== id;
    });
    writeStore(data);
    if (getActiveId() === id) setActiveId(null);
  }

  function getActiveId() {
    try {
      return (localStorage.getItem(ACTIVE_KEY) || "").trim() || null;
    } catch (e) {
      return null;
    }
  }

  function setActiveId(id) {
    if (!id) {
      localStorage.removeItem(ACTIVE_KEY);
      return;
    }
    var p = getById(id);
    if (!p) return;
    localStorage.setItem(ACTIVE_KEY, id);
  }

  function getActive() {
    return getById(getActiveId());
  }

  function instructionsForApi() {
    var p = getActive();
    if (!p || !p.instructions) return "";
    return String(p.instructions).trim().slice(0, 4000);
  }

  function talkUrl(id) {
    var base = "/pyx-talk.html";
    if (!id) return base;
    return base + "?pyxel=" + encodeURIComponent(id);
  }

  function isBannerDismissed() {
    return localStorage.getItem(DISMISS_BANNER_KEY) === "1";
  }

  function dismissBanner() {
    localStorage.setItem(DISMISS_BANNER_KEY, "1");
  }

  function applyUrlParam() {
    try {
      var q = new URLSearchParams(global.location.search);
      var id = (q.get("pyxel") || "").trim();
      if (id) setActiveId(id);
    } catch (e) {
      /* ignore */
    }
  }

  global.PyxPyxels = {
    PREMADE: PREMADE,
    listPremade: listPremade,
    listCustom: listCustom,
    listAll: listAll,
    getById: getById,
    saveCustom: saveCustom,
    removeCustom: removeCustom,
    getActiveId: getActiveId,
    setActiveId: setActiveId,
    getActive: getActive,
    instructionsForApi: instructionsForApi,
    talkUrl: talkUrl,
    isBannerDismissed: isBannerDismissed,
    dismissBanner: dismissBanner,
    applyUrlParam: applyUrlParam,
  };
})(typeof window !== "undefined" ? window : globalThis);
