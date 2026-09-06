/**
 * Pyx Assistant — spoken language understanding (SLU).
 *
 * Modular pipeline: normalize → intent + slots → local handler
 * (UI may call optional MARII cloud boost on low confidence).
 * Pattern-based NLU (Web Speech / typed text is the speech front-end).
 */
(function (root) {
  "use strict";

  var THEMES = ["aurora", "blush", "mint", "twilight", "peach", "frost", "calm-contrast"];

  var LANG_ALIASES = {
    english: "en",
    en: "en",
    spanish: "es",
    español: "es",
    espanol: "es",
    es: "es",
    french: "fr",
    français: "fr",
    francais: "fr",
    fr: "fr",
    german: "de",
    deutsch: "de",
    de: "de",
    japanese: "ja",
    日本語: "ja",
    ja: "ja",
    chinese: "zh",
    中文: "zh",
    mandarin: "zh",
    zh: "zh",
  };

  var RULES = [
    {
      intent: "greet",
      re: /^(hi|hello|hey|yo|sup|howdy|hola|bonjour|salut|hallo|guten tag|こんにちは|你好|hey pyx|hi pyx)\b/i,
    },
    {
      intent: "farewell",
      re: /^(bye|goodbye|good night|see you|adios|ciao|au revoir|tschüss|さようなら|再见)\b/i,
    },
    {
      intent: "identity",
      re: /\b(who are you|what('?s| is) your name|what are you|quién eres|qui es[- ]tu|wer bist du|あなたは誰|你是谁)\b/i,
    },
    {
      intent: "marii",
      re: /\b(what('?s| is)|who is|explain|about)\s+(marii|m\.?a\.?r\.?i\.?i)\b|\bmarii\b/i,
    },
    {
      intent: "mi",
      re: /\b(what('?s| is)|about)\s+(mainline intelligence|mi moderator)\b|mainline intelligence\b|\bmi moderator\b/i,
    },
    {
      intent: "beta",
      re: /\b(is this a beta|are you a beta|early (version|beta)|still (being )?improv)/i,
    },
    {
      intent: "help",
      re: /\b(help|what can you do|how do (i|you) use|ayuda|aide|hilfe|ヘルプ|帮助)\b/i,
    },
    {
      intent: "time",
      re: /\b(what time|what'?s the time|current time|qué hora|quelle heure|wie spät|今何時|几点|now time)\b/i,
    },
    {
      intent: "date",
      re: /\b(what('?s| is) (the )?date|what day|today'?s date|qué fecha|quelle date|welches datum|今日|几号)\b/i,
    },
    {
      intent: "weather",
      re: /\b(weather|forecast|temperature|lluvia|météo|wetter|天気|天气)\b/i,
    },
    {
      intent: "theme",
      re: /\b(theme|tema|thème|テーマ|主题|switch to|change (the )?theme|use (the )?(aurora|blush|mint|twilight|peach|frost|calm[\s-]?contrast))\b/i,
    },
    {
      intent: "language",
      re: /\b(speak|talk|switch to|change language|idioma|langue|sprache|言語|语言)\b.+\b(english|spanish|español|french|français|german|deutsch|japanese|chinese|中文|日本語)\b/i,
    },
    {
      intent: "settings",
      re: /\b(open settings|show settings|ajustes|réglages|einstellungen|設定|设置)\b/i,
    },
    {
      intent: "data",
      re: /\b(show (my )?data|open data|what do you know about me|what have you learned|what you remember about me|my data)\b/i,
    },
    {
      intent: "clear",
      re: /\b(clear (chat|conversation|history)|reset chat|borra|efface|lösche|消去|清空)\b/i,
    },
    {
      intent: "open_talk",
      re: /\b(open (pyx )?talk|switch to talk|abrir talk)\b/i,
    },
    {
      intent: "open_studio",
      re: /\b(open (pyx )?studio|go (to|home)|abrir studio)\b/i,
    },
    {
      intent: "share",
      re: /\b(share|copy (this|that|conversation)|compartir|partager|teilen)\b/i,
    },
    {
      intent: "slack_send",
      re: /\b(send (this |that |it )?to slack|post to slack)\b/i,
    },
    {
      intent: "discord_send",
      re: /\b(send (this |that |it )?to discord|post to discord)\b/i,
    },
    {
      intent: "calculator",
      re: /\b(what('?s| is)|calculate|compute|cuánto es|combien fait|was ist|計算|等于)\b.{0,40}[\d][\d\s+\-*/x×÷.^()]+/i,
    },
    {
      intent: "joke",
      re: /\b(tell me a joke|make me laugh|another joke|got a joke|chiste|blague|witz|冗談|笑话)\b/i,
    },
    {
      intent: "fact",
      re: /\b(fun fact|random fact|tell me a fact|something interesting|another fact)\b/i,
    },
    {
      intent: "riddle",
      re: /\b(tell me a riddle|another riddle|give me a riddle|riddle me)\b/i,
    },
    {
      intent: "quote",
      re: /\b(tell me a quote|inspire me|another quote|a quote)\b/i,
    },
    {
      intent: "compliment",
      re: /\b(compliment me|say something nice|encourage me|be nice)\b/i,
    },
    {
      intent: "repeat",
      re: /\b(repeat that|say that again|what did you say)\b/i,
    },
    {
      intent: "sports",
      re: /\b(mlb|nba|nfl|nhl|wnba|mls|soccer|baseball|football|basketball|hockey|batting average|\bops\b|\bera\b|standings|scoreboard|home runs?|nl central|al east|who('s| is) leading|how('s| is) \w+ (hitting|doing|pitching)|vs\.? \w+)\b/i,
    },
  ];

  var GOLDEN = [
    { text: "hey pyx", intent: "greet" },
    { text: "hello", intent: "greet" },
    { text: "hola", intent: "greet" },
    { text: "who are you", intent: "identity" },
    { text: "what can you do", intent: "help" },
    { text: "what time is it", intent: "time" },
    { text: "what's the date today", intent: "date" },
    { text: "what's the weather in austin", intent: "weather" },
    { text: "switch to mint", intent: "theme" },
    { text: "change the theme to blush", intent: "theme" },
    { text: "switch to calm contrast", intent: "theme" },
    { text: "speak Spanish", intent: "language" },
    { text: "open settings", intent: "settings" },
    { text: "show my data", intent: "data" },
    { text: "clear conversation", intent: "clear" },
    { text: "open pyx talk", intent: "open_talk" },
    { text: "open studio", intent: "open_studio" },
    { text: "share this", intent: "share" },
    { text: "send this to slack", intent: "slack_send" },
    { text: "post to discord", intent: "discord_send" },
    { text: "what's 12 times 4", intent: "calculator" },
    { text: "calculate 8 + 2 * 3", intent: "calculator" },
    { text: "tell me a joke", intent: "joke" },
    { text: "fun fact", intent: "fact" },
    { text: "tell me a riddle", intent: "riddle" },
    { text: "inspire me", intent: "quote" },
    { text: "how's Ohtani doing", intent: "sports" },
    { text: "mlb scores", intent: "sports" },
    { text: "who's leading the nl central", intent: "sports" },
    { text: "nba scores", intent: "sports" },
    { text: "compliment me", intent: "compliment" },
    { text: "say that again", intent: "repeat" },
    { text: "goodbye", intent: "farewell" },
    { text: "explain photosynthesis in simple terms", intent: "chat" },
    { text: "write a haiku about rain", intent: "chat" },
  ];

  function normalize(text) {
    return String(text || "")
      .replace(/\s+/g, " ")
      .trim();
  }

  function extractTheme(text) {
    var n = normalize(text).toLowerCase();
    if (/\bcalm[\s-]+contrast\b/.test(n) || /\bcalmcontrast\b/.test(n)) return "calm-contrast";
    var i;
    for (i = 0; i < THEMES.length; i++) {
      if (n.indexOf(THEMES[i]) !== -1) return THEMES[i];
    }
    return null;
  }

  function extractLang(text) {
    var n = normalize(text).toLowerCase();
    var key;
    for (key in LANG_ALIASES) {
      if (Object.prototype.hasOwnProperty.call(LANG_ALIASES, key) && n.indexOf(key) !== -1) {
        return LANG_ALIASES[key];
      }
    }
    return null;
  }

  function extractMath(text) {
    var n = normalize(text)
      .replace(/×|x/gi, "*")
      .replace(/÷/g, "/")
      .replace(/,/g, "");
    var m = n.match(/(-?\d+(?:\.\d+)?(?:\s*[+\-*/^()]\s*-?\d+(?:\.\d+)?)*)/);
    return m ? m[1].replace(/\s+/g, "") : null;
  }

  function safeEvalMath(expr) {
    if (!expr || !/^[-+*/^().\d]+$/.test(expr)) return null;
    if (expr.length > 80) return null;
    var js = expr.replace(/\^/g, "**");
    try {
      var fn = new Function("return (" + js + ")");
      var v = fn();
      if (typeof v !== "number" || !isFinite(v)) return null;
      return Math.round(v * 1e8) / 1e8;
    } catch (e) {
      return null;
    }
  }

  function classify(text) {
    var utterance = normalize(text);
    if (!utterance) {
      return { intent: "empty", confidence: 0, slots: {}, utterance: "" };
    }
    var i;
    var rule;
    for (i = 0; i < RULES.length; i++) {
      rule = RULES[i];
      if (rule.re.test(utterance)) {
        var slots = {};
        if (rule.intent === "theme") slots.theme = extractTheme(utterance);
        if (rule.intent === "language") slots.lang = extractLang(utterance);
        if (rule.intent === "calculator") {
          slots.expr = extractMath(utterance);
          slots.value = safeEvalMath(slots.expr);
        }
        if (rule.intent === "weather") {
          var loc = utterance.replace(rule.re, "").replace(/\b(in|for|à|en|in der)\b/gi, " ").trim();
          if (loc) slots.location = loc;
        }
        return {
          intent: rule.intent,
          confidence: 0.86,
          slots: slots,
          utterance: utterance,
        };
      }
    }
    return { intent: "chat", confidence: 0.45, slots: {}, utterance: utterance };
  }

  function formatTime(lang) {
    try {
      return new Intl.DateTimeFormat(lang || "en", {
        hour: "numeric",
        minute: "2-digit",
      }).format(new Date());
    } catch (e) {
      return new Date().toLocaleTimeString();
    }
  }

  function formatDate(lang) {
    try {
      return new Intl.DateTimeFormat(lang || "en", {
        weekday: "long",
        month: "long",
        day: "numeric",
        year: "numeric",
      }).format(new Date());
    } catch (e) {
      return new Date().toLocaleDateString();
    }
  }

  /**
   * Local handlers return a reply string, or leave reply null for the UI
   * to try KB / live data / optional MARII boost.
   * `t` is i18n.t(lang, key). Actions are side-effect flags for the UI.
   */
  function resolve(result, opts) {
    opts = opts || {};
    var lang = opts.lang || "en";
    var t = opts.t;
    var intent = result.intent;
    var out = { reply: null, action: null, useLlm: false, useWeb: false, special: null, confidence: 1 };

    switch (intent) {
      case "empty":
        out.reply = "";
        return out;
      case "greet":
        out.reply = t ? t(lang, "greeting") : "Hi, I’m Pyx.";
        return out;
      case "farewell":
        out.reply = lang === "es" ? "Hasta luego." : lang === "fr" ? "À bientôt." : lang === "de" ? "Tschüss." : lang === "ja" ? "またね。" : lang === "zh" ? "再见。" : "See you.";
        return out;
      case "identity":
        out.reply = t ? t(lang, "identity") : "I’m Pyx Assistant.";
        return out;
      case "marii":
        out.reply =
          "MARII is Mainline Artificial Realtime Instant Intelligence — instant when it matters, realtime when the world moves. I’m the first public beta. Extremely early and still being improved.";
        return out;
      case "mi":
        out.reply =
          "Mainline Intelligence (MI) is the umbrella for the new wave of Pyx — moderator, MARII, MCI, and more. Home: https://pyx-ai.web.app/mainlineintelligence";
        return out;
      case "beta":
        out.reply =
          "Yes — this is an extremely early MARII beta and it’s still being improved. Rough edges are expected. Thanks for trying it.";
        return out;
      case "help":
        out.reply = t ? t(lang, "help") : "Ask me anything.";
        return out;
      case "time":
        out.reply = (t ? t(lang, "timePrefix") : "It’s") + " " + formatTime(lang) + ".";
        return out;
      case "date":
        out.reply = (t ? t(lang, "datePrefix") : "Today is") + " " + formatDate(lang) + ".";
        return out;
      case "theme":
        if (result.slots.theme) {
          out.action = { type: "theme", theme: result.slots.theme };
          out.reply = (t ? t(lang, "themeSet") : "Theme set to") + " " + String(result.slots.theme).replace(/-/g, " ") + ".";
        } else {
          out.action = { type: "settings" };
          out.reply = t ? t(lang, "help") : "Pick a theme in settings.";
        }
        return out;
      case "language":
        if (result.slots.lang) {
          out.action = { type: "language", lang: result.slots.lang };
          out.reply = (t ? t(lang, "langSet") : "Okay — I’ll use") + " " + result.slots.lang + ".";
        } else {
          out.action = { type: "settings" };
          out.reply = t ? t(lang, "help") : "Pick a language in settings.";
        }
        return out;
      case "settings":
        out.action = { type: "settings" };
        out.reply = t ? t(lang, "settings") : "Settings";
        return out;
      case "data":
        out.action = { type: "data" };
        out.reply = t ? t(lang, "dataOpen") : "Here’s what I remember about you.";
        return out;
      case "clear":
        out.action = { type: "clear" };
        out.reply = t ? t(lang, "cleared") : "Conversation cleared.";
        return out;
      case "open_talk":
        out.action = { type: "open_talk" };
        out.reply = t ? t(lang, "openTalk") : "Opening Pyx Talk.";
        return out;
      case "open_studio":
        out.action = { type: "open_studio" };
        out.reply = "Studio";
        return out;
      case "share":
        out.action = { type: "share" };
        out.reply = t ? t(lang, "copied") : "Copied.";
        return out;
      case "slack_send":
        out.action = { type: "slack" };
        out.reply = t ? t(lang, "sendSlack") : "Slack";
        return out;
      case "discord_send":
        out.action = { type: "discord" };
        out.reply = t ? t(lang, "sendDiscord") : "Discord";
        return out;
      case "calculator":
        if (result.slots.value != null) {
          out.reply = (t ? t(lang, "calcPrefix") : "That’s") + " " + String(result.slots.value) + ". =)";
          return out;
        }
        out.confidence = 0.4;
        return out;
      case "weather":
        out.useWeb = true;
        out.confidence = 0.55;
        return out;
      case "joke":
        out.special = "__JOKE__";
        return out;
      case "fact":
        out.special = "__FACT__";
        return out;
      case "riddle":
        out.special = "__RIDDLE__";
        return out;
      case "quote":
        out.special = "__QUOTE__";
        return out;
      case "compliment":
        out.special = "__COMPLIMENT__";
        return out;
      case "repeat":
        out.special = "__REPEAT__";
        return out;
      case "sports":
        out.confidence = 0.55;
        return out;
      case "chat":
        out.confidence = 0.35;
        return out;
      default:
        out.confidence = 0.3;
        return out;
    }
  }

  function evaluateGolden() {
    var i;
    var hit = 0;
    var misses = [];
    for (i = 0; i < GOLDEN.length; i++) {
      var g = GOLDEN[i];
      var got = classify(g.text).intent;
      if (got === g.intent) hit += 1;
      else misses.push({ text: g.text, expected: g.intent, got: got });
    }
    return {
      total: GOLDEN.length,
      hit: hit,
      accuracy: GOLDEN.length ? hit / GOLDEN.length : 0,
      misses: misses,
    };
  }

  var api = {
    THEMES: THEMES,
    RULES: RULES,
    GOLDEN: GOLDEN,
    normalize: normalize,
    classify: classify,
    resolve: resolve,
    extractTheme: extractTheme,
    extractLang: extractLang,
    extractMath: extractMath,
    safeEvalMath: safeEvalMath,
    evaluateGolden: evaluateGolden,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
  root.PyxAssistantSLU = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
