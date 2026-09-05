/**
 * Pyx Assistant — local KB + math + continuous on-device voice.
 */
(function () {
  "use strict";

  var STORE_KEY = "pyx.assistant.v2";
  var slu = window.PyxAssistantSLU;
  var i18n = window.PyxAssistantI18n;
  var math = window.PyxAssistantMath;
  var kb = window.PyxAssistantKB;
  var voice = null;

  var state = {
    theme: "aurora",
    lang: "en",
    voice: true,
    voiceId: "af_heart",
    slack: "",
    discord: "",
    messages: [],
    ui: "idle",
    session: false,
    warming: false,
  };

  var els = {};
  var swirlRaf = 0;
  var handleInFlight = false;

  function t(key) {
    return i18n.t(state.lang, key);
  }

  function load() {
    try {
      var raw = localStorage.getItem(STORE_KEY);
      if (!raw) return;
      var o = JSON.parse(raw);
      if (o.theme) state.theme = o.theme;
      if (o.lang) state.lang = o.lang;
      if (typeof o.voice === "boolean") state.voice = o.voice;
      if (o.voiceId) state.voiceId = o.voiceId;
      if (typeof o.slack === "string") state.slack = o.slack;
      if (typeof o.discord === "string") state.discord = o.discord;
      if (Array.isArray(o.messages)) state.messages = o.messages.slice(-40);
    } catch (e) {}
  }

  function save() {
    try {
      localStorage.setItem(
        STORE_KEY,
        JSON.stringify({
          theme: state.theme,
          lang: state.lang,
          voice: state.voice,
          voiceId: state.voiceId,
          slack: state.slack,
          discord: state.discord,
          messages: state.messages.slice(-40),
        })
      );
    } catch (e) {}
  }

  function lastAssistant() {
    var i;
    for (i = state.messages.length - 1; i >= 0; i--) {
      if (state.messages[i].role === "assistant") return state.messages[i].content;
    }
    return "";
  }

  function toast(msg) {
    els.toast.textContent = msg;
    els.toast.classList.add("is-on");
    clearTimeout(toast._t);
    toast._t = setTimeout(function () {
      els.toast.classList.remove("is-on");
    }, 2200);
  }

  function setUi(mode, extra) {
    state.ui = mode;
    els.orb.setAttribute("data-state", mode);
    els.orb.classList.toggle("is-session", !!state.session);
    var label = extra || t("hint");
    if (!extra) {
      if (mode === "listen") label = state.session ? t("listeningHold") : t("listening");
      else if (mode === "think") label = t("thinking");
      else if (mode === "speak") label = t("speakingTap");
      else if (mode === "warm") label = extra || t("warming");
      else if (state.session) label = t("listeningHold");
    }
    els.status.textContent = label;
    els.send.disabled = mode === "think";
  }

  function applyTheme(theme) {
    state.theme = slu.THEMES.indexOf(theme) !== -1 ? theme : "aurora";
    document.documentElement.setAttribute("data-theme", state.theme);
    if (els.theme) els.theme.value = state.theme;
    save();
  }

  function applyLang(lang) {
    state.lang = i18n.LOCALES[lang] ? lang : "en";
    document.documentElement.lang = state.lang;
    if (els.lang) els.lang.value = state.lang;
    paintChrome();
    save();
  }

  function paintChrome() {
    els.back.textContent = t("back");
    els.input.placeholder = t("placeholder");
    els.send.textContent = t("send");
    els.settingsTitle.textContent = t("settings");
    els.historyTitle.textContent = t("history");
    els.themeLabel.textContent = t("theme");
    els.langLabel.textContent = t("language");
    els.voiceLabel.textContent = t("voiceReplies");
    els.modeLabel.textContent = t("voiceName");
    els.slackLabel.textContent = t("slackWebhook");
    els.discordLabel.textContent = t("discordWebhook");
    els.sendSlack.textContent = t("sendSlack");
    els.sendDiscord.textContent = t("sendDiscord");
    if (els.openTalk) els.openTalk.textContent = t("stayLocal");
    els.clearBtn.textContent = t("clear");
    if (!state.messages.length) {
      els.reply.textContent = t("greeting");
      els.userLine.textContent = "";
      els.status.textContent = t("hint");
    }
    renderHistory();
  }

  function renderHistory() {
    els.history.innerHTML = "";
    if (!state.messages.length) {
      var empty = document.createElement("li");
      empty.textContent = t("emptyHistory");
      els.history.appendChild(empty);
      return;
    }
    state.messages.forEach(function (m) {
      var li = document.createElement("li");
      var role = document.createElement("span");
      role.className = "role";
      role.textContent = m.role;
      var p = document.createElement("div");
      p.textContent = m.content;
      li.appendChild(role);
      li.appendChild(p);
      els.history.appendChild(li);
    });
    var lastUser = "";
    var lastAsst = t("greeting");
    state.messages.forEach(function (m) {
      if (m.role === "user") lastUser = m.content;
      if (m.role === "assistant") lastAsst = m.content;
    });
    els.userLine.textContent = lastUser;
    els.reply.textContent = lastAsst;
  }

  function speak(text) {
    if (!state.voice || !text) {
      if (state.session && voice) voice.startListenLoop();
      else setUi("idle");
      return;
    }
    var loc = i18n.LOCALES[state.lang] || i18n.LOCALES.en;
    if (voice && typeof voice.speak === "function") {
      setUi("speak");
      voice.speak(text, loc.tts);
      return;
    }
    if (!window.speechSynthesis) {
      setUi(state.session ? "listen" : "idle");
      return;
    }
    window.speechSynthesis.cancel();
    var u = new SpeechSynthesisUtterance(text);
    u.lang = loc.tts;
    u.rate = 1.02;
    u.onstart = function () {
      setUi("speak");
    };
    u.onend = function () {
      if (state.session && voice) voice.startListenLoop();
      else setUi("idle");
    };
    window.speechSynthesis.speak(u);
  }

  function transcript() {
    return state.messages
      .map(function (m) {
        return (m.role === "user" ? "You: " : "Pyx: ") + m.content;
      })
      .join("\n");
  }

  async function postWebhook(kind) {
    var url = kind === "slack" ? state.slack : state.discord;
    var text = lastAssistant();
    if (!url || !text) {
      toast(t("webhookFail"));
      return;
    }
    var payload = kind === "slack" ? { text: text } : { content: text.slice(0, 1900) };
    try {
      await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        mode: "no-cors",
      });
      toast(t("webhookOk"));
    } catch (e) {
      toast(t("webhookFail"));
    }
  }

  function runAction(action) {
    if (!action) return;
    switch (action.type) {
      case "theme":
        applyTheme(action.theme);
        return;
      case "language":
        applyLang(action.lang);
        return;
      case "settings":
        openSheet(els.settingsSheet);
        return;
      case "clear":
        state.messages = [];
        save();
        renderHistory();
        return;
      case "open_talk":
        toast(t("stayLocal"));
        return;
      case "open_studio":
        location.href = "/studio.html";
        return;
      case "share":
        shareConversation();
        return;
      case "slack":
        postWebhook("slack");
        return;
      case "discord":
        postWebhook("discord");
        return;
      default: {
        var unused = action.type;
        void unused;
      }
    }
  }

  async function shareConversation() {
    var text = transcript();
    if (navigator.share) {
      try {
        await navigator.share({ title: "Pyx Assistant", text: text });
        return;
      } catch (e) {}
    }
    try {
      await navigator.clipboard.writeText(text);
      toast(t("copied"));
    } catch (e) {
      toast(t("copied"));
    }
  }

  function localReply(text) {
    if (math && math.looksMath(text)) {
      var solved = math.answer(text);
      if (solved) return solved.reply;
    }
    var understood = slu.classify(text);
    var resolved = slu.resolve(understood, { lang: state.lang, t: i18n.t });
    runAction(resolved.action);
    if (resolved.action && resolved.action.type === "clear") return resolved.reply;
    if (resolved.special && kb) return kb.expandSpecial(resolved.special);
    if (resolved.reply) return resolved.reply;
    if (kb) {
      var hit = kb.retrieve(text, 0.62);
      if (hit) return hit.reply;
      return kb.warmFallback(text);
    }
    return t("identity");
  }

  async function handleUserText(raw, fromVoice) {
    var text = slu.normalize(raw);
    if (!text || handleInFlight) return;
    handleInFlight = true;
    els.userLine.textContent = text;
    setUi("think");
    try {
      var reply = localReply(text);
      if (reply == null) reply = "";
      if (kb) kb.remember(reply);
      if (!(slu.classify(text).intent === "clear")) {
        state.messages.push({ role: "user", content: text });
        if (reply) state.messages.push({ role: "assistant", content: reply });
      }
      els.reply.textContent = reply || t("greeting");
      save();
      renderHistory();
      if (fromVoice || state.session) speak(reply);
      else setUi("idle");
    } finally {
      handleInFlight = false;
    }
  }

  async function toggleOrb() {
    voice = window.PyxAssistantVoice;
    if (!voice) {
      toast(t("noSpeech"));
      els.input.focus();
      return;
    }
    if (voice.isSpeaking()) {
      await voice.interrupt();
      state.session = true;
      setUi("listen");
      return;
    }
    if (voice.isSession() || state.session) {
      await voice.stopSession();
      state.session = false;
      setUi("idle");
      toast(t("paused"));
      return;
    }
    try {
      state.session = true;
      await voice.startSession();
      setUi("listen");
    } catch (e) {
      state.session = false;
      toast(t("micDenied"));
      setUi("idle");
    }
  }

  function openSheet(sheet) {
    sheet.classList.add("is-open");
    sheet.setAttribute("aria-hidden", "false");
  }

  function closeSheets() {
    [els.settingsSheet, els.historySheet].forEach(function (s) {
      s.classList.remove("is-open");
      s.setAttribute("aria-hidden", "true");
    });
  }

  function paintLangOptions() {
    els.lang.innerHTML = "";
    Object.keys(i18n.LOCALES).forEach(function (code) {
      var opt = document.createElement("option");
      opt.value = code;
      opt.textContent = i18n.LOCALES[code].label;
      els.lang.appendChild(opt);
    });
    els.lang.value = state.lang;
  }

  function paintThemeOptions() {
    els.theme.innerHTML = "";
    slu.THEMES.forEach(function (name) {
      var opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      els.theme.appendChild(opt);
    });
    els.theme.value = state.theme;
  }

  function paintVoiceOptions() {
    els.mode.innerHTML = "";
    var names = (voice && voice.listVoices && voice.listVoices()) || [
      "af_heart",
      "af_bella",
      "am_michael",
    ];
    names.forEach(function (name) {
      var opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name.replace(/_/g, " ");
      els.mode.appendChild(opt);
    });
    els.mode.value = state.voiceId;
  }

  function startSwirl(canvas) {
    var ctx = canvas.getContext("2d");
    var blobs = [
      { x: 0.35, y: 0.38, r: 0.28, a: 0, s: 0.0007 },
      { x: 0.68, y: 0.42, r: 0.26, a: 1.2, s: 0.0009 },
      { x: 0.5, y: 0.66, r: 0.3, a: 2.4, s: 0.0006 },
      { x: 0.42, y: 0.28, r: 0.18, a: 0.6, s: 0.0011 },
    ];
    function resize() {
      canvas.width = canvas.clientWidth;
      canvas.height = canvas.clientHeight;
    }
    resize();
    window.addEventListener("resize", resize);
    function colors() {
      var cs = getComputedStyle(document.documentElement);
      return [
        cs.getPropertyValue("--swirl-a").trim() || "#ff9ec8",
        cs.getPropertyValue("--swirl-b").trim() || "#c4b5fd",
        cs.getPropertyValue("--swirl-c").trim() || "#93c5fd",
        cs.getPropertyValue("--swirl-d").trim() || "#99f6e4",
      ];
    }
    function frame(ts) {
      var w = canvas.width;
      var h = canvas.height;
      var cols = colors();
      ctx.clearRect(0, 0, w, h);
      ctx.globalCompositeOperation = "lighter";
      blobs.forEach(function (b, i) {
        var px = (b.x + Math.sin(ts * b.s + b.a) * 0.08) * w;
        var py = (b.y + Math.cos(ts * b.s * 1.15 + b.a) * 0.07) * h;
        var rad = b.r * Math.min(w, h);
        var g = ctx.createRadialGradient(px, py, 0, px, py, rad);
        g.addColorStop(0, cols[i]);
        g.addColorStop(1, "rgba(0,0,0,0)");
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(px, py, rad, 0, Math.PI * 2);
        ctx.fill();
      });
      swirlRaf = requestAnimationFrame(frame);
    }
    if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      swirlRaf = requestAnimationFrame(frame);
    } else {
      frame(0);
    }
    void swirlRaf;
  }

  function bindVoice() {
    voice = window.PyxAssistantVoice;
    if (!voice) return;
    voice.setVoice(state.voiceId);
    voice.onStatus = function (kind, extra) {
      if (kind === "listen") setUi("listen", extra);
      else if (kind === "think") setUi("think", extra);
      else if (kind === "speak") setUi("speak", extra);
      else if (kind === "idle") setUi("idle", extra);
      else if (extra) els.status.textContent = extra;
    };
    voice.onPartial = function (p) {
      if (p) els.userLine.textContent = p;
    };
    voice.onUtterance = function (text) {
      handleUserText(text, true);
    };
    voice.onSpeakEnd = function (interrupted) {
      if (interrupted) setUi("listen");
    };
    voice.onError = function () {
      toast(t("sttFail"));
    };
  }

  function bind() {
    els.orb.addEventListener("click", toggleOrb);
    els.mic.addEventListener("click", toggleOrb);
    els.send.addEventListener("click", function () {
      var v = els.input.value;
      els.input.value = "";
      handleUserText(v, false);
    });
    els.input.addEventListener("keydown", function (ev) {
      if (ev.key === "Enter") {
        ev.preventDefault();
        els.send.click();
      }
    });
    els.settingsBtn.addEventListener("click", function () {
      openSheet(els.settingsSheet);
    });
    els.historyBtn.addEventListener("click", function () {
      openSheet(els.historySheet);
    });
    document.querySelectorAll("[data-close-sheet]").forEach(function (btn) {
      btn.addEventListener("click", closeSheets);
    });
    [els.settingsSheet, els.historySheet].forEach(function (sheet) {
      sheet.addEventListener("click", function (ev) {
        if (ev.target === sheet) closeSheets();
      });
    });
    els.theme.addEventListener("change", function () {
      applyTheme(els.theme.value);
    });
    els.lang.addEventListener("change", function () {
      applyLang(els.lang.value);
    });
    els.voice.addEventListener("change", function () {
      state.voice = els.voice.checked;
      save();
    });
    els.mode.addEventListener("change", function () {
      state.voiceId = els.mode.value;
      if (voice && voice.setVoice) voice.setVoice(state.voiceId);
      save();
    });
    els.slack.addEventListener("change", function () {
      state.slack = els.slack.value.trim();
      save();
    });
    els.discord.addEventListener("change", function () {
      state.discord = els.discord.value.trim();
      save();
    });
    els.sendSlack.addEventListener("click", function () {
      postWebhook("slack");
    });
    els.sendDiscord.addEventListener("click", function () {
      postWebhook("discord");
    });
    if (els.openTalk) {
      els.openTalk.addEventListener("click", function () {
        toast(t("stayLocal"));
      });
    }
    els.clearBtn.addEventListener("click", function () {
      runAction({ type: "clear" });
      toast(t("cleared"));
    });
    document.addEventListener("keydown", function (ev) {
      if (ev.key === "Escape") closeSheets();
    });
  }

  async function bootKnowledge() {
    var res = await fetch("/betas/pyxassistant/kb/pyx-assistant-kb.json", { cache: "force-cache" });
    var data = await res.json();
    var n = kb.load(data);
    if (els.kbMeta) els.kbMeta.textContent = n + " local replies";
    return n;
  }

  async function bootVoice() {
    voice = window.PyxAssistantVoice;
    if (!voice || !voice.warmup) return;
    state.warming = true;
    setUi("warm", t("warming"));
    await voice.warmup(function (msg) {
      els.status.textContent = msg;
    });
    state.warming = false;
    bindVoice();
    paintVoiceOptions();
    if (!state.messages.length) setUi("idle");
  }

  function init() {
    els = {
      orb: document.getElementById("orb"),
      status: document.getElementById("status"),
      reply: document.getElementById("reply"),
      userLine: document.getElementById("userLine"),
      input: document.getElementById("input"),
      send: document.getElementById("send"),
      mic: document.getElementById("mic"),
      back: document.getElementById("backLink"),
      settingsBtn: document.getElementById("settingsBtn"),
      historyBtn: document.getElementById("historyBtn"),
      settingsSheet: document.getElementById("settingsSheet"),
      historySheet: document.getElementById("historySheet"),
      settingsTitle: document.getElementById("settingsTitle"),
      historyTitle: document.getElementById("historyTitle"),
      theme: document.getElementById("theme"),
      lang: document.getElementById("lang"),
      voice: document.getElementById("voice"),
      mode: document.getElementById("mode"),
      slack: document.getElementById("slack"),
      discord: document.getElementById("discord"),
      themeLabel: document.getElementById("themeLabel"),
      langLabel: document.getElementById("langLabel"),
      voiceLabel: document.getElementById("voiceLabel"),
      modeLabel: document.getElementById("modeLabel"),
      slackLabel: document.getElementById("slackLabel"),
      discordLabel: document.getElementById("discordLabel"),
      sendSlack: document.getElementById("sendSlack"),
      sendDiscord: document.getElementById("sendDiscord"),
      openTalk: document.getElementById("openTalk"),
      clearBtn: document.getElementById("clearBtn"),
      history: document.getElementById("historyList"),
      toast: document.getElementById("toast"),
      kbMeta: document.getElementById("kbMeta"),
    };
    load();
    applyTheme(state.theme);
    applyLang(state.lang);
    paintThemeOptions();
    paintLangOptions();
    paintVoiceOptions();
    els.voice.checked = state.voice;
    els.slack.value = state.slack;
    els.discord.value = state.discord;
    paintChrome();
    bind();
    startSwirl(document.getElementById("swirlCanvas"));
    bootKnowledge()
      .then(function () {
        if (window.PyxHandoff) {
          window.PyxHandoff.applyIncoming({
            app: "pyxassistant",
            onQuery: function (q) {
              handleUserText(q, false);
            },
            onText: function (text) {
              if (text) handleUserText(text, false);
            },
          });
        }
      })
      .catch(function () {
        toast("Knowledge pack didn’t load — math and built-ins still work.");
      });
    if (window.PyxAssistantVoice) bootVoice();
    else {
      window.addEventListener("pyx-voice-ready", bootVoice, { once: true });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
