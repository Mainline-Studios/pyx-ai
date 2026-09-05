/**
 * Pyx Assistant — UI, voice, Talk API, and integrations.
 */
(function () {
  "use strict";

  var STORE_KEY = "pyx.assistant.v1";
  var PYX_CLOUD_RUN = "https://pyxaiapi-574247481583.us-central1.run.app";
  var PYXEL =
    "You are Pyx Assistant, a warm, concise, Siri-like voice assistant. " +
    "Answer in the user's language. Keep replies short unless asked for depth. " +
    "Stay friendly and useful. Refuse harmful requests briefly.";

  var slu = window.PyxAssistantSLU;
  var i18n = window.PyxAssistantI18n;

  var state = {
    theme: "aurora",
    lang: "en",
    voice: true,
    mode: "fast",
    slack: "",
    discord: "",
    messages: [],
    ui: "idle",
  };

  var els = {};
  var recognition = null;
  var swirlRaf = 0;

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
      if (o.mode) state.mode = o.mode;
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
          mode: state.mode,
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

  function setUi(mode) {
    state.ui = mode;
    els.orb.setAttribute("data-state", mode);
    var label = t("hint");
    if (mode === "listen") label = t("listening");
    else if (mode === "think") label = t("thinking");
    else if (mode === "speak") label = t("speaking");
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
    els.modeLabel.textContent = t("model");
    els.slackLabel.textContent = t("slackWebhook");
    els.discordLabel.textContent = t("discordWebhook");
    els.sendSlack.textContent = t("sendSlack");
    els.sendDiscord.textContent = t("sendDiscord");
    els.openTalk.textContent = t("openTalk");
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
    if (!state.voice || !window.speechSynthesis || !text) return;
    window.speechSynthesis.cancel();
    var u = new SpeechSynthesisUtterance(text);
    var loc = i18n.LOCALES[state.lang] || i18n.LOCALES.en;
    u.lang = loc.tts;
    u.rate = 1.02;
    u.onstart = function () {
      setUi("speak");
    };
    u.onend = function () {
      setUi("idle");
    };
    window.speechSynthesis.speak(u);
  }

  function pyxIsLocalHost() {
    var h = location.hostname || "";
    return h === "localhost" || h === "127.0.0.1" || h === "";
  }

  function fetchTalk(body) {
    var opts = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      mode: "cors",
      cache: "no-store",
      credentials: "omit",
    };
    var direct = PYX_CLOUD_RUN + "/talk";
    var proxied = "/api/talk";
    if (pyxIsLocalHost()) {
      return fetch("/talk", opts).catch(function () {
        return fetch(direct, opts);
      });
    }
    function once(url) {
      return fetch(url, opts);
    }
    function withFallback(first, second) {
      return once(first)
        .then(function (res) {
          if (res.status >= 500) {
            return once(second).catch(function () {
              return res;
            });
          }
          return res;
        })
        .catch(function () {
          return once(second);
        });
    }
    return withFallback(proxied, direct).then(function (res) {
      if (res && (res.status === 502 || res.status === 503 || res.status === 429)) {
        return new Promise(function (resolve) {
          setTimeout(resolve, 900);
        }).then(function () {
          return withFallback(direct, proxied);
        });
      }
      return res;
    });
  }

  function parseSse(buffer, onEvent) {
    var parts = buffer.split("\n\n");
    var rest = parts.pop() || "";
    parts.forEach(function (part) {
      var line = part.split("\n").filter(function (l) {
        return l.indexOf("data: ") === 0;
      })[0];
      if (!line) return;
      try {
        onEvent(JSON.parse(line.slice(6)));
      } catch (e) {}
    });
    return rest;
  }

  async function askLlm(userText, useWeb) {
    var history = state.messages
      .filter(function (m) {
        return m.role === "user" || m.role === "assistant";
      })
      .slice(-16)
      .map(function (m) {
        return { role: m.role, content: m.content };
      });
    history.push({ role: "user", content: userText });
    var body = {
      messages: history,
      mode: state.mode,
      stream: true,
      use_web: !!useWeb,
      use_web_auto: true,
      pyxel_instructions: PYXEL + " Respond in language code: " + state.lang + ".",
    };
    var res = await fetchTalk(body);
    if (!res || !res.ok) {
      var raw = res ? await res.text() : "";
      throw new Error(raw || "Talk request failed");
    }
    if (!res.body || !res.body.getReader) {
      var j = await res.json();
      return j.reply || "";
    }
    var reader = res.body.getReader();
    var dec = new TextDecoder();
    var buf = "";
    var reply = "";
    els.reply.textContent = "";
    while (true) {
      var chunk = await reader.read();
      if (chunk.done) break;
      buf += dec.decode(chunk.value, { stream: true });
      buf = parseSse(buf, function (evt) {
        if (evt.type === "delta" && evt.t) {
          reply += evt.t;
          els.reply.textContent = reply;
        }
        if (evt.type === "error") {
          throw new Error(evt.message || evt.detail || "LLM error");
        }
      });
    }
    return reply || t("identity");
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
        if (window.PyxHandoff) {
          window.PyxHandoff.sendTo("talk", transcript(), "pyxassistant");
        } else {
          location.href = "/pyx-talk.html";
        }
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

  async function handleUserText(raw, fromVoice) {
    var text = slu.normalize(raw);
    if (!text) return;
    els.userLine.textContent = text;
    var understood = slu.classify(text);
    var resolved = slu.resolve(understood, { lang: state.lang, t: i18n.t });
    runAction(resolved.action);

    if (resolved.action && resolved.action.type === "clear") {
      setUi("idle");
      speak(resolved.reply);
      return;
    }

    state.messages.push({ role: "user", content: text });
    if (resolved.useLlm) {
      setUi("think");
      try {
        var reply = await askLlm(text, resolved.useWeb);
        state.messages.push({ role: "assistant", content: reply });
        els.reply.textContent = reply;
        save();
        renderHistory();
        setUi(fromVoice && state.voice ? "speak" : "idle");
        speak(reply);
      } catch (e) {
        var fallback = "I couldn’t reach Pyx Talk just now. Try again in a moment.";
        state.messages.push({ role: "assistant", content: fallback });
        els.reply.textContent = fallback;
        save();
        renderHistory();
        setUi("idle");
      }
      return;
    }

    if (resolved.reply) {
      state.messages.push({ role: "assistant", content: resolved.reply });
      els.reply.textContent = resolved.reply;
      save();
      renderHistory();
      setUi("idle");
      speak(resolved.reply);
    }
  }

  function SpeechCtor() {
    return window.SpeechRecognition || window.webkitSpeechRecognition || null;
  }

  function stopListen() {
    if (recognition) {
      try {
        recognition.stop();
      } catch (e) {}
    }
    if (state.ui === "listen") setUi("idle");
  }

  function startListen() {
    var Ctor = SpeechCtor();
    if (!Ctor) {
      toast(t("noSpeech"));
      els.input.focus();
      return;
    }
    if (state.ui === "listen") {
      stopListen();
      return;
    }
    recognition = new Ctor();
    var loc = i18n.LOCALES[state.lang] || i18n.LOCALES.en;
    recognition.lang = loc.speech;
    recognition.interimResults = true;
    recognition.continuous = false;
    recognition.onstart = function () {
      setUi("listen");
    };
    recognition.onerror = function (ev) {
      if (ev.error === "not-allowed") toast(t("micDenied"));
      setUi("idle");
    };
    recognition.onend = function () {
      if (state.ui === "listen") setUi("idle");
    };
    recognition.onresult = function (ev) {
      var i;
      var finalText = "";
      var interim = "";
      for (i = ev.resultIndex; i < ev.results.length; i++) {
        if (ev.results[i].isFinal) finalText += ev.results[i][0].transcript;
        else interim += ev.results[i][0].transcript;
      }
      if (interim) els.userLine.textContent = interim;
      if (finalText) handleUserText(finalText, true);
    };
    try {
      recognition.start();
    } catch (e) {
      toast(t("noSpeech"));
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

  function bind() {
    els.orb.addEventListener("click", startListen);
    els.mic.addEventListener("click", startListen);
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
      state.mode = els.mode.value;
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
    els.openTalk.addEventListener("click", function () {
      runAction({ type: "open_talk" });
    });
    els.clearBtn.addEventListener("click", function () {
      runAction({ type: "clear" });
      toast(t("cleared"));
    });
    document.addEventListener("keydown", function (ev) {
      if (ev.key === "Escape") closeSheets();
    });
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
    };
    load();
    applyTheme(state.theme);
    applyLang(state.lang);
    paintThemeOptions();
    paintLangOptions();
    els.voice.checked = state.voice;
    els.mode.value = state.mode;
    els.slack.value = state.slack;
    els.discord.value = state.discord;
    paintChrome();
    bind();
    startSwirl(document.getElementById("swirlCanvas"));
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
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
