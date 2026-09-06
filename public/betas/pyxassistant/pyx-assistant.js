/**
 * Pyx Assistant — local KB + math + sports + weather (MARII = local, no cloud AI).
 */
(function () {
  "use strict";

  var STORE_KEY = "pyx.assistant.v3";
  var slu = window.PyxAssistantSLU;
  var i18n = window.PyxAssistantI18n;
  var math = window.PyxAssistantMath;
  var kb = window.PyxAssistantKB;
  var learn = window.PyxAssistantLearn;
  var cookies = window.PyxAssistantCookies;
  var sports = window.PyxAssistantSports;
  var weather = window.PyxAssistantWeather;
  var wiki = window.PyxAssistantWiki;
  var voice = null;

  var state = {
    theme: "aurora",
    lang: "en",
    voice: true,
    voiceId: "en-GB",
    mariiBoost: false,
    slack: "",
    discord: "",
    messages: [],
    ui: "idle",
    session: false,
    warming: false,
    voiceReady: false,
  };

  var els = {};
  var swirlRaf = 0;
  var swirlClock = 0;
  var swirlLast = 0;
  var swirlBurstStart = 0;
  var swirlBurstDur = 0;
  var swirlReduced = false;
  var handleInFlight = false;
  var pendingHandoff = null;

  function t(key) {
    return i18n.t(state.lang, key);
  }

  function load() {
    try {
      if (cookies) {
        var packed = cookies.loadModel();
        if (packed && learn) learn.unpack(packed);
        var fromCookies = cookies.loadChats();
        if (fromCookies.length) state.messages = fromCookies;
      }
      var raw = localStorage.getItem(STORE_KEY);
      if (!raw) return;
      var o = JSON.parse(raw);
      if (o.theme) state.theme = o.theme;
      if (o.lang) state.lang = o.lang;
      if (typeof o.voice === "boolean") state.voice = o.voice;
      if (typeof o.mariiBoost === "boolean") state.mariiBoost = false;
      if (o.voiceId) {
        state.voiceId = String(o.voiceId).indexOf("af_") === 0 ? "en-GB" : o.voiceId;
      }
      if (typeof o.slack === "string") state.slack = o.slack;
      if (typeof o.discord === "string") state.discord = o.discord;
      if (!state.messages.length && Array.isArray(o.messages)) state.messages = o.messages.slice(-40);
    } catch (e) {}
  }

  function save() {
    try {
      if (cookies) {
        cookies.saveChats(state.messages);
        if (learn) cookies.saveModel(learn.pack());
      }
      localStorage.setItem(
        STORE_KEY,
        JSON.stringify({
          theme: state.theme,
          lang: state.lang,
          voice: state.voice,
          voiceId: state.voiceId,
          mariiBoost: state.mariiBoost,
          slack: state.slack,
          discord: state.discord,
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

  function shortName(full) {
    var parts = String(full || "").trim().split(/\s+/);
    if (!parts.length) return "";
    if (parts.length > 1 && /^jr\.?$/i.test(parts[parts.length - 1])) return parts.slice(-2).join(" ");
    return parts[parts.length - 1];
  }

  function fieldNode(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text) n.textContent = text;
    return n;
  }

  function countDots(kind, label, max, filled) {
    var wrap = fieldNode("span", "field-sim__stat");
    wrap.setAttribute("data-kind", kind);
    wrap.appendChild(fieldNode("em", "", label));
    var i;
    var n = Math.max(0, Number(filled) || 0);
    for (i = 0; i < max; i++) {
      var dot = fieldNode("i", i < n ? "is-on" : "");
      wrap.appendChild(dot);
    }
    return wrap;
  }

  function paintField(board) {
    if (!els.field) return;
    els.field.innerHTML = "";
    if (!board || board.kind !== "mlb-field" || !board.live) {
      els.field.hidden = true;
      document.body.classList.remove("has-field");
      return;
    }
    document.body.classList.add("has-field");
    els.field.hidden = false;
    var inning = [board.inningState, board.inning].filter(Boolean).join(" ");
    var score = board.away + " " + board.awayScore + " · " + board.home + " " + board.homeScore;
    els.field.setAttribute(
      "aria-label",
      score +
        (inning ? ", " + inning : "") +
        ". " +
        (board.pitcher ? board.pitcher + " pitching" : "Pitcher") +
        (board.batter ? " to " + board.batter : "") +
        ". " +
        board.balls +
        "-" +
        board.strikes +
        " count, " +
        board.outs +
        " out" +
        (board.outs === 1 ? "" : "s") +
        "."
    );
    els.field.appendChild(fieldNode("p", "field-sim__score", score));
    if (inning) els.field.appendChild(fieldNode("p", "field-sim__inning", inning));
    var diamond = fieldNode("div", "field-sim__diamond");
    diamond.appendChild(fieldNode("div", "field-sim__grass"));
    diamond.appendChild(fieldNode("div", "field-sim__dirt"));
    ["2", "3", "1", "h"].forEach(function (spot) {
      var base = fieldNode("span", "field-sim__base is-" + spot);
      var on = (spot === "1" && board.first) || (spot === "2" && board.second) || (spot === "3" && board.third);
      if (on) base.classList.add("is-on");
      diamond.appendChild(base);
    });
    var mound = fieldNode("span", "field-sim__mound");
    mound.appendChild(fieldNode("b", "", "P"));
    mound.appendChild(document.createTextNode(shortName(board.pitcher) || "Pitcher"));
    diamond.appendChild(mound);
    var plate = fieldNode("span", "field-sim__plate");
    plate.appendChild(fieldNode("b", "", "B"));
    plate.appendChild(document.createTextNode(shortName(board.batter) || "Batter"));
    diamond.appendChild(plate);
    els.field.appendChild(diamond);
    var count = fieldNode("div", "field-sim__count");
    count.appendChild(countDots("balls", "B", 3, board.balls));
    count.appendChild(countDots("strikes", "S", 2, board.strikes));
    count.appendChild(countDots("outs", "O", 3, board.outs));
    els.field.appendChild(count);
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
    if (els.dataTitle) els.dataTitle.textContent = t("dataTitle");
    if (els.dataBtn) {
      els.dataBtn.textContent = t("data");
      els.dataBtn.setAttribute("aria-label", t("data"));
      els.dataBtn.setAttribute("title", t("data"));
    }
    if (els.openData) els.openData.textContent = t("dataSee");
    if (els.dataForgetBtn) els.dataForgetBtn.textContent = t("forgetMe");
    els.themeLabel.textContent = t("theme");
    els.langLabel.textContent = t("language");
    els.voiceLabel.textContent = t("voiceReplies");
    if (els.mariiBoostLabel) els.mariiBoostLabel.textContent = t("mariiBoost");
    els.modeLabel.textContent = t("voiceName");
    els.slackLabel.textContent = t("slackWebhook");
    els.discordLabel.textContent = t("discordWebhook");
    els.sendSlack.textContent = t("sendSlack");
    els.sendDiscord.textContent = t("sendDiscord");
    if (els.openTalk) els.openTalk.textContent = t("stayLocal");
    els.clearBtn.textContent = t("clear");
    if (els.forgetBtn) els.forgetBtn.textContent = t("forgetMe");
    if (!state.messages.length) {
      els.reply.textContent = learn ? learn.greeting(t("greeting")) : t("greeting");
      els.userLine.textContent = "";
      els.status.textContent = t("hint");
      paintField(null);
    }
    renderHistory();
    renderData();
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
      case "data":
        renderData();
        openSheet(els.dataSheet);
        return;
      case "clear":
        state.messages = [];
        paintField(null);
        save();
        renderHistory();
        paintChrome();
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

  function setSourceChip(source) {
    if (!els.sourceChip) return;
    if (source === "marii") {
      els.sourceChip.hidden = false;
      els.sourceChip.textContent = t("mariiChip") || "marii";
    } else if (source === "wiki") {
      els.sourceChip.hidden = false;
      els.sourceChip.textContent = "wikipedia";
    } else {
      els.sourceChip.hidden = true;
    }
  }

  async function moderateOk(text) {
    try {
      var q = encodeURIComponent(String(text || "").slice(0, 400));
      if (!q) return true;
      var res = await fetch("/api/moderator/check/" + q + "?threshold=700", {
        method: "GET",
        cache: "no-store",
      });
      if (!res.ok) return true;
      var data = await res.json();
      return data.appropriate !== false;
    } catch (e) {
      return true;
    }
  }

  async function fetchMariiAsk() {
    // MARII is local-only (no cloud LLM / no Groq / no Workers AI).
    return null;
  }

  async function askMarii() {
    // Optional “boost” used to call a cloud LLM. That path is retired.
    return null;
  }

  async function localReply(text) {
    if (sports && sports.clearBoard) sports.clearBoard();
    if (learn) {
      var fb = learn.observeFeedback(text);
      var extracted = learn.ingest(text);
      if (extracted.reply) {
        if (extracted.kind) learn.observe(text, extracted.kind);
        if (/\b(what(?:'s| is) my name|who am i|do you know me|what do i like)\b/i.test(text)) {
          renderData();
          openSheet(els.dataSheet);
        }
        return { reply: learn.flavor(extracted.reply), source: "local" };
      }
      if (fb === "pos" && kb && learn.profile.lastKind === "joke") {
        return { reply: learn.flavor(kb.expandSpecial("__JOKE__")), source: "local" };
      }
    }
    if (math && math.looksMath(text)) {
      var solved = math.answer(text);
      if (learn) learn.observe(text, "math");
      if (solved) {
        return { reply: learn ? learn.flavor(solved.reply) : solved.reply, source: "local" };
      }
      return {
        reply: "I couldn’t parse that as math. Try 15% of 80, 9 times 8, or 32 F to C.",
        source: "local",
      };
    }
    var understood = slu.classify(text);
    var resolved = slu.resolve(understood, { lang: state.lang, t: i18n.t });
    runAction(resolved.action);
    if (resolved.action && resolved.action.type === "clear") {
      return { reply: resolved.reply, source: "local" };
    }
    if (understood.intent === "greet") {
      var g = learn ? learn.greeting(resolved.reply) : resolved.reply;
      if (learn) learn.observe(text, "talk");
      return { reply: g, source: "local" };
    }
    var kind = learn ? learn.kindFromIntent(understood.intent, null) : "talk";
    if (resolved.special && kb) {
      if (learn) learn.observe(text, kind);
      var special = kb.expandSpecial(resolved.special);
      return { reply: learn ? learn.flavor(special) : special, source: "local" };
    }
    if (resolved.reply) {
      if (learn) learn.observe(text, kind);
      return { reply: learn ? learn.flavor(resolved.reply) : resolved.reply, source: "local" };
    }
    if (weather && (resolved.useWeb || understood.intent === "weather")) {
      try {
        var weatherReply = await weather.answer(text);
        if (weatherReply) {
          if (learn) learn.observe(text, "talk");
          return { reply: learn ? learn.flavor(weatherReply) : weatherReply, source: "weather" };
        }
      } catch (err) {
        return {
          reply: "I couldn’t reach live weather just now. Try again in a second. =)",
          source: "weather",
        };
      }
    }
    if (sports && (understood.intent === "sports" || sports.looksSports(text))) {
      try {
        var sportReply = await sports.answer(text);
        if (sportReply) {
          if (learn) learn.observe(text, "talk");
          return { reply: learn ? learn.flavor(sportReply) : sportReply, source: "sports" };
        }
      } catch (err) {
        return {
          reply: "I couldn’t reach live sports just now. Try me again in a second. =)",
          source: "sports",
        };
      }
    }
    if (kb) {
      var priors = learn ? learn.priorsFor(text) : {};
      var likes = learn ? learn.profile.likes : [];
      var hit = kb.retrieve(text, 0.62, priors, likes);
      if (hit) {
        var recKind = kb.family ? kb.family(hit.rec.kind) : hit.rec.kind;
        if (learn) learn.observe(text, learn.kindFromIntent(understood.intent, recKind));
        return { reply: learn ? learn.flavor(hit.reply) : hit.reply, source: "local" };
      }
      if (wiki && wiki.looksWikiWorthy(text)) {
        try {
          var wikiHit = await wiki.answer(text);
          if (wikiHit && wikiHit.reply) {
            if (learn) learn.observe(text, "talk");
            return {
              reply: wikiHit.reply,
              speak: wikiHit.speak || wikiHit.reply,
              source: "wiki",
            };
          }
        } catch (wikiErr) {
          // Fall through to warm local reply — wiki is optional.
        }
      }
      var boost = await askMarii(text);
      if (boost) {
        if (learn) learn.observe(text, "talk");
        return { reply: learn ? learn.flavor(boost) : boost, source: "marii" };
      }
      if (learn) learn.observe(text, "talk");
      return { reply: learn ? learn.flavor(kb.warmFallback(text)) : kb.warmFallback(text), source: "local" };
    }
    var lastBoost = await askMarii(text);
    if (lastBoost) return { reply: lastBoost, source: "marii" };
    return { reply: t("identity"), source: "local" };
  }

  async function handleUserText(raw, fromVoice) {
    var text = slu.normalize(raw);
    if (!text || handleInFlight) return;
    if (!state.voiceReady) {
      toast("Voice is still downloading — hang on a second.");
      return;
    }
    handleInFlight = true;
    els.userLine.textContent = text;
    kickSwirl();
    setUi("think");
    setSourceChip(null);
    try {
      var result = await localReply(text);
      var reply = result && result.reply != null ? result.reply : "";
      var speakText =
        result && result.speak != null && String(result.speak).trim()
          ? String(result.speak).trim()
          : reply;
      var source = (result && result.source) || "local";
      if (kb) kb.remember(reply);
      if (!(slu.classify(text).intent === "clear")) {
        state.messages.push({ role: "user", content: text });
        if (reply) state.messages.push({ role: "assistant", content: reply });
      }
      els.reply.textContent = reply || t("greeting");
      setSourceChip(source);
      paintField(sports && sports.board);
      save();
      renderHistory();
      renderData();
      refreshKbMeta();
      if (state.voice) speak(speakText);
      else setUi(state.session ? "listen" : "idle");
    } finally {
      handleInFlight = false;
    }
  }

  async function toggleOrb() {
    if (!state.voiceReady) {
      toast("Voice is still downloading — hang on a second.");
      return;
    }
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
    [els.settingsSheet, els.historySheet, els.dataSheet].forEach(function (s) {
      if (!s) return;
      s.classList.remove("is-open");
      s.setAttribute("aria-hidden", "true");
    });
  }

  function node(tag, className, text) {
    var n = document.createElement(tag);
    if (className) n.className = className;
    if (text != null) n.textContent = text;
    return n;
  }

  function fillCount(key, n) {
    return t(key).replace("{n}", String(n));
  }

  function chips(items, emptyText) {
    if (!items.length) return node("p", "data-value is-empty", emptyText);
    var wrap = node("div", "data-chips");
    items.forEach(function (item) {
      wrap.appendChild(node("span", "data-chip", item));
    });
    return wrap;
  }

  function block(title, child) {
    var sec = node("section", "data-block");
    sec.appendChild(node("h3", "", title));
    sec.appendChild(child);
    return sec;
  }

  function renderData() {
    if (!els.dataBody) return;
    els.dataBody.innerHTML = "";
    els.dataBody.appendChild(node("p", "data-note", t("dataIntro")));
    var info = learn
      ? learn.explain()
      : {
          name: "",
          likes: [],
          dislikes: [],
          seen: 0,
          lastLabel: "",
          tastes: [],
          patterned: false,
          known: false,
        };

    if (!info.known && !state.messages.length) {
      els.dataBody.appendChild(node("p", "data-value is-empty", t("dataEmpty")));
    }

    var about = node("div");
    about.appendChild(node("p", info.name ? "data-value" : "data-value is-empty", info.name || t("dataNameNone")));
    els.dataBody.appendChild(block(t("dataName"), about.firstChild));
    els.dataBody.appendChild(block(t("dataLikes"), chips(info.likes, t("dataLikesNone"))));
    els.dataBody.appendChild(block(t("dataDislikes"), chips(info.dislikes, t("dataDislikesNone"))));

    var tasteWrap = node("div");
    tasteWrap.appendChild(node("p", "data-note", t("dataTastesHint")));
    if (!info.patterned) {
      tasteWrap.appendChild(node("p", "data-value is-empty", t("dataTastesNone")));
    } else {
      info.tastes.forEach(function (row) {
        var item = node("div", "data-taste");
        var rowEl = node("div", "data-taste__row");
        rowEl.appendChild(node("span", "data-taste__name", t("taste_" + row.id)));
        rowEl.appendChild(node("span", "data-taste__amt", t("tasteAmt_" + row.amount)));
        item.appendChild(rowEl);
        var track = node("div", "data-taste__track");
        var fill = node("div", "data-taste__fill");
        fill.style.width = row.bar + "%";
        track.appendChild(fill);
        item.appendChild(track);
        tasteWrap.appendChild(item);
      });
    }
    els.dataBody.appendChild(block(t("dataTastes"), tasteWrap));

    if (info.seen) {
      els.dataBody.appendChild(block(t("dataLast"), node("p", "data-value", t("taste_" + info.lastKind) || info.lastLabel)));
      els.dataBody.appendChild(block(t("dataTurns"), node("p", "data-value", fillCount("dataTurnsBody", info.seen))));
    }

    var chatWrap = node("div");
    if (!state.messages.length) {
      chatWrap.appendChild(node("p", "data-value is-empty", t("dataChatsNone")));
    } else {
      chatWrap.appendChild(node("p", "data-note", fillCount("dataChatsBody", state.messages.length)));
      var list = node("ul", "data-chats");
      state.messages.slice(-24).forEach(function (m) {
        var li = node("li");
        li.appendChild(node("span", "role", m.role === "user" ? t("dataYou") : t("dataPyx")));
        li.appendChild(document.createTextNode(m.content));
        list.appendChild(li);
      });
      chatWrap.appendChild(list);
    }
    els.dataBody.appendChild(block(t("dataChats"), chatWrap));

    var setup = node("ul", "data-setup");
    var loc = (i18n.LOCALES[state.lang] && i18n.LOCALES[state.lang].label) || state.lang;
    var voiceName = state.voiceId;
    if (els.mode && els.mode.selectedIndex >= 0 && els.mode.options[els.mode.selectedIndex]) {
      voiceName = els.mode.options[els.mode.selectedIndex].textContent;
    }
    setup.appendChild(node("li", "", t("dataLook") + ": " + state.theme));
    setup.appendChild(node("li", "", t("language") + ": " + loc));
    setup.appendChild(node("li", "", state.voice ? t("dataSpeakOn") : t("dataSpeakOff")));
    setup.appendChild(node("li", "", t("voiceName") + ": " + voiceName));
    if (state.slack) setup.appendChild(node("li", "", "Slack: saved on this device (link hidden)."));
    if (state.discord) setup.appendChild(node("li", "", "Discord: saved on this device (link hidden)."));
    els.dataBody.appendChild(block(t("dataSetup"), setup));
    els.dataBody.appendChild(block(t("dataWhere"), node("p", "data-note", t("dataWhereBody"))));
  }

  function forgetMe() {
    if (learn) learn.reset();
    if (sports && sports.reset) sports.reset();
    state.messages = [];
    paintField(null);
    if (cookies) cookies.clearAll();
    save();
    renderHistory();
    renderData();
    paintChrome();
    refreshKbMeta();
    closeSheets();
    toast(t("forgotten"));
  }

  function openDataSheet() {
    renderData();
    openSheet(els.dataSheet);
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
      opt.textContent = name === "calm-contrast" ? "calm contrast" : name;
      els.theme.appendChild(opt);
    });
    els.theme.value = state.theme;
  }

  function paintVoiceOptions() {
    els.mode.innerHTML = "";
    var names = (voice && voice.listVoices && voice.listVoices()) || [
      { id: "en-GB", label: "Neural British" },
      { id: "en-US", label: "Neural US" },
      { id: "en-AU", label: "Neural Australian" },
    ];
    names.forEach(function (item) {
      var id = typeof item === "string" ? item : item.id;
      var label = typeof item === "string" ? item.replace(/_/g, " ") : item.label;
      var opt = document.createElement("option");
      opt.value = id;
      opt.textContent = label;
      els.mode.appendChild(opt);
    });
    els.mode.value = state.voiceId;
  }

  function swirlSpeed(now) {
    if (!swirlBurstDur) return 1;
    var u = (now - swirlBurstStart) / swirlBurstDur;
    if (u >= 1 || u < 0) {
      swirlBurstDur = 0;
      return 1;
    }
    return 1 + 12 * Math.pow(1 - u, 1.35);
  }

  function kickSwirl() {
    if (swirlReduced) return;
    swirlBurstStart = performance.now();
    swirlBurstDur = 500 + Math.random() * 1000;
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
      var dt = swirlLast ? Math.min(48, ts - swirlLast) : 16;
      swirlLast = ts;
      var speed = swirlSpeed(ts);
      swirlClock += dt * speed;
      var amp = 0.08 + 0.06 * Math.min(1, (speed - 1) / 12);
      var w = canvas.width;
      var h = canvas.height;
      var cols = colors();
      ctx.clearRect(0, 0, w, h);
      ctx.globalCompositeOperation = "lighter";
      blobs.forEach(function (b, i) {
        var px = (b.x + Math.sin(swirlClock * b.s + b.a) * amp) * w;
        var py = (b.y + Math.cos(swirlClock * b.s * 1.15 + b.a) * (amp * 0.88)) * h;
        var rad = b.r * Math.min(w, h) * (1 + 0.08 * Math.min(1, (speed - 1) / 12));
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
    swirlReduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (!swirlReduced) {
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
    if (els.voiceBootSkip) {
      els.voiceBootSkip.addEventListener("click", function () {
        if (voice && typeof voice.unlockAudio === "function") voice.unlockAudio();
        finishVoiceBoot("Using online neural voice.");
      });
    }
    els.settingsBtn.addEventListener("click", function () {
      openSheet(els.settingsSheet);
    });
    els.historyBtn.addEventListener("click", function () {
      openSheet(els.historySheet);
    });
    if (els.dataBtn) {
      els.dataBtn.addEventListener("click", openDataSheet);
    }
    document.querySelectorAll("[data-close-sheet]").forEach(function (btn) {
      btn.addEventListener("click", closeSheets);
    });
    [els.settingsSheet, els.historySheet, els.dataSheet].forEach(function (sheet) {
      if (!sheet) return;
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
    if (els.mariiBoost) {
      els.mariiBoost.addEventListener("change", function () {
        state.mariiBoost = els.mariiBoost.checked;
        save();
      });
    }
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
    if (els.openData) {
      els.openData.addEventListener("click", function () {
        closeSheets();
        openDataSheet();
      });
    }
    els.clearBtn.addEventListener("click", function () {
      runAction({ type: "clear" });
      toast(t("cleared"));
    });
    if (els.forgetBtn) {
      els.forgetBtn.addEventListener("click", forgetMe);
    }
    if (els.dataForgetBtn) {
      els.dataForgetBtn.addEventListener("click", forgetMe);
    }
    document.addEventListener("keydown", function (ev) {
      if (ev.key === "Escape") closeSheets();
    });
  }

  function memoryCaption() {
    if (!learn) return "";
    if (learn.profile.name) return " · remembers " + learn.profile.name;
    if (learn.profile.likes.length) return " · learning what you like";
    if (learn.profile.seen >= 3) return " · learning your taste";
    return "";
  }

  function refreshKbMeta() {
    if (!els.kbMeta) return;
    var base = els.kbMeta.getAttribute("data-base");
    if (!base) return;
    var voiceBit = els.kbMeta.getAttribute("data-voice") || "";
    els.kbMeta.textContent = base + memoryCaption() + (voiceBit ? " · " + voiceBit : "");
  }

  async function bootKnowledge() {
    var res = await fetch("/betas/pyxassistant/kb/pyx-assistant-kb.json?v=3", { cache: "no-store" });
    var data = await res.json();
    var n = kb.load(data);
    if (els.kbMeta) {
      els.kbMeta.hidden = false;
      els.kbMeta.setAttribute("data-base", n.toLocaleString() + " local replies · sports · weather");
      els.kbMeta.setAttribute("data-voice", "local-first · no cloud AI");
      refreshKbMeta();
    }
    return n;
  }

  function setChatLocked(locked) {
    document.body.classList.toggle("voice-locked", !!locked);
    if (els.input) els.input.disabled = !!locked;
    if (els.send) els.send.disabled = !!locked;
    if (els.mic) els.mic.disabled = !!locked;
    if (els.orb) els.orb.disabled = !!locked;
  }

  function setVoiceBootMsg(msg) {
    if (els.voiceBootMsg) els.voiceBootMsg.textContent = msg || "";
    if (els.kbMeta && els.kbMeta.hidden === false) {
      els.kbMeta.setAttribute("data-voice", msg || "");
      refreshKbMeta();
    }
  }

  function showVoiceBootSkip(show) {
    if (!els.voiceBootSkip) return;
    els.voiceBootSkip.classList.toggle("is-hidden", !show);
  }

  function finishVoiceBoot(msg) {
    var first = !state.voiceReady;
    state.voiceReady = true;
    state.warming = false;
    setVoiceBootMsg(msg || "Voice ready.");
    showVoiceBootSkip(false);
    setChatLocked(false);
    if (els.voiceBoot) {
      els.voiceBoot.classList.add("is-done");
      els.voiceBoot.setAttribute("aria-busy", "false");
    }
    paintVoiceOptions();
    if (first && pendingHandoff) {
      var q = pendingHandoff;
      pendingHandoff = null;
      handleUserText(q, false);
    }
  }

  async function bootVoice() {
    voice = window.PyxAssistantVoice;
    if (!voice || !voice.warmup) {
      setVoiceBootMsg("Neural voice module missing — typing still works.");
      finishVoiceBoot("Typing ready.");
      return;
    }
    state.warming = true;
    state.voiceReady = false;
    setChatLocked(true);
    setVoiceBootMsg("Getting voice ready…");
    bindVoice();
    paintVoiceOptions();
    voice.setVoice(state.voiceId);
    voice.onError = function () {
      toast(t("sttFail"));
    };
    voice.onOnlineReady = function () {
      // Unlock chat as soon as online TTS works; Kokoro continues in background.
      finishVoiceBoot("Voice ready — online neural. Kokoro still loading…");
      setVoiceBootMsg("Kokoro still downloading in the background…");
    };
    voice.onKokoroReady = function () {
      // Assistant stays on online neural US by default — only refresh the voice list.
      paintVoiceOptions();
      if (voice.setVoice) voice.setVoice(state.voiceId || "en-GB");
      setVoiceBootMsg("Kokoro available in Settings · still using online British.");
    };
    try {
      await voice.warmup(function (msg) {
        if (!state.voiceReady) setVoiceBootMsg(msg || "Getting voice ready…");
        else if (els.kbMeta) {
          els.kbMeta.setAttribute("data-voice", msg || "");
          refreshKbMeta();
        }
      });
      if (!state.voiceReady) finishVoiceBoot("Voice ready.");
    } catch (err) {
      if (!state.voiceReady) finishVoiceBoot("Online neural TTS ready.");
    }
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
      dataBtn: document.getElementById("dataBtn"),
      settingsSheet: document.getElementById("settingsSheet"),
      historySheet: document.getElementById("historySheet"),
      dataSheet: document.getElementById("dataSheet"),
      settingsTitle: document.getElementById("settingsTitle"),
      historyTitle: document.getElementById("historyTitle"),
      dataTitle: document.getElementById("dataTitle"),
      dataBody: document.getElementById("dataBody"),
      openData: document.getElementById("openData"),
      dataForgetBtn: document.getElementById("dataForgetBtn"),
      theme: document.getElementById("theme"),
      lang: document.getElementById("lang"),
      voice: document.getElementById("voice"),
      mariiBoost: document.getElementById("mariiBoost"),
      mariiBoostLabel: document.getElementById("mariiBoostLabel"),
      sourceChip: document.getElementById("sourceChip"),
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
      forgetBtn: document.getElementById("forgetBtn"),
      history: document.getElementById("historyList"),
      toast: document.getElementById("toast"),
      kbMeta: document.getElementById("kbMeta"),
      field: document.getElementById("fieldSim"),
      voiceBoot: document.getElementById("voiceBoot"),
      voiceBootMsg: document.getElementById("voiceBootMsg"),
      voiceBootSkip: document.getElementById("voiceBootSkip"),
    };
    load();
    applyTheme(state.theme);
    applyLang(state.lang);
    paintThemeOptions();
    paintLangOptions();
    paintVoiceOptions();
    els.voice.checked = state.voice;
    if (els.mariiBoost) els.mariiBoost.checked = state.mariiBoost;
    els.slack.value = state.slack;
    els.discord.value = state.discord;
    paintChrome();
    bind();
    setChatLocked(true);
    startSwirl(document.getElementById("swirlCanvas"));
    bootKnowledge()
      .then(function () {
        if (window.PyxHandoff) {
          window.PyxHandoff.applyIncoming({
            app: "pyxassistant",
            onQuery: function (q) {
              if (!state.voiceReady) pendingHandoff = q;
              else handleUserText(q, false);
            },
            onText: function (text) {
              if (!text) return;
              if (!state.voiceReady) pendingHandoff = text;
              else handleUserText(text, false);
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
