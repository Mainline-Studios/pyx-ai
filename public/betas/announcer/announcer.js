/**
 * Announcer beta — live MLB play-by-play with normal / booth voice.
 * Polls Stats API every 7s. Uses Pyx neural TTS (Sound of Text + Kokoro).
 */
(function () {
  "use strict";

  var MLB = "https://statsapi.mlb.com/api/v1";
  var LIVE = "https://statsapi.mlb.com/api/v1.1/game/";
  var POLL_MS = 7000;

  var state = {
    mode: "normal", // normal | announcer
    muted: false,
    voiceId: "en-US",
    voiceReady: false,
    gamePk: null,
    gameLabel: "",
    timer: 0,
    countdownTimer: 0,
    pollLeftMs: 0,
    lastAtBatIndex: -1,
    lastEventKey: "",
    lastScoreKey: "",
    lastStatus: "",
    started: false,
    speaking: false,
    queue: [],
  };

  var els = {};
  var voice = null;

  function $(id) {
    return document.getElementById(id);
  }

  function toast(msg) {
    els.toast.textContent = msg;
    els.toast.classList.add("is-on");
    clearTimeout(toast._t);
    toast._t = setTimeout(function () {
      els.toast.classList.remove("is-on");
    }, 2200);
  }

  function todayISO() {
    var d = new Date();
    var m = String(d.getMonth() + 1).padStart(2, "0");
    var day = String(d.getDate()).padStart(2, "0");
    return d.getFullYear() + "-" + m + "-" + day;
  }

  function teamName(side) {
    if (!side || !side.team) return "TBD";
    return side.team.teamName || side.team.name || "TBD";
  }

  function isLive(g) {
    var s = (g && g.status && (g.status.detailedState || g.status.abstractGameState)) || "";
    return /progress|live|challenge|review/i.test(s);
  }

  function isFinal(g) {
    var s = (g && g.status && g.status.detailedState) || "";
    return /final|completed|game over/i.test(s);
  }

  async function fetchSchedule() {
    var url =
      MLB +
      "/schedule?sportId=1&date=" +
      encodeURIComponent(todayISO()) +
      "&hydrate=team,linescore(matchup),probablePitcher";
    var res = await fetch(url);
    if (!res.ok) throw new Error("schedule " + res.status);
    var data = await res.json();
    return ((data.dates || [])[0] && data.dates[0].games) || [];
  }

  async function fetchLive(gamePk) {
    var res = await fetch(LIVE + gamePk + "/feed/live");
    if (!res.ok) throw new Error("live " + res.status);
    return res.json();
  }

  function scoreKey(ls, teams) {
    var as = teams && teams.away && teams.away.score;
    var hs = teams && teams.home && teams.home.score;
    if (as == null && ls) as = ls.teams && ls.teams.away && ls.teams.away.runs;
    if (hs == null && ls) hs = ls.teams && ls.teams.home && ls.teams.home.runs;
    return String(as) + "-" + String(hs);
  }

  function playKey(play) {
    if (!play) return "";
    var about = play.about || {};
    return [
      about.atBatIndex,
      about.playIndex,
      about.endTime || about.startTime || "",
      (play.result && play.result.eventType) || "",
      (play.result && play.result.description) || "",
    ].join("|");
  }

  function eventKey(play) {
    if (!play) return "";
    var events = play.playEvents || [];
    var last = events.length ? events[events.length - 1] : null;
    if (!last) return playKey(play) + "|e0";
    return playKey(play) + "|e" + events.length + "|" + ((last.details && last.details.description) || last.type || "");
  }

  function paintScoreboard(feed) {
    var gd = (feed && feed.gameData) || {};
    var teams = gd.teams || {};
    var away = teams.away || {};
    var home = teams.home || {};
    var ls = (feed.liveData && feed.liveData.linescore) || {};
    var status = (gd.status && gd.status.detailedState) || "";
    var as =
      (ls.teams && ls.teams.away && ls.teams.away.runs != null
        ? ls.teams.away.runs
        : away.score != null
          ? away.score
          : "—");
    var hs =
      (ls.teams && ls.teams.home && ls.teams.home.runs != null
        ? ls.teams.home.runs
        : home.score != null
          ? home.score
          : "—");
    var an = away.teamName || (away.team && away.team.teamName) || "Away";
    var hn = home.teamName || (home.team && home.team.teamName) || "Home";
    var box = (feed.liveData && feed.liveData.boxscore && feed.liveData.boxscore.teams) || {};
    if (box.away && box.away.team) an = box.away.team.teamName || box.away.team.name || an;
    if (box.home && box.home.team) hn = box.home.team.teamName || box.home.team.name || hn;

    els.matchup.textContent = an + " at " + hn;
    els.scoreLine.textContent = an + " " + as + "  ·  " + hn + " " + hs;
    var bits = [];
    if (ls.inningState || ls.currentInningOrdinal) {
      bits.push((ls.inningState || "") + " " + (ls.currentInningOrdinal || ""));
    }
    if (ls.balls != null && ls.strikes != null) bits.push(ls.balls + "-" + ls.strikes);
    if (ls.outs != null) bits.push(ls.outs === 1 ? "1 out" : ls.outs + " outs");
    if (status) bits.push(status);
    els.metaLine.textContent = bits.filter(Boolean).join(" · ") || "—";
    state.gameLabel = an + " at " + hn;
  }

  function colorizeAnnouncer(plain, kind) {
    var t = String(plain || "").trim();
    if (!t) return t;
    if (/home run|homers|grand slam/i.test(t)) {
      return "Gone! " + t + " What a shot from the booth!";
    }
    if (/strikes out|struck out|strikeout/i.test(t)) {
      return "Got him looking — " + t;
    }
    if (/\b(doubles|triples|triple|double)\b/i.test(t)) {
      return "Extra bases! " + t;
    }
    if (/scores|comes home|walks it off/i.test(t)) {
      return "And that’ll score — " + t;
    }
    if (/walks|base on balls/i.test(t)) {
      return "Free pass — " + t;
    }
    if (/error|wild pitch|passed ball/i.test(t)) {
      return "Trouble on the field — " + t;
    }
    if (kind === "score") {
      return "New scoreboard, folks: " + t;
    }
    if (kind === "status") {
      return "From the booth: " + t;
    }
    if (kind === "pitch") {
      return "Here’s the pitch… " + t;
    }
    var openers = [
      "And now — ",
      "Folks, ",
      "Would you look at that — ",
      "From deep in the booth: ",
      "Here we go — ",
    ];
    return openers[Math.floor(Math.random() * openers.length)] + t;
  }

  function formatLine(plain, kind) {
    if (state.mode === "announcer") return colorizeAnnouncer(plain, kind);
    return plain;
  }

  function pushLog(text) {
    var li = document.createElement("li");
    var when = document.createElement("span");
    when.className = "when";
    when.textContent = new Date().toLocaleTimeString();
    li.appendChild(when);
    li.appendChild(document.createTextNode(text));
    els.callLog.insertBefore(li, els.callLog.firstChild);
    while (els.callLog.children.length > 40) {
      els.callLog.removeChild(els.callLog.lastChild);
    }
  }

  function stopNeuralSpeak() {
    if (voice && typeof voice.stopSpeak === "function") voice.stopSpeak();
    state.speaking = false;
  }

  function speakNext() {
    if (state.muted || state.speaking || !state.queue.length) return;
    if (!state.voiceReady || !voice || typeof voice.speak !== "function") {
      return;
    }
    var line = state.queue.shift();
    state.speaking = true;
    els.lastCall.textContent = line;
    els.status.textContent = "Speaking…";
    voice.onSpeakEnd = function () {
      state.speaking = false;
      refreshStatusLine();
      speakNext();
    };
    voice.speak(line, "en-US");
  }

  function enqueue(plain, kind) {
    var line = formatLine(plain, kind);
    if (!line) return;
    pushLog(line);
    els.lastCall.textContent = line;
    if (!state.muted) {
      state.queue.push(line);
      speakNext();
    }
  }

  function extractUpdates(feed, bootstrap) {
    var plays = (((feed.liveData || {}).plays || {}).allPlays) || [];
    var current = ((feed.liveData || {}).plays || {}).currentPlay || null;
    var ls = (feed.liveData || {}).linescore || {};
    var gd = feed.gameData || {};
    var status = (gd.status && gd.status.detailedState) || "";
    var teams = {
      away: { score: ls.teams && ls.teams.away && ls.teams.away.runs },
      home: { score: ls.teams && ls.teams.home && ls.teams.home.runs },
    };
    var sk = scoreKey(ls, teams);
    var out = [];

    if (bootstrap) {
      var an = els.matchup.textContent || "This matchup";
      var intro =
        an +
        ". Score " +
        (els.scoreLine.textContent || sk.replace("-", "–")) +
        ". " +
        (status || "Live") +
        ".";
      out.push({ plain: intro, kind: "status" });
      if (current && current.result && current.result.description && current.about && current.about.isComplete) {
        out.push({ plain: current.result.description, kind: "play" });
      }
      state.lastAtBatIndex = -1;
      var bi;
      for (bi = 0; bi < plays.length; bi++) {
        var bp = plays[bi];
        var babout = bp.about || {};
        var bidx = Number(babout.atBatIndex);
        if (!isNaN(bidx) && babout.isComplete) state.lastAtBatIndex = bidx;
      }
      state.lastEventKey = eventKey(current);
      state.lastScoreKey = sk;
      state.lastStatus = status;
      return out;
    }

    if (status && status !== state.lastStatus) {
      out.push({ plain: "Game status: " + status + ".", kind: "status" });
      state.lastStatus = status;
    }

    if (sk && sk !== state.lastScoreKey && sk.indexOf("undefined") === -1) {
      out.push({ plain: "Score is now " + sk.replace("-", " to ") + ".", kind: "score" });
      state.lastScoreKey = sk;
    }

    var i;
    for (i = 0; i < plays.length; i++) {
      var p = plays[i];
      var about = p.about || {};
      var idx = Number(about.atBatIndex);
      if (isNaN(idx)) continue;
      if (idx <= state.lastAtBatIndex) continue;
      if (about.isComplete && p.result && p.result.description) {
        out.push({ plain: p.result.description, kind: "play" });
        state.lastAtBatIndex = idx;
      }
    }

    if (state.mode === "announcer" && current) {
      var ek = eventKey(current);
      if (ek && ek !== state.lastEventKey) {
        var events = current.playEvents || [];
        var last = events.length ? events[events.length - 1] : null;
        var desc = last && last.details && last.details.description;
        var isPitch = last && last.isPitch;
        if (desc && isPitch && !(current.about && current.about.isComplete)) {
          out.push({ plain: desc, kind: "pitch" });
        }
        state.lastEventKey = ek;
      }
    } else if (current) {
      state.lastEventKey = eventKey(current);
    }

    return out;
  }

  function setPollUi(kind, secs) {
    if (!els.pollSpinner || !els.pollCountdown || !els.pollSecs) return;
    var active = !!state.gamePk;
    els.pollSpinner.classList.toggle("is-on", active);
    els.pollSpinner.classList.toggle("is-fetch", kind === "fetch");
    els.pollCountdown.classList.toggle("is-hidden", !active || kind === "idle");
    if (typeof secs === "number") {
      els.pollSecs.textContent = String(Math.max(0, Math.ceil(secs)));
    }
  }

  function stopCountdown() {
    clearInterval(state.countdownTimer);
    state.countdownTimer = 0;
    state.pollLeftMs = 0;
    setPollUi("idle");
  }

  function startCountdown() {
    stopCountdown();
    if (!state.gamePk) return;
    state.pollLeftMs = POLL_MS;
    setPollUi("wait", POLL_MS / 1000);
    state.countdownTimer = setInterval(function () {
      state.pollLeftMs = Math.max(0, state.pollLeftMs - 250);
      setPollUi(state.pollLeftMs <= 0 ? "fetch" : "wait", state.pollLeftMs / 1000);
    }, 250);
  }

  function refreshStatusLine() {
    if (state.speaking) {
      els.status.textContent = "Speaking…";
      return;
    }
    if (!state.gamePk) {
      els.status.textContent = "Idle";
      return;
    }
    els.status.textContent = "Listening";
  }

  async function tick(bootstrap) {
    if (!state.gamePk) return;
    setPollUi("fetch", 0);
    try {
      var feed = await fetchLive(state.gamePk);
      paintScoreboard(feed);
      var updates = extractUpdates(feed, !!bootstrap);
      updates.forEach(function (u) {
        enqueue(u.plain, u.kind);
      });
      refreshStatusLine();
    } catch (err) {
      els.status.textContent = "Feed hiccup — retrying…";
    }
    if (state.gamePk) startCountdown();
  }

  function stopPolling() {
    clearInterval(state.timer);
    state.timer = 0;
    state.gamePk = null;
    state.started = false;
    state.queue = [];
    stopCountdown();
    stopNeuralSpeak();
  }

  function startGame(game) {
    stopPolling();
    if (voice && typeof voice.unlockAudio === "function") voice.unlockAudio();
    state.gamePk = game.gamePk;
    state.lastAtBatIndex = -1;
    state.lastEventKey = "";
    state.lastScoreKey = "";
    state.lastStatus = "";
    state.started = true;
    els.pickerPanel.classList.add("is-hidden");
    els.livePanel.classList.remove("is-hidden");
    els.status.textContent = "Connecting to live feed…";
    els.lastCall.textContent = "Warming up the booth…";
    setPollUi("fetch", 0);
    tick(true);
    state.timer = setInterval(function () {
      tick(false);
    }, POLL_MS);
  }

  function renderGames(games) {
    els.gameList.innerHTML = "";
    if (!games.length) {
      els.gamesHint.textContent = "No MLB games on today’s slate.";
      return;
    }
    els.gamesHint.textContent = games.length + " games · tap one to listen";
    games.forEach(function (g) {
      var away = teamName(g.teams && g.teams.away);
      var home = teamName(g.teams && g.teams.home);
      var as = g.teams && g.teams.away && g.teams.away.score;
      var hs = g.teams && g.teams.home && g.teams.home.score;
      var stateLabel = (g.status && g.status.detailedState) || "Scheduled";
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "game-card";
      btn.setAttribute("role", "listitem");
      var title = document.createElement("div");
      title.className = "game-card__title";
      title.textContent = away + " @ " + home;
      var badge = document.createElement("span");
      badge.className = "game-card__state";
      if (isLive(g)) badge.textContent = "Live";
      else if (isFinal(g)) {
        badge.textContent = "Final";
        badge.classList.add("is-final");
      } else {
        badge.textContent = "Upcoming";
        badge.classList.add("is-sched");
      }
      var meta = document.createElement("div");
      meta.className = "game-card__meta";
      var scoreBit = as != null && hs != null ? as + "–" + hs + " · " : "";
      meta.textContent = scoreBit + stateLabel;
      btn.appendChild(title);
      btn.appendChild(badge);
      btn.appendChild(meta);
      btn.addEventListener("click", function () {
        startGame(g);
      });
      els.gameList.appendChild(btn);
    });
  }

  async function loadGames() {
    els.gamesHint.textContent = "Loading schedule…";
    try {
      var games = await fetchSchedule();
      games.sort(function (a, b) {
        return (isLive(b) ? 1 : 0) - (isLive(a) ? 1 : 0);
      });
      renderGames(games);
    } catch (err) {
      els.gamesHint.textContent = "Couldn’t load MLB schedule. Try refresh.";
    }
  }

  function setMode(mode) {
    state.mode = mode === "announcer" ? "announcer" : "normal";
    els.modeNormal.classList.toggle("is-on", state.mode === "normal");
    els.modeAnnouncer.classList.toggle("is-on", state.mode === "announcer");
    try {
      localStorage.setItem("pyx.announcer.mode", state.mode);
    } catch (e) {}
  }

  function setMuted(on) {
    state.muted = !!on;
    els.muteBtn.textContent = state.muted ? "🔇" : "🔊";
    els.muteBtn.setAttribute("aria-label", state.muted ? "Unmute" : "Mute");
    if (state.muted) {
      state.queue = [];
      stopNeuralSpeak();
    }
    try {
      localStorage.setItem("pyx.announcer.muted", state.muted ? "1" : "0");
    } catch (e) {}
  }

  function paintVoiceOptions() {
    if (!els.voiceSelect) return;
    var names =
      (voice && voice.listVoices && voice.listVoices()) || [
        { id: "en-US", label: "Online neural · US" },
        { id: "en-GB", label: "Online neural · British" },
        { id: "en-AU", label: "Online neural · Australian" },
        { id: "en-IN", label: "Online neural · Indian English" },
      ];
    var cur = state.voiceId;
    els.voiceSelect.innerHTML = "";
    names.forEach(function (v) {
      var opt = document.createElement("option");
      opt.value = v.id;
      opt.textContent = v.label;
      els.voiceSelect.appendChild(opt);
    });
    if (names.some(function (v) { return v.id === cur; })) {
      els.voiceSelect.value = cur;
    } else if (names.length) {
      state.voiceId = names[0].id;
      els.voiceSelect.value = state.voiceId;
    }
  }

  function setVoiceId(id) {
    state.voiceId = id || "en-US";
    if (voice && voice.setVoice) voice.setVoice(state.voiceId);
    try {
      localStorage.setItem("pyx.announcer.voiceId", state.voiceId);
    } catch (e) {}
  }

  function setVoiceBootMsg(msg) {
    if (els.voiceBootMsg) els.voiceBootMsg.textContent = msg || "";
    if (els.voiceHint) els.voiceHint.textContent = msg || "";
  }

  function showVoiceBootSkip(show) {
    if (!els.voiceBootSkip) return;
    els.voiceBootSkip.classList.toggle("is-hidden", !show);
  }

  function finishVoiceBoot(msg) {
    state.voiceReady = true;
    setVoiceBootMsg(msg || "Neural TTS ready.");
    showVoiceBootSkip(false);
    if (els.voiceBoot) {
      els.voiceBoot.classList.add("is-done");
      els.voiceBoot.setAttribute("aria-busy", "false");
    }
    speakNext();
  }

  function bootVoice() {
    voice = window.PyxAssistantVoice;
    if (!voice || !voice.warmup) {
      setVoiceBootMsg("Neural voice module missing.");
      showVoiceBootSkip(false);
      if (els.voiceBoot) els.voiceBoot.classList.add("is-done");
      return;
    }
    voice.allowBrowserFallback = false;
    setVoiceId(state.voiceId);
    paintVoiceOptions();
    voice.onError = function () {
      toast("Neural voice failed — try another voice or wait for Kokoro.");
    };
    voice.onOnlineReady = function () {
      setVoiceBootMsg("Online neural voice ready. Still loading Kokoro…");
      showVoiceBootSkip(true);
      if (!state.voiceReady) {
        state.voiceReady = true;
        speakNext();
      }
    };
    setVoiceBootMsg("Downloading neural TTS…");
    voice
      .warmup(function (msg) {
        setVoiceBootMsg(msg || "Getting voice ready…");
        if (voice.ready && voice.ready.kokoro) {
          paintVoiceOptions();
          if (voice.voiceId && voice.voiceId.indexOf("af_") === 0) {
            var locked = false;
            try {
              locked = !!localStorage.getItem("pyx.announcer.voiceId");
            } catch (e) {}
            if (!locked) {
              state.voiceId = voice.voiceId;
              paintVoiceOptions();
              els.voiceSelect.value = state.voiceId;
            } else {
              voice.setVoice(state.voiceId);
            }
          }
        }
      })
      .then(function () {
        paintVoiceOptions();
        setVoiceId(state.voiceId);
        var doneMsg =
          voice.ready && voice.ready.kokoro
            ? "Voice ready — Kokoro on-device."
            : "Voice ready — online neural TTS.";
        finishVoiceBoot(doneMsg);
      })
      .catch(function () {
        finishVoiceBoot("Online neural TTS ready.");
      });
  }

  function bind() {
    els.refreshGames.addEventListener("click", loadGames);
    els.backToGames.addEventListener("click", function () {
      stopPolling();
      els.livePanel.classList.add("is-hidden");
      els.pickerPanel.classList.remove("is-hidden");
      els.status.textContent = "Idle";
      loadGames();
    });
    els.modeNormal.addEventListener("click", function () {
      setMode("normal");
    });
    els.modeAnnouncer.addEventListener("click", function () {
      setMode("announcer");
    });
    els.muteBtn.addEventListener("click", function () {
      setMuted(!state.muted);
      toast(state.muted ? "Muted" : "Unmuted");
    });
    els.clearLog.addEventListener("click", function () {
      els.callLog.innerHTML = "";
    });
    els.voiceSelect.addEventListener("change", function () {
      setVoiceId(els.voiceSelect.value);
      toast("Voice updated");
    });
    if (els.voiceBootSkip) {
      els.voiceBootSkip.addEventListener("click", function () {
        if (voice && typeof voice.unlockAudio === "function") voice.unlockAudio();
        finishVoiceBoot("Using online neural voice.");
      });
    }
    function unlockOnce() {
      if (voice && typeof voice.unlockAudio === "function") voice.unlockAudio();
      document.removeEventListener("pointerdown", unlockOnce);
    }
    document.addEventListener("pointerdown", unlockOnce);
  }

  function init() {
    els = {
      pickerPanel: $("pickerPanel"),
      livePanel: $("livePanel"),
      gameList: $("gameList"),
      gamesHint: $("gamesHint"),
      refreshGames: $("refreshGames"),
      backToGames: $("backToGames"),
      modeNormal: $("modeNormal"),
      modeAnnouncer: $("modeAnnouncer"),
      muteBtn: $("muteBtn"),
      matchup: $("matchup"),
      scoreLine: $("scoreLine"),
      metaLine: $("metaLine"),
      status: $("status"),
      pollSpinner: $("pollSpinner"),
      pollCountdown: $("pollCountdown"),
      pollSecs: $("pollSecs"),
      lastCall: $("lastCall"),
      callLog: $("callLog"),
      clearLog: $("clearLog"),
      toast: $("toast"),
      voiceSelect: $("voiceSelect"),
      voiceHint: $("voiceHint"),
      voiceBoot: $("voiceBoot"),
      voiceBootMsg: $("voiceBootMsg"),
      voiceBootSkip: $("voiceBootSkip"),
    };
    try {
      var m = localStorage.getItem("pyx.announcer.mode");
      if (m) setMode(m);
      if (localStorage.getItem("pyx.announcer.muted") === "1") setMuted(true);
      var vid = localStorage.getItem("pyx.announcer.voiceId");
      if (vid) state.voiceId = vid;
    } catch (e) {}
    bind();
    loadGames();
    if (window.PyxAssistantVoice) bootVoice();
    else window.addEventListener("pyx-voice-ready", bootVoice, { once: true });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
