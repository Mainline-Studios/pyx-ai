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
    voiceId: "bm_lewis",
    voiceReady: false,
    gamePk: null,
    gameLabel: "",
    lastFeed: null,
    histCache: null,
    lastAskAnswer: "",
    asking: false,
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
      state.lastFeed = feed;
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
    state.lastFeed = null;
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
    state.lastAskAnswer = "";
    state.histCache = null;
    state.started = true;
    els.pickerPanel.classList.add("is-hidden");
    els.liveLayout.classList.remove("is-hidden");
    els.status.textContent = "Connecting to live feed…";
    els.lastCall.textContent = "Warming up the booth…";
    if (els.askAnswer) {
      els.askAnswer.innerHTML = '<p class="ask-answer__empty">Pick a chip or type a question about this game.</p>';
    }
    if (els.askSpeak) els.askSpeak.disabled = true;
    if (els.askInput) els.askInput.value = "";
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

  function playerFromBox(teamBox, id) {
    if (!teamBox || !teamBox.players || id == null) return null;
    return teamBox.players["ID" + id] || teamBox.players[String(id)] || null;
  }

  function playerName(entry) {
    if (!entry) return "";
    if (entry.person && entry.person.fullName) return entry.person.fullName;
    if (entry.fullName) return entry.fullName;
    return "";
  }

  function lineupLines(feed, side) {
    var box = (((feed || {}).liveData || {}).boxscore || {}).teams || {};
    var teamBox = box[side] || {};
    var order = teamBox.battingOrder || [];
    var lines = [];
    var i;
    for (i = 0; i < order.length; i++) {
      var p = playerFromBox(teamBox, order[i]);
      var name = playerName(p) || "Unknown";
      var pos =
        (p && p.position && (p.position.abbreviation || p.position.name)) || "";
      var bat =
        p && p.stats && p.stats.batting
          ? p.stats.batting
          : {};
      var bit =
        i +
        1 +
        ". " +
        name +
        (pos ? " (" + pos + ")" : "");
      if (bat.atBats != null) {
        bit +=
          " — " +
          (bat.hits != null ? bat.hits : "0") +
          "-" +
          bat.atBats +
          (bat.homeRuns ? ", " + bat.homeRuns + " HR" : "") +
          (bat.rbi ? ", " + bat.rbi + " RBI" : "");
      }
      lines.push(bit);
    }
    return lines;
  }

  function teamLabel(feed, side) {
    var box = (((feed || {}).liveData || {}).boxscore || {}).teams || {};
    var t = box[side] && box[side].team;
    if (t) return t.teamName || t.name || side;
    var gd = ((feed || {}).gameData || {}).teams || {};
    var g = gd[side];
    if (g) return g.teamName || (g.team && g.team.teamName) || side;
    return side;
  }

  function teamAliases(feed, side) {
    var out = [];
    function add(s) {
      s = String(s || "")
        .toLowerCase()
        .trim();
      if (!s || out.indexOf(s) !== -1) return;
      out.push(s);
      var parts = s.split(/\s+/).filter(Boolean);
      if (parts.length > 1) {
        var nick = parts[parts.length - 1];
        if (nick.length > 2 && out.indexOf(nick) === -1) out.push(nick);
      }
    }
    var box = (((feed || {}).liveData || {}).boxscore || {}).teams || {};
    var bt = box[side] && box[side].team;
    if (bt) {
      add(bt.teamName);
      add(bt.name);
      add(bt.abbreviation);
      add(bt.locationName);
      add(bt.shortName);
    }
    var gd = ((feed || {}).gameData || {}).teams || {};
    var g = gd[side];
    if (g) {
      add(g.teamName);
      add(g.name);
      add(g.abbreviation);
      if (g.team) {
        add(g.team.teamName);
        add(g.team.name);
        add(g.team.abbreviation);
      }
    }
    add(teamLabel(feed, side));
    return out;
  }

  function questionMentionsSide(q, feed, side) {
    var t = String(q || "").toLowerCase();
    return teamAliases(feed, side).some(function (a) {
      if (a.length <= 3) return new RegExp("\\b" + a.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "\\b").test(t);
      return t.indexOf(a) !== -1;
    });
  }

  function liveSeasonRecord(feed, side) {
    var gd = ((feed || {}).gameData || {}).teams || {};
    var g = gd[side] || {};
    var rec = g.record || (g.team && g.team.record) || {};
    var league = rec.leagueRecord || rec;
    var w = league.wins != null ? Number(league.wins) : null;
    var l = league.losses != null ? Number(league.losses) : null;
    if (w == null || l == null) return null;
    return { w: w, l: l, pct: league.pct || null };
  }

  function formBlurb(label, form, seasonRec) {
    var bits = [];
    bits.push("MARII on " + label + ":");
    if (seasonRec) {
      bits.push(
        "season mark " +
          seasonRec.w +
          "-" +
          seasonRec.l +
          (seasonRec.pct ? " (" + seasonRec.pct + ")" : "") +
          "."
      );
    }
    if (form && form.games) {
      bits.push(
        "Last " +
          form.games +
          " finals: " +
          form.w +
          "-" +
          form.l +
          ", averaging " +
          round1(form.rpg) +
          " RPG / " +
          round1(form.rapg) +
          " RA/G (recent stretch " +
          round1(form.recentRpg) +
          " / " +
          round1(form.recentRapg) +
          ")."
      );
      if (form.lastScores && form.lastScores.length) {
        bits.push("Runs scored last " + form.lastScores.length + ": " + form.lastScores.join(", ") + ".");
      }
    } else {
      bits.push("Not enough recent finals in the lookback window for a form line yet.");
    }
    return bits.join(" ");
  }

  async function answerFormAlgo(feed, q) {
    var wantsAway = questionMentionsSide(q, feed, "away");
    var wantsHome = questionMentionsSide(q, feed, "home");
    var sides =
      wantsAway && !wantsHome ? ["away"] : wantsHome && !wantsAway ? ["home"] : ["away", "home"];
    try {
      var hist = await loadHist(feed);
      if (!hist) {
        return "MARII couldn’t load recent finals yet — try again in a moment.";
      }
      var lines = sides.map(function (side) {
        var form = side === "away" ? hist.awayForm : hist.homeForm;
        return formBlurb(teamLabel(feed, side), form, liveSeasonRecord(feed, side));
      });
      lines.push("Game-scoped MARII form from the live board + recent finals — not a full season scouting report.");
      return lines.join(" ");
    } catch (err) {
      return mariiUnknown();
    }
  }

  function teamBox(feed, side) {
    return ((((feed || {}).liveData || {}).boxscore || {}).teams || {})[side] || {};
  }

  function scoreSnap(feed) {
    var ls = (feed.liveData && feed.liveData.linescore) || {};
    var away = teamLabel(feed, "away");
    var home = teamLabel(feed, "home");
    var ar = Number((ls.teams && ls.teams.away && ls.teams.away.runs) || 0);
    var hr = Number((ls.teams && ls.teams.home && ls.teams.home.runs) || 0);
    return {
      ls: ls,
      away: away,
      home: home,
      ar: ar,
      hr: hr,
      lead: ar === hr ? 0 : ar > hr ? "away" : "home",
      margin: Math.abs(ar - hr),
      inning: Number(ls.currentInning || 0),
      inningState: ls.inningState || "",
      inningOrd: ls.currentInningOrdinal || "",
      balls: ls.balls,
      strikes: ls.strikes,
      outs: ls.outs,
      status: (((feed.gameData || {}).status || {}).detailedState) || "",
      offense: ls.offense || {},
      defense: ls.defense || {},
    };
  }

  function battingEntries(feed, side) {
    var box = teamBox(feed, side);
    var order = box.battingOrder || [];
    var out = [];
    var i;
    for (i = 0; i < order.length; i++) {
      var p = playerFromBox(box, order[i]);
      if (!p) continue;
      var bat = (p.stats && p.stats.batting) || {};
      out.push({
        name: playerName(p),
        pos: (p.position && (p.position.abbreviation || p.position.name)) || "",
        hits: Number(bat.hits || 0),
        abs: Number(bat.atBats || 0),
        hr: Number(bat.homeRuns || 0),
        rbi: Number(bat.rbi || 0),
        runs: Number(bat.runs || 0),
        bb: Number(bat.baseOnBalls || 0),
        so: Number(bat.strikeOuts || 0),
      });
    }
    return out;
  }

  function pitchingEntries(feed, side) {
    var box = teamBox(feed, side);
    var ids = box.pitchers || [];
    return ids
      .map(function (id) {
        var p = playerFromBox(box, id);
        if (!p) return null;
        var st = (p.stats && p.stats.pitching) || {};
        return {
          name: playerName(p),
          ip: st.inningsPitched != null ? String(st.inningsPitched) : "",
          er: Number(st.earnedRuns || 0),
          h: Number(st.hits || 0),
          so: Number(st.strikeOuts || 0),
          bb: Number(st.baseOnBalls || 0),
          pitches: Number(st.numberOfPitches || st.pitchesThrown || 0),
        };
      })
      .filter(Boolean);
  }

  function runnersText(offense) {
    var runners = [];
    if (offense.first && offense.first.fullName) runners.push("1B " + offense.first.fullName);
    if (offense.second && offense.second.fullName) runners.push("2B " + offense.second.fullName);
    if (offense.third && offense.third.fullName) runners.push("3B " + offense.third.fullName);
    return runners.length ? runners.join(", ") : "bases empty";
  }

  function preferredSides(q, feed) {
    var wantsAway = questionMentionsSide(q, feed, "away");
    var wantsHome = questionMentionsSide(q, feed, "home");
    if (wantsAway && !wantsHome) return ["away"];
    if (wantsHome && !wantsAway) return ["home"];
    return ["away", "home"];
  }

  function teamStatPack(feed, side) {
    var box = teamBox(feed, side);
    var bat = (box.teamStats && box.teamStats.batting) || {};
    var pit = (box.teamStats && box.teamStats.pitching) || {};
    var fld = (box.teamStats && box.teamStats.fielding) || {};
    return {
      hits: Number(bat.hits || 0),
      runs: Number(bat.runs || 0),
      hr: Number(bat.homeRuns || 0),
      bb: Number(bat.baseOnBalls || 0),
      so: Number(bat.strikeOuts || 0),
      lob: Number(bat.leftOnBase || 0),
      avg: bat.avg || "",
      er: Number(pit.earnedRuns || 0),
      pitchSo: Number(pit.strikeOuts || 0),
      pitchBb: Number(pit.baseOnBalls || 0),
      hitsAllowed: Number(pit.hits || 0),
      errors: Number(fld.errors || 0),
    };
  }

  function findPlayersInFeed(feed, needle) {
    var n = String(needle || "")
      .toLowerCase()
      .trim();
    if (n.length < 2) return [];
    var hits = [];
    ["away", "home"].forEach(function (side) {
      var box = teamBox(feed, side);
      var players = box.players || {};
      Object.keys(players).forEach(function (key) {
        var p = players[key];
        var name = playerName(p);
        if (!name) return;
        var low = name.toLowerCase();
        var last = low.split(/\s+/).pop();
        if (low.indexOf(n) === -1 && last !== n && n.indexOf(last) === -1) return;
        if (n.length <= 2 && last !== n) return;
        var bat = (p.stats && p.stats.batting) || {};
        var pit = (p.stats && p.stats.pitching) || {};
        hits.push({
          side: side,
          name: name,
          pos: (p.position && (p.position.abbreviation || p.position.name)) || "",
          bat: bat,
          pit: pit,
        });
      });
    });
    return hits;
  }

  function extractPlayerNeedle(q, feed) {
    var t = String(q || "");
    var cleaned = t
      .replace(/\b(how|what|who|where|when|is|are|was|were|the|a|an|about|for|with|of|on|in|this|that|game|tonight|today|doing|looking|stats?|line|box)\b/gi, " ")
      .replace(/[?.!,]/g, " ")
      .replace(/\s+/g, " ")
      .trim();
    teamAliases(feed, "away")
      .concat(teamAliases(feed, "home"))
      .forEach(function (a) {
        cleaned = cleaned.replace(new RegExp("\\b" + a.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "\\b", "gi"), " ");
      });
    cleaned = cleaned.replace(/\s+/g, " ").trim();
    if (cleaned.length < 3) return "";
    return cleaned;
  }

  function answerLineup(feed, q) {
    var sides = preferredSides(q, feed);
    var chunks = [];
    sides.forEach(function (side) {
      var lines = lineupLines(feed, side);
      if (!lines.length) return;
      chunks.push(teamLabel(feed, side) + " batting order:\n" + lines.join("\n"));
    });
    if (!chunks.length) return "Lineups aren’t posted in the feed yet.";
    return chunks.join("\n\n");
  }

  function answerMatchup(feed) {
    var s = scoreSnap(feed);
    var pitcher = (s.defense.pitcher && s.defense.pitcher.fullName) || "Unknown";
    var batter = (s.offense.batter && s.offense.batter.fullName) || "Unknown";
    var onDeck = (s.offense.onDeck && s.offense.onDeck.fullName) || "";
    var inHole = (s.offense.inHole && s.offense.inHole.fullName) || "";
    var count =
      s.balls != null && s.strikes != null
        ? " Count " + s.balls + "-" + s.strikes + ", " + (s.outs != null ? s.outs : "?") + " out."
        : "";
    var parts = [
      pitcher + " is on the mound against " + batter + "." + count,
      "Runners: " + runnersText(s.offense) + ".",
    ];
    if (onDeck) parts.push("On deck: " + onDeck + ".");
    if (inHole) parts.push("In the hole: " + inHole + ".");
    return parts.join(" ");
  }

  function answerDueUp(feed) {
    var s = scoreSnap(feed);
    var batter = (s.offense.batter && s.offense.batter.fullName) || "";
    var onDeck = (s.offense.onDeck && s.offense.onDeck.fullName) || "";
    var inHole = (s.offense.inHole && s.offense.inHole.fullName) || "";
    if (!batter && !onDeck) return "Due-up order isn’t in the live feed yet.";
    return (
      "Due up: " +
      (batter || "current batter n/a") +
      (onDeck ? ", on deck " + onDeck : "") +
      (inHole ? ", in the hole " + inHole : "") +
      "."
    );
  }

  function answerSituation(feed) {
    var s = scoreSnap(feed);
    return (
      s.away +
      " " +
      s.ar +
      ", " +
      s.home +
      " " +
      s.hr +
      ". " +
      ((s.inningState + " " + s.inningOrd).trim() || "Inning n/a") +
      (s.status ? " (" + s.status + ")" : "") +
      ". " +
      (s.balls != null && s.strikes != null
        ? "Count " + s.balls + "-" + s.strikes + ", " + (s.outs != null ? s.outs : "?") + " out. "
        : "") +
      "Runners: " +
      runnersText(s.offense) +
      "."
    );
  }

  function answerWhoWinning(feed) {
    var s = scoreSnap(feed);
    if (s.ar === s.hr) {
      return (
        "Tied " +
        s.ar +
        "-" +
        s.hr +
        " in the " +
        ((s.inningState + " " + s.inningOrd).trim() || "current inning") +
        "."
      );
    }
    var leader = s.lead === "away" ? s.away : s.home;
    return (
      leader +
      " lead " +
      s.away +
      " " +
      s.ar +
      ", " +
      s.home +
      " " +
      s.hr +
      " (" +
      s.margin +
      "-run game) — " +
      ((s.inningState + " " + s.inningOrd).trim() || "inning n/a") +
      "."
    );
  }

  function answerHotBats(feed, q) {
    var sides = preferredSides(q, feed);
    var all = [];
    sides.forEach(function (side) {
      battingEntries(feed, side).forEach(function (b) {
        if (!b.name) return;
        var score = b.hits * 3 + b.hr * 4 + b.rbi * 2 + b.runs + b.bb;
        all.push({ side: side, b: b, score: score });
      });
    });
    all.sort(function (a, b) {
      return b.score - a.score || b.b.hits - a.b.hits;
    });
    var hot = all.filter(function (x) {
      return x.score > 0;
    }).slice(0, 5);
    if (!hot.length) {
      return "Nobody’s piled up counting stats yet — early or quiet bats so far.";
    }
    var lines = hot.map(function (x) {
      var b = x.b;
      return (
        b.name +
        " (" +
        teamLabel(feed, x.side) +
        "): " +
        b.hits +
        "-" +
        b.abs +
        (b.hr ? ", " + b.hr + " HR" : "") +
        (b.rbi ? ", " + b.rbi + " RBI" : "") +
        (b.runs ? ", " + b.runs + " R" : "")
      );
    });
    return "Hot bats so far from the box score:\n" + lines.join("\n");
  }

  function answerColdBats(feed, q) {
    var sides = preferredSides(q, feed);
    var all = [];
    sides.forEach(function (side) {
      battingEntries(feed, side).forEach(function (b) {
        if (!b.name || b.abs < 2) return;
        all.push({ side: side, b: b, score: b.hits * 3 + b.hr * 4 + b.rbi - b.so });
      });
    });
    all.sort(function (a, b) {
      return a.score - b.score || b.b.abs - a.b.abs;
    });
    var cold = all.filter(function (x) {
      return x.b.hits === 0 || x.score <= 0;
    }).slice(0, 5);
    if (!cold.length) return "No clear cold bats in the box yet.";
    return (
      "Quiet bats so far:\n" +
      cold
        .map(function (x) {
          return (
            x.b.name +
            " (" +
            teamLabel(feed, x.side) +
            "): " +
            x.b.hits +
            "-" +
            x.b.abs +
            (x.b.so ? ", " + x.b.so + " K" : "")
          );
        })
        .join("\n")
    );
  }

  function answerLeaders(feed, q) {
    var t = String(q || "").toLowerCase();
    var kind = /\b(home ?runs?|homers?|hr)\b/.test(t)
      ? "hr"
      : /\b(rbi|ribbies)\b/.test(t)
        ? "rbi"
        : /\b(strikeouts?|punchouts?|k'?s)\b/.test(t)
          ? "so"
          : /\b(walks?|bb)\b/.test(t)
            ? "bb"
            : /\b(runs?\b)/.test(t)
              ? "runs"
              : "hits";
    var sides = preferredSides(q, feed);
    var all = [];
    sides.forEach(function (side) {
      battingEntries(feed, side).forEach(function (b) {
        var v = Number(b[kind] || 0);
        if (!v) return;
        all.push({ side: side, b: b, v: v });
      });
    });
    if (kind === "so" && !all.length) {
      sides.forEach(function (side) {
        pitchingEntries(feed, side).forEach(function (p) {
          if (!p.so) return;
          all.push({ side: side, b: { name: p.name, so: p.so }, v: p.so, pitching: true });
        });
      });
    }
    all.sort(function (a, b) {
      return b.v - a.v;
    });
    if (!all.length) return "No " + kind.toUpperCase() + " leaders on the board yet.";
    var label =
      kind === "hr" ? "HR" : kind === "rbi" ? "RBI" : kind === "so" ? "K" : kind === "bb" ? "BB" : kind === "runs" ? "R" : "H";
    return (
      "Leaders (" +
      label +
      ") tonight:\n" +
      all
        .slice(0, 6)
        .map(function (x) {
          return x.b.name + " (" + teamLabel(feed, x.side) + "): " + x.v + " " + label;
        })
        .join("\n")
    );
  }

  function answerBullpen(feed, q) {
    var s = scoreSnap(feed);
    var sides = preferredSides(q, feed);
    var bits = [];
    sides.forEach(function (side) {
      var staff = pitchingEntries(feed, side);
      if (!staff.length) return;
      var label = teamLabel(feed, side);
      var lines = staff.map(function (p, idx) {
        var role = idx === 0 ? "starter/first" : "relief";
        return (
          p.name +
          " (" +
          role +
          ")" +
          (p.ip ? " " + p.ip + " IP" : "") +
          ", " +
          p.er +
          " ER, " +
          p.so +
          " K" +
          (p.pitches ? ", " + p.pitches + " pitches" : "")
        );
      });
      bits.push(label + " pitchers used:\n" + lines.join("\n"));
      var last = staff[staff.length - 1];
      if (last && last.pitches >= 25 && staff.length === 1) {
        bits.push(label + " still on the starter — a bullpen arm could be next if the pitch count climbs.");
      } else if (staff.length > 1) {
        bits.push(label + " already into the pen; current arm is " + last.name + ".");
      }
    });
    var current = (s.defense.pitcher && s.defense.pitcher.fullName) || "";
    if (current) bits.unshift("Pitching now: " + current + ".");
    if (!bits.length) return "Bullpen detail isn’t in the feed yet.";
    return bits.join("\n\n");
  }

  function answerTeamBox(feed, q) {
    var sides = preferredSides(q, feed);
    var lines = sides.map(function (side) {
      var st = teamStatPack(feed, side);
      return (
        teamLabel(feed, side) +
        ": " +
        st.runs +
        " R, " +
        st.hits +
        " H, " +
        st.errors +
        " E, " +
        st.hr +
        " HR, " +
        st.bb +
        " BB, " +
        st.so +
        " K, " +
        st.lob +
        " LOB."
      );
    });
    return "Team box tonight:\n" + lines.join("\n");
  }

  function answerLinescore(feed) {
    var ls = ((feed || {}).liveData || {}).linescore || {};
    var inns = ls.innings || [];
    if (!inns.length) return "Inning-by-inning linescore isn’t posted yet.";
    var s = scoreSnap(feed);
    var header = "Inn " + inns.map(function (inn) { return inn.num; }).join(" ");
    var away = s.away + " " + inns.map(function (inn) {
      return inn.away && inn.away.runs != null ? inn.away.runs : "·";
    }).join(" ") + " | R " + s.ar;
    var home = s.home + " " + inns.map(function (inn) {
      return inn.home && inn.home.runs != null ? inn.home.runs : "·";
    }).join(" ") + " | R " + s.hr;
    return "Linescore:\n" + header + "\n" + away + "\n" + home;
  }

  function answerVenue(feed) {
    var gd = (feed && feed.gameData) || {};
    var venue = (gd.venue && gd.venue.name) || "Venue n/a";
    var loc = gd.venue && (gd.venue.location || {});
    var city = loc.city || "";
    var stateName = loc.stateAbbrev || loc.state || "";
    var where = [city, stateName].filter(Boolean).join(", ");
    var dt = gd.datetime || {};
    var first = dt.time || dt.dateTime || "";
    return (
      "Ballpark: " +
      venue +
      (where ? " (" + where + ")" : "") +
      "." +
      (first ? " Listed first pitch / game time: " + first + "." : "")
    );
  }

  function answerWeather(feed) {
    var w = ((feed && feed.gameData) || {}).weather || {};
    if (!w.condition && w.temp == null && !w.wind) {
      return "Ballpark weather isn’t in this feed yet.";
    }
    return (
      "Ballpark weather: " +
      (w.condition || "n/a") +
      (w.temp != null ? ", " + w.temp + "°" : "") +
      (w.wind ? ", wind " + w.wind : "") +
      "."
    );
  }

  function answerProbables(feed) {
    var gd = (feed && feed.gameData) || {};
    var pp = gd.probablePitchers || {};
    var away = (pp.away && (pp.away.fullName || pp.away.name)) || "";
    var home = (pp.home && (pp.home.fullName || pp.home.name)) || "";
    var bits = [];
    if (away) bits.push(teamLabel(feed, "away") + " probable/listed: " + away);
    if (home) bits.push(teamLabel(feed, "home") + " probable/listed: " + home);
    if (!bits.length) {
      return answerStarter(feed);
    }
    return bits.join(". ") + ". " + answerStarter(feed);
  }

  function answerLastPlay(feed) {
    var plays = ((((feed || {}).liveData || {}).plays || {}).allPlays) || [];
    var cur = ((((feed || {}).liveData || {}).plays || {}).currentPlay) || null;
    var play = cur || plays[plays.length - 1];
    if (!play) return "No play description in the feed yet.";
    var result = (play.result && (play.result.description || play.result.event)) || "";
    var about = play.about || {};
    var half = about.halfInning || "";
    var inn = about.inning != null ? about.inning : "";
    return (
      "Latest play" +
      (inn ? " (" + half + " " + inn + ")" : "") +
      ": " +
      (result || "n/a") +
      "."
    );
  }

  function answerRecap(feed) {
    return (
      answerSituation(feed) +
      " " +
      answerMatchup(feed) +
      " " +
      answerTeamBox(feed, "") +
      " Tap Projections for the MARII prior-games model, or Form for recent finals."
    );
  }

  function answerDecisions(feed) {
    var dec = ((((feed || {}).liveData || {}).decisions) || {});
    var w = dec.winner && (dec.winner.fullName || dec.winner.name);
    var l = dec.loser && (dec.loser.fullName || dec.loser.name);
    var sv = dec.save && (dec.save.fullName || dec.save.name);
    if (!w && !l) return "Win/loss/save decisions aren’t posted yet (usually after the final).";
    return (
      "Decisions: " +
      (w ? "W " + w : "W n/a") +
      ", " +
      (l ? "L " + l : "L n/a") +
      (sv ? ", S " + sv : "") +
      "."
    );
  }

  function answerUmpires(feed) {
    var officials = ((((feed || {}).liveData || {}).boxscore || {}).officials) || [];
    if (!officials.length) {
      var gdOff = ((((feed || {}).gameData || {}).officials) || []);
      officials = gdOff;
    }
    if (!officials.length) return "Umpire crew isn’t listed in the feed yet.";
    return (
      "Umpires: " +
      officials
        .map(function (o) {
          var name = (o.official && o.official.fullName) || o.fullName || "n/a";
          var type = o.officialType || o.type || "";
          return name + (type ? " (" + type + ")" : "");
        })
        .join("; ") +
      "."
    );
  }

  function answerSeries(feed) {
    var gd = (feed && feed.gameData) || {};
    var series = gd.seriesDescription || gd.series || "";
    var gNum = gd.gameNumber != null ? gd.gameNumber : null;
    var gamesIn = gd.gamesInSeries != null ? gd.gamesInSeries : null;
    var desc = [];
    if (series) desc.push(String(series));
    if (gNum != null && gamesIn != null) desc.push("game " + gNum + " of " + gamesIn);
    else if (gNum != null) desc.push("game " + gNum);
    if (!desc.length) return "Series metadata isn’t in this feed.";
    return "Series: " + desc.join(", ") + ".";
  }

  function answerAttendance(feed) {
    var gd = (feed && feed.gameData) || {};
    var att = gd.attendance || (gd.gameInfo && gd.gameInfo.attendance);
    if (att == null) return "Attendance isn’t posted in the feed yet.";
    return "Attendance: " + att + ".";
  }

  function answerPlayer(feed, q) {
    var needle = extractPlayerNeedle(q, feed);
    if (!needle) return "";
    var hits = findPlayersInFeed(feed, needle);
    if (!hits.length) {
      var parts = needle.split(/\s+/);
      if (parts.length > 1) hits = findPlayersInFeed(feed, parts[parts.length - 1]);
    }
    if (!hits.length) return "";
    return hits
      .slice(0, 4)
      .map(function (h) {
        var bits = [h.name + " (" + teamLabel(feed, h.side) + (h.pos ? ", " + h.pos : "") + ")"];
        if (h.bat && (h.bat.atBats != null || h.bat.hits != null)) {
          bits.push(
            "batting " +
              (h.bat.hits != null ? h.bat.hits : "0") +
              "-" +
              (h.bat.atBats != null ? h.bat.atBats : "?") +
              (h.bat.homeRuns ? ", " + h.bat.homeRuns + " HR" : "") +
              (h.bat.rbi ? ", " + h.bat.rbi + " RBI" : "")
          );
        }
        if (h.pit && (h.pit.inningsPitched != null || h.pit.pitchesThrown != null || h.pit.numberOfPitches != null)) {
          bits.push(
            "pitching " +
              (h.pit.inningsPitched != null ? h.pit.inningsPitched + " IP" : "") +
              ", " +
              (h.pit.earnedRuns != null ? h.pit.earnedRuns : "?") +
              " ER, " +
              (h.pit.strikeOuts != null ? h.pit.strikeOuts : "?") +
              " K" +
              (h.pit.numberOfPitches || h.pit.pitchesThrown
                ? ", " + (h.pit.numberOfPitches || h.pit.pitchesThrown) + " pitches"
                : "")
          );
        }
        return bits.join(" — ");
      })
      .join("\n");
  }

  async function answerH2H(feed) {
    try {
      var hist = await loadHist(feed);
      if (!hist || !hist.h2h) return "No head-to-head sample in the lookback window.";
      var s = scoreSnap(feed);
      if (!hist.h2h.n) {
        return (
          "No recent " +
          s.away +
          " vs " +
          s.home +
          " finals in the MARII lookback. Use Form or Projections for team-level color."
        );
      }
      return (
        "MARII H2H (lookback finals): " +
        hist.h2h.n +
        " game(s), avg " +
        s.away +
        " " +
        round1(hist.h2h.awayAvg) +
        " – " +
        s.home +
        " " +
        round1(hist.h2h.homeAvg) +
        "."
      );
    } catch (err) {
      return mariiUnknown();
    }
  }

  function answerProjection(feed) {
    var s = scoreSnap(feed);
    var pitcher = (s.defense.pitcher && s.defense.pitcher.fullName) || "the pitcher";
    var batter = (s.offense.batter && s.offense.batter.fullName) || "the batter";
    var onDeck = (s.offense.onDeck && s.offense.onDeck.fullName) || "";
    var outs = s.outs != null ? s.outs : 0;
    var countBit =
      s.balls != null && s.strikes != null ? "a " + s.balls + "-" + s.strikes + " count" : "this at-bat";
    var inn =
      (s.inningState + " " + s.inningOrd).trim() ||
      (s.inning ? "inning " + s.inning : "this inning");

    var offenseSide =
      /bottom|mid/i.test(s.inningState) || /bot/i.test(s.inningState) ? "home" : "away";
    if (/top/i.test(s.inningState)) offenseSide = "away";
    if (/bottom|bot/i.test(s.inningState)) offenseSide = "home";
    var offenseName = offenseSide === "home" ? s.home : s.away;
    var defenseName = offenseSide === "home" ? s.away : s.home;

    var sentences = [];
    sentences.push(
      "With " +
        (outs === 1 ? "one out" : outs === 2 ? "two outs" : outs + " outs") +
        " and " +
        countBit +
        ", " +
        offenseName +
        " " +
        (outs >= 2 ? "need a clean swing to keep the inning alive" : "have a chance to build something") +
        "."
    );
    sentences.push(
      pitcher +
        " is on the mound for " +
        defenseName +
        " against " +
        batter +
        (onDeck ? ", with " + onDeck + " on deck" : "") +
        "."
    );

    var runState = runnersText(s.offense);
    if (runState !== "bases empty") {
      sentences.push("Traffic on the bases (" + runState + ") raises the leverage of this pitch.");
    } else {
      sentences.push("Bases are empty, so the focus is on getting the inning started.");
    }

    // Simple lean from score + inning (not a model — booth color from live state).
    var leanSide = s.lead || null;
    var leanName = leanSide === "away" ? s.away : leanSide === "home" ? s.home : null;
    var late = s.inning >= 7 || /final/i.test(s.status);
    if (!leanName) {
      sentences.push(
        "It’s tied " +
          s.ar +
          "-" +
          s.hr +
          " in the " +
          inn +
          ". Next run swings the whole look of this game."
      );
      sentences.push(
        "A plausible finish from here is still a one-run game either way — call it roughly " +
          (s.ar + 1) +
          "-" +
          s.hr +
          " or " +
          s.ar +
          "-" +
          (s.hr + 1) +
          " depending who breaks through first. Booth color only, not a forecast model."
      );
    } else {
      var trailName = leanSide === "away" ? s.home : s.away;
      var leadScore = leanSide === "away" ? s.ar : s.hr;
      var trailScore = leanSide === "away" ? s.hr : s.ar;
      var edge =
        late && s.margin >= 3
          ? leanName + " look firmly in control"
          : late && s.margin >= 1
            ? leanName + " hold a late edge"
            : leanName + " have the current edge";
      sentences.push(
        "Scoreboard says " +
          s.away +
          " " +
          s.ar +
          ", " +
          s.home +
          " " +
          s.hr +
          " in the " +
          inn +
          " — " +
          edge +
          "."
      );
      var projLead = leadScore + (offenseSide === leanSide && outs < 2 ? 1 : 0);
      var projTrail = trailScore + (offenseSide !== leanSide && outs < 2 && runState !== "bases empty" ? 1 : 0);
      if (projLead === leadScore && projTrail === trailScore) {
        projLead = leadScore;
        projTrail = trailScore;
      }
      // Nudge a finishing score that mirrors current margin.
      var finLead = Math.max(projLead, leadScore);
      var finTrail = Math.min(projTrail, trailScore + (late && s.margin >= 2 ? 0 : 1));
      if (finTrail >= finLead) finTrail = Math.max(0, finLead - 1);
      sentences.push(
        trailName +
          " can still push, but from this spot a " +
          finLead +
          "-" +
          finTrail +
          " finish for " +
          leanName +
          " looks plausible. That’s booth color from the live board — not betting advice."
      );
    }

    var hot = battingEntries(feed, offenseSide)
      .filter(function (b) {
        return b.hits > 0 || b.hr > 0 || b.rbi > 0;
      })
      .sort(function (a, b) {
        return b.hits + b.hr * 2 + b.rbi - (a.hits + a.hr * 2 + a.rbi);
      })[0];
    if (hot && hot.name) {
      sentences.push(
        offenseName +
          " have gotten production from " +
          hot.name +
          " (" +
          hot.hits +
          "-" +
          hot.abs +
          (hot.rbi ? ", " + hot.rbi + " RBI" : "") +
          ") already tonight."
      );
    }

    return sentences.join(" ");
  }

  function answerStarter(feed) {
    var s = scoreSnap(feed);
    var bits = [];
    ["away", "home"].forEach(function (side) {
      var staff = pitchingEntries(feed, side);
      if (!staff.length) return;
      var p = staff[0];
      bits.push(
        teamLabel(feed, side) +
          " first pitcher " +
          p.name +
          ": " +
          (p.ip || "?") +
          " IP, " +
          p.h +
          " H, " +
          p.er +
          " ER, " +
          p.bb +
          " BB, " +
          p.so +
          " K" +
          (p.pitches ? ", " + p.pitches + " pitches" : "") +
          "."
      );
    });
    var cur = (s.defense.pitcher && s.defense.pitcher.fullName) || "";
    if (cur) bits.push("On the mound right now: " + cur + ".");
    return bits.length ? bits.join(" ") : "Pitching lines aren’t in the feed yet.";
  }

  function teamIds(feed) {
    var gd = ((feed || {}).gameData || {}).teams || {};
    var box = ((((feed || {}).liveData || {}).boxscore || {}).teams || {});
    var away =
      (gd.away && (gd.away.id || (gd.away.team && gd.away.team.id))) ||
      (box.away && box.away.team && box.away.team.id) ||
      null;
    var home =
      (gd.home && (gd.home.id || (gd.home.team && gd.home.team.id))) ||
      (box.home && box.home.team && box.home.team.id) ||
      null;
    return { awayId: away, homeId: home };
  }

  function isoDaysAgo(n) {
    var d = new Date();
    d.setUTCDate(d.getUTCDate() - n);
    var m = String(d.getUTCMonth() + 1).padStart(2, "0");
    var day = String(d.getUTCDate()).padStart(2, "0");
    return d.getUTCFullYear() + "-" + m + "-" + day;
  }

  async function mlbGet(path) {
    var res = await fetch(MLB + path);
    if (!res.ok) throw new Error("mlb " + res.status);
    return res.json();
  }

  async function fetchFinalGames(teamId, days) {
    if (!teamId) return [];
    var data = await mlbGet(
      "/schedule?sportId=1&teamId=" +
        encodeURIComponent(teamId) +
        "&startDate=" +
        encodeURIComponent(isoDaysAgo(days || 28)) +
        "&endDate=" +
        encodeURIComponent(todayISO()) +
        "&hydrate=linescore,team"
    );
    var out = [];
    (data.dates || []).forEach(function (day) {
      (day.games || []).forEach(function (g) {
        var st = (g.status && g.status.detailedState) || "";
        if (!/final|completed|game over/i.test(st)) return;
        var a = g.teams && g.teams.away;
        var h = g.teams && g.teams.home;
        if (!a || !h || a.score == null || h.score == null) return;
        out.push({
          date: g.officialDate || day.date,
          awayId: a.team && a.team.id,
          homeId: h.team && h.team.id,
          awayScore: Number(a.score),
          homeScore: Number(h.score),
          gamePk: g.gamePk,
        });
      });
    });
    return out;
  }

  function formFromGames(games, teamId) {
    var scoredFor = [];
    var scoredAgainst = [];
    var w = 0;
    var l = 0;
    games.forEach(function (g) {
      var mine = g.awayId === teamId ? g.awayScore : g.homeId === teamId ? g.homeScore : null;
      var theirs = g.awayId === teamId ? g.homeScore : g.homeId === teamId ? g.awayScore : null;
      if (mine == null || theirs == null) return;
      scoredFor.push(mine);
      scoredAgainst.push(theirs);
      if (mine > theirs) w += 1;
      else if (mine < theirs) l += 1;
    });
    function avg(arr) {
      if (!arr.length) return 4.5;
      return arr.reduce(function (s, x) { return s + x; }, 0) / arr.length;
    }
    var last5 = scoredFor.slice(-5);
    var last5a = scoredAgainst.slice(-5);
    return {
      games: scoredFor.length,
      w: w,
      l: l,
      rpg: avg(scoredFor),
      rapg: avg(scoredAgainst),
      recentRpg: avg(last5.length ? last5 : scoredFor),
      recentRapg: avg(last5a.length ? last5a : scoredAgainst),
      lastScores: scoredFor.slice(-5),
    };
  }

  function h2hFromGames(gamesA, gamesB, awayId, homeId) {
    var map = {};
    gamesA.concat(gamesB).forEach(function (g) {
      if (!g.gamePk) return;
      map[g.gamePk] = g;
    });
    var pair = Object.keys(map)
      .map(function (k) { return map[k]; })
      .filter(function (g) {
        return (
          (g.awayId === awayId && g.homeId === homeId) ||
          (g.awayId === homeId && g.homeId === awayId)
        );
      });
    var awayRuns = [];
    var homeRuns = [];
    pair.forEach(function (g) {
      if (g.awayId === awayId) {
        awayRuns.push(g.awayScore);
        homeRuns.push(g.homeScore);
      } else {
        awayRuns.push(g.homeScore);
        homeRuns.push(g.awayScore);
      }
    });
    function avg(arr) {
      if (!arr.length) return null;
      return arr.reduce(function (s, x) { return s + x; }, 0) / arr.length;
    }
    return {
      n: pair.length,
      awayAvg: avg(awayRuns),
      homeAvg: avg(homeRuns),
    };
  }

  function gameLeftFactor(s) {
    // Rough fraction of a 9-inning game still to play (0–1).
    var inn = Math.max(1, Math.min(12, s.inning || 1));
    var state = (s.inningState || "").toLowerCase();
    var within = 0.5;
    if (/top/.test(state)) within = 0.85;
    else if (/middle|mid/.test(state)) within = 0.5;
    else if (/bottom|bot|end/.test(state)) within = 0.2;
    var outs = s.outs != null ? Number(s.outs) : 0;
    within = Math.max(0.05, within - outs * 0.12);
    var leftInnings = Math.max(0, 9 - inn) + within;
    if (/final/i.test(s.status)) return 0;
    return Math.max(0, Math.min(1, leftInnings / 9));
  }

  function round1(n) {
    return Math.round(n * 10) / 10;
  }

  async function loadHist(feed) {
    var ids = teamIds(feed);
    if (!ids.awayId || !ids.homeId) return null;
    var key = ids.awayId + "-" + ids.homeId;
    if (state.histCache && state.histCache.key === key && Date.now() - state.histCache.at < 5 * 60 * 1000) {
      return state.histCache.data;
    }
    var pair = await Promise.all([
      fetchFinalGames(ids.awayId, 28),
      fetchFinalGames(ids.homeId, 28),
    ]);
    var data = {
      awayId: ids.awayId,
      homeId: ids.homeId,
      awayForm: formFromGames(pair[0], ids.awayId),
      homeForm: formFromGames(pair[1], ids.homeId),
      h2h: h2hFromGames(pair[0], pair[1], ids.awayId, ids.homeId),
    };
    state.histCache = { key: key, at: Date.now(), data: data };
    return data;
  }

  function projectFinalAlgo(feed, hist) {
    var s = scoreSnap(feed);
    var af = hist.awayForm;
    var hf = hist.homeForm;
    var left = gameLeftFactor(s);

    // Expected runs/game from blend of season-ish window + recent form.
    var awayOff = af.rpg * 0.45 + af.recentRpg * 0.55;
    var homeOff = hf.rpg * 0.45 + hf.recentRpg * 0.55;
    var awayDef = af.rapg * 0.45 + af.recentRapg * 0.55;
    var homeDef = hf.rapg * 0.45 + hf.recentRapg * 0.55;

    // Matchup expected full-game scoring (offense vs opponent defense), slight home bump.
    var expAwayFull = (awayOff + homeDef) / 2;
    var expHomeFull = (homeOff + awayDef) / 2 * 1.04;

    if (hist.h2h.n >= 2 && hist.h2h.awayAvg != null) {
      expAwayFull = expAwayFull * 0.7 + hist.h2h.awayAvg * 0.3;
      expHomeFull = expHomeFull * 0.7 + hist.h2h.homeAvg * 0.3;
    }

    var remAway = expAwayFull * left;
    var remHome = expHomeFull * left;

    // Leverage: runners + outs nudge remaining offense for the batting side.
    var offenseSide =
      /bottom|bot/i.test(s.inningState) ? "home" : /top/i.test(s.inningState) ? "away" : null;
    var runFactor = runnersText(s.offense) === "bases empty" ? 1 : 1.15;
    if (offenseSide === "away") remAway *= runFactor;
    if (offenseSide === "home") remHome *= runFactor;
    if (s.outs >= 2) {
      if (offenseSide === "away") remAway *= 0.85;
      if (offenseSide === "home") remHome *= 0.85;
    }

    var predAway = s.ar + remAway;
    var predHome = s.hr + remHome;

    // Discrete final score (integers); avoid ties by leaning to projected edge.
    var awayFinal = Math.max(s.ar, Math.round(predAway));
    var homeFinal = Math.max(s.hr, Math.round(predHome));
    if (awayFinal === homeFinal) {
      if (predAway >= predHome) awayFinal += 1;
      else homeFinal += 1;
    }

    var edge = predAway - predHome;
    // Crude win chance from projected margin (not a market model).
    var winAway = 1 / (1 + Math.exp(-edge * 0.55));
    winAway = Math.max(0.08, Math.min(0.92, winAway));

    var insights = [];
    insights.push(
      s.away +
        " recent form (last " +
        af.games +
        " finals): " +
        af.w +
        "-" +
        af.l +
        ", " +
        round1(af.recentRpg) +
        " RPG / " +
        round1(af.recentRapg) +
        " RA/G."
    );
    insights.push(
      s.home +
        " recent form (last " +
        hf.games +
        " finals): " +
        hf.w +
        "-" +
        hf.l +
        ", " +
        round1(hf.recentRpg) +
        " RPG / " +
        round1(hf.recentRapg) +
        " RA/G."
    );
    if (hist.h2h.n) {
      insights.push(
        "Season head-to-head sample: " +
          hist.h2h.n +
          " game(s), avg score " +
          s.away +
          " " +
          round1(hist.h2h.awayAvg) +
          " – " +
          s.home +
          " " +
          round1(hist.h2h.homeAvg) +
          "."
      );
    } else {
      insights.push("No recent head-to-head finals in the lookback window — using team form only.");
    }
    insights.push(
      "Live board " +
        s.ar +
        "-" +
        s.hr +
        " with ~" +
        Math.round(left * 100) +
        "% of a regulation game left → remaining expected runs " +
        s.away +
        " +" +
        round1(remAway) +
        ", " +
        s.home +
        " +" +
        round1(remHome) +
        "."
    );

    return {
      awayFinal: awayFinal,
      homeFinal: homeFinal,
      winAway: winAway,
      insights: insights,
      predAway: predAway,
      predHome: predHome,
    };
  }

  async function answerProjectionAlgo(feed) {
    var s = scoreSnap(feed);
    if (/final/i.test(s.status)) {
      return (
        "This one’s final: " +
        s.away +
        " " +
        s.ar +
        ", " +
        s.home +
        " " +
        s.hr +
        ". No projection left — only the box score."
      );
    }
    try {
      var hist = await loadHist(feed);
      if (!hist || !hist.awayForm.games || !hist.homeForm.games) {
        return answerProjection(feed) + " (Not enough prior finals yet for the full algorithm.)";
      }
      var p = projectFinalAlgo(feed, hist);
      var fav = p.winAway >= 0.5 ? s.away : s.home;
      var favPct = Math.round((p.winAway >= 0.5 ? p.winAway : 1 - p.winAway) * 100);
      var lines = [];
      lines.push(
        "MARII game model (prior finals + live state): projected final " +
          s.away +
          " " +
          p.awayFinal +
          ", " +
          s.home +
          " " +
          p.homeFinal +
          "."
      );
      lines.push(
        "Lean: " +
          fav +
          " (~" +
          favPct +
          "% from the margin model). Continuous mark " +
          round1(p.predAway) +
          "–" +
          round1(p.predHome) +
          "."
      );
      lines.push(p.insights.join(" "));
      lines.push("Algorithmic booth color only — not betting advice.");
      return lines.join(" ");
    } catch (err) {
      return answerProjection(feed) + " (History fetch failed — fell back to live-board color.)";
    }
  }

  function wantsFormQuestion(t) {
    var s = String(t || "").toLowerCase();
    if (/\b(starter|pitch(er|ing)? line|on the mound|era tonight|pitch count)\b/.test(s)) return false;
    if (/\b(season|record|standings|form|this year|playing lately|recent (games|form|stretch))\b/.test(s)) {
      return true;
    }
    return /\bhow (are|have|is|'re|'ve)\b/.test(s) && /\b(team|they|clubs?|doing|looking|year)\b/.test(s);
  }

  function wantsProjectionQuestion(t) {
    return /\b(projection|project|predict|win probability|who wins|who('?s| is) (gonna|going to|favored)|forecast|plausible|final score|score prediction|who should win|pick (a |the )?winner)\b/i.test(
      String(t || "")
    );
  }

  function wantsH2HQuestion(t) {
    return /\b(head.?to.?head|h2h|vs each other|against each other|when (they|these teams) (play|met|meet)|series history)\b/i.test(
      String(t || "")
    );
  }

  function isOffTopicQuestion(t) {
    return /\b(stock market|nasdaq|recipe|cookbook|python code|javascript code|politics|president|bitcoin|crypto wallet|homework|math problem|chatgpt|write (me )?an essay|tell me a joke|who are you|what is ai)\b/i.test(
      String(t || "")
    );
  }

  function isGameScopedQuestion(q, feed) {
    var t = String(q || "").toLowerCase().trim();
    if (!t) return false;
    if (isOffTopicQuestion(t)) return false;
    if (
      /\b(lineup|batting|pitch|batter|mound|inning|out|count|runner|score|bullpen|projection|project|predict|forecast|win|hot bat|cold bat|starter|matchup|rbi|homer|plate|this game|tonight|box|standings|form|season|record|head.?to.?head|h2h|final|weather|ballpark|stadium|venue|umpire|attendance|series|linescore|recap|summar|due up|on deck|leading|ahead|behind|error|walk|strikeout|left on base|lob|probable|decision|save|winner|loser|play|at.?bat|diamond|mlb|baseball)\b/i.test(
        t
      )
    ) {
      return true;
    }
    if (wantsFormQuestion(t) || wantsProjectionQuestion(t) || wantsH2HQuestion(t)) return true;
    if (questionMentionsSide(q, feed, "away") || questionMentionsSide(q, feed, "home")) return true;
    if (feed && extractPlayerNeedle(q, feed) && findPlayersInFeed(feed, extractPlayerNeedle(q, feed)).length) {
      return true;
    }
    return false;
  }

  function mariiUnknown() {
    return "That question is currently for a league above me. MARII is still in beta and it will improve slowly.";
  }

  function mariiRefuse() {
    return mariiUnknown();
  }

  function classifyAskIntent(q) {
    var t = String(q || "").toLowerCase();
    var rules = [
      { intent: "projection", re: /\b(projection|project|predict|win probability|who wins|who('?s| is) (gonna|going to|favored)|forecast|plausible|final score|score prediction|who should win|pick (a |the )?winner)\b/ },
      { intent: "h2h", re: /\b(head.?to.?head|h2h|vs each other|against each other|when (they|these teams) (play|met|meet)|series history)\b/ },
      { intent: "form", re: /\b(season|record|standings|form|this year|playing lately|recent (games|form|stretch))\b/ },
      { intent: "form", re: /\bhow (are|have|is|'re|'ve)\b.*\b(team|they|clubs?|doing|looking|year)\b/ },
      { intent: "lineup", re: /\b(lineup|batting order|who('?s| is) (in the lineup|hitting)|starting nine|batting order)\b/ },
      { intent: "matchup", re: /\b(pitcher|mound|on the hill).*(batter|hitting|at bat)|batter.*pitcher|who('?s| is) (pitching|batting|at bat|up)|current matchup|who('?s| is) facing\b/ },
      { intent: "dueup", re: /\b(due up|on deck|in the hole|who('?s| is) next|batting next)\b/ },
      { intent: "situation", re: /\b(count|outs?|runners?|on base|situation|scoreboard|what('?s| is) (the )?score|score\??$|how('?s| is) it (going|looking)|game state|status of the game)\b/ },
      { intent: "lead", re: /\b(who('?s| is) (winning|leading|ahead)|who('?s| is) behind|what('?s| is) the (lead|margin)|tied\b)/ },
      { intent: "hot", re: /\b(hot|heating|best bat|who('?s| is) hitting|production at the plate|raking|torching)\b/ },
      { intent: "cold", re: /\b(cold|quiet bat|ice cold|struggling at the plate|who('?s| is) (slumping|struggling))\b/ },
      { intent: "leaders", re: /\b(who (has|leads|got|got the most)|most (hits|runs|rbi|home ?runs|homers|walks|strikeouts)|leader(board)?|home ?runs? tonight|homers? tonight)\b/ },
      { intent: "bullpen", re: /\b(bullpen|relief|reliever|who('?s| is) (next|coming in)|pitch count|arms left)\b/ },
      { intent: "starter", re: /\b(starter|probable|how has .* (looked|pitched)|pitching line|era tonight|listed pitcher|opening pitcher)\b/ },
      { intent: "teambox", re: /\b(team (stats?|totals?|box)|hits and errors|left on base|\blob\b|errors?|team batting)\b/ },
      { intent: "linescore", re: /\b(linescore|by inning|inning.?by.?inning|run chart)\b/ },
      { intent: "venue", re: /\b(where|ballpark|stadium|venue|park name|playing at)\b/ },
      { intent: "weather", re: /\b(weather|temperature|wind|how('?s| is) it outside|climate)\b/ },
      { intent: "lastplay", re: /\b(last play|what just happened|latest play|previous play|what happened)\b/ },
      { intent: "recap", re: /\b(recap|summar(y|ize)|overview|catch me up|what('?s| is) going on|brief me|big picture)\b/ },
      { intent: "decisions", re: /\b(decision|who (got|earned) the (win|loss|save)|winning pitcher|losing pitcher|save\b)/ },
      { intent: "umpires", re: /\b(umpire|ump crew|home plate ump|officials)\b/ },
      { intent: "series", re: /\b(series|game number|how many games in)\b/ },
      { intent: "attendance", re: /\b(attendance|crowd|how many (fans|people)|packed)\b/ },
      { intent: "player", re: /\b(how (is|has|did)|stats? for|what about|tell me about|line for)\b/ },
    ];
    var i;
    for (i = 0; i < rules.length; i++) {
      if (rules[i].re.test(t)) {
        if (rules[i].intent === "form" && /\b(starter|pitch(er|ing)? line|on the mound|era tonight)\b/.test(t)) {
          continue;
        }
        return rules[i].intent;
      }
    }
    return "unknown";
  }

  function localAskAnswer(q) {
    var feed = state.lastFeed;
    if (!feed) return "Live feed isn’t ready yet — wait a second and ask again.";
    if (!isGameScopedQuestion(q, feed)) return mariiRefuse();

    var intent = classifyAskIntent(q);
    switch (intent) {
      case "projection":
      case "h2h":
      case "form":
        return null;
      case "lineup":
        return answerLineup(feed, q);
      case "matchup":
        return answerMatchup(feed);
      case "dueup":
        return answerDueUp(feed);
      case "situation":
        return answerSituation(feed);
      case "lead":
        return answerWhoWinning(feed);
      case "hot":
        return answerHotBats(feed, q);
      case "cold":
        return answerColdBats(feed, q);
      case "leaders":
        return answerLeaders(feed, q);
      case "bullpen":
        return answerBullpen(feed, q);
      case "starter":
        return answerProbables(feed);
      case "teambox":
        return answerTeamBox(feed, q);
      case "linescore":
        return answerLinescore(feed);
      case "venue":
        return answerVenue(feed);
      case "weather":
        return answerWeather(feed);
      case "lastplay":
        return answerLastPlay(feed);
      case "recap":
        return answerRecap(feed);
      case "decisions":
        return answerDecisions(feed);
      case "umpires":
        return answerUmpires(feed);
      case "series":
        return answerSeries(feed);
      case "attendance":
        return answerAttendance(feed);
      case "player": {
        var playerAns = answerPlayer(feed, q);
        return playerAns || mariiUnknown();
      }
      case "unknown": {
        var byName = answerPlayer(feed, q);
        if (byName) return byName;
        return mariiUnknown();
      }
      default: {
        var _exhaustive = intent;
        void _exhaustive;
        return mariiUnknown();
      }
    }
  }

  function showAskAnswer(question, answer) {
    state.lastAskAnswer = answer || "";
    if (!els.askAnswer) return;
    els.askAnswer.innerHTML = "";
    var qEl = document.createElement("p");
    qEl.className = "ask-answer__q";
    qEl.textContent = "Q: " + question;
    var body = document.createElement("p");
    body.className = "ask-answer__body";
    body.textContent = answer;
    els.askAnswer.appendChild(qEl);
    els.askAnswer.appendChild(body);
    if (els.askSpeak) els.askSpeak.disabled = !answer;
  }

  async function handleAsk(raw) {
    var question = String(raw || "").trim();
    if (!question) {
      toast("Type a question first.");
      return;
    }
    if (!state.gamePk) {
      toast("Pick a game first.");
      return;
    }
    if (state.asking) return;
    state.asking = true;
    if (els.askSubmit) els.askSubmit.disabled = true;
    try {
      if (!state.lastFeed) {
        showAskAnswer(question, "Live feed isn’t ready yet — wait a second and ask again.");
        return;
      }
      if (!isGameScopedQuestion(question, state.lastFeed)) {
        showAskAnswer(question, mariiRefuse());
        return;
      }
      var intent = classifyAskIntent(question);
      if (intent === "projection" || wantsProjectionQuestion(question)) {
        showAskAnswer(question, "Running MARII projection…");
        showAskAnswer(question, await answerProjectionAlgo(state.lastFeed));
        return;
      }
      if (intent === "h2h" || wantsH2HQuestion(question)) {
        showAskAnswer(question, "Checking MARII head-to-head…");
        showAskAnswer(question, await answerH2H(state.lastFeed));
        return;
      }
      if (intent === "form" || wantsFormQuestion(question)) {
        showAskAnswer(question, "Asking MARII about form…");
        showAskAnswer(question, await answerFormAlgo(state.lastFeed, question));
        return;
      }
      var local = localAskAnswer(question);
      showAskAnswer(question, local || mariiUnknown());
    } finally {
      state.asking = false;
      if (els.askSubmit) els.askSubmit.disabled = false;
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

  function isOnlineVoiceId(id) {
    return !id || /^en-[A-Z]{2}$/.test(id);
  }

  function preferDefaultKokoro(id) {
    return !id || isOnlineVoiceId(id) || id === "am_fenrir";
  }

  function paintVoiceOptions() {
    if (!els.voiceSelect) return;
    var names =
      (voice && voice.listVoices && voice.listVoices()) || [
        { id: "bm_lewis", label: "Kokoro · Lewis (UK male)" },
        { id: "en-US", label: "Online neural · US" },
        { id: "en-GB", label: "Online neural · British" },
        { id: "en-AU", label: "Online neural · Australian" },
        { id: "en-IN", label: "Online neural · Indian English" },
      ];
    // Keep Kokoro voices ahead of online neural once available.
    names = names.slice().sort(function (a, b) {
      var ak = /^[a-z]{2}_/.test(a.id) ? 0 : 1;
      var bk = /^[a-z]{2}_/.test(b.id) ? 0 : 1;
      if (ak !== bk) return ak - bk;
      if (a.id === "bm_lewis") return -1;
      if (b.id === "bm_lewis") return 1;
      return 0;
    });
    var cur = state.voiceId || "bm_lewis";
    els.voiceSelect.innerHTML = "";
    // If Lewis is preferred but Kokoro isn't listed yet, keep a placeholder so we don't snap to en-US.
    if (!names.some(function (v) { return v.id === cur; }) && /^[a-z]{2}_/.test(cur)) {
      var hold = document.createElement("option");
      hold.value = cur;
      hold.textContent = cur === "bm_lewis" ? "Kokoro · Lewis (loading…)" : "Kokoro · " + cur + " (loading…)";
      els.voiceSelect.appendChild(hold);
    }
    names.forEach(function (v) {
      var opt = document.createElement("option");
      opt.value = v.id;
      opt.textContent = v.label;
      els.voiceSelect.appendChild(opt);
    });
    if (names.some(function (v) { return v.id === cur; }) || els.voiceSelect.querySelector('option[value="' + cur + '"]')) {
      els.voiceSelect.value = cur;
    } else if (names.length) {
      // Never overwrite a pending Kokoro default with the first online voice.
      if (!preferDefaultKokoro(cur)) {
        els.voiceSelect.value = names[0].id;
      } else {
        state.voiceId = "bm_lewis";
        els.voiceSelect.value = els.voiceSelect.querySelector('option[value="bm_lewis"]')
          ? "bm_lewis"
          : names[0].id;
      }
    }
  }

  function setVoiceId(id) {
    state.voiceId = id || "bm_lewis";
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
    var first = !state.voiceReady;
    state.voiceReady = true;
    setVoiceBootMsg(msg || "Neural TTS ready.");
    showVoiceBootSkip(false);
    if (els.voiceBoot) {
      els.voiceBoot.classList.add("is-done");
      els.voiceBoot.setAttribute("aria-busy", "false");
    }
    if (first) speakNext();
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
      finishVoiceBoot("Voice ready — online neural. Kokoro still loading…");
      setVoiceBootMsg("Kokoro still downloading in the background…");
      if (els.voiceHint) els.voiceHint.textContent = "Kokoro loading in background…";
    };
    voice.onKokoroReady = function () {
      paintVoiceOptions();
      var saved = "";
      try {
        saved = localStorage.getItem("pyx.announcer.voiceId") || "";
      } catch (e) {}
      if (preferDefaultKokoro(saved) || preferDefaultKokoro(state.voiceId)) {
        setVoiceId("bm_lewis");
        paintVoiceOptions();
        if (els.voiceSelect) els.voiceSelect.value = "bm_lewis";
      } else {
        voice.setVoice(state.voiceId);
      }
      setVoiceBootMsg("Kokoro ready — Lewis.");
      if (els.voiceHint) els.voiceHint.textContent = "Kokoro · Lewis ready.";
      toast("Kokoro Lewis ready.");
    };
    setVoiceBootMsg("Getting voice ready…");
    voice
      .warmup(function (msg) {
        if (!state.voiceReady) setVoiceBootMsg(msg || "Getting voice ready…");
        else if (els.voiceHint) els.voiceHint.textContent = msg || "";
      })
      .then(function () {
        if (!state.voiceReady) finishVoiceBoot("Voice ready — online neural TTS.");
      })
      .catch(function () {
        if (!state.voiceReady) finishVoiceBoot("Online neural TTS ready.");
      });
  }

  function bind() {
    els.refreshGames.addEventListener("click", loadGames);
    els.backToGames.addEventListener("click", function () {
      stopPolling();
      els.liveLayout.classList.add("is-hidden");
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
    if (els.askForm) {
      els.askForm.addEventListener("submit", function (ev) {
        ev.preventDefault();
        handleAsk(els.askInput && els.askInput.value);
      });
    }
    if (els.askChips) {
      els.askChips.addEventListener("click", function (ev) {
        var btn = ev.target && ev.target.closest ? ev.target.closest(".ask-chip") : null;
        if (!btn) return;
        var q = btn.getAttribute("data-q") || btn.textContent;
        if (els.askInput) els.askInput.value = q;
        handleAsk(q);
      });
    }
    if (els.askSpeak) {
      els.askSpeak.addEventListener("click", function () {
        if (!state.lastAskAnswer || !state.voiceReady || !voice) {
          toast("Nothing to read yet.");
          return;
        }
        if (voice.unlockAudio) voice.unlockAudio();
        state.queue = [];
        stopNeuralSpeak();
        state.speaking = true;
        els.status.textContent = "Speaking…";
        voice.onSpeakEnd = function () {
          state.speaking = false;
          refreshStatusLine();
          speakNext();
        };
        voice.speak(state.lastAskAnswer, "en-US");
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
      liveLayout: $("liveLayout"),
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
      askPanel: $("askPanel"),
      askForm: $("askForm"),
      askInput: $("askInput"),
      askSubmit: $("askSubmit"),
      askSpeak: $("askSpeak"),
      askAnswer: $("askAnswer"),
      askChips: $("askChips"),
    };
    try {
      var m = localStorage.getItem("pyx.announcer.mode");
      if (m) setMode(m);
      if (localStorage.getItem("pyx.announcer.muted") === "1") setMuted(true);
      var vid = localStorage.getItem("pyx.announcer.voiceId");
      // Default / migrate soft defaults to Kokoro Lewis.
      if (!vid || vid === "en-US" || vid === "en-GB" || vid === "am_fenrir") {
        state.voiceId = "bm_lewis";
        try {
          localStorage.setItem("pyx.announcer.voiceId", "bm_lewis");
        } catch (e2) {}
      } else {
        state.voiceId = vid;
      }
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
