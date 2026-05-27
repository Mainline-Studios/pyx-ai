/**
 * Pyx Write — GPT-OSS instrumental composer + Tone.js playback.
 */
(function (global) {
  "use strict";

  var API = "/api/write/compose";
  var INSTRUMENTS = [
    "piano",
    "synth",
    "bass",
    "drums",
    "guitar",
    "strings",
    "brass",
    "pad",
    "bells",
    "organ",
  ];

  var state = {
    score: null,
    model: "",
    playing: false,
    stopTimer: null,
    synths: [],
  };

  function $(id) {
    return document.getElementById(id);
  }

  function selectedInstruments() {
    var out = [];
    INSTRUMENTS.forEach(function (id) {
      var el = $("writeInst_" + id);
      if (el && el.checked) out.push(id);
    });
    return out.length ? out : ["piano", "bass", "pad"];
  }

  function setStatus(msg, kind) {
    var el = $("writeStatus");
    if (!el) return;
    el.textContent = msg || "";
    el.className = "write-status" + (kind ? " write-status--" + kind : "");
  }

  function setMeta(score, model) {
    var el = $("writeMeta");
    if (!el) return;
    if (!score) {
      el.textContent = "";
      return;
    }
    var parts = [
      score.title || "Untitled",
      score.bpm + " BPM",
      score.key || "",
      (score.bars || "?") + " bars",
      (score.tracks || []).length + " tracks",
    ];
    if (model) parts.push(model);
    el.textContent = parts.filter(Boolean).join(" · ");
  }

  function renderTracks(score) {
    var el = $("writeTracks");
    if (!el) return;
    if (!score || !score.tracks) {
      el.innerHTML = "";
      return;
    }
    el.innerHTML = score.tracks
      .map(function (tr) {
        return (
          '<span class="write-track-chip">' +
          escapeHtml(tr.instrument) +
          " (" +
          (tr.notes ? tr.notes.length : 0) +
          " notes)</span>"
        );
      })
      .join("");
  }

  function escapeHtml(s) {
    var d = document.createElement("div");
    d.textContent = s == null ? "" : String(s);
    return d.innerHTML;
  }

  function createInstrument(instrumentId) {
    switch (instrumentId) {
      case "bass":
        return new Tone.MonoSynth({
          oscillator: { type: "sawtooth" },
          filter: { Q: 2 },
          envelope: { attack: 0.02, decay: 0.15, sustain: 0.35, release: 0.4 },
        }).toDestination();
      case "drums":
        return {
          type: "drums",
          kick: new Tone.MembraneSynth({ pitchDecay: 0.02, octaves: 4 }).toDestination(),
          snare: new Tone.NoiseSynth({
            noise: { type: "white" },
            envelope: { attack: 0.001, decay: 0.12, sustain: 0 },
          }).toDestination(),
          hat: new Tone.MetalSynth({
            frequency: 320,
            envelope: { attack: 0.001, decay: 0.04, sustain: 0 },
          }).toDestination(),
        };
      case "guitar":
        return new Tone.PluckSynth({
          attackNoise: 0.5,
          dampening: 2800,
          resonance: 0.85,
        }).toDestination();
      case "strings":
        return new Tone.PolySynth(Tone.FMSynth, {
          harmonicity: 2,
          modulationIndex: 1.2,
          envelope: { attack: 0.08, decay: 0.3, sustain: 0.45, release: 1.2 },
        }).toDestination();
      case "brass":
        return new Tone.PolySynth(Tone.FMSynth, {
          harmonicity: 1,
          modulationIndex: 0.8,
          envelope: { attack: 0.05, decay: 0.2, sustain: 0.5, release: 0.5 },
        }).toDestination();
      case "pad":
        return new Tone.PolySynth(Tone.Synth, {
          oscillator: { type: "sine" },
          envelope: { attack: 0.4, decay: 0.2, sustain: 0.7, release: 1.8 },
        }).toDestination();
      case "bells":
        return new Tone.PolySynth(Tone.FMSynth, {
          harmonicity: 3.5,
          modulationIndex: 1.8,
          envelope: { attack: 0.01, decay: 1.2, sustain: 0.05, release: 2 },
        }).toDestination();
      case "organ":
        return new Tone.PolySynth(Tone.Synth, {
          oscillator: { type: "square" },
          envelope: { attack: 0.02, decay: 0.1, sustain: 0.85, release: 0.3 },
        }).toDestination();
      case "synth":
        return new Tone.PolySynth(Tone.Synth, {
          oscillator: { type: "sawtooth" },
          envelope: { attack: 0.03, decay: 0.2, sustain: 0.4, release: 0.6 },
        }).toDestination();
      case "piano":
      default:
        return new Tone.PolySynth(Tone.Synth, {
          oscillator: { type: "triangle" },
          envelope: { attack: 0.01, decay: 0.12, sustain: 0.25, release: 0.9 },
        }).toDestination();
    }
  }

  function scheduleDrumHit(drums, pitch, time, velocity) {
    var p = String(pitch || "C1").toUpperCase();
    var v = Math.max(0.15, Math.min(1, velocity || 0.8));
    if (p.charAt(0) === "C") {
      drums.kick.triggerAttackRelease("C1", "8n", time, v);
    } else if (p.charAt(0) === "D") {
      drums.snare.triggerAttackRelease("16n", time, v * 0.9);
    } else {
      drums.hat.triggerAttackRelease("32n", time, v * 0.75);
    }
  }

  function scheduleNote(synth, instrumentId, pitch, time, duration, velocity) {
    var v = Math.max(0.15, Math.min(1, velocity || 0.75));
    if (instrumentId === "drums" && synth && synth.type === "drums") {
      scheduleDrumHit(synth, pitch, time, v);
      return;
    }
    if (synth && typeof synth.triggerAttackRelease === "function") {
      synth.triggerAttackRelease(pitch, duration, time, v);
    }
  }

  function disposeSynth(s) {
    if (!s) return;
    if (s.type === "drums") {
      if (s.kick) s.kick.dispose();
      if (s.snare) s.snare.dispose();
      if (s.hat) s.hat.dispose();
      return;
    }
    if (typeof s.dispose === "function") s.dispose();
  }

  function stopPlayback() {
    state.playing = false;
    if (state.stopTimer) {
      clearTimeout(state.stopTimer);
      state.stopTimer = null;
    }
    state.synths.forEach(disposeSynth);
    state.synths = [];
    var playBtn = $("writePlayBtn");
    if (playBtn) playBtn.textContent = "Play";
    setStatus(state.score ? "Ready to play." : "", "");
  }

  function playScore() {
    var score = state.score;
    if (!score || !score.tracks || !score.tracks.length) {
      setStatus("Generate a piece first.", "err");
      return;
    }
    if (typeof Tone === "undefined") {
      setStatus("Audio engine failed to load.", "err");
      return;
    }
    stopPlayback();
    Tone.start()
      .then(function () {
        var bpm = score.bpm || 120;
        var beatSec = 60 / bpm;
        var beatsPerBar =
          score.time_signature && score.time_signature[0] ? score.time_signature[0] : 4;
        var totalBeats = (score.bars || 16) * beatsPerBar;
        var startAt = Tone.now() + 0.2;
        state.playing = true;
        var playBtn = $("writePlayBtn");
        if (playBtn) playBtn.textContent = "Stop";

        score.tracks.forEach(function (tr) {
          var synth = createInstrument(tr.instrument);
          state.synths.push(synth);
          (tr.notes || []).forEach(function (n) {
            var t = startAt + (n.start || 0) * beatSec;
            var dur = Math.max(0.05, (n.duration || 0.25) * beatSec);
            scheduleNote(synth, tr.instrument, n.pitch, t, dur, n.velocity);
          });
        });

        setStatus("Playing…", "ok");
        state.stopTimer = setTimeout(function () {
          stopPlayback();
          setStatus("Finished.", "ok");
        }, totalBeats * beatSec * 1000 + 800);
      })
      .catch(function (e) {
        setStatus(e.message || "Could not start audio.", "err");
        stopPlayback();
      });
  }

  function downloadScoreJson() {
    if (!state.score) return;
    var blob = new Blob([JSON.stringify(state.score, null, 2)], {
      type: "application/json",
    });
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download =
      (state.score.title || "pyx-write").replace(/[^\w\-]+/g, "_").slice(0, 40) + ".json";
    a.click();
    URL.revokeObjectURL(a.href);
  }

  function compose() {
    var prompt = ($("writePrompt") && $("writePrompt").value) || "";
    prompt = prompt.trim();
    if (!prompt) {
      setStatus("Describe the music you want.", "err");
      return;
    }
    var style = ($("writeStyle") && $("writeStyle").value) || "";
    var bars = Number(($("writeBars") && $("writeBars").value) || 16) || 16;
    var btn = $("writeComposeBtn");
    if (btn) btn.disabled = true;
    setStatus("Composing with GPT-OSS… (10–40s)", "");
    stopPlayback();
    state.score = null;
    renderTracks(null);
    setMeta(null, "");

    fetch(API, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      cache: "no-store",
      body: JSON.stringify({
        prompt: prompt,
        style: style,
        instruments: selectedInstruments(),
        bars: bars,
      }),
    })
      .then(function (r) {
        return r.json().then(function (j) {
          return { ok: r.ok, data: j };
        });
      })
      .then(function (x) {
        if (!x.ok || !x.data.ok) {
          throw new Error((x.data && x.data.error) || "Compose failed");
        }
        state.score = x.data.score;
        state.model = x.data.model || "";
        setMeta(state.score, state.model);
        renderTracks(state.score);
        setStatus("Ready — press Play.", "ok");
        try {
          localStorage.setItem("pyx.write.lastPrompt", prompt);
          localStorage.setItem("pyx.write.lastAt", String(Date.now()));
        } catch (e) {}
      })
      .catch(function (e) {
        setStatus(e.message || String(e), "err");
      })
      .finally(function () {
        if (btn) btn.disabled = false;
      });
  }

  function renderInstrumentChecks() {
    var el = $("writeInstruments");
    if (!el) return;
    var defaults = { piano: true, bass: true, pad: true, synth: false, drums: false };
    el.innerHTML = INSTRUMENTS.map(function (id) {
      var checked = defaults[id] ? " checked" : "";
      return (
        '<label class="write-inst"><input type="checkbox" id="writeInst_' +
        id +
        '"' +
        checked +
        " /> " +
        id +
        "</label>"
      );
    }).join("");
  }

  function init() {
    renderInstrumentChecks();
    var composeBtn = $("writeComposeBtn");
    var playBtn = $("writePlayBtn");
    var dlBtn = $("writeDownloadBtn");
    if (composeBtn) composeBtn.addEventListener("click", compose);
    if (playBtn) {
      playBtn.addEventListener("click", function () {
        if (state.playing) stopPlayback();
        else playScore();
      });
    }
    if (dlBtn) dlBtn.addEventListener("click", downloadScoreJson);
    document.querySelectorAll(".write-preset").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var p = $("writePrompt");
        if (p) p.value = btn.getAttribute("data-prompt") || "";
        var s = $("writeStyle");
        if (s) s.value = btn.getAttribute("data-style") || "";
      });
    });
    try {
      var last = localStorage.getItem("pyx.write.lastPrompt");
      if (last && $("writePrompt") && !$("writePrompt").value) $("writePrompt").value = last;
    } catch (e) {}
    if (global.PyxHandoff && global.PyxHandoff.applyIncoming) {
      global.PyxHandoff.applyIncoming({
        target: "write",
        onText: function (text) {
          if ($("writePrompt")) $("writePrompt").value = text;
        },
      });
    }
    global.PyxWrite = { compose: compose, play: playScore, stop: stopPlayback };
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})(typeof window !== "undefined" ? window : globalThis);
