/**
 * Pyx Assistant voice — Web Speech STT + Sound of Text neural TTS,
 * optional on-device Kokoro when it finishes loading.
 */
(function (root) {
  "use strict";

  var SOT_POST = "https://api.soundoftext.com/sounds";
  var SOT_GET = "https://api.soundoftext.com/sounds/";

  var api = {
    ready: { stt: true, tts: true, kokoro: false },
    status: "idle",
    voiceId: "en-GB",
    /** When false, skip Web Speech TTS fallback (neural/Kokoro only). */
    allowBrowserFallback: true,
    onStatus: null,
    onPartial: null,
    onUtterance: null,
    onSpeakEnd: null,
    onError: null,
    onOnlineReady: null,
  };

  var session = false;
  var listening = false;
  var speaking = false;
  var interruptFlag = false;
  var recognition = null;
  var speakEl = null;
  var kokoro = null;
  var restartTimer = 0;

  function emit(kind, extra) {
    api.status = kind;
    if (typeof api.onStatus === "function") api.onStatus(kind, extra || "");
  }

  function SpeechCtor() {
    return root.SpeechRecognition || root.webkitSpeechRecognition || null;
  }

  function listVoices() {
    var list = [];
    if (api.ready.kokoro) {
      list.push(
        { id: "bm_lewis", label: "Kokoro · Lewis (UK male)" },
        { id: "am_fenrir", label: "Kokoro · Fenrir (US male)" },
        { id: "am_michael", label: "Kokoro · Michael (US male)" },
        { id: "am_adam", label: "Kokoro · Adam (US male)" },
        { id: "am_onyx", label: "Kokoro · Onyx (US male)" },
        { id: "bm_george", label: "Kokoro · George (UK male)" },
        { id: "af_heart", label: "Kokoro · Heart (US female)" },
        { id: "af_bella", label: "Kokoro · Bella (US female)" },
        { id: "bf_emma", label: "Kokoro · Emma (UK female)" }
      );
    }
    list.push(
      { id: "en-US", label: "Online neural · US" },
      { id: "en-GB", label: "Online neural · British" },
      { id: "en-AU", label: "Online neural · Australian" },
      { id: "en-IN", label: "Online neural · Indian English" }
    );
    return list;
  }

  function setVoice(id) {
    api.voiceId = id || "en-GB";
  }

  function isKokoroVoice(id) {
    return !!(id && /^[a-z]{2}_/.test(id));
  }

  function kokoroSpeechLang(id) {
    if (!id) return "en-US";
    if (id.indexOf("b") === 0) return "en-GB";
    return "en-US";
  }

  function chunkText(text) {
    var s = String(text || "").replace(/\s+/g, " ").trim();
    if (!s) return [];
    if (s.length <= 180) return [s];
    var parts = [];
    var buf = "";
    s.split(/(?<=[.!?])\s+/).forEach(function (sent) {
      if ((buf + " " + sent).trim().length > 180) {
        if (buf) parts.push(buf.trim());
        buf = sent;
      } else {
        buf = (buf + " " + sent).trim();
      }
    });
    if (buf) parts.push(buf.trim());
    return parts.length ? parts : [s.slice(0, 180)];
  }

  function sleep(ms) {
    return new Promise(function (r) {
      setTimeout(r, ms);
    });
  }

  async function soundOfTextUrl(text, voice) {
    var res = await fetch(SOT_POST, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        engine: "Google",
        data: { text: text, voice: voice || "en-GB" },
      }),
    });
    var j = await res.json();
    if (!j || !j.success || !j.id) throw new Error("tts create failed");
    var i;
    for (i = 0; i < 20; i++) {
      var g = await fetch(SOT_GET + j.id);
      var st = await g.json();
      if (st && st.status === "Done" && st.location) return st.location;
      if (st && st.status === "Error") throw new Error("tts error");
      await sleep(250);
    }
    throw new Error("tts timeout");
  }

  function playUrl(url) {
    return new Promise(function (resolve, reject) {
      stopSpeakEl();
      var a = new Audio(url);
      speakEl = a;
      a.onended = function () {
        speakEl = null;
        resolve();
      };
      a.onerror = function () {
        speakEl = null;
        reject(new Error("audio play failed"));
      };
      var p = a.play();
      if (p && p.catch) p.catch(reject);
    });
  }

  /** Call from a user gesture so later TTS playback isn’t blocked. */
  function unlockAudio() {
    try {
      var Ctx = root.AudioContext || root.webkitAudioContext;
      if (!Ctx) return Promise.resolve();
      if (!api._audioCtx) api._audioCtx = new Ctx();
      if (api._audioCtx.state === "suspended") return api._audioCtx.resume();
    } catch (e) {}
    return Promise.resolve();
  }

  function stopSpeakEl() {
    if (speakEl) {
      try {
        speakEl.pause();
        speakEl.src = "";
      } catch (e) {}
      speakEl = null;
    }
    if (root.speechSynthesis) root.speechSynthesis.cancel();
  }

  function speakBrowser(text, lang) {
    return new Promise(function (resolve) {
      if (!root.speechSynthesis) {
        resolve();
        return;
      }
      root.speechSynthesis.cancel();
      var u = new SpeechSynthesisUtterance(text);
      u.lang = lang || "en-GB";
      u.rate = 0.96;
      u.pitch = 1.05;
      var voices = root.speechSynthesis.getVoices() || [];
      var prefer = voices.find(function (v) {
        return /en-GB|Google UK|British/i.test(v.lang + " " + v.name);
      });
      if (prefer) u.voice = prefer;
      u.onend = resolve;
      u.onerror = resolve;
      root.speechSynthesis.speak(u);
    });
  }

  async function speakNeural(text, lang) {
    var voice = api.voiceId || "en-GB";
    if (isKokoroVoice(voice)) voice = kokoroSpeechLang(voice);
    if (lang && lang.indexOf("es") === 0) voice = "es-ES";
    else if (lang && lang.indexOf("fr") === 0) voice = "fr-FR";
    else if (lang && lang.indexOf("de") === 0) voice = "de-DE";
    else if (lang && lang.indexOf("ja") === 0) voice = "ja-JP";
    else if (lang && lang.indexOf("zh") === 0) voice = "zh-CN";
    else if (!isKokoroVoice(api.voiceId) && lang && lang.indexOf("en") === 0) {
      // keep selected online locale unless lang override needed
    }
    var chunks = chunkText(text);
    var i;
    for (i = 0; i < chunks.length; i++) {
      if (interruptFlag) return;
      var url = await soundOfTextUrl(chunks[i], voice);
      if (interruptFlag) return;
      await playUrl(url);
    }
  }

  function playBuffer(float32, sampleRate) {
    return new Promise(function (resolve, reject) {
      try {
        var ctx = new (root.AudioContext || root.webkitAudioContext)();
        var buf = ctx.createBuffer(1, float32.length, sampleRate);
        buf.getChannelData(0).set(float32);
        var src = ctx.createBufferSource();
        src.buffer = buf;
        src.connect(ctx.destination);
        speakEl = { pause: function () { try { src.stop(); } catch (e) {} }, src: "" };
        src.onended = function () {
          speakEl = null;
          resolve();
        };
        src.start();
      } catch (e) {
        reject(e);
      }
    });
  }

  async function speak(text, lang) {
    if (!text) return;
    interruptFlag = false;
    speaking = true;
    listening = false;
    stopListenEngine();
    emit("speak");
    try {
      if (kokoro && isKokoroVoice(api.voiceId)) {
        var audio = await kokoro.generate(text, { voice: api.voiceId });
        if (interruptFlag) return;
        var data = audio.audio || audio.data;
        var sr = audio.sampling_rate || 24000;
        if (data && data.length) await playBuffer(data, sr);
      } else {
        await speakNeural(text, lang);
      }
    } catch (e) {
      if (!interruptFlag && api.allowBrowserFallback !== false) {
        await speakBrowser(text, lang || "en-GB");
      } else if (!interruptFlag && typeof api.onError === "function") {
        api.onError(e || new Error("neural tts failed"));
      }
    } finally {
      speaking = false;
      if (typeof api.onSpeakEnd === "function") api.onSpeakEnd(interruptFlag);
      if (session && !interruptFlag) startListenLoop();
      else if (!session) emit("idle");
    }
  }

  function stopListenEngine() {
    listening = false;
    if (recognition) {
      try {
        recognition.onend = null;
        recognition.stop();
      } catch (e) {}
    }
  }

  function startListenLoop() {
    if (!session || speaking) return;
    var Ctor = SpeechCtor();
    if (!Ctor) {
      emit("listen", "Type below — this browser has no speech recognition.");
      return Promise.resolve();
    }
    stopListenEngine();
    recognition = new Ctor();
    recognition.lang =
      api.voiceId && api.voiceId.indexOf("en-") === 0
        ? api.voiceId
        : isKokoroVoice(api.voiceId)
          ? kokoroSpeechLang(api.voiceId)
          : "en-GB";
    recognition.interimResults = true;
    recognition.continuous = false;
    listening = true;
    emit("listen");
    recognition.onresult = function (ev) {
      var i;
      var finalText = "";
      var interim = "";
      for (i = ev.resultIndex; i < ev.results.length; i++) {
        if (ev.results[i].isFinal) finalText += ev.results[i][0].transcript;
        else interim += ev.results[i][0].transcript;
      }
      if (interim && typeof api.onPartial === "function") api.onPartial(interim);
      if (finalText && typeof api.onUtterance === "function") {
        listening = false;
        api.onUtterance(finalText.trim());
      }
    };
    recognition.onerror = function (ev) {
      if (ev.error === "not-allowed" && typeof api.onError === "function") api.onError(ev);
      if (ev.error === "no-speech" && session && !speaking) scheduleRestart();
    };
    recognition.onend = function () {
      if (session && listening && !speaking) scheduleRestart();
    };
    try {
      recognition.start();
    } catch (e) {
      scheduleRestart();
    }
    return Promise.resolve();
  }

  function scheduleRestart() {
    clearTimeout(restartTimer);
    restartTimer = setTimeout(function () {
      if (session && !speaking) startListenLoop();
    }, 280);
  }

  async function startSession() {
    session = true;
    interruptFlag = false;
    await startListenLoop();
  }

  async function stopSession() {
    session = false;
    clearTimeout(restartTimer);
    interruptFlag = true;
    stopSpeak();
    stopListenEngine();
    emit("idle");
  }

  function stopSpeak() {
    interruptFlag = true;
    speaking = false;
    stopSpeakEl();
  }

  async function interrupt() {
    var wasSpeaking = speaking;
    var wasListening = listening;
    stopSpeak();
    if (wasListening && !wasSpeaking) {
      await stopSession();
      return "stopped";
    }
    if (!session) {
      await startSession();
      return "started";
    }
    emit("listen");
    await startListenLoop();
    return "barge-in";
  }

  async function warmup(onProgress) {
    var note = onProgress || function () {};
    note("Warming online neural voice…");
    emit("ready");
    var onlineReady = false;
    try {
      // Prime Sound of Text so the first spoken reply isn’t a cold round-trip.
      await soundOfTextUrl("Hi.", api.voiceId && api.voiceId.indexOf("en-") === 0 ? api.voiceId : "en-GB");
      onlineReady = true;
      note("Online neural voice ready. Loading Kokoro…");
    } catch (e) {
      note("Online voice warm-up skipped. Loading Kokoro…");
    }
    if (typeof api.onOnlineReady === "function") {
      try {
        api.onOnlineReady({ online: onlineReady });
      } catch (e2) {}
    }
    try {
      var mod = await import("https://cdn.jsdelivr.net/npm/kokoro-js@1.2.1/+esm");
      note("Kokoro downloading model weights…");
      kokoro = await mod.KokoroTTS.from_pretrained("onnx-community/Kokoro-82M-v1.0-ONNX", {
        dtype: "q8",
        device: "wasm",
      });
      api.ready.kokoro = true;
      if (!api.voiceId || api.voiceId === "en-GB" || api.voiceId === "en-US" || api.voiceId === "am_fenrir") {
        api.voiceId = "bm_lewis";
        note("Kokoro is ready — using Lewis.");
      } else {
        note("Kokoro is ready — pick a voice in the list.");
      }
    } catch (e) {
      api.ready.kokoro = false;
      note(
        onlineReady
          ? "Online voice is on (Sound of Text). Kokoro didn’t load on this device."
          : "Voice warm-up finished with limited TTS. Typing still works."
      );
    }
    return api.ready;
  }

  root.PyxAssistantVoice = Object.assign(api, {
    warmup: warmup,
    startSession: startSession,
    stopSession: stopSession,
    interrupt: interrupt,
    startListenLoop: startListenLoop,
    stopMic: stopListenEngine,
    stopSpeak: stopSpeak,
    speak: speak,
    unlockAudio: unlockAudio,
    listVoices: listVoices,
    setVoice: setVoice,
    isListening: function () {
      return listening;
    },
    isSpeaking: function () {
      return speaking;
    },
    isSession: function () {
      return session;
    },
  });
  root.dispatchEvent(new Event("pyx-voice-ready"));
})(typeof window !== "undefined" ? window : globalThis);
