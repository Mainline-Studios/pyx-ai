/**
 * Pyx Assistant — on-device Whisper STT + Kokoro TTS + VAD loop.
 */
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.1";
import { KokoroTTS } from "https://cdn.jsdelivr.net/npm/kokoro-js@1.2.1/+esm";

env.allowLocalModels = false;
env.useBrowserCache = true;

const api = {
  ready: { stt: false, tts: false },
  status: "idle",
  voiceId: "af_heart",
  onStatus: null,
  onPartial: null,
  onUtterance: null,
  onSpeakEnd: null,
  onError: null,
};

let asr = null;
let tts = null;
let session = false;
let listening = false;
let speaking = false;
let interruptFlag = false;
let audioCtx = null;
let mediaStream = null;
let processor = null;
let sourceNode = null;
let speakSource = null;
let speakCtx = null;
let pending = [];
let speechFrames = 0;
let silenceFrames = 0;
let collecting = false;

function emit(kind, extra) {
  api.status = kind;
  if (typeof api.onStatus === "function") api.onStatus(kind, extra || "");
}

function resample(input, fromRate, toRate) {
  if (fromRate === toRate) return input;
  const ratio = fromRate / toRate;
  const outLen = Math.max(1, Math.round(input.length / ratio));
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const x = i * ratio;
    const i0 = Math.floor(x);
    const i1 = Math.min(i0 + 1, input.length - 1);
    const f = x - i0;
    out[i] = input[i0] * (1 - f) + input[i1] * f;
  }
  return out;
}

function rms(buf) {
  let s = 0;
  for (let i = 0; i < buf.length; i++) s += buf[i] * buf[i];
  return Math.sqrt(s / Math.max(buf.length, 1));
}

async function ensureCtx() {
  if (!audioCtx || audioCtx.state === "closed") {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (audioCtx.state === "suspended") await audioCtx.resume();
  return audioCtx;
}

export async function warmup(onProgress) {
  const note = onProgress || function () {};
  try {
    note("Loading Whisper (on-device STT)…");
    asr = await pipeline("automatic-speech-recognition", "Xenova/whisper-tiny.en", {
      dtype: "q8",
      progress_callback: function (p) {
        if (p && p.status === "progress" && p.file) {
          note("Whisper " + Math.round((p.progress || 0)) + "%");
        }
      },
    });
    api.ready.stt = true;
    note("Whisper ready.");
  } catch (e) {
    api.ready.stt = false;
    note("Whisper unavailable — type, or I’ll try the browser mic.");
  }
  try {
    note("Loading Kokoro (warmer voice)…");
    tts = await KokoroTTS.from_pretrained("onnx-community/Kokoro-82M-v1.0-ONNX", {
      dtype: "q8",
      device: "wasm",
      progress_callback: function (p) {
        if (p && p.status === "progress") note("Kokoro " + Math.round((p.progress || 0)) + "%");
      },
    });
    api.ready.tts = true;
    note("Voice ready.");
  } catch (e) {
    api.ready.tts = false;
    note("Kokoro unavailable — using the system voice.");
  }
  emit(api.ready.stt || api.ready.tts ? "ready" : "degraded");
  return api.ready;
}

function stopCaptureNodes() {
  if (processor) {
    try {
      processor.disconnect();
    } catch (e) {}
    processor.onaudioprocess = null;
    processor = null;
  }
  if (sourceNode) {
    try {
      sourceNode.disconnect();
    } catch (e) {}
    sourceNode = null;
  }
}

export async function stopMic() {
  listening = false;
  collecting = false;
  pending = [];
  stopCaptureNodes();
  if (mediaStream) {
    mediaStream.getTracks().forEach(function (t) {
      t.stop();
    });
    mediaStream = null;
  }
}

export function stopSpeak() {
  interruptFlag = true;
  speaking = false;
  if (speakSource) {
    try {
      speakSource.stop();
    } catch (e) {}
    speakSource = null;
  }
  if (window.speechSynthesis) window.speechSynthesis.cancel();
}

export function isListening() {
  return listening;
}

export function isSpeaking() {
  return speaking;
}

export function isSession() {
  return session;
}

async function transcribe(float32, sampleRate) {
  if (!asr) return "";
  const audio16 = resample(float32, sampleRate, 16000);
  const result = await asr(audio16, { sampling_rate: 16000 });
  const text = (result && (result.text || result)) || "";
  return String(text).trim();
}

function finishUtterance(ctxRate) {
  if (!pending.length) return;
  const len = pending.reduce(function (n, a) {
    return n + a.length;
  }, 0);
  const merged = new Float32Array(len);
  let o = 0;
  pending.forEach(function (a) {
    merged.set(a, o);
    o += a.length;
  });
  pending = [];
  collecting = false;
  speechFrames = 0;
  silenceFrames = 0;
  if (len < ctxRate * 0.28) return;
  emit("think");
  transcribe(merged, ctxRate)
    .then(function (text) {
      if (!session) return;
      if (text && typeof api.onUtterance === "function") api.onUtterance(text);
      else if (session) startListenLoop();
    })
    .catch(function (err) {
      if (typeof api.onError === "function") api.onError(err);
      if (session) startListenLoop();
    });
}

export async function startListenLoop() {
  if (!session || speaking) return;
  await ensureCtx();
  if (!mediaStream) {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
    });
  }
  stopCaptureNodes();
  sourceNode = audioCtx.createMediaStreamSource(mediaStream);
  processor = audioCtx.createScriptProcessor(2048, 1, 1);
  const rate = audioCtx.sampleRate;
  const startRms = 0.018;
  const holdRms = 0.01;
  listening = true;
  collecting = false;
  pending = [];
  speechFrames = 0;
  silenceFrames = 0;
  emit("listen");
  processor.onaudioprocess = function (ev) {
    if (!listening || !session || speaking) return;
    const input = ev.inputBuffer.getChannelData(0);
    const copy = new Float32Array(input);
    const level = rms(copy);
    if (!collecting) {
      if (level > startRms) {
        speechFrames += 1;
        if (speechFrames > 3) {
          collecting = true;
          pending.push(copy);
          if (typeof api.onPartial === "function") api.onPartial("…");
        }
      } else {
        speechFrames = 0;
      }
      return;
    }
    pending.push(copy);
    if (level < holdRms) {
      silenceFrames += 1;
      if (silenceFrames > 14) finishUtterance(rate);
    } else {
      silenceFrames = 0;
    }
    if (pending.length > Math.ceil((rate * 12) / 2048)) finishUtterance(rate);
  };
  sourceNode.connect(processor);
  processor.connect(audioCtx.destination);
}

export async function startSession() {
  session = true;
  interruptFlag = false;
  await startListenLoop();
}

export async function stopSession() {
  session = false;
  stopSpeak();
  await stopMic();
  emit("idle");
}

export async function interrupt() {
  const wasSpeaking = speaking;
  const wasListening = listening;
  stopSpeak();
  if (wasListening && !wasSpeaking) {
    await stopSession();
    return "stopped";
  }
  if (!session) {
    await startSession();
    return "started";
  }
  listening = false;
  pending = [];
  collecting = false;
  emit("listen");
  await startListenLoop();
  return "barge-in";
}

function playBuffer(float32, sampleRate) {
  return new Promise(function (resolve, reject) {
    try {
      speakCtx = speakCtx && speakCtx.state !== "closed" ? speakCtx : new (window.AudioContext || window.webkitAudioContext)();
      if (speakCtx.state === "suspended") speakCtx.resume();
      const buf = speakCtx.createBuffer(1, float32.length, sampleRate);
      buf.getChannelData(0).set(float32);
      const src = speakCtx.createBufferSource();
      src.buffer = buf;
      src.connect(speakCtx.destination);
      speakSource = src;
      src.onended = function () {
        speakSource = null;
        resolve();
      };
      src.start();
    } catch (e) {
      reject(e);
    }
  });
}

function speakBrowser(text, lang) {
  return new Promise(function (resolve) {
    if (!window.speechSynthesis) {
      resolve();
      return;
    }
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = lang || "en-US";
    u.rate = 1.02;
    u.onend = resolve;
    u.onerror = resolve;
    window.speechSynthesis.speak(u);
  });
}

export async function speak(text, lang) {
  if (!text) return;
  interruptFlag = false;
  speaking = true;
  listening = false;
  emit("speak");
  try {
    if (tts && (!lang || lang === "en" || lang === "en-US")) {
      const audio = await tts.generate(text, { voice: api.voiceId || "af_heart" });
      if (interruptFlag) return;
      const data = audio.audio || audio.data;
      const sr = audio.sampling_rate || 24000;
      if (data && data.length) await playBuffer(data, sr);
    } else {
      await speakBrowser(text, lang);
    }
  } catch (e) {
    await speakBrowser(text, lang);
  } finally {
    speaking = false;
    speakSource = null;
    if (typeof api.onSpeakEnd === "function") api.onSpeakEnd(interruptFlag);
    if (session && !interruptFlag) startListenLoop();
    else if (!session) emit("idle");
  }
}

export function listVoices() {
  return ["af_heart", "af_bella", "af_nicole", "am_michael", "bf_emma", "bm_george"];
}

export function setVoice(id) {
  api.voiceId = id;
}

window.PyxAssistantVoice = Object.assign(api, {
  warmup,
  startSession,
  stopSession,
  interrupt,
  startListenLoop,
  stopMic,
  stopSpeak,
  speak,
  listVoices,
  setVoice,
  isListening,
  isSpeaking,
  isSession,
});
window.dispatchEvent(new Event("pyx-voice-ready"));
