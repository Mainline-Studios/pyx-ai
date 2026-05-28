/**
 * Pyx Write — GPT-OSS composer + Tone.js playback.
 * 1.0 adds a formant-synthesis singing voice (vowel/syllable melody); 0.5/0.25 are instrumental.
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

  function writeProfile() {
    var sel = $("writeModel");
    var v = (sel && sel.value) || "0.5";
    if (v === "0.25") return "0.25";
    if (v === "1.0") return "1.0";
    return "0.5";
  }

  function updateModelHint() {
    var hint = $("writeModelHint");
    var bars = $("writeBars");
    var lyricRow = $("writeLyricRow");
    var p = writeProfile();
    if (lyricRow) lyricRow.style.display = p === "1.0" ? "" : "none";
    if (!hint) return;
    if (p === "0.25") {
      hint.textContent =
        "0.25 uses a smaller Llama model — quicker instrumental drafts (about 5–20s). Max 16 bars.";
      if (bars && Number(bars.value) > 16) bars.value = "16";
      if (bars) bars.max = "16";
    } else if (p === "1.0") {
      hint.textContent =
        "1.0 adds a synthesized singing voice (vowel/syllable melody) over the backing instruments — GPT-OSS (about 15–45s). Max 20 bars.";
      if (bars && Number(bars.value) > 20) bars.value = "20";
      if (bars) bars.max = "20";
    } else {
      hint.textContent = "0.5 is instrumental only — GPT-OSS for richer arrangements (about 10–40s).";
      if (bars) bars.max = "24";
    }
  }

  function setMeta(score, model, profileLabel) {
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
    if (profileLabel) parts.push("Write " + profileLabel);
    if (model) parts.push(model);
    el.textContent = parts.filter(Boolean).join(" · ");
  }

  function renderTracks(score) {
    var el = $("writeTracks");
    var lyricEl = $("writeLyrics");
    if (lyricEl) lyricEl.textContent = "";
    if (!el) return;
    if (!score || !score.tracks) {
      el.innerHTML = "";
      return;
    }
    var voiceTrack = null;
    el.innerHTML = score.tracks
      .map(function (tr) {
        var isVoice = tr.instrument === "voice";
        if (isVoice) voiceTrack = tr;
        var label = isVoice ? "🎤 voice" : tr.instrument;
        return (
          '<span class="write-track-chip' +
          (isVoice ? " write-track-chip--voice" : "") +
          '">' +
          escapeHtml(label) +
          " (" +
          (tr.notes ? tr.notes.length : 0) +
          " notes)</span>"
        );
      })
      .join("");
    if (lyricEl && voiceTrack && voiceTrack.notes && voiceTrack.notes.length) {
      var words = voiceTrack.notes.map(function (n) {
        return n.text || n.syllable || "ah";
      });
      lyricEl.textContent = "♪ " + words.join(" ");
    }
  }

  function escapeHtml(s) {
    var d = document.createElement("div");
    d.textContent = s == null ? "" : String(s);
    return d.innerHTML;
  }

  // ===== Singing-voice phoneme engine =====
  // The lead voice sings REAL WORDS: each note's lyric syllable is converted to phonemes
  // (grapheme→phoneme), then rendered as pitched vowel formants + consonant noise/plosives.

  // Vowel formants [F1, F2, F3] (Hz, sung values). glide = second vowel target for diphthongs.
  var VOWELS = {
    IY: { f: [270, 2290, 3010] }, // ee  (see)
    IH: { f: [390, 1990, 2550] }, // i   (sit)
    EH: { f: [530, 1840, 2480] }, // e   (bed)
    AE: { f: [660, 1720, 2410] }, // a   (cat)
    AA: { f: [730, 1090, 2440] }, // ah  (hot/father)
    AO: { f: [570, 840, 2410] }, //  aw  (saw)
    UH: { f: [440, 1020, 2240] }, // oo  (book)
    UW: { f: [300, 870, 2240] }, //  oo  (too)
    AH: { f: [640, 1190, 2390] }, // u   (cup)
    ER: { f: [490, 1350, 1690] }, // er  (her)
    EY: { f: [400, 2080, 2560], glide: "IY" }, // ay (say)
    OW: { f: [450, 870, 2440], glide: "UW" }, //  oh (go)
    AY: { f: [660, 1200, 2480], glide: "IY" }, // i_e (my)
    AW: { f: [680, 1100, 2440], glide: "UW" }, // ow (cow)
    OY: { f: [550, 960, 2440], glide: "IY" }, //  oy (boy)
  };

  // Consonants. type: stop | fric | nasal | liquid | glide. voiced bool.
  // hz/q = noise band for fricatives/bursts. f = formants for voiced consonants. asp = aspiration.
  var CONS = {
    M: { type: "nasal", voiced: true, f: [250, 1000, 2200], amp: 0.45, dur: 0.07 },
    N: { type: "nasal", voiced: true, f: [250, 1600, 2600], amp: 0.45, dur: 0.07 },
    NG: { type: "nasal", voiced: true, f: [250, 2000, 2600], amp: 0.45, dur: 0.07 },
    L: { type: "liquid", voiced: true, f: [360, 1300, 2700], amp: 0.6, dur: 0.06 },
    R: { type: "liquid", voiced: true, f: [480, 1280, 1600], amp: 0.6, dur: 0.06 },
    W: { type: "glide", voiced: true, f: [300, 610, 2200], amp: 0.52, dur: 0.055 },
    Y: { type: "glide", voiced: true, f: [270, 2290, 3010], amp: 0.52, dur: 0.05 },
    S: { type: "fric", voiced: false, hz: 7000, q: 3.4, amp: 0.5, dur: 0.095 },
    Z: { type: "fric", voiced: true, hz: 6500, q: 3.2, amp: 0.34, dur: 0.08 },
    SH: { type: "fric", voiced: false, hz: 3300, q: 2.2, amp: 0.55, dur: 0.095 },
    ZH: { type: "fric", voiced: true, hz: 3100, q: 2.2, amp: 0.36, dur: 0.08 },
    F: { type: "fric", voiced: false, hz: 5000, q: 0.9, amp: 0.32, dur: 0.08 },
    V: { type: "fric", voiced: true, hz: 4800, q: 0.9, amp: 0.24, dur: 0.07 },
    TH: { type: "fric", voiced: false, hz: 6200, q: 0.8, amp: 0.28, dur: 0.08 },
    DH: { type: "fric", voiced: true, hz: 5600, q: 0.8, amp: 0.24, dur: 0.07 },
    HH: { type: "fric", voiced: false, hz: 1900, q: 0.4, amp: 0.26, dur: 0.055, breath: true },
    P: { type: "stop", voiced: false, hz: 1200, q: 0.9, amp: 0.4, dur: 0.085, asp: 0.024 },
    B: { type: "stop", voiced: true, hz: 1000, q: 0.9, amp: 0.32, dur: 0.07 },
    T: { type: "stop", voiced: false, hz: 4000, q: 1.4, amp: 0.46, dur: 0.075, asp: 0.022 },
    D: { type: "stop", voiced: true, hz: 3000, q: 1.3, amp: 0.34, dur: 0.065 },
    K: { type: "stop", voiced: false, hz: 1900, q: 1.1, amp: 0.44, dur: 0.085, asp: 0.03 },
    G: { type: "stop", voiced: true, hz: 1700, q: 1.1, amp: 0.34, dur: 0.07 },
    JH: { type: "affricate", voiced: true, dur: 0.09 },
    CH: { type: "affricate", voiced: false, dur: 0.095 },
  };

  var VOICELESS = { P: 1, T: 1, K: 1, F: 1, TH: 1, S: 1, SH: 1, CH: 1, HH: 1 };

  function isVowelPh(ph) {
    return Object.prototype.hasOwnProperty.call(VOWELS, ph);
  }
  function isVoicelessPh(ph) {
    return !!VOICELESS[ph];
  }

  // Pronunciation dictionary for common / irregular song words (ARPAbet-style tokens).
  var DICT = {
    the: "DH AH", a: "AH", an: "AE N", and: "AE N D", to: "T UW", too: "T UW", two: "T UW",
    of: "AH V", or: "AO R", for: "F AO R", you: "Y UW", your: "Y AO R", youre: "Y AO R",
    i: "AY", im: "AY M", ive: "AY V", ill: "AY L", id: "AY D", my: "M AY", me: "M IY",
    we: "W IY", were: "W ER", he: "HH IY", she: "SH IY", be: "B IY", been: "B IH N",
    is: "IH Z", are: "AA R", am: "AE M", was: "W AH Z", will: "W IH L", would: "W UH D",
    could: "K UH D", should: "SH UH D", do: "D UW", dont: "D OW N T", does: "D AH Z",
    did: "D IH D", go: "G OW", going: "G OW IH NG", gonna: "G AO N AH", got: "G AA T",
    no: "N OW", not: "N AA T", so: "S OW", on: "AA N", in: "IH N", it: "IH T", its: "IH T S",
    this: "DH IH S", that: "DH AE T", with: "W IH DH", what: "W AH T", when: "W EH N",
    where: "W EH R", here: "HH IY R", there: "DH EH R", their: "DH EH R", theyre: "DH EH R",
    they: "DH EY", them: "DH EH M", then: "DH EH N", than: "DH AE N", out: "AW T",
    about: "AH B AW T", up: "AH P", down: "D AW N", all: "AO L", call: "K AO L", fall: "F AO L",
    small: "S M AO L", love: "L AH V", live: "L IH V", life: "L AY F", like: "L AY K",
    light: "L AY T", night: "N AY T", right: "R AY T", might: "M AY T", fight: "F AY T",
    sight: "S AY T", bright: "B R AY T", high: "HH AY", sky: "S K AY", fly: "F L AY",
    why: "W AY", eye: "AY", eyes: "AY Z", fire: "F AY ER", desire: "D IH Z AY ER",
    higher: "HH AY ER", time: "T AY M", mine: "M AY N", shine: "SH AY N", find: "F AY N D",
    mind: "M AY N D", kind: "K AY N D", behind: "B IH HH AY N D", one: "W AH N", once: "W AH N S",
    none: "N AH N", done: "D AH N", come: "K AH M", coming: "K AH M IH NG", some: "S AH M",
    home: "HH OW M", alone: "AH L OW N", hold: "HH OW L D", cold: "K OW L D", gold: "G OW L D",
    old: "OW L D", soul: "S OW L", whole: "HH OW L", hope: "HH OW P", glow: "G L OW",
    slow: "S L OW", show: "SH OW", know: "N OW", grow: "G R OW", flow: "F L OW", low: "L OW",
    snow: "S N OW", throw: "TH R OW", heart: "HH AA R T", start: "S T AA R T", apart: "AH P AA R T",
    dark: "D AA R K", rain: "R EY N", pain: "P EY N", again: "AH G EH N", stay: "S T EY",
    away: "AH W EY", day: "D EY", way: "W EY", say: "S EY", play: "P L EY", pray: "P R EY",
    gray: "G R EY", today: "T AH D EY", maybe: "M EY B IY", baby: "B EY B IY", dream: "D R IY M",
    dreams: "D R IY M Z", dreaming: "D R IY M IH NG", feel: "F IY L", feeling: "F IY L IH NG",
    real: "R IY L", deal: "D IY L", need: "N IY D", free: "F R IY", see: "S IY", sea: "S IY",
    believe: "B IH L IY V", breathe: "B R IY DH", please: "P L IY Z", leave: "L IY V",
    world: "W ER L D", word: "W ER D", bird: "B ER D", heard: "HH ER D", turn: "T ER N",
    burn: "B ER N", learn: "L ER N", girl: "G ER L", first: "F ER S T", heaven: "HH EH V AH N",
    forever: "F ER EH V ER", together: "T AH G EH DH ER", never: "N EH V ER", ever: "EH V ER",
    over: "OW V ER", under: "AH N D ER", water: "W AO T ER", tonight: "T AH N AY T",
    beautiful: "B Y UW T AH F AH L", remember: "R IH M EH M B ER", another: "AH N AH DH ER",
    wonder: "W AH N D ER", thunder: "TH AH N D ER", young: "Y AH NG", song: "S AO NG",
    long: "L AO NG", strong: "S T R AO NG", wrong: "R AO NG", along: "AH L AO NG",
    sing: "S IH NG", bring: "B R IH NG", king: "K IH NG", thing: "TH IH NG",
    everything: "EH V R IY TH IH NG", nothing: "N AH TH IH NG", something: "S AH M TH IH NG",
    forever2: "", oh: "OW", yeah: "Y AE", hey: "HH EY", ooh: "UW", ah: "AA", la: "L AA",
    na: "N AA", woah: "W OW", now: "N AW", how: "HH AW", our: "AW ER", hour: "AW ER",
    around: "AH R AW N D", found: "F AW N D", sound: "S AW N D", ground: "G R AW N D",
    good: "G UH D", girl2: "", boy: "B OY", joy: "JH OY", true: "T R UW", blue: "B L UW",
    you2: "", new: "N UW", through: "TH R UW", you3: ""
  };

  // Heuristic English grapheme→phoneme: dictionary → suffix rules → letter rules.
  function isVowelLetter(c) {
    return "aeiou".indexOf(c) >= 0;
  }

  function g2pLetters(w) {
    if (!w) return [];
    var magicE = /[aeiou][^aeiouy]e$/.test(w) || /[aeiou][^aeiouy][lr]e$/.test(w);
    var ph = [];
    var i = 0, n = w.length;
    // silent onset clusters
    if (n >= 2) {
      var st = w.substr(0, 2);
      if (st === "kn" || st === "gn" || st === "pn") { ph.push("N"); i = 2; }
      else if (st === "wr") { ph.push("R"); i = 2; }
      else if (st === "ps") { ph.push("S"); i = 2; }
    }
    while (i < n) {
      var two = w.substr(i, 2);
      var three = w.substr(i, 3);
      var c = w[i];
      var next = w[i + 1] || "";
      if (three === "tch") { ph.push("CH"); i += 3; continue; }
      if (three === "igh") { ph.push("AY"); i += 3; continue; }
      if (three === "dge") { ph.push("JH"); i += 3; continue; }
      if (two === "sh") { ph.push("SH"); i += 2; continue; }
      if (two === "ch") { ph.push("CH"); i += 2; continue; }
      if (two === "th") { ph.push("TH"); i += 2; continue; }
      if (two === "ph") { ph.push("F"); i += 2; continue; }
      if (two === "wh") { ph.push("W"); i += 2; continue; }
      if (two === "ng") { ph.push("NG"); i += 2; continue; }
      if (two === "ck") { ph.push("K"); i += 2; continue; }
      if (two === "qu") { ph.push("K", "W"); i += 2; continue; }
      if (two === "gh") { i += 2; continue; }
      if (two === "mb" && i + 2 >= n) { ph.push("M"); i += 2; continue; }
      if (two === "ee" || two === "ea" || two === "ie") { ph.push("IY"); i += 2; continue; }
      if (two === "oo") { ph.push("UW"); i += 2; continue; }
      if ((two === "ow" || two === "ou") && !isVowelLetter(w[i + 2] || "")) { ph.push("AW"); i += 2; continue; }
      if (two === "aw" && !isVowelLetter(w[i + 2] || "")) { ph.push("AO"); i += 2; continue; }
      if (two === "oa") { ph.push("OW"); i += 2; continue; }
      if (two === "oi" || two === "oy") { ph.push("OY"); i += 2; continue; }
      if (two === "ai" || two === "ay" || two === "ei" || two === "ey") { ph.push("EY"); i += 2; continue; }
      if (two === "au") { ph.push("AO"); i += 2; continue; }
      if (two === "ar" && (i + 2 >= n || !isVowelLetter(w[i + 2]))) { ph.push("AA", "R"); i += 2; continue; }
      if (two === "or" && (i + 2 >= n || !isVowelLetter(w[i + 2]))) { ph.push("AO", "R"); i += 2; continue; }
      if ((two === "er" || two === "ir" || two === "ur") && (i + 2 >= n || !isVowelLetter(w[i + 2]))) {
        ph.push("ER"); i += 2; continue;
      }
      if (isVowelLetter(c)) {
        if (c === "e" && i === n - 1 && ph.length) { i += 1; continue; }
        if (magicE && i < n - 2 && !isVowelLetter(next)) {
          if (c === "a") ph.push("EY");
          else if (c === "i") ph.push("AY");
          else if (c === "o") ph.push("OW");
          else if (c === "u") ph.push("UW");
          else ph.push("IY");
          i += 1; continue;
        }
        if (c === "a") ph.push("AE");
        else if (c === "e") ph.push("EH");
        else if (c === "i") ph.push("IH");
        else if (c === "o") ph.push("AA");
        else if (c === "u") ph.push("AH");
        i += 1; continue;
      }
      if (c === "y") {
        if (i === 0) { ph.push("Y"); i += 1; continue; }
        ph.push(i === n - 1 && n <= 3 ? "AY" : "IY");
        i += 1; continue;
      }
      if (c === next) { i += 1; continue; }
      switch (c) {
        case "b": ph.push("B"); break;
        case "c": ph.push("eiy".indexOf(next) >= 0 ? "S" : "K"); break;
        case "d": ph.push("D"); break;
        case "f": ph.push("F"); break;
        case "g": ph.push("eiy".indexOf(next) >= 0 ? "JH" : "G"); break;
        case "h": ph.push("HH"); break;
        case "j": ph.push("JH"); break;
        case "k": ph.push("K"); break;
        case "l": ph.push("L"); break;
        case "m": ph.push("M"); break;
        case "n": ph.push("N"); break;
        case "p": ph.push("P"); break;
        case "q": ph.push("K"); break;
        case "r": ph.push("R"); break;
        case "s": ph.push("S"); break;
        case "t": ph.push("T"); break;
        case "v": ph.push("V"); break;
        case "w": ph.push("W"); break;
        case "x": ph.push("K", "S"); break;
        case "z": ph.push("Z"); break;
        default: break;
      }
      i += 1;
    }
    if (!ph.some(isVowelPh)) ph.push("AH");
    return ph;
  }

  function g2pStem(w) {
    if (DICT[w]) return DICT[w].split(" ");
    return g2pLetters(w);
  }

  function wordToPhonemes(word) {
    var w = String(word || "").toLowerCase().replace(/[^a-z]/g, "");
    if (!w) return [];
    if (DICT[w]) return DICT[w].split(" ");
    var ph;
    // suffix transforms (recursive on the stem)
    if (w.length > 5 && /tion$/.test(w)) ph = g2pStem(w.slice(0, -4)).concat(["SH", "AH", "N"]);
    else if (w.length > 5 && /sion$/.test(w)) ph = g2pStem(w.slice(0, -4)).concat(["ZH", "AH", "N"]);
    else if (w.length > 5 && /(cious|tious)$/.test(w)) ph = g2pStem(w.slice(0, -5)).concat(["SH", "AH", "S"]);
    else if (w.length > 4 && /ing$/.test(w)) ph = g2pStem(w.slice(0, -3)).concat(["IH", "NG"]);
    else if (w.length > 4 && / ment$/.test(w)) ph = g2pStem(w.slice(0, -4)).concat(["M", "AH", "N", "T"]);
    else if (w.length > 4 && /ness$/.test(w)) ph = g2pStem(w.slice(0, -4)).concat(["N", "AH", "S"]);
    else if (w.length > 4 && /ful$/.test(w)) ph = g2pStem(w.slice(0, -3)).concat(["F", "AH", "L"]);
    else if (w.length > 4 && /ous$/.test(w)) ph = g2pStem(w.slice(0, -3)).concat(["AH", "S"]);
    else if (w.length > 3 && /ly$/.test(w)) ph = g2pStem(w.slice(0, -2)).concat(["L", "IY"]);
    else if (w.length > 3 && /le$/.test(w) && !isVowelLetter(w[w.length - 3])) {
      ph = g2pStem(w.slice(0, -2)).concat(["AH", "L"]);
    } else if (w.length > 3 && /ed$/.test(w)) {
      var st = g2pStem(w.slice(0, -2));
      var last = st[st.length - 1];
      if (last === "T" || last === "D") st = st.concat(["IH", "D"]);
      else if (isVoicelessPh(last)) st = st.concat(["T"]);
      else st = st.concat(["D"]);
      ph = st;
    } else if (w.length > 3 && /(ches|shes|sses|zes|xes)$/.test(w)) {
      ph = g2pStem(w.slice(0, -2)).concat(["IH", "Z"]);
    } else if (w.length > 2 && /s$/.test(w) && !/(ss|us|is)$/.test(w)) {
      var st2 = g2pStem(w.slice(0, -1));
      var last2 = st2[st2.length - 1];
      if (["S", "Z", "SH", "ZH", "CH", "JH"].indexOf(last2) >= 0) st2 = st2.concat(["IH", "Z"]);
      else if (isVoicelessPh(last2)) st2 = st2.concat(["S"]);
      else st2 = st2.concat(["Z"]);
      ph = st2;
    } else {
      ph = g2pLetters(w);
    }
    return ph.slice(0, 10);
  }

  // Fallback: old vowel/syllable names → phonemes (for notes lacking a word).
  var SYLL_PH = {
    ah: ["AA"], aah: ["AA"], aa: ["AA"], eh: ["EH"], ih: ["IH"], ee: ["IY"],
    oh: ["OW"], oo: ["UW"], ooh: ["UW"], mm: ["M", "AH"], hmm: ["HH", "AH"],
    la: ["L", "AA"], na: ["N", "AA"], ya: ["Y", "AA"], ba: ["B", "AA"],
    da: ["D", "AA"], doo: ["D", "UW"], doot: ["D", "UW", "T"],
  };
  function syllablePhonemes(syll) {
    var key = String(syll || "ah").toLowerCase();
    return (SYLL_PH[key] || ["AA"]).slice();
  }

  // High-quality monophonic formant singer: spectral-tilted glottal source,
  // 3 dynamic formants + 2 static singer's-formant peaks, nasal anti-formant,
  // parallel consonant-noise path, chorus + reverb polish.
  function createVoice() {
    var master = new Tone.Gain(0.85).toDestination();
    var reverb = null;
    try {
      reverb = new Tone.Freeverb({ roomSize: 0.72, dampening: 3000, wet: 0.18 });
      reverb.connect(master);
    } catch (e) {
      reverb = master;
    }
    var chorus = null;
    try {
      chorus = new Tone.Chorus({ frequency: 1.1, delayTime: 3.5, depth: 0.35, wet: 0.22 }).start();
      chorus.connect(reverb);
    } catch (e) {
      chorus = reverb;
    }
    var bus = new Tone.Gain(1).connect(chorus);
    var tone = new Tone.Filter({ type: "lowpass", frequency: 6500, Q: 0.3 }).connect(bus);

    // Voiced path: 2 detuned saws → spectral tilt → voicedGain → nasal notch → formants.
    var voicedGain = new Tone.Gain(0.0001);
    var nasalNotch = new Tone.Filter({ type: "notch", frequency: 8000, Q: 0.0001 });
    var tilt = new Tone.Filter({ type: "highshelf", frequency: 2200, gain: -10 });
    var f1 = new Tone.Filter({ type: "bandpass", frequency: 600, Q: 8 });
    var f2 = new Tone.Filter({ type: "bandpass", frequency: 1100, Q: 9 });
    var f3 = new Tone.Filter({ type: "bandpass", frequency: 2500, Q: 10 });
    var f4 = new Tone.Filter({ type: "bandpass", frequency: 3300, Q: 5 }); // singer's formant (broad to avoid whistle)
    var f5 = new Tone.Filter({ type: "bandpass", frequency: 3850, Q: 5 });
    var g1 = new Tone.Gain(1.0), g2 = new Tone.Gain(0.78), g3 = new Tone.Gain(0.42),
      g4 = new Tone.Gain(0.14), g5 = new Tone.Gain(0.08);
    var osc1 = new Tone.Oscillator({ type: "sawtooth", frequency: 220 }).start();
    var osc2 = new Tone.Oscillator({ type: "sawtooth", frequency: 220, detune: -9 }).start();
    var vib = new Tone.LFO({ frequency: 5.3, min: -13, max: 13, type: "sine" }).start();
    var drift = new Tone.LFO({ frequency: 0.7, min: -3, max: 3, type: "sine" }).start();
    vib.connect(osc1.detune); vib.connect(osc2.detune);
    drift.connect(osc1.detune);
    osc1.connect(tilt); osc2.connect(tilt);
    tilt.connect(voicedGain);
    voicedGain.connect(nasalNotch);
    nasalNotch.connect(f1); nasalNotch.connect(f2); nasalNotch.connect(f3);
    nasalNotch.connect(f4); nasalNotch.connect(f5);
    f1.connect(g1); f2.connect(g2); f3.connect(g3); f4.connect(g4); f5.connect(g5);
    g1.connect(tone); g2.connect(tone); g3.connect(tone); g4.connect(tone); g5.connect(tone);

    // Consonant path: white noise → tunable bandpass → noiseGain → bus.
    var noiseGain = new Tone.Gain(0.0001);
    var consBP = new Tone.Filter({ type: "bandpass", frequency: 4000, Q: 2 });
    var noise = new Tone.Noise({ type: "white" }).start();
    noise.connect(consBP); consBP.connect(noiseGain); noiseGain.connect(bus);

    return {
      type: "voice",
      nodes: [osc1, osc2, noise, vib, drift, voicedGain, noiseGain, tilt, nasalNotch,
        f1, f2, f3, f4, f5, g1, g2, g3, g4, g5, consBP, tone, bus, chorus, reverb, master],
      osc1: osc1, osc2: osc2,
      voicedGain: voicedGain, noiseGain: noiseGain,
      f1: f1, f2: f2, f3: f3, consBP: consBP, nasalNotch: nasalNotch,
    };
  }

  function scheduleVoiceNote(voice, pitch, time, duration, velocity, syllable, text) {
    var freq;
    try {
      freq = Tone.Frequency(pitch).toFrequency();
    } catch (e) {
      return;
    }
    if (!freq || !isFinite(freq)) return;
    var v = Math.max(0.15, Math.min(1, velocity || 0.8)) * 0.82;

    // Pitch with a tiny scoop into the note (natural attack).
    var scoop = Math.min(0.04, duration * 0.18);
    [voice.osc1, voice.osc2].forEach(function (osc) {
      osc.frequency.cancelScheduledValues(time);
      osc.frequency.setValueAtTime(freq * 0.972, time);
      osc.frequency.linearRampToValueAtTime(freq, time + scoop);
    });

    var phs = text ? wordToPhonemes(text) : syllablePhonemes(syllable);
    if (!phs.length) phs = ["AA"];

    var firstV = -1, lastV = -1;
    for (var k = 0; k < phs.length; k++) {
      if (isVowelPh(phs[k])) { if (firstV < 0) firstV = k; lastV = k; }
    }
    if (firstV < 0) { phs = ["AH"]; firstV = 0; lastV = 0; }
    var onset = phs.slice(0, firstV);
    var nucleus = phs.slice(firstV, lastV + 1).filter(isVowelPh);
    var coda = phs.slice(lastV + 1);

    function consDur(ph) {
      var info = CONS[ph];
      return info ? info.dur : 0.06;
    }
    var onsetSum = onset.reduce(function (s, p) { return s + consDur(p); }, 0);
    var codaSum = coda.reduce(function (s, p) { return s + consDur(p); }, 0);
    var onsetTotal = onsetSum, codaTotal = codaSum;
    var minVowel = 0.11;
    var maxCons = Math.max(0, duration - minVowel);
    if (onsetTotal + codaTotal > maxCons && onsetTotal + codaTotal > 0) {
      var scale = maxCons / (onsetTotal + codaTotal);
      onsetTotal *= scale; codaTotal *= scale;
    }
    var vowelStart = time + onsetTotal;
    var vowelEnd = Math.max(vowelStart + minVowel, time + duration - codaTotal);

    var prev = {
      voiced: 0, noise: 0,
      f: [voice.f1.frequency.value, voice.f2.frequency.value, voice.f3.frequency.value],
    };
    voice.voicedGain.gain.cancelScheduledValues(time);
    voice.noiseGain.gain.cancelScheduledValues(time);
    voice.f1.frequency.cancelScheduledValues(time);
    voice.f2.frequency.cancelScheduledValues(time);
    voice.f3.frequency.cancelScheduledValues(time);
    if (voice.nasalNotch) {
      voice.nasalNotch.frequency.cancelScheduledValues(time);
      voice.nasalNotch.frequency.setValueAtTime(8000, time);
      voice.nasalNotch.Q.setValueAtTime(0.0001, time);
    }

    function ramp(t0, dur, opts) {
      var rp = Math.min(opts.ramp != null ? opts.ramp : 0.018, Math.max(0.004, dur * 0.6));
      if (opts.voiced != null) {
        var vv = Math.max(0, opts.voiced);
        voice.voicedGain.gain.setValueAtTime(prev.voiced, t0);
        voice.voicedGain.gain.linearRampToValueAtTime(vv, t0 + rp);
        prev.voiced = vv;
      }
      if (opts.noise != null) {
        var nn = Math.max(0, opts.noise);
        voice.noiseGain.gain.setValueAtTime(prev.noise, t0);
        voice.noiseGain.gain.linearRampToValueAtTime(nn, t0 + rp);
        prev.noise = nn;
      }
      if (opts.formants) {
        var ff = [voice.f1, voice.f2, voice.f3];
        for (var j = 0; j < 3; j++) {
          ff[j].frequency.setValueAtTime(prev.f[j], t0);
          ff[j].frequency.linearRampToValueAtTime(opts.formants[j], t0 + rp);
          prev.f[j] = opts.formants[j];
        }
      }
      if (opts.consBP) {
        voice.consBP.frequency.setValueAtTime(opts.consBP[0], t0);
        voice.consBP.Q.setValueAtTime(opts.consBP[1], t0);
      }
      if (opts.notch && voice.nasalNotch) {
        voice.nasalNotch.frequency.setValueAtTime(opts.notch[0], t0);
        voice.nasalNotch.Q.setValueAtTime(opts.notch[1], t0);
      }
    }

    function scheduleConsonant(ph, t0, t1, towardF) {
      var info = CONS[ph];
      if (!info) return;
      var dur = t1 - t0;
      if (dur <= 0.004) return;
      if (info.type === "affricate") {
        if (ph === "CH") {
          scheduleConsonant("T", t0, t0 + dur * 0.42, towardF);
          scheduleConsonant("SH", t0 + dur * 0.42, t1, towardF);
        } else {
          scheduleConsonant("D", t0, t0 + dur * 0.42, towardF);
          scheduleConsonant("ZH", t0 + dur * 0.42, t1, towardF);
        }
        return;
      }
      if (info.type === "stop") {
        var closure = dur * 0.5;
        // voiced stops keep a low "voice bar" during closure
        ramp(t0, closure, { voiced: info.voiced ? v * 0.14 : 0.0001, noise: 0.0001, ramp: 0.006 });
        var burstAt = t0 + closure;
        var burstDur = Math.max(0.014, dur - closure);
        ramp(burstAt, burstDur * 0.45, {
          noise: info.amp * v,
          consBP: [info.hz, info.q],
          voiced: info.voiced ? v * 0.4 : 0.0001,
          ramp: 0.003,
        });
        var aspAt = burstAt + burstDur * 0.45;
        if (info.asp && !info.voiced) {
          // aspiration: brief breathy noise into the vowel
          ramp(aspAt, burstDur * 0.55, { noise: v * 0.18, consBP: [2400, 0.6], ramp: 0.004 });
        } else {
          ramp(aspAt, burstDur * 0.55, { noise: 0.0001, ramp: 0.012 });
        }
      } else if (info.type === "fric") {
        ramp(t0, dur, {
          noise: info.amp * v,
          consBP: [info.hz, info.q],
          voiced: info.voiced ? v * 0.4 : 0.0001,
          formants: info.voiced ? (towardF || null) : null,
          ramp: 0.012,
        });
      } else if (info.type === "nasal") {
        ramp(t0, dur, { voiced: info.amp * v, noise: 0.0001, formants: info.f, notch: [1100, 1.4], ramp: 0.022 });
      } else {
        // liquid / glide
        ramp(t0, dur, { voiced: info.amp * v, noise: 0.0001, formants: info.f, ramp: 0.02 });
        if (info.type === "glide" && towardF) {
          ramp(t0 + dur * 0.4, dur * 0.6, { formants: towardF, ramp: dur * 0.5 });
        }
      }
    }

    var nucF = VOWELS[nucleus[0]] ? VOWELS[nucleus[0]].f : VOWELS.AH.f;

    var t = time;
    for (var oi = 0; oi < onset.length; oi++) {
      var od = onsetTotal * (consDur(onset[oi]) / (onsetSum || 1));
      scheduleConsonant(onset[oi], t, t + od, nucF);
      t += od;
    }

    // Re-open the nasal notch when entering the vowel (in case last onset was nasal).
    if (voice.nasalNotch) {
      voice.nasalNotch.frequency.setValueAtTime(8000, vowelStart);
      voice.nasalNotch.Q.setValueAtTime(0.0001, vowelStart);
    }

    var atk = Math.min(0.055, (vowelEnd - vowelStart) * 0.4);
    var segCount = nucleus.length || 1;
    var segLen = (vowelEnd - vowelStart) / segCount;
    for (var vi = 0; vi < segCount; vi++) {
      var vd = VOWELS[nucleus[vi]] || VOWELS.AH;
      var segStart = vowelStart + vi * segLen;
      var segOpts = { voiced: v, formants: vd.f, ramp: vi === 0 ? atk : 0.045 };
      if (vi === 0) segOpts.noise = 0; // kill onset consonant noise once the vowel begins
      ramp(segStart, vi === 0 ? atk : 0.045, segOpts);
      if (vd.glide && VOWELS[vd.glide]) {
        ramp(segStart + segLen * 0.5, segLen * 0.45, { formants: VOWELS[vd.glide].f, ramp: segLen * 0.4 });
      }
    }

    t = vowelEnd;
    for (var ci = 0; ci < coda.length; ci++) {
      var cd = codaTotal * (consDur(coda[ci]) / (codaSum || 1));
      scheduleConsonant(coda[ci], t, t + cd, null);
      t += cd;
    }

    var rel = Math.min(0.08, duration * 0.28);
    var relAt = Math.max(time, time + duration - rel);
    voice.voicedGain.gain.setValueAtTime(prev.voiced, relAt);
    voice.voicedGain.gain.linearRampToValueAtTime(0, time + duration);
    voice.noiseGain.gain.setValueAtTime(prev.noise, relAt);
    voice.noiseGain.gain.linearRampToValueAtTime(0, time + duration);
    prev.voiced = 0;
    prev.noise = 0;
  }

  function createInstrument(instrumentId) {
    switch (instrumentId) {
      case "voice":
        return createVoice();
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

  function scheduleNote(synth, instrumentId, pitch, time, duration, velocity, syllable, text) {
    var v = Math.max(0.15, Math.min(1, velocity || 0.75));
    if (instrumentId === "voice" && synth && synth.type === "voice") {
      scheduleVoiceNote(synth, pitch, time, duration, v, syllable, text);
      return;
    }
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
    if (s.type === "voice") {
      (s.nodes || []).forEach(function (n) {
        try {
          if (n && typeof n.stop === "function") n.stop();
        } catch (e) {}
        try {
          if (n && typeof n.dispose === "function") n.dispose();
        } catch (e) {}
      });
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
            scheduleNote(synth, tr.instrument, n.pitch, t, dur, n.velocity, n.syllable, n.text);
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
    var profile = writeProfile();
    var lyricTheme = ($("writeLyricTheme") && $("writeLyricTheme").value) || "";
    var composingMsg =
      profile === "0.25"
        ? "Composing with Write 0.25 (fast)… (5–20s)"
        : profile === "1.0"
        ? "Composing a song with Write 1.0 (singing voice)… (15–45s)"
        : "Composing with Write 0.5 (GPT-OSS)… (10–40s)";
    if (btn) btn.disabled = true;
    setStatus(composingMsg, "");
    stopPlayback();
    state.score = null;
    renderTracks(null);
    setMeta(null, "", "");

    fetch(API, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      cache: "no-store",
      body: JSON.stringify({
        prompt: prompt,
        style: style,
        instruments: selectedInstruments(),
        bars: bars,
        write_profile: profile,
        lyric_theme: lyricTheme,
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
        var prof = x.data.write_profile || x.data.version || profile;
        setMeta(state.score, state.model, prof);
        renderTracks(state.score);
        var hasVoice = (state.score.tracks || []).some(function (t) {
          return t.instrument === "voice";
        });
        setStatus(hasVoice ? "Ready — press Play to hear the vocals." : "Ready — press Play.", "ok");
        try {
          localStorage.setItem("pyx.write.lastPrompt", prompt);
          localStorage.setItem("pyx.write.lastAt", String(Date.now()));
          localStorage.setItem("pyx.write.profile", profile);
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
    var modelSel = $("writeModel");
    if (modelSel) {
      try {
        var saved = localStorage.getItem("pyx.write.profile");
        if (saved === "0.25" || saved === "0.5" || saved === "1.0") modelSel.value = saved;
      } catch (e) {}
      modelSel.addEventListener("change", updateModelHint);
      updateModelHint();
    }
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
