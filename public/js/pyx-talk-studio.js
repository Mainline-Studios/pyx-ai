/**
 * Pyx Talk — Studio handoffs, safety strip, inline TTS.
 */
(function (global) {
  "use strict";

  var INLINE_TTS_MAX = 280;
  var styleInjected = false;

  function injectStyles() {
    if (styleInjected) return;
    styleInjected = true;
    var s = document.createElement("style");
    s.textContent =
      ".pyx-msg-handoff{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px;padding-top:8px;border-top:1px solid rgba(148,163,184,.2)}" +
      ".pyx-msg-handoff button{font:inherit;font-size:.72rem;font-weight:700;padding:4px 8px;border-radius:999px;border:1px solid rgba(129,140,248,.35);background:rgba(99,102,241,.15);color:#c7d2fe;cursor:pointer}" +
      ".pyx-msg-handoff button:hover{border-color:rgba(167,139,250,.55)}" +
      ".pyx-safety-row{margin-top:8px;padding:8px 10px;border-radius:10px;border:1px solid rgba(248,113,113,.35);background:rgba(127,29,29,.2);font-size:.82rem;color:#fecaca}" +
      ".pyx-safety-row.is-ok{border-color:rgba(52,211,153,.35);background:rgba(6,78,59,.25);color:#a7f3d0}";
    document.head.appendChild(s);
  }

  function attachHandoffToolbar(bubbleEl, text) {
    if (!global.PyxHandoff || !bubbleEl || !text) return;
    injectStyles();
    if (bubbleEl.querySelector(".pyx-msg-handoff")) return;
    var bar = document.createElement("div");
    bar.className = "pyx-msg-handoff";
    var plain = text.replace(/```[\s\S]*?```/g, " ").trim();
    if (!plain) return;

    function btn(label, fn) {
      var b = document.createElement("button");
      b.type = "button";
      b.textContent = label;
      b.addEventListener("click", fn);
      bar.appendChild(b);
    }

    btn("Speak", function () {
      global.PyxHandoff.sendTo("speak", plain, "talk");
    });
    btn("Pyxel", function () {
      global.PyxHandoff.sendTo("pyxel", "10x10 pixel art: " + plain.slice(0, 200), "talk");
    });
    var block = global.PyxHandoff.extractFirstCodeBlock(text);
    if (block) {
      btn("Code", function () {
        global.PyxHandoff.setHandoff({
          source: "talk",
          target: "code",
          text: block.code,
          meta: { lang: block.lang || global.PyxHandoff.guessLangFromCode(block.code) },
        });
        global.location.href = "/pyx-code.html?handoff=code";
      });
    }
    btn("Gallery", function () {
      global.PyxHandoff.saveGalleryItem({
        kind: "talk",
        title: plain.slice(0, 40),
        preview: plain.slice(0, 500),
      });
      alert("Saved to Pyx Gallery!");
    });
    if (plain.length <= INLINE_TTS_MAX) {
      btn("▶ Listen", function () {
        playInlineTts(plain, bar);
      });
    }
    bubbleEl.appendChild(bar);
  }

  function playInlineTts(text, bar) {
    var prev = bar.querySelector(".pyx-inline-audio");
    if (prev) prev.remove();
    var audio = document.createElement("audio");
    audio.className = "pyx-inline-audio";
    audio.controls = true;
    bar.appendChild(audio);
    fetch("/api/speak/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text, format: "mp3" }),
    })
      .then(function (r) {
        if (!r.ok) throw new Error("TTS failed");
        return r.blob();
      })
      .then(function (blob) {
        audio.src = URL.createObjectURL(blob);
        audio.play();
      })
      .catch(function () {
        global.PyxHandoff.sendTo("speak", text, "talk");
      });
  }

  function installSendGuard() {
    var sendBtn = document.getElementById("send");
    var inputEl = document.getElementById("input");
    var safetyEl = document.getElementById("pyxSafetyCheck");
    if (!sendBtn || !inputEl) return;

    sendBtn.addEventListener(
      "click",
      function (e) {
        if (!safetyEl || !safetyEl.checked) return;
        var text = (inputEl.value || "").trim();
        if (!text) return;
        e.stopImmediatePropagation();
        e.preventDefault();
        checkScore(text, function (ok, censored) {
          if (!ok) {
            showSafety(inputEl, "Message blocked: " + (censored || "doesn’t pass the safety check"), false);
            return;
          }
          showSafety(inputEl, "Looks good — sending…", true);
          safetyEl.checked = false;
          sendBtn.click();
        });
      },
      true
    );
  }

  function showSafety(anchor, msg, ok) {
    var row = document.getElementById("pyxSafetyStatus");
    if (!row) {
      row = document.createElement("div");
      row.id = "pyxSafetyStatus";
      row.className = "pyx-safety-row";
      var webRow = document.getElementById("webRow");
      if (webRow && webRow.parentNode) webRow.parentNode.insertBefore(row, webRow.nextSibling);
    }
    row.textContent = msg;
    row.classList.toggle("is-ok", !!ok);
  }

  function checkScore(text, cb) {
    fetch("/api/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text }),
    })
      .then(function (r) {
        return r.json();
      })
      .then(function (j) {
        cb(!j.bad, j.censored || "");
      })
      .catch(function () {
        cb(true, "");
      });
  }

  function installGalleryDock() {
    var btn = document.getElementById("dockGalleryBtn");
    if (btn) {
      btn.title = "Pyx Gallery";
      btn.addEventListener("click", function () {
        global.location.href = "/pyx-gallery.html";
      });
    }
  }

  function install() {
    injectStyles();
    installSendGuard();
    installGalleryDock();
    if (global.PyxShell) global.PyxShell.init({ active: "talk" });
    global.PyxTalkStudio = global.PyxTalkStudio || {};
    global.PyxTalkStudio.attachHandoffToolbar = attachHandoffToolbar;
  }

  global.PyxTalkStudio = {
    install: install,
    attachHandoffToolbar: attachHandoffToolbar,
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", install);
  } else {
    install();
  }
})(typeof window !== "undefined" ? window : globalThis);
