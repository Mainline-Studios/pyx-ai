/**
 * PyxCaptcha — compact traffic-light captcha (iframe-friendly).
 * User label always trains Pyx; disagreement yields a second challenge.
 */
(function (global) {
  "use strict";

  var API = "/api/dev-workshop/traffic/captcha";
  var FEATURE_W = 120;
  var FEATURE_H = 90;
  var COLORS = ["red", "yellow", "green", "off"];

  var state = {
    challengeId: null,
    publicUrl: null,
    busy: false,
  };

  function $(id) {
    return document.getElementById(id);
  }

  function api(path, body) {
    return fetch(API + path, {
      method: body ? "POST" : "GET",
      headers: body ? { "Content-Type": "application/json" } : {},
      body: body ? JSON.stringify(body) : undefined,
    }).then(function (r) {
      return r.json().then(function (j) {
        return { ok: r.ok, data: j };
      });
    });
  }

  function extractFeaturesFromImageData(d, w, h) {
    var topH = Math.floor(h * 0.55);
    var n = 0;
    var meanR = 0;
    var meanG = 0;
    var meanB = 0;
    var cntR = 0;
    var cntY = 0;
    var cntG = 0;
    var brightTop = 0;
    var brightBot = 0;
    var botN = 0;
    var y;
    var x;
    var i;
    var r;
    var g;
    var b;
    for (y = 0; y < topH; y++) {
      for (x = 0; x < w; x++) {
        i = (y * w + x) * 4;
        r = d[i];
        g = d[i + 1];
        b = d[i + 2];
        meanR += r;
        meanG += g;
        meanB += b;
        brightTop += (r + g + b) / 3;
        n++;
        if (r > 165 && g < 115 && b < 115) cntR++;
        else if (g > 145 && r < 150 && b < 130) cntG++;
        else if (r > 145 && g > 125 && b < 95) cntY++;
      }
    }
    for (y = topH; y < h; y++) {
      for (x = 0; x < w; x++) {
        i = (y * w + x) * 4;
        r = d[i];
        g = d[i + 1];
        b = d[i + 2];
        brightBot += (r + g + b) / 3;
        botN++;
      }
    }
    if (n < 1) n = 1;
    if (botN < 1) botN = 1;
    return [
      meanR / n / 255,
      meanG / n / 255,
      meanB / n / 255,
      cntR / n,
      cntY / n,
      cntG / n,
      brightTop / n / 255,
      brightBot / botN / 255,
    ];
  }

  function extractFeatures(img) {
    var c = document.createElement("canvas");
    c.width = FEATURE_W;
    c.height = FEATURE_H;
    var ctx = c.getContext("2d");
    ctx.drawImage(img, 0, 0, FEATURE_W, FEATURE_H);
    var data = ctx.getImageData(0, 0, FEATURE_W, FEATURE_H);
    return extractFeaturesFromImageData(data.data, FEATURE_W, FEATURE_H);
  }

  function loadImage(url) {
    return new Promise(function (resolve, reject) {
      var img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = function () {
        resolve(img);
      };
      img.onerror = function () {
        reject(new Error("Could not load image"));
      };
      img.src = url;
    });
  }

  function escapeHtml(s) {
    var d = document.createElement("div");
    d.textContent = s == null ? "" : String(s);
    return d.innerHTML;
  }

  function notifyParent(detail) {
    try {
      if (global.parent && global.parent !== global) {
        global.parent.postMessage(
          { type: "pyx-captcha-done", agreed: !!detail.agreed, color: detail.color },
          "*"
        );
      }
    } catch (e) {
      /* ignore */
    }
  }

  function setStatus(text, kind) {
    var el = $("capStatus");
    if (!el) return;
    el.textContent = text || "";
    el.className = "cap-status" + (kind ? " cap-status--" + kind : "");
  }

  function setBusy(busy) {
    state.busy = busy;
    var root = $("capRoot");
    if (!root) return;
    root.querySelectorAll(".cap-colors button").forEach(function (btn) {
      btn.disabled = busy;
    });
    var next = $("capNext");
    if (next) next.disabled = busy;
  }

  function applyChallenge(ch) {
    state.challengeId = ch.challenge_id;
    state.publicUrl = ch.public_url;
    renderUI(ch);
  }

  function renderUI(ch) {
    var root = $("capRoot");
    if (!root) return;
    var hint = ch.hint ? '<span class="cap-badge">' + escapeHtml(ch.hint) + "</span>" : "";
    root.className = "";
    root.innerHTML =
      '<div class="cap-head"><strong>PyxCaptcha</strong>' +
      hint +
      "</div>" +
      '<div class="cap-body">' +
      '<div class="cap-img-wrap"><img id="capImg" src="' +
      escapeHtml(ch.public_url) +
      '" alt="Traffic signal" crossorigin="anonymous" /></div>' +
      '<div class="cap-side">' +
      '<p class="cap-prompt">Which light is on?</p>' +
      '<div class="cap-colors" id="capColors">' +
      COLORS.map(function (c) {
        return (
          '<button type="button" data-c="' +
          c +
          '">' +
          c.charAt(0).toUpperCase() +
          c.slice(1) +
          "</button>"
        );
      }).join("") +
      "</div>" +
      '<p class="cap-status" id="capStatus"></p>' +
      '<button type="button" class="cap-next" id="capNext">Next challenge</button>' +
      "</div></div>";

    $("capColors").addEventListener("click", function (e) {
      var btn = e.target.closest("button[data-c]");
      if (!btn || state.busy) return;
      submitColor(btn.getAttribute("data-c"));
    });
    $("capNext").addEventListener("click", function () {
      loadChallenge();
    });
  }

  function loadChallenge(hint) {
    setBusy(true);
    var root = $("capRoot");
    if (root && !state.challengeId) {
      root.className = "cap-loading";
      root.textContent = "Loading challenge…";
    }
    var q = hint ? "?hint=" + encodeURIComponent(hint) : "";
    api("/challenge" + q, null)
      .then(function (x) {
        if (!x.ok || !x.data.ok) {
          throw new Error((x.data && x.data.error) || "Could not load challenge");
        }
        applyChallenge(x.data);
        setStatus("");
        setBusy(false);
      })
      .catch(function (e) {
        if (root) {
          root.className = "cap-loading";
          root.textContent = e.message || String(e);
        }
        setBusy(false);
      });
  }

  function showNextButton(show) {
    var next = $("capNext");
    if (next) next.style.display = show ? "inline-block" : "none";
  }

  function submitColor(color) {
    if (!state.challengeId || !state.publicUrl) return;
    setBusy(true);
    setStatus("Checking…");
    loadImage(state.publicUrl)
      .then(function (img) {
        var features = extractFeatures(img);
        return api("/submit", {
          challenge_id: state.challengeId,
          color: color,
          features: features,
        });
      })
      .then(function (x) {
        var d = x.data || {};
        if (!x.ok || !d.ok) {
          throw new Error(d.error || "Submit failed");
        }
        state.challengeId = null;
        if (d.agreed) {
          setStatus("Correct — thanks for training Pyx.", "ok");
          notifyParent({ agreed: true, color: color });
          showNextButton(true);
          if (d.next_challenge) {
            setTimeout(function () {
              applyChallenge(d.next_challenge);
              setStatus("");
              showNextButton(false);
            }, 1400);
          } else {
            setTimeout(function () {
              loadChallenge();
              showNextButton(false);
            }, 1400);
          }
        } else {
          setStatus(
            "Pyx saw " +
              (d.pyx_color || "?") +
              " — training with your " +
              color +
              ". One more…",
            "warn"
          );
          notifyParent({ agreed: false, color: color });
          if (d.next_challenge) {
            setTimeout(function () {
              applyChallenge(d.next_challenge);
              setStatus("Pick the lit color on this signal.");
              showNextButton(false);
            }, 900);
          } else {
            showNextButton(true);
          }
        }
        setBusy(false);
      })
      .catch(function (e) {
        setStatus(e.message || String(e), "err");
        setBusy(false);
      });
  }

  function init() {
    var root = document.getElementById("capRoot");
    if (!root) return;
    root.id = "capRoot";
    loadChallenge();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})(typeof window !== "undefined" ? window : globalThis);
