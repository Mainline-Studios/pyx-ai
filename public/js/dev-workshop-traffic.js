/**
 * Dev Workshop — traffic light analyzer (web images + live video preview).
 * Live path: canvas frame → features → POST /traffic/frame → emit (same as still images).
 */
(function (global) {
  "use strict";

  var API = "/api/dev-workshop/traffic";
  var COLORS = ["red", "yellow", "green", "off", "unknown"];
  var HEX = {
    red: "#ef4444",
    yellow: "#eab308",
    green: "#22c55e",
    off: "#64748b",
    unknown: "#94a3b8",
  };
  var FEATURE_W = 120;
  var FEATURE_H = 90;

  var WEB_PRESETS = [
    "green traffic light",
    "red traffic light",
    "yellow traffic light",
    "traffic light at night",
    "horizontal traffic signal green",
  ];

  var STARTER_IMAGES = [
    {
      url: "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Red_traffic_light.jpg/320px-Red_traffic_light.jpg",
      hint: "Red (Wikimedia)",
    },
    {
      url: "https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Yellow_traffic_light.jpg/320px-Yellow_traffic_light.jpg",
      hint: "Yellow (Wikimedia)",
    },
    {
      url: "https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Green_traffic_light.jpg/320px-Green_traffic_light.jpg",
      hint: "Green (Wikimedia)",
    },
    {
      url: "https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/Traffic_lights_at_Night.jpg/320px-Traffic_lights_at_Night.jpg",
      hint: "Night scene",
    },
  ];

  var liveState = {
    stream: null,
    timer: null,
    running: false,
    inFlight: false,
    frameSeq: 0,
    lastEmitColor: null,
    lastEmitAt: 0,
    holdMs: 400,
    intervalMs: 200,
    autoEmit: true,
    source: "camera",
  };

  function $(id) {
    return document.getElementById(id);
  }

  function log(msg, isErr) {
    var el = $("trafficLog");
    if (!el) return;
    var line = document.createElement("div");
    line.textContent = new Date().toLocaleTimeString() + " — " + msg;
    if (isErr) line.style.color = "#f87171";
    el.prepend(line);
  }

  function api(path, body) {
    return fetch(API + path, {
      method: body ? "POST" : "GET",
      headers: body ? { "Content-Type": "application/json" } : {},
      body: body ? JSON.stringify(body) : undefined,
    }).then(function (r) {
      return r.json().then(function (j) {
        return { ok: r.ok, status: r.status, data: j };
      });
    });
  }

  /** Shared feature extraction (still image + live video frames). */
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

  function extractFeaturesFromCanvasSource(drawable) {
    var c = document.createElement("canvas");
    c.width = FEATURE_W;
    c.height = FEATURE_H;
    var ctx = c.getContext("2d");
    ctx.drawImage(drawable, 0, 0, FEATURE_W, FEATURE_H);
    var img = ctx.getImageData(0, 0, FEATURE_W, FEATURE_H);
    return extractFeaturesFromImageData(img.data, FEATURE_W, FEATURE_H);
  }

  function extractFeatures(img) {
    return extractFeaturesFromCanvasSource(img);
  }

  function broadcastColor(detail, extra) {
    detail = detail || {};
    extra = extra || {};
    global.__pyxLastTrafficColor = {
      color: detail.color,
      hex: detail.hex,
      confidence: detail.confidence,
      mode: detail.mode || extra.mode,
      frame_id: detail.frame_id || extra.frame_id,
      source: detail.source || extra.source,
    };
    try {
      global.dispatchEvent(
        new CustomEvent("pyx-traffic-color", { detail: global.__pyxLastTrafficColor })
      );
    } catch (e) {
      /* ignore */
    }
    try {
      if (global.parent && global.parent !== global) {
        global.parent.postMessage(
          {
            type: "pyx-traffic-color",
            color: detail.color,
            hex: detail.hex,
            confidence: detail.confidence,
            mode: detail.mode,
            frame_id: detail.frame_id,
            source: detail.source,
          },
          "*"
        );
      }
    } catch (e2) {
      /* ignore */
    }
  }

  function setResult(data, opts) {
    opts = opts || {};
    var swatch = $("trafficSwatch");
    var label = $("trafficResultLabel");
    var meta = $("trafficResultMeta");
    var liveSwatch = $("trafficLiveSwatch");
    var liveLabel = $("trafficLiveLabel");
    if (!data || !data.ok) {
      if (swatch) swatch.style.background = "#334155";
      if (label) label.textContent = data && data.error ? data.error : "No result";
      if (meta) meta.textContent = "";
      if (liveSwatch && opts.live) liveSwatch.style.background = "#334155";
      if (liveLabel && opts.live) liveLabel.textContent = data && data.error ? data.error : "—";
      return;
    }
    var text =
      (data.traffic_light_detected ? "Signal: " : "Guess: ") +
      String(data.color || "unknown").toUpperCase() +
      " · " +
      (data.hex || "");
    var metaText =
      "Confidence " +
      Math.round((data.confidence || 0) * 100) +
      "% · " +
      (data.method || "?") +
      " · " +
      (data.training_samples || 0) +
      " samples" +
      (data.mode === "frame" || data.mode === "live" ? " · live frame" : "");
    if (swatch && !opts.liveOnly) {
      swatch.style.background = data.hex || HEX.unknown;
      if (label) label.textContent = text;
      if (meta) meta.textContent = metaText;
    }
    if (liveSwatch) liveSwatch.style.background = data.hex || HEX.unknown;
    if (liveLabel) liveLabel.textContent = text;
    if (!opts.skipBroadcast) {
      broadcastColor(data, { mode: data.mode, frame_id: data.frame_id, source: data.source });
    }
  }

  function analyzeFeaturesOnServer(features, opts) {
    opts = opts || {};
    var body = {
      features: features,
      mode: opts.mode || "image",
      source: opts.source || null,
      frame_id: opts.frame_id || null,
      image_url: opts.image_url || null,
    };
    var path = opts.mode === "frame" || opts.mode === "live" ? "/frame" : "/analyze";
    return api(path, body).then(function (x) {
      return x.data;
    });
  }

  function maybeEmitLive(data) {
    if (!liveState.autoEmit || !data || !data.ok) return;
    var now = Date.now();
    if (
      data.color === liveState.lastEmitColor &&
      now - liveState.lastEmitAt < liveState.holdMs
    ) {
      return;
    }
    liveState.lastEmitColor = data.color;
    liveState.lastEmitAt = now;
    api("/emit", {
      color: data.color,
      hex: data.hex,
      source: liveState.source,
      mode: "live",
      frame_id: data.frame_id,
    }).catch(function () {
      /* emit is best-effort during live preview */
    });
  }

  function loadImage(url) {
    return new Promise(function (resolve, reject) {
      var img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = function () {
        resolve(img);
      };
      img.onerror = function () {
        reject(
          new Error(
            "Image blocked or failed to load (CORS?). Try another URL or server-side analyze."
          )
        );
      };
      img.src = url;
    });
  }

  function setPreview(url) {
    var el = $("trafficPreview");
    if (!el) return;
    el.src = url || "";
    el.hidden = !url;
  }

  function currentUrl() {
    return (($("trafficImageUrl") && $("trafficImageUrl").value) || "").trim();
  }

  function runAnalyze() {
    var url = currentUrl();
    if (!url) {
      log("Paste an image URL first.", true);
      return;
    }
    setPreview(url);
    log("Loading image…");
    loadImage(url)
      .then(function (img) {
        var features = extractFeatures(img);
        return analyzeFeaturesOnServer(features, {
          mode: "image",
          image_url: url,
        });
      })
      .then(function (data) {
        if (!data.ok) {
          setResult(data);
          log(data.error || "Analyze failed", true);
          return;
        }
        setResult(data);
        log("Analyzed: " + data.color + " " + data.hex);
      })
      .catch(function () {
        log("Client load failed — trying server fetch…", true);
        api("/analyze", { image_url: url, mode: "image" })
          .then(function (x) {
            setResult(x.data);
            if (x.data.ok) log("Server analyzed: " + x.data.color);
            else log(x.data.error || "Server analyze failed", true);
          })
          .catch(function (err) {
            log(err.message || String(err), true);
          });
      });
  }

  function runTrain(color) {
    var url = currentUrl();
    if (!url) {
      log("Paste an image URL to train.", true);
      return;
    }
    log("Training as " + color + "…");
    loadImage(url)
      .then(function (img) {
        var features = extractFeatures(img);
        return api("/train", {
          image_url: url,
          color: color,
          features: features,
        });
      })
      .then(function (x) {
        if (!x.ok || !x.data.ok) {
          log(x.data.error || "Train failed", true);
          return;
        }
        log("Saved training sample " + x.data.sample.id);
        refreshSamples();
      })
      .catch(function (e) {
        log(e.message || String(e), true);
      });
  }

  function trainFromLiveFrame(color) {
    var video = $("trafficLiveVideo");
    if (!video || video.readyState < 2) {
      log("Start live preview first.", true);
      return;
    }
    var features = extractFeaturesFromCanvasSource(video);
    var ref = "live:" + liveState.source + "#" + Date.now();
    api("/train", { image_url: ref, color: color, features: features }).then(function (x) {
      if (!x.data.ok) {
        log(x.data.error || "Train failed", true);
        return;
      }
      log("Live frame saved as " + color);
      refreshSamples();
    });
  }

  function sendColor() {
    var last = global.__pyxLastTrafficColor;
    if (!last || !last.hex) {
      log("Analyze an image or run live preview first.", true);
      return;
    }
    api("/emit", {
      color: last.color,
      hex: last.hex,
      source: last.source || "workshop-ui",
      mode: last.mode || "image",
      frame_id: last.frame_id,
    }).then(function (x) {
      if (x.data && x.data.ok) log("Sent color " + last.hex);
      else log((x.data && x.data.error) || "Emit failed", true);
    });
  }

  function refreshStats() {
    return api("/samples", null).then(function (x) {
      var el = $("trafficStats");
      if (!el || !x.data || !x.data.stats) return;
      var st = x.data.stats;
      el.textContent =
        "Training set — red: " +
        (st.red || 0) +
        ", yellow: " +
        (st.yellow || 0) +
        ", green: " +
        (st.green || 0) +
        ", off: " +
        (st.off || 0) +
        ", unknown: " +
        (st.unknown || 0) +
        " (label more greens/reds for best results)";
    });
  }

  function runWebSearch() {
    var q = ($("trafficWebQuery") && $("trafficWebQuery").value) || "";
    q = q.trim();
    if (!q) {
      log("Enter a search query.", true);
      return;
    }
    var grid = $("trafficWebGrid");
    if (grid) grid.innerHTML = '<p class="traffic-muted">Searching and caching images…</p>';
    log("Searching web for: " + q);
    api("/search-images", { query: q, max: 14 })
      .then(function (x) {
        if (!x.ok || !x.data.ok) {
          if (grid) {
            grid.innerHTML =
              '<p class="traffic-muted">' +
              escapeHtml((x.data && x.data.error) || "Search failed") +
              "</p>";
          }
          log((x.data && x.data.error) || "Search failed", true);
          return;
        }
        renderWebGrid(x.data.images || []);
        log("Loaded " + (x.data.count || 0) + " images — label each card.");
      })
      .catch(function (e) {
        log(e.message || String(e), true);
      });
  }

  function renderWebGrid(images) {
    var grid = $("trafficWebGrid");
    if (!grid) return;
    if (!images.length) {
      grid.innerHTML = '<p class="traffic-muted">No images returned.</p>';
      return;
    }
    grid.innerHTML = images
      .map(function (img) {
        var url = img.public_url || img.thumbnail_url || "";
        return (
          '<article class="traffic-web-card" data-public-url="' +
          escapeAttr(url) +
          '">' +
          '<img src="' +
          escapeAttr(url) +
          '" alt="" loading="lazy" crossorigin="anonymous" />' +
          '<div class="traffic-web-card__body">' +
          '<p class="traffic-web-card__title">' +
          escapeHtml(img.title || img.query || "Image") +
          "</p>" +
          '<div class="traffic-web-card__btns">' +
          '<button type="button" class="btn btn-xs" data-label="red" style="border-color:#ef4444">Red</button>' +
          '<button type="button" class="btn btn-xs" data-label="yellow" style="border-color:#eab308">Yellow</button>' +
          '<button type="button" class="btn btn-xs" data-label="green" style="border-color:#22c55e">Green</button>' +
          '<button type="button" class="btn btn-xs" data-label="off">Off</button>' +
          '<button type="button" class="btn btn-xs" data-label="unknown">Skip</button>' +
          "</div></div></article>"
        );
      })
      .join("");
    grid.querySelectorAll("[data-label]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var card = btn.closest(".traffic-web-card");
        var publicUrl = card && card.getAttribute("data-public-url");
        var color = btn.getAttribute("data-label");
        if (!publicUrl || !color) return;
        labelWebImage(publicUrl, color, card, btn);
      });
    });
  }

  function labelWebImage(publicUrl, color, card, btn) {
    if (color === "unknown") {
      if (card) card.style.opacity = "0.45";
      log("Skipped image.");
      return;
    }
    btn.disabled = true;
    log("Labeling as " + color + "…");
    loadImage(publicUrl)
      .then(function (img) {
        var features = extractFeatures(img);
        return api("/train", {
          image_url: publicUrl,
          color: color,
          features: features,
        });
      })
      .then(function (x) {
        if (!x.data.ok) {
          log(x.data.error || "Train failed", true);
          btn.disabled = false;
          return;
        }
        if (card) {
          card.style.boxShadow = "0 0 0 2px " + (HEX[color] || "#fff");
          card.setAttribute("data-labeled", color);
        }
        log("Saved as " + color + " — re-test on Test image tab.");
        refreshSamples();
        refreshStats();
      })
      .catch(function (e) {
        log(e.message || String(e), true);
        btn.disabled = false;
      });
  }

  function renderWebPresets() {
    var el = $("trafficWebPresets");
    if (!el) return;
    el.innerHTML = WEB_PRESETS.map(function (q) {
      return (
        '<button type="button" data-q="' +
        escapeAttr(q) +
        '">' +
        escapeHtml(q) +
        "</button>"
      );
    }).join("");
    el.querySelectorAll("button").forEach(function (b) {
      b.addEventListener("click", function () {
        if ($("trafficWebQuery")) $("trafficWebQuery").value = b.getAttribute("data-q") || "";
        runWebSearch();
      });
    });
  }

  function refreshSamples() {
    return api("/samples", null).then(function (x) {
      var grid = $("trafficTrainGrid");
      if (!grid) return;
      var samples = (x.data && x.data.samples) || [];
      refreshStats();
      if (!samples.length) {
        grid.innerHTML =
          '<p class="traffic-muted">No labels yet — use <strong>Train from web</strong> above.</p>';
        return;
      }
      grid.innerHTML = samples
        .slice()
        .reverse()
        .map(function (s) {
          var hex = HEX[s.color] || HEX.unknown;
          var url = s.image_url || "";
          var thumb =
            String(url).indexOf("live:") === 0
              ? '<div class="traffic-train-card__live">LIVE</div>'
              : '<img src="' + escapeAttr(url) + '" alt="" loading="lazy" crossorigin="anonymous" />';
          return (
            '<div class="traffic-train-card">' +
            thumb +
            '<div class="traffic-train-card__body">' +
            '<span class="traffic-signal-dot" style="background:' +
            hex +
            '"></span> ' +
            escapeHtml(s.color) +
            '<button type="button" class="btn btn-ghost btn-xs" data-del="' +
            escapeAttr(s.id) +
            '">Delete</button>' +
            "</div></div>"
          );
        })
        .join("");
    });
  }

  function escapeHtml(s) {
    var d = document.createElement("div");
    d.textContent = s == null ? "" : String(s);
    return d.innerHTML;
  }

  function escapeAttr(s) {
    return escapeHtml(s).replace(/"/g, "&quot;");
  }

  function renderStarters() {
    var el = $("trafficStarters");
    if (!el) return;
    el.innerHTML = STARTER_IMAGES.map(function (item) {
      return (
        '<button type="button" class="traffic-starter" data-url="' +
        escapeAttr(item.url) +
        '"><img src="' +
        escapeAttr(item.url) +
        '" alt="" /><span>' +
        escapeHtml(item.hint) +
        "</span></button>"
      );
    }).join("");
    el.querySelectorAll(".traffic-starter").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var u = btn.getAttribute("data-url");
        if ($("trafficImageUrl")) $("trafficImageUrl").value = u;
        setPreview(u);
      });
    });
  }

  function readLiveSettings() {
    var fps = $("trafficLiveFps");
    var hold = $("trafficLiveHold");
    if (fps) liveState.intervalMs = Math.max(100, Math.round(1000 / Math.max(1, Number(fps.value) || 5)));
    if (hold) liveState.holdMs = Math.max(0, Number(hold.value) || 400);
    liveState.autoEmit = !($("trafficLiveAutoEmit") && !$("trafficLiveAutoEmit").checked);
  }

  function stopLive() {
    liveState.running = false;
    if (liveState.timer) {
      clearInterval(liveState.timer);
      liveState.timer = null;
    }
    if (liveState.stream) {
      liveState.stream.getTracks().forEach(function (t) {
        t.stop();
      });
      liveState.stream = null;
    }
    var video = $("trafficLiveVideo");
    if (video) video.srcObject = null;
    var wrap = $("trafficLiveWrap");
    if (wrap) wrap.hidden = true;
    var cam = $("trafficLiveCamera");
    var stop = $("trafficLiveStop");
    if (cam) cam.disabled = false;
    var fileBtn = $("trafficLiveFileBtn");
    if (fileBtn) fileBtn.disabled = false;
    if (stop) stop.disabled = true;
    log("Live preview stopped.");
  }

  function processLiveTick() {
    if (!liveState.running || liveState.inFlight) return;
    var video = $("trafficLiveVideo");
    if (!video || video.readyState < 2) return;
    liveState.inFlight = true;
    liveState.frameSeq += 1;
    var frameId = "f" + liveState.frameSeq;
    var features = extractFeaturesFromCanvasSource(video);
    analyzeFeaturesOnServer(features, {
      mode: "frame",
      source: liveState.source,
      frame_id: frameId,
    })
      .then(function (data) {
        setResult(data, { live: true, liveOnly: false });
        if (data.ok) {
          maybeEmitLive(
            Object.assign({}, data, { frame_id: frameId, source: liveState.source })
          );
          var stat = $("trafficLiveStat");
          if (stat) {
            stat.textContent =
              "Frame " +
              frameId +
              " · ~" +
              Math.round(1000 / liveState.intervalMs) +
              " fps cap";
          }
        }
      })
      .catch(function (e) {
        log(e.message || "Live frame failed", true);
      })
      .finally(function () {
        liveState.inFlight = false;
      });
  }

  function startLiveCamera() {
    readLiveSettings();
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      log("Camera API not available in this browser.", true);
      return;
    }
    stopLive();
    navigator.mediaDevices
      .getUserMedia({ video: { facingMode: "environment" }, audio: false })
      .then(function (stream) {
        liveState.stream = stream;
        liveState.source = "camera";
        var video = $("trafficLiveVideo");
        var wrap = $("trafficLiveWrap");
        if (video) {
          video.srcObject = stream;
          return video.play();
        }
      })
      .then(function () {
        var wrap = $("trafficLiveWrap");
        if (wrap) wrap.hidden = false;
        liveState.running = true;
        liveState.frameSeq = 0;
        liveState.lastEmitColor = null;
        $("trafficLiveCamera") && ($("trafficLiveCamera").disabled = true);
        $("trafficLiveFileBtn") && ($("trafficLiveFileBtn").disabled = true);
        $("trafficLiveStop") && ($("trafficLiveStop").disabled = false);
        liveState.timer = setInterval(processLiveTick, liveState.intervalMs);
        log("Live camera on — analyzing ~" + Math.round(1000 / liveState.intervalMs) + " fps.");
        processLiveTick();
      })
      .catch(function (e) {
        log(e.message || "Camera permission denied", true);
      });
  }

  function startLiveFile() {
    var inp = $("trafficLiveFile");
    if (!inp || !inp.files || !inp.files[0]) {
      log("Choose a video file first.", true);
      return;
    }
    readLiveSettings();
    stopLive();
    var url = URL.createObjectURL(inp.files[0]);
    var video = $("trafficLiveVideo");
    liveState.source = "file:" + inp.files[0].name.slice(0, 40);
    if (!video) return;
    video.srcObject = null;
    video.src = url;
    video.loop = true;
    video.muted = true;
    video.onloadeddata = function () {
      var wrap = $("trafficLiveWrap");
      if (wrap) wrap.hidden = false;
      video.play();
      liveState.running = true;
      liveState.frameSeq = 0;
      $("trafficLiveCamera") && ($("trafficLiveCamera").disabled = true);
      $("trafficLiveFileBtn") && ($("trafficLiveFileBtn").disabled = true);
      $("trafficLiveStop") && ($("trafficLiveStop").disabled = false);
      liveState.timer = setInterval(processLiveTick, liveState.intervalMs);
      log("Live file playback — same frame pipeline as camera.");
      processLiveTick();
    };
  }

  function setInputMode(mode) {
    var trainPanel = $("trafficPanelTrainWeb");
    var imgPanel = $("trafficPanelImage");
    var livePanel = $("trafficPanelLive");
    var tabTrain = $("trafficTabTrainWeb");
    var tabImg = $("trafficTabImage");
    var tabLive = $("trafficTabLive");
    function offTabs() {
      [tabTrain, tabImg, tabLive].forEach(function (t) {
        if (t) t.classList.remove("is-active");
      });
    }
    if (trainPanel) trainPanel.hidden = mode !== "trainWeb";
    if (imgPanel) imgPanel.hidden = mode !== "image";
    if (livePanel) livePanel.hidden = mode !== "live";
    offTabs();
    if (mode === "live") {
      if (tabLive) tabLive.classList.add("is-active");
    } else if (mode === "image") {
      if (tabImg) tabImg.classList.add("is-active");
      stopLive();
    } else {
      if (tabTrain) tabTrain.classList.add("is-active");
      stopLive();
    }
  }

  function init() {
    renderStarters();
    renderWebPresets();
    refreshSamples();

    $("trafficWebSearchBtn") &&
      $("trafficWebSearchBtn").addEventListener("click", runWebSearch);
    $("trafficWebQuery") &&
      $("trafficWebQuery").addEventListener("keydown", function (e) {
        if (e.key === "Enter") runWebSearch();
      });

    $("trafficAnalyzeBtn") &&
      $("trafficAnalyzeBtn").addEventListener("click", runAnalyze);
    $("trafficSendBtn") && $("trafficSendBtn").addEventListener("click", sendColor);
    $("trafficTabTrainWeb") &&
      $("trafficTabTrainWeb").addEventListener("click", function () {
        setInputMode("trainWeb");
      });
    $("trafficTabImage") &&
      $("trafficTabImage").addEventListener("click", function () {
        setInputMode("image");
      });
    $("trafficTabLive") &&
      $("trafficTabLive").addEventListener("click", function () {
        setInputMode("live");
      });
    $("trafficLiveCamera") &&
      $("trafficLiveCamera").addEventListener("click", startLiveCamera);
    $("trafficLiveFileBtn") &&
      $("trafficLiveFileBtn").addEventListener("click", startLiveFile);
    $("trafficLiveStop") && $("trafficLiveStop").addEventListener("click", stopLive);
    $("trafficLiveFps") &&
      $("trafficLiveFps").addEventListener("change", function () {
        readLiveSettings();
        if (liveState.running) {
          clearInterval(liveState.timer);
          liveState.timer = setInterval(processLiveTick, liveState.intervalMs);
        }
      });

    COLORS.forEach(function (c) {
      var btn = $("trafficTrain_" + c);
      if (btn)
        btn.addEventListener("click", function () {
          runTrain(c);
        });
      var lbtn = $("trafficLiveTrain_" + c);
      if (lbtn)
        lbtn.addEventListener("click", function () {
          trainFromLiveFrame(c);
        });
    });

    var grid = $("trafficTrainGrid");
    if (grid) {
      grid.addEventListener("click", function (e) {
        var del = e.target.closest("[data-del]");
        if (!del) return;
        var id = del.getAttribute("data-del");
        fetch(API + "/samples/" + encodeURIComponent(id), { method: "DELETE" })
          .then(function () {
            return refreshSamples();
          })
          .then(function () {
            log("Deleted sample " + id);
          });
      });
    }

    if (/[?&]input=live/.test(location.search)) setInputMode("live");
    else if (/[?&]input=test/.test(location.search)) setInputMode("image");
    else setInputMode("trainWeb");
  }

  global.PyxDevWorkshopTraffic = {
    init: init,
    extractFeatures: extractFeatures,
    extractFeaturesFromCanvasSource: extractFeaturesFromCanvasSource,
    analyzeFeaturesOnServer: analyzeFeaturesOnServer,
    HEX: HEX,
    stopLive: stopLive,
  };
})(typeof window !== "undefined" ? window : globalThis);
