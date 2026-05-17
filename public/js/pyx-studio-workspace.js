/**
 * Pyx Studio Workspace — productivity (essay helper, research browser, fill blanks).
 */
(function (global) {
  "use strict";

  var STORAGE_KEY = "pyx.studio.workspace.v1";
  var PINNED_KEY = "pyx.studio.essay.pinned";
  var TASKS_KEY = "pyx.studio.tasks.v1";

  var currentEssay = null;
  var currentGuide = null;
  var coachStepDone = {};

  function setStatus(msg, kind) {
    var statusEl = document.getElementById("wsStatus");
    if (!statusEl) return;
    statusEl.textContent = msg || "";
    statusEl.className = "ws-status" + (kind ? " ws-status--" + kind : "");
  }

  function api(path, body) {
    return fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(body || {}),
    }).then(function (r) {
      return r.text().then(function (text) {
        var trimmed = (text || "").trim();
        if (trimmed.charAt(0) === "<") {
          throw new Error(
            "Pyx API returned a web page instead of JSON. Studio routes may need a hosting deploy — " +
              "if you just updated, wait a minute and refresh. (Path: " + path + ")"
          );
        }
        var j;
        try {
          j = trimmed ? JSON.parse(trimmed) : {};
        } catch (parseErr) {
          throw new Error("Bad API response: " + trimmed.slice(0, 120));
        }
        if (!r.ok) throw new Error((j && j.error) || r.statusText || "Request failed");
        return j;
      });
    });
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function formatPlanForKids(essay) {
    if (!essay || typeof essay !== "object") return "";
    var lines = [];
    lines.push("TOPIC");
    lines.push(essay.topic || "(your topic)");
    lines.push("");
    lines.push("MAIN IDEA (thesis)");
    lines.push(essay.thesis || "(fill in your main idea)");
    lines.push("");
    if (essay.outline && essay.outline.length) {
      lines.push("OUTLINE — parts of your essay");
      essay.outline.forEach(function (sec, i) {
        var draft = sec.writer_draft || "";
        var goal = sec.goal || "";
        lines.push("");
        lines.push(i + 1 + ". " + (sec.section || "Section"));
        if (draft) lines.push("   Your words: " + draft);
        else if (goal) lines.push("   What to cover: " + goal);
      });
    }
    if (essay.fill_blanks && essay.fill_blanks.length) {
      lines.push("");
      lines.push("YOUR ANSWERS");
      essay.fill_blanks.forEach(function (b) {
        if ((b.user_fill || "").trim()) {
          lines.push("");
          lines.push("• " + (b.label || b.id));
          lines.push("  " + b.user_fill.trim());
        }
      });
    }
    if (essay.citations && essay.citations.length) {
      lines.push("");
      lines.push("SOURCES YOU USED");
      essay.citations.forEach(function (c, i) {
        lines.push(i + 1 + ". " + (c.title || c.url || "Source"));
        if (c.url) lines.push("   " + c.url);
      });
    }
    return lines.join("\n");
  }

  function updatePlanViews(essay, pyText) {
    var planOut = document.getElementById("wsPlanOut");
    var jsonOut = document.getElementById("wsJsonOut");
    var pyOut = document.getElementById("wsPyOut");
    if (planOut) planOut.value = formatPlanForKids(essay);
    if (jsonOut && essay) jsonOut.value = JSON.stringify(essay, null, 2);
    if (pyOut && pyText) pyOut.value = pyText;
  }

  function loadState() {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    } catch (e) {
      return {};
    }
  }

  function saveState(patch) {
    var s = loadState();
    Object.keys(patch).forEach(function (k) {
      s[k] = patch[k];
    });
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(s));
    } catch (e) {}
    return s;
  }

  function getPinned() {
    try {
      var arr = JSON.parse(localStorage.getItem(PINNED_KEY) || "[]");
      return Array.isArray(arr) ? arr : [];
    } catch (e) {
      return [];
    }
  }

  function setPinned(arr) {
    try {
      localStorage.setItem(PINNED_KEY, JSON.stringify(arr.slice(0, 24)));
    } catch (e) {}
  }

  function updatePinCount() {
    var el = document.getElementById("wsPinCount");
    if (el) el.textContent = String(getPinned().length);
  }

  function markStep(step, done) {
    var li = document.querySelector('#wsSteps li[data-step="' + step + '"]');
    if (li) li.classList.toggle("is-done", !!done);
  }

  function collectFillsFromDom() {
    var blanks = [];
    document.querySelectorAll(".ws-blank[data-blank-id]").forEach(function (el) {
      var ta = el.querySelector("textarea");
      blanks.push({
        id: el.getAttribute("data-blank-id"),
        user_fill: ta ? ta.value : "",
      });
    });
    return blanks;
  }

  function syncEssayFromDom() {
    if (!currentEssay || !currentEssay.fill_blanks) return currentEssay;
    var byId = {};
    collectFillsFromDom().forEach(function (f) {
      byId[f.id] = f.user_fill;
    });
    currentEssay.fill_blanks.forEach(function (b) {
      if (byId[b.id] !== undefined) b.user_fill = byId[b.id];
    });
    return currentEssay;
  }

  function updateBlankProgress() {
    var bar = document.getElementById("wsBlankProgress");
    if (!bar || !currentEssay || !currentEssay.fill_blanks) return;
    var blanks = currentEssay.fill_blanks;
    var filled = blanks.filter(function (b) {
      return (b.user_fill || "").trim().length > 0;
    }).length;
    var pct = blanks.length ? Math.round((filled / blanks.length) * 100) : 0;
    bar.style.width = pct + "%";
  }

  function renderPyxMade(essay) {
    var el = document.getElementById("wsPyxMade");
    if (!el) return;
    if (!essay) {
      el.innerHTML =
        '<p class="ws-muted">Build your essay to see Pyx\u2019s outline here.</p>';
      return;
    }
    var parts = [];
    parts.push(
      "<p><strong>Topic:</strong> " + escapeHtml(essay.topic || "") + "</p>"
    );
    if (essay.thesis) {
      parts.push(
        "<p><strong>Main idea Pyx suggests you argue:</strong> " +
          escapeHtml(essay.thesis) +
          "</p>"
      );
    }
    if (essay.outline && essay.outline.length) {
      parts.push("<p><strong>Essay structure Pyx planned:</strong></p><ul>");
      essay.outline.forEach(function (sec) {
        parts.push("<li><strong>" + escapeHtml(sec.section || "Section") + "</strong>");
        if (sec.goal) parts.push(" — " + escapeHtml(sec.goal));
        if (sec.writer_draft) {
          parts.push(
            '<br><span class="ws-muted">Starter note: ' +
              escapeHtml(sec.writer_draft) +
              "</span>"
          );
        }
        parts.push("</li>");
      });
      parts.push("</ul>");
    }
    if (essay.key_points && essay.key_points.length) {
      parts.push("<p><strong>Notes from your research:</strong></p><ul>");
      essay.key_points.forEach(function (kp) {
        parts.push("<li>" + escapeHtml(String(kp)) + "</li>");
      });
      parts.push("</ul>");
    }
    if (essay.citations && essay.citations.length) {
      parts.push("<p><strong>Sources in your plan:</strong></p><ul>");
      essay.citations.forEach(function (c) {
        parts.push("<li>" + escapeHtml(c.title || c.url || "Source"));
        if (c.url) {
          parts.push(' <span class="ws-muted">' + escapeHtml(c.url) + "</span>");
        }
        parts.push("</li>");
      });
      parts.push("</ul>");
    }
    if (essay.disclaimer) {
      parts.push(
        '<p class="ws-muted" style="margin-top:10px;">' +
          escapeHtml(essay.disclaimer) +
          "</p>"
      );
    }
    el.innerHTML = parts.join("");
    renderRiverside();
  }

  function buildDraftStreamText(essay) {
    if (!essay) return "";
    var lines = [];
    lines.push(essay.topic || "Your topic");
    lines.push("");
    if (essay.thesis) {
      lines.push("Main idea: " + essay.thesis);
      lines.push("");
    }
    var fillsBySection = {};
    (essay.fill_blanks || []).forEach(function (b) {
      var sec = b.section || "Essay";
      if (!fillsBySection[sec]) fillsBySection[sec] = [];
      fillsBySection[sec].push(b);
    });
    (essay.outline || []).forEach(function (sec, i) {
      var title = sec.section || "Section " + (i + 1);
      lines.push("—— " + title + " ——");
      if (sec.goal) lines.push("(Goal: " + sec.goal + ")");
      var draft = (sec.writer_draft || "").trim();
      if (draft) lines.push(draft);
      var gaps = fillsBySection[title] || [];
      gaps.forEach(function (b) {
        var fill = (b.user_fill || "").trim();
        if (fill) {
          lines.push("");
          lines.push(fill);
        } else if (b.label) {
          lines.push("");
          lines.push("[" + b.label + " — you write here]");
        }
      });
      lines.push("");
    });
    return lines.join("\n").trim();
  }

  function renderResearchRiver() {
    var el = document.getElementById("wsResearchRiver");
    if (!el) return;
    var pinned = getPinned();
    var essay = currentEssay;
    var outline = (essay && essay.outline) || [];
    if (!pinned.length && !outline.length) {
      el.innerHTML =
        '<p class="ws-muted" style="min-width:200px;">Save links and build an essay plan — your river appears here.</p>';
      return;
    }
    var html = [];
    pinned.forEach(function (s, i) {
      if (i) html.push('<span class="ws-river__arrow" aria-hidden="true">→</span>');
      html.push(
        '<div class="ws-river__node ws-river__node--source">' +
          "<strong>" +
          escapeHtml((s.title || "Source").slice(0, 48)) +
          "</strong>" +
          "<span>" +
          (s.read_ok ? "Pyx read ✓" : "not read yet") +
          "</span></div>"
      );
    });
    if (pinned.length && outline.length) {
      html.push('<span class="ws-river__arrow" aria-hidden="true">⇢</span>');
    }
    outline.forEach(function (sec, i) {
      if (i) html.push('<span class="ws-river__arrow" aria-hidden="true">→</span>');
      html.push(
        '<div class="ws-river__node ws-river__node--section">' +
          "<strong>" +
          escapeHtml(sec.section || "Section") +
          "</strong>" +
          "<span>" +
          escapeHtml((sec.goal || "").slice(0, 80) || "section") +
          "</span></div>"
      );
    });
    el.innerHTML = html.join("");
  }

  function renderDraftStream() {
    var el = document.getElementById("wsDraftStream");
    if (!el) return;
    syncEssayFromDom();
    if (!currentEssay) {
      el.textContent = "Build your essay plan first — then your draft stream flows here.";
      return;
    }
    var raw = buildDraftStreamText(currentEssay);
    var parts = raw.split(/\n—— (.+?) ——\n/);
    if (parts.length < 2) {
      el.textContent = raw;
      return;
    }
    var html = "<h4>" + escapeHtml(parts[0].trim()) + "</h4>";
    for (var i = 1; i < parts.length; i += 2) {
      html += "<h4>" + escapeHtml(parts[i]) + "</h4>";
      html += "<p>" + escapeHtml(parts[i + 1] || "").replace(/\n/g, "<br>") + "</p>";
    }
    el.innerHTML = html;
  }

  function renderRiverside() {
    renderResearchRiver();
    renderDraftStream();
  }

  function runFlowCheck() {
    var topicEl = document.getElementById("wsTopic");
    var topic = (topicEl && topicEl.value) || (currentEssay && currentEssay.topic) || "";
    var flowOut = document.getElementById("wsFlowOut");
    if (!currentEssay) {
      setStatus("Build your essay plan first.", "err");
      return Promise.resolve();
    }
    syncEssayFromDom();
    setStatus("Pyx is checking how your essay flows…", "info");
    return api("/api/studio/flow", {
      topic: topic,
      sources: getPinned(),
      essay: currentEssay,
    })
      .then(function (j) {
        if (flowOut) {
          flowOut.hidden = false;
          flowOut.textContent = j.flow_notes || "";
        }
        setStatus("Flow check ready — smooth the bends in your own words.", "ok");
        markStep("riverside", true);
        switchTab("riverside");
      })
      .catch(function (e) {
        setStatus(e.message, "err");
        throw e;
      });
  }

  function renderBlanks(essay) {
    var list = document.getElementById("wsBlanksList");
    if (!list) return;
    var blanks = (essay && essay.fill_blanks) || [];
    if (!blanks.length) {
      list.innerHTML =
        '<p class="ws-muted">Make an essay plan first, or tap <strong>Read my links &amp; build my essay</strong> on the left.</p>';
      return;
    }
    list.innerHTML = blanks
      .map(function (b) {
        var filled = (b.user_fill || "").trim().length > 0;
        return (
          '<article class="ws-blank' +
          (filled ? " is-filled" : "") +
          '" data-blank-id="' +
          escapeHtml(b.id) +
          '">' +
          '<span class="ws-blank__tag">' +
          escapeHtml(b.section || "section") +
          "</span>" +
          "<strong>" +
          escapeHtml(b.label || b.id) +
          "</strong>" +
          (b.hint ? '<p class="ws-muted">' + escapeHtml(b.hint) + "</p>" : "") +
          '<textarea placeholder="' +
          escapeHtml(b.placeholder || "Your answer…") +
          '">' +
          escapeHtml(b.user_fill || "") +
          "</textarea>" +
          '<div class="ws-blank-hint-box" hidden></div>' +
          '<div class="ws-blank-actions">' +
          '<button type="button" class="btn secondary ws-blank-hint">Hint from Pyx</button>' +
          "</div>" +
          "</article>"
        );
      })
      .join("");


    list.querySelectorAll(".ws-blank textarea").forEach(function (ta) {
      ta.addEventListener("input", function () {
        var card = ta.closest(".ws-blank");
        if (card) card.classList.toggle("is-filled", ta.value.trim().length > 0);
        syncEssayFromDom();
        updateBlankProgress();
        renderDraftStream();
        markStep("essay", true);
      });
    });

    list.querySelectorAll(".ws-blank-hint").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var card = btn.closest(".ws-blank");
        var id = card && card.getAttribute("data-blank-id");
        var blank = blanks.find(function (x) {
          return x.id === id;
        });
        if (!blank) return;
        hintForBlank(blank, card, btn);
      });
    });
    updateBlankProgress();
    renderPyxMade(essay);
  }

  function hintForBlank(blank, cardEl, btnEl) {
    var topicEl = document.getElementById("wsTopic");
    var topic = (topicEl && topicEl.value) || (currentEssay && currentEssay.topic) || "";
    syncEssayFromDom();
    if (btnEl) btnEl.disabled = true;
    setStatus("Pyx is thinking of a hint (not the answer)…", "info");
    return api("/api/studio/hint", {
      topic: topic,
      blank: blank,
      sources: getPinned(),
      essay: currentEssay,
    })
      .then(function (j) {
        if (cardEl) {
          var box = cardEl.querySelector(".ws-blank-hint-box");
          if (box) {
            box.hidden = false;
            box.innerHTML =
              '<strong>Hint from Pyx</strong><div class="ws-blank-hint-text"></div>';
            var hintBody = box.querySelector(".ws-blank-hint-text");
            if (hintBody) {
              hintBody.style.whiteSpace = "pre-wrap";
              hintBody.style.marginTop = "6px";
              hintBody.textContent = j.hints || "";
            }
          }
        }
        setStatus("Hint ready — write your own answer in the box above!", "ok");
        markStep("essay", true);
      })
      .catch(function (e) {
        setStatus(e.message, "err");
        throw e;
      })
      .finally(function () {
        if (btnEl) btnEl.disabled = false;
      });
  }

  function runHelpFromPyx() {
    var topicEl = document.getElementById("wsTopic");
    var topic = (topicEl && topicEl.value) || (currentEssay && currentEssay.topic) || "";
    var helpOut = document.getElementById("wsPyxHelpOut");
    if (!currentEssay) {
      setStatus("Build your essay plan first.", "err");
      return Promise.resolve();
    }
    syncEssayFromDom();
    setStatus("Pyx is reviewing your sources…", "info");
    return api("/api/studio/help", {
      topic: topic,
      sources: getPinned(),
      essay: currentEssay,
    })
      .then(function (j) {
        if (helpOut) {
          helpOut.hidden = false;
          var intro = "";
          if (j.source_titles && j.source_titles.length) {
            intro =
              "Sources Pyx looked at: " + j.source_titles.join(", ") + "\n\n";
          }
          helpOut.textContent = intro + (j.hints || "");
        }
        setStatus("Hints ready — Pyx never writes the answer for you.", "ok");
        markStep("essay", true);
        switchTab("essay");
      })
      .catch(function (e) {
        setStatus(e.message, "err");
        throw e;
      });
  }

  function refreshExportFromFills() {
    var pyOut = document.getElementById("wsPyOut");
    if (!currentEssay) {
      setStatus("Make an essay plan first.", "err");
      return Promise.resolve();
    }
    syncEssayFromDom();
    setStatus("Updating your plan…", "info");
    return api("/api/studio/export", {
      essay: currentEssay,
      fills: collectFillsFromDom(),
    })
      .then(function (j) {
        currentEssay = j.essay || j.json;
        updatePlanViews(currentEssay, j.python || (pyOut && pyOut.value));
        saveState({ lastEssay: currentEssay });
        setStatus("Your plan is updated!", "ok");
        markStep("export", true);
      })
      .catch(function (e) {
        setStatus(e.message, "err");
        throw e;
      });
  }

  function initWorkspace() {
    var topicEl = document.getElementById("wsTopic");
    var notesEl = document.getElementById("wsNotes");
    var searchInput = document.getElementById("wsSearchInput");
    var searchBtn = document.getElementById("wsSearchBtn");
    var resultsEl = document.getElementById("wsSearchResults");
    var readerEl = document.getElementById("wsReaderText");
    var readerTitle = document.getElementById("wsReaderTitle");
    var browserUrl = document.getElementById("wsBrowserUrl");
    var browserPlaceholder = document.getElementById("wsBrowserPlaceholder");
    var lastExternalSearchUrl = "";
    var pinnedEl = document.getElementById("wsPinned");
    var buildBtn = document.getElementById("wsBuildEssay");
    var jsonOut = document.getElementById("wsJsonOut");
    var pyOut = document.getElementById("wsPyOut");
    var statusEl = document.getElementById("wsStatus");
    var tabBtns = document.querySelectorAll("[data-ws-tab]");
    var panels = document.querySelectorAll("[data-ws-panel]");

    var state = loadState();
    if (topicEl && state.topic) topicEl.value = state.topic;
    if (notesEl && state.notes) notesEl.value = state.notes;
    if (state.lastEssay) {
      currentEssay = state.lastEssay;
      renderPyxMade(currentEssay);
      renderBlanks(currentEssay);
      renderRiverside();
      updatePlanViews(currentEssay, state.lastPython);
    }

    var coachEl = document.getElementById("wsCoach");
    var coachMsg = document.getElementById("wsCoachMsg");
    var coachSteps = document.getElementById("wsCoachSteps");
    var coachAfter = document.getElementById("wsCoachAfter");
    var startPyxBtn = document.getElementById("wsStartPyx");

    function switchTab(id) {
      tabBtns.forEach(function (b) {
        b.classList.toggle("is-active", b.getAttribute("data-ws-tab") === id);
      });
      panels.forEach(function (p) {
        p.hidden = p.getAttribute("data-ws-panel") !== id;
      });
      if (id === "riverside") renderRiverside();
    }

    function renderGuide(guide) {
      currentGuide = guide;
      if (!coachEl || !guide) return;
      coachEl.hidden = false;
      if (coachMsg) coachMsg.textContent = guide.pyx_message || "";
      if (coachAfter) coachAfter.textContent = guide.after_pins || "";
      if (!coachSteps) return;
      var steps = guide.search_steps || [];
      coachSteps.innerHTML = steps
        .map(function (s, idx) {
          var done = coachStepDone[s.step];
          return (
            '<button type="button" class="ws-coach-step' +
            (done ? " is-done" : "") +
            '" data-coach-step="' +
            s.step +
            '" data-coach-idx="' +
            idx +
            '">' +
            "<strong>Search " +
            s.step +
            ": " +
            escapeHtml(s.query) +
            "</strong>" +
            "<span>" +
            escapeHtml(s.instruction) +
            "</span></button>"
          );
        })
        .join("");
      coachSteps.querySelectorAll(".ws-coach-step").forEach(function (btn) {
        btn.addEventListener("click", function () {
          var idx = parseInt(btn.getAttribute("data-coach-idx"), 10);
          var step = parseInt(btn.getAttribute("data-coach-step"), 10);
          var s = steps[idx];
          if (!s) return;
          if (searchInput) searchInput.value = s.query;
          runSearch(s.query);
          coachStepDone[step] = true;
          btn.classList.add("is-done");
        });
      });
    }

    function loadGuide(topic, autoFirstSearch) {
      return api("/api/studio/guide", { topic: topic }).then(function (guide) {
        renderGuide(guide);
        markStep("topic", true);
        if (autoFirstSearch && guide.search_steps && guide.search_steps[0]) {
          var first = guide.search_steps[0];
          if (searchInput) searchInput.value = first.query;
          return runSearch(first.query).then(function () {
            coachStepDone[1] = true;
            setStatus("Search 1 done — check Search results and Save link, or use Open on web for a new tab.", "info");
          });
        }
      });
    }

    tabBtns.forEach(function (btn) {
      btn.addEventListener("click", function () {
        switchTab(btn.getAttribute("data-ws-tab"));
      });
    });

    function renderPinned() {
      if (!pinnedEl) return;
      updatePinCount();
      var pinned = getPinned();
      if (!pinned.length) {
        pinnedEl.innerHTML =
          '<p class="ws-muted">Save links from search results. Pyx will read them when you tap <strong>Read my links &amp; build my essay</strong>.</p>';
        return;
      }
      pinnedEl.innerHTML = pinned
        .map(function (s, i) {
          return (
            '<article class="ws-pin' +
            (s.read_ok ? " is-read" : "") +
            '">' +
            (s.read_ok
              ? '<span class="ws-pin__badge">Pyx read this ✓</span> '
              : "") +
            '<strong>' +
            escapeHtml(s.title || s.url || "Source") +
            "</strong>" +
            '<p class="ws-muted">' +
            escapeHtml((s.snippet || "").slice(0, 160)) +
            "</p>" +
            '<textarea class="ws-pin-note" data-pin="' +
            i +
            '" rows="2" placeholder="Your note on this source…">' +
            escapeHtml(s.user_note || "") +
            "</textarea>" +
            '<div class="ws-pin-actions">' +
            '<button type="button" class="btn secondary ws-pin-read" data-pin="' +
            i +
            '">Read page</button>' +
            '<button type="button" class="btn secondary ws-pin-remove" data-pin="' +
            i +
            '">Remove</button>' +
            "</div>" +
            "</article>"
          );
        })
        .join("");

      pinnedEl.querySelectorAll(".ws-pin-remove").forEach(function (btn) {
        btn.addEventListener("click", function () {
          var idx = parseInt(btn.getAttribute("data-pin"), 10);
          var arr = getPinned();
          arr.splice(idx, 1);
          setPinned(arr);
          renderPinned();
        });
      });

      pinnedEl.querySelectorAll(".ws-pin-read").forEach(function (btn) {
        btn.addEventListener("click", function () {
          var idx = parseInt(btn.getAttribute("data-pin"), 10);
          var s = getPinned()[idx];
          if (s && s.url) loadReader(s.url, s.title);
        });
      });

      pinnedEl.querySelectorAll(".ws-pin-note").forEach(function (ta) {
        ta.addEventListener("change", function () {
          var idx = parseInt(ta.getAttribute("data-pin"), 10);
          var arr = getPinned();
          if (arr[idx]) {
            arr[idx].user_note = ta.value;
            setPinned(arr);
          }
        });
      });
    }

    function pinSource(item) {
      var arr = getPinned();
      if (
        arr.some(function (x) {
          return x.url === item.url;
        })
      ) {
        setStatus("You already saved that link.", "info");
        return;
      }
      arr.unshift({
        title: item.title,
        url: item.url,
        snippet: item.snippet,
        user_note: "",
        pinned_at: Date.now(),
      });
      setPinned(arr);
      renderPinned();
      renderResearchRiver();
      setStatus("Saved link: " + (item.title || item.url), "ok");
      markStep("search", true);
    }

    function ddgSearchUrl(query) {
      return "https://html.duckduckgo.com/html/?q=" + encodeURIComponent(query);
    }

    function rememberSearchUrl(url) {
      lastExternalSearchUrl = url || "";
      if (browserUrl && url) browserUrl.value = url;
    }

    function openExternal(url) {
      if (!url) return;
      rememberSearchUrl(url);
      global.open(url, "_blank", "noopener,noreferrer");
      setStatus("Opened in a new tab — come back here to Save link on Search results.", "ok");
    }

    function showBrowserPanel(topicHint) {
      switchTab("browser");
      if (browserPlaceholder) {
        var msg = browserPlaceholder.querySelector("p");
        if (msg) {
          var hint = topicHint
            ? " (" + escapeHtml(topicHint) + ")"
            : "";
          msg.innerHTML =
            "<strong>Search results are in the first tab.</strong>" +
            hint +
            " Tap <strong>Save link</strong> there, or open the full search in a new tab.";
        }
      }
    }

    function loadBrowserSearch(query) {
      rememberSearchUrl(ddgSearchUrl(query));
      showBrowserPanel(query);
    }

    if (document.getElementById("wsOpenSearchExt")) {
      document.getElementById("wsOpenSearchExt").addEventListener("click", function () {
        var q = (searchInput && searchInput.value) || (topicEl && topicEl.value) || "";
        var url = lastExternalSearchUrl || (q.trim() ? ddgSearchUrl(q.trim()) : "");
        if (url) openExternal(url);
        else setStatus("Run a search first.", "info");
      });
    }
    if (document.getElementById("wsBackToResults")) {
      document.getElementById("wsBackToResults").addEventListener("click", function () {
        switchTab("results");
      });
    }

    if (document.getElementById("wsBrowserGo")) {
      document.getElementById("wsBrowserGo").addEventListener("click", function () {
        var u = (browserUrl && browserUrl.value) || "";
        u = u.trim();
        if (!u) return;
        if (!/^https?:\/\//i.test(u)) u = "https://" + u;
        openExternal(u);
      });
    }
    if (document.getElementById("wsBrowserSearch")) {
      document.getElementById("wsBrowserSearch").addEventListener("click", function () {
        var q = (topicEl && topicEl.value) || (searchInput && searchInput.value) || "";
        if (!q.trim()) return;
        var url = ddgSearchUrl(q.trim());
        openExternal(url);
      });
    }

    function loadReader(url, title) {
      if (!readerEl) return;
      switchTab("reader");
      readerTitle.textContent = title || url;
      readerEl.textContent = "Loading page text through Pyx reader…";
      setStatus("Reading " + url, "info");
      api("/api/studio/read", { url: url })
        .then(function (j) {
          readerEl.textContent = j.text || "(No readable text returned.)";
          readerEl.dataset.lastUrl = url;
          if (j.error) setStatus("Reader note: " + j.error, "info");
          else setStatus("Loaded " + (j.chars || 0) + " characters.", "ok");
        })
        .catch(function (e) {
          readerEl.textContent = "Could not read this page: " + e.message;
          setStatus(e.message, "err");
        });
    }

    function appendToNotes(text) {
      if (!notesEl || !text) return;
      notesEl.value = (notesEl.value + "\n\n" + text).trim().slice(0, 8000);
      saveState({ notes: notesEl.value });
      setStatus("Added to your notes.", "ok");
    }

    if (document.getElementById("wsQuoteToNotes")) {
      document.getElementById("wsQuoteToNotes").addEventListener("click", function () {
        var sel = global.getSelection && global.getSelection().toString();
        if (sel && sel.trim()) appendToNotes("> " + sel.trim());
        else setStatus("Select text in the reader first.", "info");
      });
    }
    if (document.getElementById("wsQuoteAll")) {
      document.getElementById("wsQuoteAll").addEventListener("click", function () {
        var t = (readerEl && readerEl.textContent) || "";
        var para = t.split(/\n\n+/)[0] || t.slice(0, 400);
        appendToNotes("> " + para.trim());
      });
    }

    function renderResults(results) {
      if (!resultsEl) return;
      if (!results || !results.length) {
        resultsEl.innerHTML = '<p class="ws-muted">No results yet. Try a search.</p>';
        return;
      }
      resultsEl.innerHTML = results
        .map(function (r, i) {
          return (
            '<article class="ws-result">' +
            '<a href="' +
            escapeHtml(r.url) +
            '" target="_blank" rel="noopener noreferrer"><strong>' +
            escapeHtml(r.title || "Result") +
            "</strong></a>" +
            '<p class="ws-muted">' +
            escapeHtml(r.snippet || "") +
            "</p>" +
            '<div class="ws-result-actions">' +
            '<button type="button" class="btn secondary ws-open-read" data-i="' +
            i +
            '">Read in Pyx</button>' +
            '<button type="button" class="btn secondary ws-open-ext" data-url="' +
            escapeHtml(r.url) +
            '">Open site</button>' +
            '<button type="button" class="btn ws-pin" data-i="' +
            i +
            '">Save link</button>' +
            "</div>" +
            "</article>"
          );
        })
        .join("");

      var lastResults = results;
      resultsEl.querySelectorAll(".ws-pin").forEach(function (btn) {
        btn.addEventListener("click", function () {
          pinSource(lastResults[parseInt(btn.getAttribute("data-i"), 10)]);
        });
      });
      resultsEl.querySelectorAll(".ws-open-read").forEach(function (btn) {
        btn.addEventListener("click", function () {
          var r = lastResults[parseInt(btn.getAttribute("data-i"), 10)];
          loadReader(r.url, r.title);
        });
      });
      resultsEl.querySelectorAll(".ws-open-ext").forEach(function (btn) {
        btn.addEventListener("click", function () {
          global.open(btn.getAttribute("data-url"), "_blank", "noopener,noreferrer");
        });
      });
    }

    function runSearch(queryOverride) {
      var q =
        queryOverride ||
        (searchInput && searchInput.value) ||
        (topicEl && topicEl.value) ||
        "";
      q = String(q).trim();
      if (!q) return Promise.resolve();
      if (searchInput) searchInput.value = q;
      setStatus("Searching the web…", "info");
      saveState({ lastSearch: q });
      if (topicEl && !topicEl.value.trim()) topicEl.value = q.slice(0, 500);
      markStep("topic", true);
      return api("/api/studio/search", { query: q })
        .then(function (j) {
          renderResults(j.results || []);
          switchTab("results");
          var extUrl = j.browser_url || ddgSearchUrl(j.search_query || q);
          rememberSearchUrl(extUrl);
          if (j.error && !(j.results && j.results.length))
            setStatus("Search: " + j.error, "err");
          else {
            setStatus(
              "Found " +
                (j.results || []).length +
                " results — tap Save link, or Open on web for a new tab.",
              "ok"
            );
            markStep("search", true);
          }
          return j;
        })
        .catch(function (e) {
          setStatus(e.message, "err");
          throw e;
        });
    }

    if (startPyxBtn) {
      startPyxBtn.addEventListener("click", function () {
        var topic = (topicEl && topicEl.value) || "";
        topic = topic.trim();
        if (!topic) {
          setStatus("Enter your essay topic first.", "err");
          return;
        }
        saveState({ topic: topic });
        coachStepDone = {};
        setStatus("Pyx is planning your research…", "info");
        startPyxBtn.disabled = true;
        loadGuide(topic, true)
          .catch(function (e) {
            setStatus(e.message, "err");
          })
          .finally(function () {
            startPyxBtn.disabled = false;
          });
      });
    }

    if (searchBtn) searchBtn.addEventListener("click", function () {
      runSearch();
    });
    if (searchInput) {
      searchInput.addEventListener("keydown", function (e) {
        if (e.key === "Enter") runSearch();
      });
    }

    function readAllPinnedSources() {
      var arr = getPinned();
      if (!arr.length) {
        return Promise.reject(
          new Error("Save at least one link first — search the web, then tap Save link on a result.")
        );
      }
      setStatus("Pyx is reading your saved links…", "info");
      function finishRead(enriched) {
        setPinned(enriched);
        renderPinned();
        var n = enriched.filter(function (s) {
          return s.read_ok;
        }).length;
        setStatus("Pyx read " + n + " of " + enriched.length + " saved links.", n ? "ok" : "info");
        markStep("search", true);
        return enriched;
      }
      return api("/api/studio/read-sources", { sources: arr })
        .then(function (j) {
          return finishRead(j.sources || arr);
        })
        .catch(function () {
          var chain = Promise.resolve([]);
          var updated = arr.slice();
          updated.forEach(function (s, i) {
            chain = chain.then(function () {
              setStatus("Reading link " + (i + 1) + " of " + updated.length + "…", "info");
              if (!s.url) {
                s.read_ok = false;
                return updated;
              }
              return api("/api/studio/read", { url: s.url })
                .then(function (j) {
                  s.page_text = (j.text || "").slice(0, 4000);
                  s.read_ok = !!(j.text && j.text.length > 50);
                  s.read_chars = j.chars || 0;
                  return updated;
                })
                .catch(function () {
                  s.read_ok = false;
                  return updated;
                });
            });
          });
          return chain.then(finishRead);
        });
    }

    function buildEssayPlan() {
      var topic = (topicEl && topicEl.value) || "";
      topic = topic.trim();
      if (!topic) {
        return Promise.reject(new Error("Enter what you're writing about first."));
      }
      var notes = (notesEl && notesEl.value) || "";
      var sources = getPinned();
      saveState({ topic: topic, notes: notes });
      markStep("topic", true);
      setStatus("Pyx is making your essay plan…", "info");
      return api("/api/studio/essay", {
        topic: topic,
        notes: notes,
        sources: sources,
        search: sources.length === 0,
      }).then(function (j) {
        currentEssay = j.json || j.essay;
        saveState({ lastEssay: currentEssay, lastPython: j.python });
        updatePlanViews(currentEssay, j.python);
        renderPyxMade(currentEssay);
        renderBlanks(currentEssay);
        renderRiverside();
        setStatus("Essay ready — visit Riverside or write your gaps!", "ok");
        if (global.PyxHandoff && global.PyxHandoff.saveGalleryItem) {
          global.PyxHandoff.saveGalleryItem({
            type: "essay-pack",
            title: topic.slice(0, 80),
            json: currentEssay,
            python: j.python,
            at: Date.now(),
          });
        }
        markStep("pack", true);
        switchTab("essay");
        return j;
      });
    }

    function runReadAndFill() {
      return readAllPinnedSources()
        .then(function () {
          return buildEssayPlan();
        })
        .then(function () {
          switchTab("essay");
        });
    }

    var readAndFillBtn = document.getElementById("wsReadAndFill");
    var readPinsBtn = document.getElementById("wsReadPins");

    if (readAndFillBtn) {
      readAndFillBtn.addEventListener("click", function () {
        var topic = (topicEl && topicEl.value) || "";
        if (!topic.trim()) {
          setStatus("Enter your topic first.", "err");
          return;
        }
        readAndFillBtn.disabled = true;
        runReadAndFill()
          .catch(function (e) {
            setStatus(e.message, "err");
          })
          .finally(function () {
            readAndFillBtn.disabled = false;
          });
      });
    }

    if (readPinsBtn) {
      readPinsBtn.addEventListener("click", function () {
        readPinsBtn.disabled = true;
        readAllPinnedSources()
          .catch(function (e) {
            setStatus(e.message, "err");
          })
          .finally(function () {
            readPinsBtn.disabled = false;
          });
      });
    }

    if (buildBtn) {
      buildBtn.addEventListener("click", function () {
        buildBtn.disabled = true;
        buildEssayPlan()
          .catch(function (e) {
            setStatus(e.message, "err");
          })
          .finally(function () {
            buildBtn.disabled = false;
          });
      });
    }

    var helpFromPyxBtn = document.getElementById("wsHelpFromPyx");
    var flowCheckBtn = document.getElementById("wsFlowCheck");
    if (flowCheckBtn) {
      flowCheckBtn.addEventListener("click", function () {
        flowCheckBtn.disabled = true;
        runFlowCheck()
          .catch(function (e) {
            setStatus(e.message, "err");
          })
          .finally(function () {
            flowCheckBtn.disabled = false;
          });
      });
    }

    var copyDraftBtn = document.getElementById("wsCopyDraftStream");
    if (copyDraftBtn) {
      copyDraftBtn.addEventListener("click", function () {
        syncEssayFromDom();
        var text = buildDraftStreamText(currentEssay);
        if (text) navigator.clipboard.writeText(text);
        setStatus("Copied draft stream!", "ok");
        markStep("riverside", true);
      });
    }

    if (helpFromPyxBtn) {
      helpFromPyxBtn.addEventListener("click", function () {
        helpFromPyxBtn.disabled = true;
        var chain = getPinned().some(function (s) {
          return !s.read_ok && s.url;
        })
          ? readAllPinnedSources()
          : Promise.resolve();
        chain
          .then(function () {
            return runHelpFromPyx();
          })
          .catch(function (e) {
            setStatus(e.message, "err");
          })
          .finally(function () {
            helpFromPyxBtn.disabled = false;
          });
      });
    }

    if (document.getElementById("wsRefreshExport")) {
      document.getElementById("wsRefreshExport").addEventListener("click", refreshExportFromFills);
    }

    document.getElementById("wsCopyPlan") &&
      document.getElementById("wsCopyPlan").addEventListener("click", function () {
        var planOut = document.getElementById("wsPlanOut");
        if (planOut && planOut.value) navigator.clipboard.writeText(planOut.value);
        setStatus("Copied your essay plan!", "ok");
      });
    document.getElementById("wsCopyJson") &&
      document.getElementById("wsCopyJson").addEventListener("click", function () {
        if (jsonOut && jsonOut.value) navigator.clipboard.writeText(jsonOut.value);
        setStatus("Copied technical file.", "ok");
      });
    document.getElementById("wsSendTalk") &&
      document.getElementById("wsSendTalk").addEventListener("click", function () {
        syncEssayFromDom();
        var topic = (topicEl && topicEl.value) || "my essay";
        function goTalk() {
          var plan = document.getElementById("wsPlanOut");
          var outline = plan && plan.value ? plan.value.slice(0, 6000) : formatPlanForKids(currentEssay);
          if (global.PyxHandoff) {
            global.PyxHandoff.sendTo(
              "talk",
              "Help me write an essay on: " +
                topic +
                "\n\nHere is my essay plan from Pyx Studio (I researched and started my gaps):\n\n" +
                outline +
                "\n\nAsk me which part to write first and help me turn it into full paragraphs.",
              "workspace"
            );
          }
        }
        if (currentEssay) {
          refreshExportFromFills().then(goTalk).catch(goTalk);
        } else {
          goTalk();
        }
      });

    renderPinned();

    var url = new URLSearchParams(global.location.search);
    if (url.get("topic") && topicEl) {
      topicEl.value = url.get("topic");
      markStep("topic", true);
    }
    if (url.get("q") && searchInput) {
      searchInput.value = url.get("q");
      runSearch();
    }
    if (url.get("tab")) switchTab(url.get("tab"));
    if (url.get("start") === "1" && topicEl && topicEl.value.trim() && startPyxBtn) {
      startPyxBtn.click();
    } else if (topicEl && topicEl.value.trim()) {
      loadGuide(topicEl.value.trim(), false).catch(function () {});
    }

    if (global.PyxHandoff) {
      global.PyxHandoff.applyIncoming({
        app: "workspace",
        onText: function (text) {
          if (topicEl && !topicEl.value.trim()) topicEl.value = text.slice(0, 500);
          else if (notesEl) notesEl.value = (notesEl.value + "\n\n" + text).trim().slice(0, 8000);
        },
      });
      global.PyxHandoff.touchRecent(
        "workspace",
        (topicEl && topicEl.value) || "Essay project"
      );
    }
  }

  function initQuickTools() {
    var flashBtn = document.getElementById("qtFlashcards");
    var flashOut = document.getElementById("qtFlashOut");
    if (flashBtn && flashOut) {
      flashBtn.addEventListener("click", function () {
        var raw = (document.getElementById("qtFlashIn") || {}).value || "";
        var lines = raw.split(/\n+/).filter(Boolean).slice(0, 20);
        var cards = lines.map(function (line, i) {
          var parts = line.split("|");
          return { id: i + 1, front: (parts[0] || line).trim(), back: (parts[1] || "…").trim() };
        });
        var text = cards.map(function (c) {
          return c.front + " → " + c.back;
        }).join("\n");
        flashOut.value = text || "(no cards)";
      });
    }

    var studyBtn = document.getElementById("qtStudyPlan");
    var studyOut = document.getElementById("qtStudyOut");
    if (studyBtn && studyOut) {
      studyBtn.addEventListener("click", function () {
        var topic = (document.getElementById("wsTopic") || {}).value || "your topic";
        var days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
        var plan = {
          topic: topic.trim(),
          generated_by: "Pyx Studio",
          blocks: days.map(function (d, i) {
            var tasks = [
              "Research & pin 2 sources",
              "Fill outline blanks",
              "Draft one section",
              "Revise & cite",
              "Peer review / read aloud",
              "Polish introduction",
              "Final proofread",
            ];
            return { day: d, focus: tasks[i] || "Review" };
          }),
        };
        studyOut.value = plan.blocks
          .map(function (b) {
            return b.day + ": " + b.focus;
          })
          .join("\n");
      });
    }

    var taskIn = document.getElementById("qtTaskIn");
    var taskAdd = document.getElementById("qtTaskAdd");
    var taskList = document.getElementById("qtTaskList");

    function loadTasks() {
      try {
        return JSON.parse(localStorage.getItem(TASKS_KEY) || "[]");
      } catch (e) {
        return [];
      }
    }
    function saveTasks(arr) {
      try {
        localStorage.setItem(TASKS_KEY, JSON.stringify(arr.slice(0, 40)));
      } catch (e) {}
    }
    function renderTasks() {
      if (!taskList) return;
      var tasks = loadTasks();
      taskList.innerHTML = tasks
        .map(function (t, i) {
          return (
            "<li>" +
            escapeHtml(t.text) +
            ' <button type="button" data-task-done="' +
            i +
            '" style="font-size:0.75rem;margin-left:6px;">✓</button></li>'
          );
        })
        .join("");
      taskList.querySelectorAll("[data-task-done]").forEach(function (btn) {
        btn.addEventListener("click", function () {
          var idx = parseInt(btn.getAttribute("data-task-done"), 10);
          var arr = loadTasks();
          arr.splice(idx, 1);
          saveTasks(arr);
          renderTasks();
        });
      });
    }
    if (taskAdd && taskIn) {
      taskAdd.addEventListener("click", function () {
        var text = taskIn.value.trim();
        if (!text) return;
        var arr = loadTasks();
        arr.unshift({ text: text, at: Date.now() });
        saveTasks(arr);
        taskIn.value = "";
        renderTasks();
      });
      taskIn.addEventListener("keydown", function (e) {
        if (e.key === "Enter") taskAdd.click();
      });
    }
    renderTasks();

    var timerBtn = document.getElementById("qtTimerStart");
    var timerDisplay = document.getElementById("qtTimerDisplay");
    if (timerBtn && timerDisplay) {
      var timerId = null;
      var endAt = 0;
      timerBtn.addEventListener("click", function () {
        var mins = parseInt((document.getElementById("qtTimerMins") || {}).value || "25", 10);
        endAt = Date.now() + mins * 60 * 1000;
        if (timerId) clearInterval(timerId);
        timerId = setInterval(function () {
          var left = Math.max(0, endAt - Date.now());
          var m = Math.floor(left / 60000);
          var s = Math.floor((left % 60000) / 1000);
          timerDisplay.textContent = m + ":" + (s < 10 ? "0" : "") + s;
          if (left <= 0) {
            clearInterval(timerId);
            timerId = null;
            timerDisplay.textContent = "Done! Take a break.";
          }
        }, 500);
      });
    }
  }

  global.PyxStudioWorkspace = {
    initWorkspace: initWorkspace,
    initQuickTools: initQuickTools,
  };
})(typeof window !== "undefined" ? window : globalThis);
