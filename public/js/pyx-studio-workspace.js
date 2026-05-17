/**
 * Pyx Studio Workspace — productivity (essay helper, research browser, fill blanks).
 */
(function (global) {
  "use strict";

  var STORAGE_KEY = "pyx.studio.workspace.v1";
  var PINNED_KEY = "pyx.studio.essay.pinned";
  var TASKS_KEY = "pyx.studio.tasks.v1";

  var currentEssay = null;

  function setStatus(msg, kind) {
    var statusEl = document.getElementById("wsStatus");
    if (!statusEl) return;
    statusEl.textContent = msg || "";
    statusEl.className = "ws-status" + (kind ? " ws-status--" + kind : "");
  }

  function api(path, body) {
    return fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    }).then(function (r) {
      return r.json().then(function (j) {
        if (!r.ok) throw new Error((j && j.error) || r.statusText);
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

  function renderBlanks(essay) {
    var list = document.getElementById("wsBlanksList");
    if (!list) return;
    var blanks = (essay && essay.fill_blanks) || [];
    if (!blanks.length) {
      list.innerHTML =
        '<p class="ws-muted">Build a data pack first — Pyx will create fill-in-the-blank prompts from your outline.</p>';
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
          (b.suggested && !b.user_fill
            ? '<p class="ws-muted">Pyx hint: ' + escapeHtml(b.suggested.slice(0, 200)) + "</p>"
            : "") +
          '<div class="ws-blank-actions">' +
          '<button type="button" class="btn secondary ws-blank-pyx">Ask Pyx</button>' +
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
        markStep("blanks", true);
      });
    });

    list.querySelectorAll(".ws-blank-pyx").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var card = btn.closest(".ws-blank");
        var id = card && card.getAttribute("data-blank-id");
        var blank = blanks.find(function (x) {
          return x.id === id;
        });
        if (!blank) return;
        fillOneBlank(blank, card);
      });
    });
    updateBlankProgress();
  }

  function fillOneBlank(blank, cardEl) {
    var topicEl = document.getElementById("wsTopic");
    var topic = (topicEl && topicEl.value) || (currentEssay && currentEssay.topic) || "";
    var statusEl = document.getElementById("wsStatus");
    if (cardEl) {
      var ta = cardEl.querySelector("textarea");
      if (ta) ta.disabled = true;
    }
    if (statusEl) statusEl.textContent = "Pyx is drafting a suggestion…";
    api("/api/studio/fill", {
      topic: topic,
      blank: blank,
      sources: getPinned(),
      essay: currentEssay,
    })
      .then(function (j) {
        if (cardEl) {
          var ta2 = cardEl.querySelector("textarea");
          if (ta2) {
            ta2.value = j.suggestion || "";
            ta2.disabled = false;
            cardEl.classList.add("is-filled");
          }
        }
        blank.user_fill = j.suggestion || "";
        syncEssayFromDom();
        updateBlankProgress();
        setStatus("Suggestion added — edit before you submit.", "ok");
        markStep("blanks", true);
      })
      .catch(function (e) {
        if (cardEl) {
          var ta3 = cardEl.querySelector("textarea");
          if (ta3) ta3.disabled = false;
        }
        if (statusEl) statusEl.textContent = e.message;
      });
  }

  function refreshExportFromFills() {
    var topicEl = document.getElementById("wsTopic");
    var jsonOut = document.getElementById("wsJsonOut");
    var pyOut = document.getElementById("wsPyOut");
    var statusEl = document.getElementById("wsStatus");
    if (!currentEssay) {
      if (statusEl) statusEl.textContent = "Build a data pack first.";
      return;
    }
    syncEssayFromDom();
    if (statusEl) statusEl.textContent = "Updating export…";
    api("/api/studio/export", {
      essay: currentEssay,
      fills: collectFillsFromDom(),
    })
      .then(function (j) {
        currentEssay = j.essay || j.json;
        if (jsonOut) jsonOut.value = JSON.stringify(currentEssay, null, 2);
        if (pyOut) pyOut.value = j.python || "";
        saveState({ lastEssay: currentEssay });
        setStatus("Export updated with your fills.", "ok");
        markStep("export", true);
      })
      .catch(function (e) {
        if (statusEl) statusEl.textContent = e.message;
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
    var browserFrame = document.getElementById("wsBrowserFrame");
    var browserUrl = document.getElementById("wsBrowserUrl");
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
      renderBlanks(currentEssay);
      if (jsonOut) jsonOut.value = JSON.stringify(currentEssay, null, 2);
    }

    function switchTab(id) {
      tabBtns.forEach(function (b) {
        b.classList.toggle("is-active", b.getAttribute("data-ws-tab") === id);
      });
      panels.forEach(function (p) {
        p.hidden = p.getAttribute("data-ws-panel") !== id;
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
          '<p class="ws-muted">Pin sources from search results. Pyx uses them when building your JSON/Python data pack and when filling blanks.</p>';
        return;
      }
      pinnedEl.innerHTML = pinned
        .map(function (s, i) {
          return (
            '<article class="ws-pin">' +
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
        setStatus("Already pinned.", "info");
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
      setStatus("Pinned: " + (item.title || item.url), "ok");
      markStep("search", true);
    }

    function loadBrowserUrl(url) {
      if (!browserFrame) return;
      browserFrame.src = url;
      if (browserUrl) browserUrl.value = url;
      switchTab("browser");
    }

    function loadBrowserSearch(query) {
      var q = encodeURIComponent(query);
      loadBrowserUrl("https://html.duckduckgo.com/html/?q=" + q);
    }

    if (document.getElementById("wsBrowserGo")) {
      document.getElementById("wsBrowserGo").addEventListener("click", function () {
        var u = (browserUrl && browserUrl.value) || "";
        u = u.trim();
        if (!u) return;
        if (!/^https?:\/\//i.test(u)) u = "https://" + u;
        loadBrowserUrl(u);
      });
    }
    if (document.getElementById("wsBrowserSearch")) {
      document.getElementById("wsBrowserSearch").addEventListener("click", function () {
        var q = (topicEl && topicEl.value) || (searchInput && searchInput.value) || "";
        if (q.trim()) loadBrowserSearch(q.trim());
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
            '">Pin source</button>' +
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

    function runSearch() {
      var q = (searchInput && searchInput.value) || (topicEl && topicEl.value) || "";
      q = q.trim();
      if (!q) return;
      setStatus("Searching the web…", "info");
      saveState({ lastSearch: q });
      if (topicEl && !topicEl.value.trim()) topicEl.value = q.slice(0, 500);
      markStep("topic", true);
      api("/api/studio/search", { query: q })
        .then(function (j) {
          renderResults(j.results || []);
          if (j.browser_url) loadBrowserUrl(j.browser_url);
          else loadBrowserSearch(j.search_query || q);
          if (j.error && !(j.results && j.results.length)) setStatus("Search: " + j.error, "err");
          else {
            setStatus("Found " + (j.results || []).length + " results.", "ok");
            markStep("search", true);
          }
        })
        .catch(function (e) {
          setStatus(e.message, "err");
        });
    }

    if (searchBtn) searchBtn.addEventListener("click", runSearch);
    if (searchInput) {
      searchInput.addEventListener("keydown", function (e) {
        if (e.key === "Enter") runSearch();
      });
    }

    if (buildBtn) {
      buildBtn.addEventListener("click", function () {
        var topic = (topicEl && topicEl.value) || "";
        topic = topic.trim();
        if (!topic) {
          setStatus("Enter an essay topic first.", "err");
          return;
        }
        var notes = (notesEl && notesEl.value) || "";
        var sources = getPinned();
        saveState({ topic: topic, notes: notes });
        markStep("topic", true);
        setStatus("Pyx is building your JSON + Python data pack…", "info");
        buildBtn.disabled = true;
        api("/api/studio/essay", {
          topic: topic,
          notes: notes,
          sources: sources,
          search: sources.length === 0,
        })
          .then(function (j) {
            currentEssay = j.json || j.essay;
            saveState({ lastEssay: currentEssay });
            if (jsonOut) jsonOut.value = JSON.stringify(currentEssay, null, 2);
            if (pyOut) pyOut.value = j.python || "";
            renderBlanks(currentEssay);
            setStatus(
              "Data pack ready" +
                (j.model ? " · " + j.model : "") +
                (j.web_search && j.web_search.used ? " · web research" : "") +
                " — fill in the blanks next.",
              "ok"
            );
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
            switchTab("blanks");
          })
          .catch(function (e) {
            setStatus(e.message, "err");
          })
          .finally(function () {
            buildBtn.disabled = false;
          });
      });
    }

    if (document.getElementById("wsFillAll")) {
      document.getElementById("wsFillAll").addEventListener("click", function () {
        if (!currentEssay || !currentEssay.fill_blanks) {
          setStatus("Build a data pack first.", "err");
          return;
        }
        setStatus("Pyx is filling blanks one by one…", "info");
        var chain = Promise.resolve();
        currentEssay.fill_blanks.forEach(function (blank) {
          chain = chain.then(function () {
            var card = document.querySelector('.ws-blank[data-blank-id="' + blank.id + '"]');
            return new Promise(function (resolve) {
              fillOneBlank(blank, card);
              setTimeout(resolve, 400);
            });
          });
        });
        chain.then(function () {
          refreshExportFromFills();
        });
      });
    }

    if (document.getElementById("wsRefreshExport")) {
      document.getElementById("wsRefreshExport").addEventListener("click", refreshExportFromFills);
    }

    document.getElementById("wsCopyJson") &&
      document.getElementById("wsCopyJson").addEventListener("click", function () {
        if (jsonOut && jsonOut.value) navigator.clipboard.writeText(jsonOut.value);
        setStatus("JSON copied.", "ok");
      });
    document.getElementById("wsCopyPy") &&
      document.getElementById("wsCopyPy").addEventListener("click", function () {
        if (pyOut && pyOut.value) navigator.clipboard.writeText(pyOut.value);
        setStatus("Python copied.", "ok");
      });
    document.getElementById("wsSendTalk") &&
      document.getElementById("wsSendTalk").addEventListener("click", function () {
        syncEssayFromDom();
        var topic = (topicEl && topicEl.value) || "my essay";
        function goTalk() {
          var outline = jsonOut && jsonOut.value ? jsonOut.value.slice(0, 5000) : "";
          if (global.PyxHandoff) {
            global.PyxHandoff.sendTo(
              "talk",
              "Help me write an essay on: " +
                topic +
                "\n\nUse this Pyx Studio research pack (outline + my filled blanks):\n```json\n" +
                outline +
                "\n```\n\nAsk me which sections to expand first.",
              "workspace"
            );
          }
        }
        if (currentEssay) {
          api("/api/studio/export", {
            essay: currentEssay,
            fills: collectFillsFromDom(),
          })
            .then(function (j) {
              currentEssay = j.essay || j.json;
              if (jsonOut) jsonOut.value = JSON.stringify(currentEssay, null, 2);
              if (pyOut) pyOut.value = j.python || "";
              goTalk();
            })
            .catch(goTalk);
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
        flashOut.value = JSON.stringify({ deck: "Pyx Studio", cards: cards }, null, 2);
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
        studyOut.value = JSON.stringify(plan, null, 2);
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
