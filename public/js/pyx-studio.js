/**
 * Pyx Studio hub — recents, recipes, onboarding coach.
 */
(function (global) {
  "use strict";

  var ONBOARD_KEY = "pyx.studio.onboarded";

  function formatWhen(ts) {
    if (!ts) return "—";
    var d = Date.now() - ts;
    if (d < 60000) return "just now";
    if (d < 3600000) return Math.floor(d / 60000) + "m ago";
    if (d < 86400000) return Math.floor(d / 3600000) + "h ago";
    return Math.floor(d / 86400000) + "d ago";
  }

  function readTalkRecent() {
    try {
      var idx = JSON.parse(localStorage.getItem("pyx.talk.threadIndex.v1") || "{}");
      var threads = idx && Array.isArray(idx.threads) ? idx.threads : [];
      if (!threads.length) return null;
      var t = threads[0];
      return {
        label: (t && (t.title || t.label)) || "Talk snippet",
        href: "/pyx-talk.html",
        at: (t && (t.updated || t.created)) || null,
      };
    } catch (e) {
      return null;
    }
  }

  function readCodeRecent() {
    try {
      var ed = localStorage.getItem("pyx.code.editor");
      if (!ed) return null;
      return {
        label: truncateLine(ed, 48),
        href: "/pyx-code.html",
        at: parseInt(localStorage.getItem("pyx.code.editor.at") || "0", 10) || null,
      };
    } catch (e) {
      return null;
    }
  }

  function readPyxelRecent() {
    try {
      var p = localStorage.getItem("pyx.pyxel.lastPrompt");
      if (!p) return null;
      return {
        label: truncateLine(p, 48),
        href: "/pyxel-image.html",
        at: parseInt(localStorage.getItem("pyx.pyxel.lastAt") || "0", 10) || null,
      };
    } catch (e) {
      return null;
    }
  }

  function readWorkspaceRecent() {
    try {
      var s = JSON.parse(localStorage.getItem("pyx.studio.workspace.v1") || "{}");
      if (!s.topic && !s.lastEssay) return null;
      return {
        label: truncateLine(s.topic || "Essay project", 48),
        href: "/pyx-workspace.html",
        at: s.lastSearch ? Date.now() : null,
      };
    } catch (e) {
      return null;
    }
  }

  function readWriteRecent() {
    try {
      var p = localStorage.getItem("pyx.write.lastPrompt");
      if (!p) return null;
      return {
        label: truncateLine(p, 48),
        href: "/pyx-write.html",
        at: parseInt(localStorage.getItem("pyx.write.lastAt") || "0", 10) || null,
      };
    } catch (e) {
      return null;
    }
  }

  function readSpeakRecent() {
    try {
      var s = localStorage.getItem("pyx.speak.lastScript");
      if (!s) return null;
      return {
        label: truncateLine(s, 48),
        href: "/pyx-speak.html",
        at: parseInt(localStorage.getItem("pyx.speak.lastAt") || "0", 10) || null,
      };
    } catch (e) {
      return null;
    }
  }

  function truncateLine(s, n) {
    var t = (s || "").replace(/\s+/g, " ").trim();
    if (t.length <= n) return t || "—";
    return t.slice(0, n) + "…";
  }

  function collectContinues() {
    var items = [];
    var talk = readTalkRecent();
    var workspace = readWorkspaceRecent();
    var code = readCodeRecent();
    var pyxel = readPyxelRecent();
    var speak = readSpeakRecent();
    var write = readWriteRecent();
    if (workspace) items.push({ app: "Workspace", item: workspace });
    if (talk) items.push({ app: "Talk", item: talk });
    if (code) items.push({ app: "Code", item: code });
    if (write) items.push({ app: "Write", item: write });
    if (pyxel) items.push({ app: "Pyxel", item: pyxel });
    if (speak) items.push({ app: "Speak", item: speak });
    items.sort(function (a, b) {
      return (b.item.at || 0) - (a.item.at || 0);
    });
    return items;
  }

  var RECIPES = [
    {
      title: "Essay with web research",
      desc: "Topic → search & pin sources → essay plan → Talk",
      steps: [
        {
          label: "1. Workspace",
          href: "/pyx-workspace.html",
        },
        {
          label: "2. Demo topic",
          href: "/pyx-workspace.html?topic=renewable%20energy%20for%20cities",
        },
        {
          label: "3. Write in Talk",
          action: function () {
            global.location.href = "/pyx-talk.html";
          },
        },
      ],
    },
    {
      title: "Riverside essay flow",
      desc: "Research river → draft stream → flow check → Talk",
      steps: [
        {
          label: "1. Start topic",
          href: "/pyx-workspace.html?topic=the%20water%20cycle&start=1",
        },
        {
          label: "2. Riverside",
          href: "/pyx-workspace.html?tab=riverside",
        },
        {
          label: "3. Talk",
          action: function () {
            var s = {};
            try {
              s = JSON.parse(localStorage.getItem("pyx.studio.workspace.v1") || "{}");
            } catch (e) {}
            var topic = (s.topic || "my essay").trim();
            global.PyxHandoff.sendTo(
              "talk",
              "Help me turn my Riverside draft stream into full paragraphs for: " + topic,
              "studio"
            );
          },
        },
      ],
    },
    {
      title: "Talk → Pyxel → Speak",
      desc: "Brainstorm, paint pixels, then hear it aloud",
      steps: [
        { label: "1. Talk", href: "/pyx-talk.html", handoff: null },
        {
          label: "2. Pyxel",
          action: function () {
            global.PyxHandoff.sendTo(
              "pyxel",
              "cosmic cafe with neon frog DJ, 10x10 pixel art",
              "studio",
              { recipe: "chain" }
            );
          },
        },
        {
          label: "3. Speak",
          action: function () {
            global.PyxHandoff.sendTo(
              "speak",
              "Here is your tiny cosmic cafe pixel scene, brought to life by Pyx.",
              "studio",
              { recipe: "chain" }
            );
          },
        },
      ],
    },
    {
      title: "Explain code in Talk",
      desc: "Draft in Code, unpack it in chat",
      steps: [
        {
          label: "Open Code",
          action: function () {
            global.location.href = "/pyx-code.html";
          },
        },
        {
          label: "Send to Talk",
          action: function () {
            var ed = localStorage.getItem("pyx.code.editor") || "// your code";
            global.PyxHandoff.sendTo(
              "talk",
              "Explain this code clearly:\n\n```javascript\n" + ed.slice(0, 4000) + "\n```",
              "studio"
            );
          },
        },
      ],
    },
    {
      title: "Narrate a Talk reply",
      desc: "Turn your last message into a Speak script",
      steps: [
        {
          label: "Open Talk",
          href: "/pyx-talk.html?handoff=speak",
        },
        {
          label: "Open Speak",
          href: "/pyx-speak.html",
        },
      ],
    },
  ];

  function renderContinues(container) {
    if (!container) return;
    var items = collectContinues();
    if (!items.length) {
      container.innerHTML = '<p class="studio-muted">Nothing to continue yet — open an app below to get started.</p>';
      return;
    }
    container.innerHTML = items
      .map(function (row) {
        return (
          '<a class="studio-continue" href="' +
          row.item.href +
          '"><span class="studio-continue__app">' +
          row.app +
          '</span><span class="studio-continue__label">' +
          escapeHtml(row.item.label) +
          '</span><span class="studio-continue__when">' +
          formatWhen(row.item.at) +
          "</span></a>"
        );
      })
      .join("");
  }

  function renderRecents(container) {
    if (!container || !global.PyxHandoff) return;
    var rec = global.PyxHandoff.getRecents();
    var apps = [
      { id: "workspace", label: "Workspace", href: "/pyx-workspace.html" },
      { id: "talk", label: "Talk", href: "/pyx-talk.html" },
      { id: "code", label: "Code", href: "/pyx-code.html" },
      { id: "pyxel", label: "Pyxel", href: "/pyxel-image.html" },
      { id: "speak", label: "Speak", href: "/pyx-speak.html" },
    ];
    container.innerHTML = apps
      .map(function (a) {
        var r = rec[a.id];
        return (
          '<a class="studio-launch" href="' +
          a.href +
          '"><strong>' +
          a.label +
          "</strong><span>" +
          (r ? formatWhen(r.at) : "not opened in this browser yet") +
          "</span></a>"
        );
      })
      .join("");
  }

  function renderRecipes(container) {
    if (!container) return;
    container.innerHTML = RECIPES.map(function (recipe, ri) {
      var steps = recipe.steps
        .map(function (step, si) {
          if (step.href) {
            return (
              '<a class="studio-recipe-step" href="' +
              step.href +
              '">' +
              escapeHtml(step.label) +
              "</a>"
            );
          }
          return (
            '<button type="button" class="studio-recipe-step" data-recipe="' +
            ri +
            '" data-step="' +
            si +
            '">' +
            escapeHtml(step.label) +
            "</button>"
          );
        })
        .join("");
      return (
        '<article class="studio-recipe"><h3>' +
        escapeHtml(recipe.title) +
        "</h3><p>" +
        escapeHtml(recipe.desc) +
        '</p><div class="studio-recipe-steps">' +
        steps +
        "</div></article>"
      );
    }).join("");

    container.querySelectorAll("button[data-recipe]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var ri = parseInt(btn.getAttribute("data-recipe"), 10);
        var si = parseInt(btn.getAttribute("data-step"), 10);
        var step = RECIPES[ri] && RECIPES[ri].steps[si];
        if (step && step.action) step.action();
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

  function initHub() {
    renderContinues(document.getElementById("studioContinue"));
    renderRecents(document.getElementById("studioRecents"));
    renderRecipes(document.getElementById("studioRecipes"));

    var search = document.getElementById("studioSearch");
    var searchBtn = document.getElementById("studioSearchBtn");
    function runSearch() {
      var q = (search && search.value) || "";
      if (!q.trim()) return;
      var app = global.PyxHandoff.routeQueryToApp(q);
      global.PyxHandoff.sendTo(app, q.trim(), "studio");
    }
    if (searchBtn) searchBtn.addEventListener("click", runSearch);
    if (search) {
      search.addEventListener("keydown", function (e) {
        if (e.key === "Enter") runSearch();
      });
    }

    var url = global.PyxHandoff.parseUrlHandoff();
    if (url.q && search) search.value = url.q;

    maybeShowOnboarding();
    maybeShowShortcuts();
  }

  function maybeShowOnboarding() {
    try {
      if (localStorage.getItem(ONBOARD_KEY) === "1") return;
    } catch (e) {
      return;
    }
    var el = document.getElementById("studioCoach");
    if (!el) return;
    el.hidden = false;
    var dismiss = document.getElementById("studioCoachDismiss");
    if (dismiss) {
      dismiss.addEventListener("click", function () {
        el.hidden = true;
        try {
          localStorage.setItem(ONBOARD_KEY, "1");
        } catch (e) {}
      });
    }
  }

  function maybeShowShortcuts() {
    var btn = document.getElementById("studioShortcutsBtn");
    var modal = document.getElementById("studioShortcuts");
    if (!btn || !modal) return;
    btn.addEventListener("click", function () {
      modal.hidden = !modal.hidden;
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "?" && !e.metaKey && !e.ctrlKey) {
        var t = e.target;
        if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA")) return;
        modal.hidden = !modal.hidden;
      }
    });
  }

  global.PyxStudio = {
    initHub: initHub,
    collectContinues: collectContinues,
    formatWhen: formatWhen,
  };
})(typeof window !== "undefined" ? window : globalThis);
