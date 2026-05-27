/**
 * Pyxels manager page UI.
 */
(function (global) {
  "use strict";

  var Pyx = global.PyxPyxels;
  if (!Pyx) return;

  function esc(s) {
    var d = document.createElement("div");
    d.textContent = s == null ? "" : String(s);
    return d.innerHTML;
  }

  function cardHtml(p, opts) {
    opts = opts || {};
    var menu =
      p.premade && !opts.allowEditPremade
        ? ""
        : '<button type="button" class="pyxels-card__menu" data-menu="' +
          esc(p.id) +
          '" aria-label="Options">⋯</button>';
    var badge = p.experiment
      ? '<span class="pyxels-card__badge">Experiment</span>'
      : "";
    return (
      '<article class="pyxels-card" data-id="' +
      esc(p.id) +
      '">' +
      menu +
      '<div class="pyxels-card__icon" aria-hidden="true">' +
      esc(p.emoji || "✦") +
      "</div>" +
      badge +
      "<h3>" +
      esc(p.name) +
      "</h3>" +
      "<p>" +
      esc(p.description || "") +
      "</p>" +
      '<div class="pyxels-card__actions">' +
      '<a class="btn" href="' +
      esc(Pyx.talkUrl(p.id)) +
      '">Chat</a>' +
      '<button type="button" class="btn secondary" data-use="' +
      esc(p.id) +
      '">Use in Talk</button>' +
      (p.premade
        ? '<button type="button" class="btn secondary" data-duplicate="' +
          esc(p.id) +
          '">Customize copy</button>'
        : '<button type="button" class="btn secondary" data-edit="' +
          esc(p.id) +
          '">Edit</button>') +
      "</div></article>"
    );
  }

  function renderPremade(container, showAll) {
    var list = Pyx.listPremade();
    if (!showAll) list = list.slice(0, 4);
    container.innerHTML = list.map(function (p) {
      return cardHtml(p);
    }).join("");
  }

  function renderCustom(container) {
    var list = Pyx.listCustom();
    if (!list.length) {
      container.innerHTML =
        '<p class="pyxels-empty">No custom Pyxels yet. Tap <strong>+ New Pyxel</strong> or duplicate a premade one.</p>';
      return;
    }
    container.innerHTML = list
      .map(function (p) {
        return cardHtml(p, { allowEditPremade: true });
      })
      .join("");
  }

  function openEditor(entry) {
    var dlg = document.getElementById("pyxelsEditor");
    if (!dlg) return;
    entry = entry || {
      id: "",
      name: "",
      emoji: "✦",
      description: "",
      instructions: "",
    };
    document.getElementById("pyxelsEditId").value = entry.id || "";
    document.getElementById("pyxelsEditName").value = entry.name || "";
    document.getElementById("pyxelsEditEmoji").value = entry.emoji || "✦";
    document.getElementById("pyxelsEditDesc").value = entry.description || "";
    document.getElementById("pyxelsEditInstr").value = entry.instructions || "";
    document.getElementById("pyxelsEditorTitle").textContent = entry.id
      ? "Edit Pyxel"
      : "New Pyxel";
    dlg.hidden = false;
  }

  function closeEditor() {
    var dlg = document.getElementById("pyxelsEditor");
    if (dlg) dlg.hidden = true;
  }

  function duplicatePremade(id) {
    var p = Pyx.getById(id);
    if (!p) return;
    openEditor({
      id: "",
      name: p.name + " (my copy)",
      emoji: p.emoji,
      description: p.description,
      instructions: p.instructions,
    });
  }

  function refresh() {
    renderPremade(document.getElementById("pyxelsPremadeGrid"), showAllPremade);
    renderCustom(document.getElementById("pyxelsMineGrid"));
    var active = Pyx.getActive();
    var pill = document.getElementById("pyxelsActivePill");
    if (pill) {
      pill.textContent = active
        ? "Active in Talk: " + active.emoji + " " + active.name
        : "Active in Talk: default Pyx";
    }
  }

  var showAllPremade = false;

  function init() {
    if (global.PyxShell) PyxShell.init({ active: "pyxels" });

    var banner = document.getElementById("pyxelsBanner");
    if (banner && Pyx.isBannerDismissed()) banner.hidden = true;

    document.getElementById("pyxelsBannerDismiss")?.addEventListener("click", function () {
      Pyx.dismissBanner();
      if (banner) banner.hidden = true;
    });

    document.getElementById("pyxelsNewBtn")?.addEventListener("click", function () {
      openEditor(null);
    });

    document.getElementById("pyxelsShowMore")?.addEventListener("click", function () {
      showAllPremade = !showAllPremade;
      var btn = document.getElementById("pyxelsShowMore");
      btn.textContent = showAllPremade ? "Show less" : "Show more";
      renderPremade(document.getElementById("pyxelsPremadeGrid"), showAllPremade);
    });

    document.getElementById("pyxelsEditorForm")?.addEventListener("submit", function (e) {
      e.preventDefault();
      var res = Pyx.saveCustom({
        id: document.getElementById("pyxelsEditId").value,
        name: document.getElementById("pyxelsEditName").value,
        emoji: document.getElementById("pyxelsEditEmoji").value,
        description: document.getElementById("pyxelsEditDesc").value,
        instructions: document.getElementById("pyxelsEditInstr").value,
      });
      var err = document.getElementById("pyxelsEditError");
      if (!res.ok) {
        if (err) {
          err.textContent = res.error || "Could not save.";
          err.hidden = false;
        }
        return;
      }
      if (err) err.hidden = true;
      Pyx.setActiveId(res.item.id);
      closeEditor();
      refresh();
    });

    document.getElementById("pyxelsEditorCancel")?.addEventListener("click", closeEditor);
    document.getElementById("pyxelsClearActive")?.addEventListener("click", function () {
      Pyx.setActiveId(null);
      refresh();
    });

    function onGridClick(e) {
      var t = e.target.closest("[data-use],[data-edit],[data-duplicate],[data-delete],[data-menu]");
      if (!t) return;
      var id = t.getAttribute("data-use") || t.getAttribute("data-edit") || t.getAttribute("data-duplicate");
      if (t.hasAttribute("data-use")) {
        Pyx.setActiveId(id);
        refresh();
        return;
      }
      if (t.hasAttribute("data-edit")) {
        openEditor(Pyx.getById(id));
        return;
      }
      if (t.hasAttribute("data-duplicate")) {
        duplicatePremade(id);
        return;
      }
      if (t.hasAttribute("data-delete")) {
        if (confirm("Delete this Pyxel?")) {
          Pyx.removeCustom(id);
          refresh();
        }
        return;
      }
      if (t.hasAttribute("data-menu")) {
        id = t.getAttribute("data-menu");
        var p = Pyx.getById(id);
        if (!p || p.premade) return;
        var action = prompt("Type delete to remove this Pyxel, or cancel.");
        if (action && action.toLowerCase() === "delete") {
          Pyx.removeCustom(id);
          refresh();
        }
      }
    }

    document.getElementById("pyxelsPremadeGrid")?.addEventListener("click", onGridClick);
    document.getElementById("pyxelsMineGrid")?.addEventListener("click", onGridClick);

    refresh();
  }

  global.PyxPyxelsPage = { init: init };
})(typeof window !== "undefined" ? window : globalThis);
