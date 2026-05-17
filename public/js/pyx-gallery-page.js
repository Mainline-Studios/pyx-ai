/**
 * Pyx Gallery page logic.
 */
(function (global) {
  "use strict";

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function render() {
    var grid = document.getElementById("galleryGrid");
    var empty = document.getElementById("galleryEmpty");
    if (!grid || !global.PyxHandoff) return;
    var items = global.PyxHandoff.getGallery();
    if (!items.length) {
      grid.innerHTML = "";
      if (empty) empty.hidden = false;
      return;
    }
    if (empty) empty.hidden = true;
    grid.innerHTML = items
      .map(function (it) {
        var thumb = "";
        if (it.kind === "pyxel" && it.pngDataUrl) {
          thumb = '<img src="' + it.pngDataUrl + '" alt="" class="gallery-thumb" />';
        } else {
          thumb = '<div class="gallery-thumb gallery-thumb--text">' + escapeHtml((it.title || it.kind || "item").slice(0, 1)) + "</div>";
        }
        return (
          '<article class="gallery-card" data-id="' +
          escapeHtml(it.id) +
          '">' +
          thumb +
          '<div class="gallery-card__body"><h3>' +
          escapeHtml(it.title || "Untitled") +
          "</h3><p>" +
          escapeHtml((it.preview || "").slice(0, 120)) +
          '</p><div class="gallery-card__actions">' +
          '<button type="button" class="gallery-open" data-id="' +
          escapeHtml(it.id) +
          '">Open</button>' +
          '<button type="button" class="gallery-del" data-id="' +
          escapeHtml(it.id) +
          '">Remove</button>' +
          "</div></div></article>"
        );
      })
      .join("");

    grid.querySelectorAll(".gallery-del").forEach(function (btn) {
      btn.addEventListener("click", function () {
        global.PyxHandoff.removeGalleryItem(btn.getAttribute("data-id"));
        render();
      });
    });
    grid.querySelectorAll(".gallery-open").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var id = btn.getAttribute("data-id");
        var it = global.PyxHandoff.getGallery().find(function (x) {
          return x.id === id;
        });
        if (!it) return;
        if (it.kind === "pyxel") {
          global.PyxHandoff.sendTo("pyxel", it.prompt || it.preview || "", "gallery", { pixels: it.pixels });
        } else if (it.kind === "speak") {
          global.PyxHandoff.sendTo("speak", it.preview || it.title || "", "gallery");
        } else {
          global.PyxHandoff.sendTo("talk", it.preview || it.title || "", "gallery");
        }
      });
    });
  }

  function init() {
    if (global.PyxShell) global.PyxShell.init({ active: "gallery" });
    render();

    var exp = document.getElementById("galleryExport");
    var imp = document.getElementById("galleryImport");
    var impFile = document.getElementById("galleryImportFile");
    if (exp) {
      exp.addEventListener("click", function () {
        var blob = new Blob([global.PyxHandoff.exportGalleryPack()], { type: "application/json" });
        var a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = "pyx-pack.json";
        a.click();
        URL.revokeObjectURL(a.href);
      });
    }
    if (imp && impFile) {
      imp.addEventListener("click", function () {
        impFile.click();
      });
      impFile.addEventListener("change", function () {
        var f = impFile.files && impFile.files[0];
        if (!f) return;
        var reader = new FileReader();
        reader.onload = function () {
          if (global.PyxHandoff.importGalleryPack(reader.result)) render();
        };
        reader.readAsText(f);
      });
    }
  }

  global.PyxGalleryPage = { init: init, render: render };
})(typeof window !== "undefined" ? window : globalThis);
