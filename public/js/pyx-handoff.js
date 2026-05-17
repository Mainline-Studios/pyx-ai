/**
 * Pyx Studio — cross-app handoff bus (localStorage + URL deep links).
 */
(function (global) {
  "use strict";

  var HANDOFF_KEY = "pyx.handoff.v1";
  var RECENTS_KEY = "pyx.studio.recents";
  var GALLERY_KEY = "pyx.gallery.items";
  var MAX_HANDOFF = 24000;
  var MAX_GALLERY = 50;

  var APP_PATHS = {
    studio: "/",
    workspace: "/pyx-workspace.html",
    talk: "/pyx-talk.html",
    code: "/pyx-code.html",
    pyxel: "/pyxel-image.html",
    speak: "/pyx-speak.html",
    gallery: "/pyx-gallery.html",
    downloads: "/pyx-download.html",
  };

  function safeJsonParse(raw, fallback) {
    try {
      return JSON.parse(raw);
    } catch (e) {
      return fallback;
    }
  }

  function truncate(s, n) {
    if (!s || s.length <= n) return s || "";
    return s.slice(0, n) + "…";
  }

  function stripSecrets(text) {
    if (!text) return "";
    return text
      .replace(/\bsk-[a-zA-Z0-9]{20,}\b/g, "[redacted]")
      .replace(/\bAIza[0-9A-Za-z\-_]{20,}\b/g, "[redacted]");
  }

  function setHandoff(payload) {
    if (!payload || !payload.target) return;
    var p = {
      source: payload.source || "studio",
      target: payload.target,
      text: truncate(stripSecrets(payload.text || ""), MAX_HANDOFF),
      meta: payload.meta && typeof payload.meta === "object" ? payload.meta : {},
      at: Date.now(),
    };
    try {
      localStorage.setItem(HANDOFF_KEY, JSON.stringify(p));
    } catch (e) {}
    return p;
  }

  function getHandoff() {
    try {
      var raw = localStorage.getItem(HANDOFF_KEY);
      if (!raw) return null;
      var o = JSON.parse(raw);
      if (!o || !o.target) return null;
      return o;
    } catch (e) {
      return null;
    }
  }

  function clearHandoff() {
    try {
      localStorage.removeItem(HANDOFF_KEY);
    } catch (e) {}
  }

  function touchRecent(app, label, extra) {
    try {
      var rec = safeJsonParse(localStorage.getItem(RECENTS_KEY), {});
      rec[app] = {
        label: label || app,
        at: Date.now(),
        extra: extra || null,
      };
      localStorage.setItem(RECENTS_KEY, JSON.stringify(rec));
    } catch (e) {}
  }

  function getRecents() {
    return safeJsonParse(localStorage.getItem(RECENTS_KEY), {});
  }

  function navigateTo(target, payload) {
    if (payload) setHandoff(payload);
    var path = APP_PATHS[target] || APP_PATHS.studio;
    var q = payload && payload.urlQuery ? payload.urlQuery : "";
    if (payload && payload.target === target && !q) {
      q = "?handoff=" + encodeURIComponent(target);
    }
    global.location.href = path + (q || "");
  }

  function sendTo(target, text, source, meta) {
    navigateTo(target, {
      source: source || "studio",
      target: target,
      text: text || "",
      meta: meta || {},
    });
  }

  function extractFirstCodeBlock(text) {
    if (!text) return null;
    var m = text.match(/```(\w+)?\n([\s\S]*?)```/);
    if (m) return { lang: (m[1] || "").toLowerCase(), code: m[2].trim() };
    return null;
  }

  function guessLangFromCode(code) {
    if (!code) return "javascript";
    if (/^\s*def\s+\w+|^\s*import\s+\w+/m.test(code)) return "python";
    if (/^\s*fn\s+\w+|^\s*let\s+mut\s/m.test(code)) return "rust";
    if (/^\s*package\s+\w+|^\s*func\s+\w+/m.test(code)) return "go";
    if (/void\s+main\s*\(\s*\)/.test(code)) return "glsl";
    if (/^\s*<!DOCTYPE|^\s*<html/i.test(code)) return "html";
    return "javascript";
  }

  function parseUrlHandoff() {
    try {
      var sp = new URLSearchParams(global.location.search);
      var h = sp.get("handoff");
      var q = sp.get("q");
      return { handoff: h, q: q };
    } catch (e) {
      return { handoff: null, q: null };
    }
  }

  function applyIncoming(handlers) {
    handlers = handlers || {};
    var url = parseUrlHandoff();
    var payload = getHandoff();
    var target = url.handoff || (payload && payload.target);
    var used = false;

    if (url.q && handlers.onQuery) {
      handlers.onQuery(url.q);
      used = true;
    }

    if (payload && (!target || payload.target === handlers.app || url.handoff)) {
      if (handlers.onText && payload.text) {
        handlers.onText(payload.text, payload);
        used = true;
      }
      if (handlers.onMeta && payload.meta) handlers.onMeta(payload.meta, payload);
    }

    if (used || url.handoff) clearHandoff();
    return payload;
  }

  function getGallery() {
    var arr = safeJsonParse(localStorage.getItem(GALLERY_KEY), []);
    return Array.isArray(arr) ? arr : [];
  }

  function saveGalleryItem(item) {
    var arr = getGallery();
    item.id = item.id || "g_" + Date.now() + "_" + Math.random().toString(36).slice(2, 8);
    item.at = item.at || Date.now();
    arr.unshift(item);
    if (arr.length > MAX_GALLERY) arr = arr.slice(0, MAX_GALLERY);
    try {
      localStorage.setItem(GALLERY_KEY, JSON.stringify(arr));
    } catch (e) {}
    return item;
  }

  function removeGalleryItem(id) {
    var arr = getGallery().filter(function (x) {
      return x.id !== id;
    });
    try {
      localStorage.setItem(GALLERY_KEY, JSON.stringify(arr));
    } catch (e) {}
  }

  function exportGalleryPack() {
    return JSON.stringify({ version: 1, exported: Date.now(), items: getGallery() }, null, 2);
  }

  function importGalleryPack(jsonStr) {
    var o = safeJsonParse(jsonStr, null);
    if (!o || !Array.isArray(o.items)) return false;
    var merged = o.items.concat(getGallery()).slice(0, MAX_GALLERY);
    try {
      localStorage.setItem(GALLERY_KEY, JSON.stringify(merged));
    } catch (e) {
      return false;
    }
    return true;
  }

  function routeQueryToApp(q) {
    var t = (q || "").toLowerCase();
    if (/essay|research|report|thesis|homework|study guide|data pack|outline|fill in the blank|workspace/.test(t))
      return "workspace";
    if (/code|javascript|python|function|debug/.test(t)) return "code";
    if (/pixel|pyxel|art|sprite|10x10/.test(t)) return "pyxel";
    if (/speak|voice|tts|read aloud|narrat/.test(t)) return "speak";
    return "talk";
  }

  global.PyxHandoff = {
    HANDOFF_KEY: HANDOFF_KEY,
    APP_PATHS: APP_PATHS,
    setHandoff: setHandoff,
    getHandoff: getHandoff,
    clearHandoff: clearHandoff,
    touchRecent: touchRecent,
    getRecents: getRecents,
    navigateTo: navigateTo,
    sendTo: sendTo,
    extractFirstCodeBlock: extractFirstCodeBlock,
    guessLangFromCode: guessLangFromCode,
    applyIncoming: applyIncoming,
    parseUrlHandoff: parseUrlHandoff,
    routeQueryToApp: routeQueryToApp,
    getGallery: getGallery,
    saveGalleryItem: saveGalleryItem,
    removeGalleryItem: removeGalleryItem,
    exportGalleryPack: exportGalleryPack,
    importGalleryPack: importGalleryPack,
    truncate: truncate,
    stripSecrets: stripSecrets,
  };
})(typeof window !== "undefined" ? window : globalThis);
