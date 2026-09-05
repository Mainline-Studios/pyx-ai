/**
 * Chunked cookie store for chats + the on-device learner snapshot.
 * Cookies cap ~4KB each; we split across pa.c0..pa.c7 and pa.ml.
 */
(function (root) {
  "use strict";

  var PREFIX = "pa.c";
  var ML_PREFIX = "pa.m";
  var ML_KEY = "pa.ml";
  var CHUNKS = 8;
  var ML_CHUNKS = 4;
  var MAX = 3500;
  var YEAR = 60 * 60 * 24 * 365;

  function canCookie() {
    return typeof document !== "undefined" && typeof document.cookie === "string";
  }

  function setRaw(name, value) {
    if (!canCookie()) return;
    document.cookie =
      name +
      "=" +
      encodeURIComponent(value || "") +
      "; path=/betas/pyxassistant; max-age=" +
      YEAR +
      "; SameSite=Lax";
  }

  function getRaw(name) {
    if (!canCookie()) return "";
    var parts = document.cookie.split("; ");
    var i;
    for (i = 0; i < parts.length; i++) {
      if (parts[i].indexOf(name + "=") === 0) {
        try {
          return decodeURIComponent(parts[i].slice(name.length + 1));
        } catch (e) {
          return "";
        }
      }
    }
    return "";
  }

  function delRaw(name) {
    if (!canCookie()) return;
    document.cookie = name + "=; path=/betas/pyxassistant; max-age=0; SameSite=Lax";
    document.cookie = name + "=; path=/; max-age=0; SameSite=Lax";
  }

  function compactMessages(messages) {
    return (messages || [])
      .filter(function (m) {
        return m && (m.role === "user" || m.role === "assistant") && m.content;
      })
      .slice(-24)
      .map(function (m) {
        return [m.role === "user" ? "u" : "a", String(m.content).slice(0, 280)];
      });
  }

  function expandMessages(rows) {
    if (!Array.isArray(rows)) return [];
    return rows
      .map(function (row) {
        if (!Array.isArray(row) || row.length < 2) return null;
        return { role: row[0] === "u" ? "user" : "assistant", content: String(row[1] || "") };
      })
      .filter(Boolean);
  }

  function writeChunks(prefix, str, n) {
    var i;
    for (i = 0; i < n; i++) {
      var slice = str.slice(i * MAX, (i + 1) * MAX);
      if (slice) setRaw(prefix + i, slice);
      else delRaw(prefix + i);
    }
  }

  function readChunks(prefix, n) {
    var out = "";
    var i;
    for (i = 0; i < n; i++) out += getRaw(prefix + i);
    return out;
  }

  function saveChats(messages) {
    var packed = JSON.stringify(compactMessages(messages));
    if (packed.length > CHUNKS * MAX) {
      packed = JSON.stringify(compactMessages(messages).slice(-12));
    }
    writeChunks(PREFIX, packed, CHUNKS);
  }

  function loadChats() {
    var raw = readChunks(PREFIX, CHUNKS);
    if (!raw) return [];
    try {
      return expandMessages(JSON.parse(raw));
    } catch (e) {
      return [];
    }
  }

  function saveModel(obj) {
    try {
      var packed = JSON.stringify(obj);
      writeChunks(ML_PREFIX, packed, ML_CHUNKS);
      delRaw(ML_KEY);
    } catch (e) {}
  }

  function loadModel() {
    var raw = readChunks(ML_PREFIX, ML_CHUNKS) || getRaw(ML_KEY);
    if (!raw) return null;
    try {
      return JSON.parse(raw);
    } catch (e) {
      return null;
    }
  }

  function clearAll() {
    var i;
    for (i = 0; i < CHUNKS; i++) delRaw(PREFIX + i);
    for (i = 0; i < ML_CHUNKS; i++) delRaw(ML_PREFIX + i);
    delRaw(ML_KEY);
  }

  var api = {
    saveChats: saveChats,
    loadChats: loadChats,
    saveModel: saveModel,
    loadModel: loadModel,
    clearAll: clearAll,
    compactMessages: compactMessages,
    expandMessages: expandMessages,
  };

  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.PyxAssistantCookies = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
