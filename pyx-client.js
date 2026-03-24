/**
 * Pyx API client — easy drop-in for games.
 * Get an API key, set baseUrl + apiKey, then call score(), aiDecide(), etc.
 *
 * Usage:
 *   const pyx = new PyxClient({ baseUrl: "https://...", apiKey: "your-key" });
 *   const res = await pyx.score("hello");  // { score, bad, censored }
 *   const res = await pyx.aiDecide("user message");
 *
 * If the API doesn't require a key, omit apiKey or pass null.
 */
(function (global) {
  "use strict";

  function PyxClient(options) {
    options = options || {};
    this.baseUrl = (options.baseUrl || "").replace(/\/$/, "");
    this.apiKey = options.apiKey || null;
  }

  function headers(apiKey) {
    var h = { "Content-Type": "application/json" };
    if (apiKey) {
      h["X-API-Key"] = apiKey;
      h["Authorization"] = "Bearer " + apiKey;
    }
    return h;
  }

  function request(method, url, body, apiKey) {
    var opts = { method: method, headers: headers(apiKey) };
    if (body && (method === "POST" || method === "PUT")) {
      opts.body = typeof body === "string" ? body : JSON.stringify(body);
    }
    return fetch(url, opts).then(function (res) {
      if (!res.ok) {
        return res.json().then(function (j) {
          throw new Error(j.error || res.statusText || "Request failed");
        }).catch(function () {
          throw new Error(res.statusText || "Request failed");
        });
      }
      return res.json();
    });
  }

  PyxClient.prototype._post = function (path, body) {
    return request("POST", this.baseUrl + path, body, this.apiKey);
  };

  PyxClient.prototype._get = function (path) {
    return request("GET", this.baseUrl + path, null, this.apiKey);
  };

  // --- Moderator ---
  PyxClient.prototype.score = function (text) {
    return this._post("/score", { text: text });
  };

  PyxClient.prototype.aiDecide = function (text, category) {
    var body = { text: text };
    if (category) body.category = category;
    return this._post("/ai-decide", body);
  };

  PyxClient.prototype.feedback = function (text, safe, category) {
    var body = { text: text, safe: !!safe };
    if (category) body.category = category;
    return this._post("/feedback", body);
  };

  // --- Code ---
  PyxClient.prototype.complete = function (prompt, maxTokens) {
    return this._post("/code/complete", { prompt: prompt, max_tokens: maxTokens || 256 }).then(function (r) { return r.completion; });
  };

  PyxClient.prototype.explain = function (snippet) {
    return this._post("/code/explain", { snippet: snippet }).then(function (r) { return r.explanation; });
  };

  PyxClient.prototype.refactor = function (snippet, instruction) {
    var body = { snippet: snippet };
    if (instruction) body.instruction = instruction;
    return this._post("/code/refactor", body).then(function (r) { return r.refactored; });
  };

  // --- Check ---
  PyxClient.prototype.check = function (source, language) {
    return this._post("/check", { source: source, language: language || "javascript" });
  };

  PyxClient.prototype.checkThree = function (source) {
    return this._post("/check/three", { source: source });
  };

  // --- Analyze ---
  PyxClient.prototype.analyze = function (source, language) {
    return this._post("/analyze", { source: source, language: language || "javascript" });
  };

  PyxClient.prototype.analyzeThree = function (source) {
    return this._post("/analyze/three", { source: source });
  };

  // --- Health ---
  PyxClient.prototype.health = function () {
    return this._get("/health");
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = PyxClient;
  } else {
    global.PyxClient = PyxClient;
  }
})(typeof window !== "undefined" ? window : this);
