/**
 * Pyx AI Chat — drop-in chat UI + Pyx moderator + optional web context + learning.
 *
 * Add to ANY page:
 *   <div id="pyx-chat"></div>
 *   <script src="pyx-ai-chat.js"></script>
 *   <script>
 *     PyxAIChat.mount({
 *       el: "#pyx-chat",
 *       baseUrl: "http://127.0.0.1:8765",
 *       apiKey: "your-key-or-null",
 *       title: "Pyx AI Chat",
 *       // Optional: same-origin or CORS-friendly URL whose text/JSON feeds the reply
 *       contextUrl: "/api/chat-context.json",
 *       // Or async (userMessage) => string — use your server to fetch the web safely
 *       getContext: null,
 *     });
 *   </script>
 *
 * If nothing appears: pyx-ai-chat.js must load (same folder or full URL); online HTML
 * sandboxes often 404 local scripts — use http://localhost with python -m http.server or
 * open pyx-chat-demo.html from this repo. Check the browser console.
 *
 * Flow: user sends → POST /ai-decide (scores + trains Pyx) → if safe, optional context
 * fetch → friendly reply. Unsafe messages show censored text + a gentle bot line.
 */
(function (global) {
  "use strict";

  var CSS =
    ".pyx-ai-chat{font-family:system-ui,-apple-system,sans-serif;max-width:420px;border:1px solid #e2e8f0;border-radius:12px;overflow:hidden;background:#fff;box-shadow:0 4px 24px rgba(0,0,0,.08)}" +
    ".pyx-ai-chat__head{padding:12px 14px;background:linear-gradient(135deg,#0f172a,#334155);color:#f8fafc;font-size:15px;font-weight:600}" +
    ".pyx-ai-chat__msgs{height:280px;overflow-y:auto;padding:12px;background:#f8fafc}" +
    ".pyx-ai-chat__row{margin-bottom:10px;max-width:92%}" +
    ".pyx-ai-chat__row--user{margin-left:auto;text-align:right}" +
    ".pyx-ai-chat__bubble{display:inline-block;padding:8px 12px;border-radius:12px;font-size:14px;line-height:1.45;word-break:break-word}" +
    ".pyx-ai-chat__row--user .pyx-ai-chat__bubble{background:#3b82f6;color:#fff;border-bottom-right-radius:4px}" +
    ".pyx-ai-chat__row--bot .pyx-ai-chat__bubble{background:#e2e8f0;color:#0f172a;border-bottom-left-radius:4px}" +
    ".pyx-ai-chat__row--warn .pyx-ai-chat__bubble{background:#fef3c7;color:#92400e;border:1px solid #fcd34d}" +
    ".pyx-ai-chat__foot{display:flex;gap:8px;padding:10px;border-top:1px solid #e2e8f0;background:#fff}" +
    ".pyx-ai-chat__input{flex:1;border:1px solid #cbd5e1;border-radius:8px;padding:8px 12px;font-size:14px;outline:none}" +
    ".pyx-ai-chat__input:focus{border-color:#3b82f6}" +
    ".pyx-ai-chat__send{background:#0f172a;color:#fff;border:none;border-radius:8px;padding:8px 14px;font-size:14px;cursor:pointer;font-weight:600}" +
    ".pyx-ai-chat__send:disabled{opacity:.5;cursor:not-allowed}" +
    ".pyx-ai-chat__hint{font-size:11px;color:#64748b;padding:0 12px 8px}";

  function headers(apiKey) {
    var h = { "Content-Type": "application/json" };
    if (apiKey) {
      h["X-API-Key"] = apiKey;
      h["Authorization"] = "Bearer " + apiKey;
    }
    return h;
  }

  function post(baseUrl, apiKey, path, body) {
    return fetch(baseUrl.replace(/\/$/, "") + path, {
      method: "POST",
      headers: headers(apiKey),
      body: JSON.stringify(body),
    }).then(function (res) {
      if (!res.ok) {
        return res
          .json()
          .then(function (j) {
            throw new Error(j.error || res.statusText);
          })
          .catch(function () {
            throw new Error(res.statusText || "Request failed");
          });
      }
      return res.json();
    });
  }

  function clip(s, max) {
    s = String(s || "").trim();
    if (s.length <= max) return s;
    return s.slice(0, max - 1) + "…";
  }

  /** Try to turn fetch response into plain text for context. */
  function parseContextBody(text, contentType) {
    text = String(text || "").trim();
    if (!text) return "";
    if (contentType && contentType.indexOf("json") >= 0) {
      try {
        var j = JSON.parse(text);
        if (typeof j === "string") return clip(j, 800);
        if (j && typeof j.summary === "string") return clip(j.summary, 800);
        if (j && typeof j.text === "string") return clip(j.text, 800);
        return clip(JSON.stringify(j), 800);
      } catch (e) {
        return clip(text, 800);
      }
    }
    return clip(text.replace(/\s+/g, " "), 800);
  }

  function fetchContextUrl(url) {
    if (!url) return Promise.resolve("");
    return fetch(url, { method: "GET", credentials: "omit" })
      .then(function (res) {
        if (!res.ok) return "";
        var ct = res.headers.get("content-type") || "";
        return res.text().then(function (t) {
          return parseContextBody(t, ct);
        });
      })
      .catch(function () {
        return "";
      });
  }

  var FRIENDLY_OPENERS = [
    "Thanks for sharing!",
    "Got it.",
    "I hear you.",
    "Interesting!",
    "Cool — here's what I can say:",
  ];

  function pick(arr) {
    return arr[Math.floor(Math.random() * arr.length)];
  }

  function buildBotReply(userText, webContext, pyxSafe) {
    var parts = [];
    parts.push(pick(FRIENDLY_OPENERS));
    if (userText) {
      parts.push(' You said: "' + clip(userText, 120) + '"');
    }
    if (webContext) {
      parts.push(
        "\n\nFrom what I could pull in:\n" + clip(webContext, 500)
      );
    }
    if (pyxSafe === true) {
      parts.push("\n\n(Pyx moderated this message as OK for your community.)");
    }
    parts.push(
      "\n\nI'm a Pyx-safe assistant: messages are checked so we keep things friendly. Ask me anything appropriate!"
    );
    return parts.join("");
  }

  function injectStyle() {
    if (document.getElementById("pyx-ai-chat-styles")) return;
    var s = document.createElement("style");
    s.id = "pyx-ai-chat-styles";
    s.textContent = CSS;
    document.head.appendChild(s);
  }

  function mount(opts) {
    opts = opts || {};
    var el =
      typeof opts.el === "string"
        ? document.querySelector(opts.el)
        : opts.el;
    if (!el) throw new Error("PyxAIChat.mount: el not found");

    var baseUrl = (opts.baseUrl || "").replace(/\/$/, "");
    var apiKey = opts.apiKey || null;
    var title = opts.title || "Pyx AI Chat";
    var contextUrl = opts.contextUrl || null;
    var category = opts.category || "phrases";
    var getContext =
      typeof opts.getContext === "function" ? opts.getContext : null;

    injectStyle();

    el.className = (el.className ? el.className + " " : "") + "pyx-ai-chat";
    el.innerHTML =
      '<div class="pyx-ai-chat__head"></div>' +
      '<div class="pyx-ai-chat__msgs"></div>' +
      '<div class="pyx-ai-chat__hint">Messages are moderated. Optional web context improves replies (CORS or your getContext).</div>' +
      '<div class="pyx-ai-chat__foot">' +
      '<input type="text" class="pyx-ai-chat__input" placeholder="Type a message…" autocomplete="off" />' +
      '<button type="button" class="pyx-ai-chat__send">Send</button>' +
      "</div>";

    el.querySelector(".pyx-ai-chat__head").textContent = title;

    var msgs = el.querySelector(".pyx-ai-chat__msgs");
    var input = el.querySelector(".pyx-ai-chat__input");
    var sendBtn = el.querySelector(".pyx-ai-chat__send");

    function addRow(role, text, isWarn) {
      var row = document.createElement("div");
      row.className =
        "pyx-ai-chat__row pyx-ai-chat__row--" +
        role +
        (isWarn ? " pyx-ai-chat__row--warn" : "");
      var bubble = document.createElement("div");
      bubble.className = "pyx-ai-chat__bubble";
      bubble.textContent = text;
      row.appendChild(bubble);
      msgs.appendChild(row);
      msgs.scrollTop = msgs.scrollHeight;
    }

    function setBusy(b) {
      sendBtn.disabled = b;
      input.disabled = b;
    }

    function runSend() {
      var text = (input.value || "").trim();
      if (!text || !baseUrl) return;
      input.value = "";
      addRow("user", text);
      setBusy(true);

      post(baseUrl, apiKey, "/ai-decide", {
        text: text,
        category: category,
      })
        .then(function (decide) {
          if (decide.bad) {
            addRow(
              "bot",
              (decide.censored || text) +
                "\n\nLet's keep chat friendly — try rephrasing.",
              true
            );
            return;
          }
          var ctxP = Promise.resolve("");
          if (getContext) {
            ctxP = Promise.resolve(getContext(text)).then(function (c) {
              return String(c || "");
            });
          } else if (contextUrl) {
            ctxP = fetchContextUrl(contextUrl);
          }
          return ctxP.then(function (webCtx) {
            var reply = buildBotReply(
              text,
              webCtx,
              decide.safe === true
            );
            addRow("bot", reply);
          });
        })
        .catch(function (err) {
          addRow(
            "bot",
            "Couldn't reach Pyx API: " + (err.message || String(err)),
            true
          );
        })
        .finally(function () {
          setBusy(false);
          input.focus();
        });
    }

    sendBtn.addEventListener("click", runSend);
    input.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        runSend();
      }
    });

    return {
      destroy: function () {
        el.innerHTML = "";
        el.className = el.className.replace(/\bpyx-ai-chat\b/g, "").trim();
      },
    };
  }

  global.PyxAIChat = { mount: mount };
})(typeof window !== "undefined" ? window : this);
