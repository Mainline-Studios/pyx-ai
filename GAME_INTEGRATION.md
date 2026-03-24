# Use the Pyx API in a game (one API key)

Easiest way: add the client script, set your **base URL** and **API key**, then call the API from your game.

## 1. Get an API key

**Local dev:** From the pyx-ai repo run:
```bash
npm run dev
```
The server starts and prints an API key. Use that key in your client; it’s valid for that run. Copy it into your game config.

**Production:** If the API is **open** (no key set on the server), omit the key or pass `null`. If the API **requires a key**, use the key your team set (e.g. `PYX_API_KEY` on Cloud Run). Don’t commit keys to public repos.

## 2. Add the client

**Browser / web game:** include the script, then use `PyxClient`:

```html
<script src="https://your-cdn-or-static/pyx-client.js"></script>
<script>
  var pyx = new PyxClient({
    baseUrl: "https://pyxaiapi-574247481583.us-central1.run.app",
    apiKey: "your-api-key"   // omit if the API doesn't use a key
  });

  // Content filter (chat, user text)
  pyx.score("hello world").then(function(res) {
    if (res.bad) {
      // show res.censored or block
    } else {
      // show original text
    }
  });

  // Game AI decision (trains the filter)
  pyx.aiDecide("user message").then(function(res) {
    if (res.bad) showCensored(res.censored); else show(res);
  });
</script>
```

**Node / bundler:** `const PyxClient = require("./pyx-client.js");` or import, then same usage.

## 3. What you can call

| Method        | Use in game |
|---------------|-------------|
| `score(text)` | Check if text is OK (chat, names). Returns `{ score, bad, censored }`. |
| `aiDecide(text)` | Same as score but also trains the filter. Use for in-game AI content. |
| `feedback(text, safe)` | Send moderator override (safe = true/false). |
| `complete(prompt)` | Code completion (e.g. in-game script editor). |
| `explain(snippet)` | Explain a code snippet. |
| `refactor(snippet, instruction)` | Refactor code. |
| `check(source, language)` | Get tips on code. |
| `checkThree(source)` | Tips for three.js code. |
| `analyze(source)` | Check code for inappropriate content. |
| `analyzeThree(source)` | Same for three.js. |
| `health()` | Check if the API is up. |

All methods return a **Promise** with the JSON response (or throw on error).

## 4. One config, one client

Keep your base URL and API key in one place (e.g. config object or env), create the client once at startup, and use it everywhere:

```js
var config = {
  pyxBaseUrl: "https://pyxaiapi-574247481583.us-central1.run.app",
  pyxApiKey: "your-key"  // or null if no key
};
var pyx = new PyxClient({ baseUrl: config.pyxBaseUrl, apiKey: config.pyxApiKey });
// then: pyx.score("..."), pyx.aiDecide("..."), etc.
```

That’s all you need to easily implement the API in a game.

## 5. Drop-in “Pyx AI Chat” (full UI)

For a **ready-made chat box** (moderation + learning + optional web context + bot replies), copy **`pyx-ai-chat.js`** next to your page and mount it:

```html
<div id="pyx-chat"></div>
<script src="pyx-ai-chat.js"></script>
<script>
  PyxAIChat.mount({
    el: "#pyx-chat",
    baseUrl: "https://your-pyx-api.example.com",
    apiKey: "your-key-or-null",
  });
</script>
```

Full options, `contextUrl`, and `getContext` (recommended for pulling web data via your backend): **[PYX_CHAT.md](./PYX_CHAT.md)**.
