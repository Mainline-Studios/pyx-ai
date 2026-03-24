# Pyx AI Chat — drop-in widget

One file (**`pyx-ai-chat.js`**) adds a moderated chat UI you can embed in **any HTML page** or web game.

## Quick start

```html
<div id="pyx-chat"></div>
<script src="pyx-ai-chat.js"></script>
<script>
  PyxAIChat.mount({
    el: "#pyx-chat",
    baseUrl: "http://127.0.0.1:8765",
    apiKey: null,
  });
</script>
```

### “Nothing happens” / blank page

Most of the time **`pyx-ai-chat.js` never loads**, so `PyxAIChat` is undefined and the browser throws on `mount` (easy to miss in a playground).

| Cause | Fix |
|--------|-----|
| **Online “HTML compiler” / CodePen / JSFiddle** | Those sites **cannot** load `pyx-ai-chat.js` from your disk. Upload the file as an asset, paste a **hosted** URL in `src=`, or test **locally** (below). |
| **Wrong path** | `src="pyx-ai-chat.js"` only works if that file is in the **same folder** as the HTML (or use a correct relative path / full URL). |
| **Opened as `file://`** | Prefer serving the folder over **http://**: `python3 -m http.server 8080` then open `http://127.0.0.1:8080/your-page.html`. |
| **API not running** | The UI still appears; only **Send** fails. Start Pyx: `npm run dev` or `python3 app.py` on **8765**. |

**Local test in this repo:** `python3 -m http.server 8080` from the `pyx-ai` root, then open **`http://127.0.0.1:8080/pyx-chat-demo.html`** (includes error messages if the script fails). Use DevTools → **Console** for red errors.

## What it does

1. **Learns / moderates** — Each user message is sent to **`POST /ai-decide`**, so Pyx scores it and updates training (same pipeline as games).
2. **Unsafe content** — Shows censored text and asks the user to rephrase.
3. **Safe content** — Builds a short, friendly reply that echoes the user.
4. **Internet / external data** — Optionally merges **context** into the reply:
   - **`contextUrl`** — `GET` this URL (must allow **CORS** from your site). Plain text or JSON (`summary`, `text`, or whole object) is clipped and appended.
   - **`getContext(userMessage)`** — Preferred for real web use: your server fetches APIs or pages, then returns a string. No CORS issues for third-party sites.

The bot is **template-based** (not a large generative model); the value is **consistent moderation + learning + optional facts** you supply.

## `PyxAIChat.mount(options)`

| Option | Required | Description |
|--------|----------|-------------|
| `el` | yes | CSS selector string or DOM element for the container |
| `baseUrl` | yes | Pyx API origin (no trailing slash) |
| `apiKey` | no | Same as `PyxClient`; omit if API is open |
| `title` | no | Header text (default: `Pyx AI Chat`) |
| `category` | no | Passed to `/ai-decide` as `category` (default: `phrases`) |
| `contextUrl` | no | URL to fetch for context (CORS must allow your origin) |
| `getContext` | no | `async (userMessage) => string` — if set, used **instead of** `contextUrl` |

Returns **`{ destroy() }`** — call `destroy()` to remove the widget from the DOM.

## Example: context from your backend

```js
PyxAIChat.mount({
  el: "#pyx-chat",
  baseUrl: "https://your-pyx.run.app",
  apiKey: "secret",
  getContext: async function (msg) {
    const res = await fetch("/api/wiki-snippet", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ q: msg }),
    });
    const data = await res.json();
    return data.snippet || "";
  },
});
```

## React / Vue / Svelte

Mount after the container exists (e.g. `useEffect` / `onMounted`):

```js
useEffect(() => {
  const chat = PyxAIChat.mount({ el: ref.current, baseUrl, apiKey });
  return () => chat.destroy();
}, []);
```

## See also

- **[GAME_INTEGRATION.md](./GAME_INTEGRATION.md)** — `PyxClient` for custom UIs
- **`pyx-client.js`** — programmatic API without the chat box
