# Pyx AI Chat — instructions for your web team

## What you’re adding

A small **JavaScript widget** that shows a chat box on your site. Messages go to **your Pyx API** for moderation and learning; safe messages get a friendly reply.

## What you need from Pyx / backend

1. **API base URL** — e.g. `https://your-pyx-api.example.com` (no trailing slash).  
2. **API key** — if the server requires it (`PYX_API_KEY`); otherwise use `null`.  
3. **File `pyx-ai-chat.js`** — copy from the Pyx AI repo into your static assets (or CDN).

The Pyx API already sends **CORS** headers so browsers on your domain can call it.

## Drop this into the page (or your layout template)

Put the **widget container** where you want the chat (sidebar, footer, modal, etc.):

```html
<div id="pyx-chat"></div>
<script src="/static/pyx-ai-chat.js"></script>
<script>
  PyxAIChat.mount({
    el: "#pyx-chat",
    baseUrl: "https://YOUR-PYX-API-HOST",
    apiKey: "YOUR_KEY_OR_NULL",
    title: "Chat",
  });
</script>
```

- Change **`src`** to wherever you host `pyx-ai-chat.js` (same path rules as any other script).  
- Replace **`baseUrl`** and **`apiKey`** with real values (or `apiKey: null` if the API is open).

## Checklist

| Step | Done |
|------|------|
| Deploy / run Pyx API and confirm it’s reachable from the internet (or your staging URL). | ☐ |
| Add `pyx-ai-chat.js` to your site’s static files or CDN. | ☐ |
| Paste the snippet; fix `src`, `baseUrl`, `apiKey`. | ☐ |
| Test: open the page, type a message, press Send — you should see a bot reply or a moderation message. | ☐ |

## If the chat doesn’t appear

- Open **DevTools → Console** — often `pyx-ai-chat.js` 404 (wrong `src` path).  
- Full troubleshooting: **`PYX_CHAT.md`** in the Pyx AI repo.

## Optional (later)

- **`getContext`** — your backend returns text/snippets so replies can include live data (search, FAQs, etc.). See **`PYX_CHAT.md`**.

---

*Single-file widget: no npm package required unless you bundle it yourself.*
