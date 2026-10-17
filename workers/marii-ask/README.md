# MARII ask Worker

GET `/api/marii/ask/<request>` (also `/marii/ask/<request>` and `/ask/<request>`) returns `{ ok, request, reply, echo }` from a local helper. No Groq, no Workers AI.

`POST /ask` and `POST /api/marii/ask` still return **501** `{ error: "MARII does not use cloud AI", ai: false }`.

The live site path is Hosting-only (`/marii-ask.html` + `public/js/marii-ask.js`). This Worker is optional.

```bash
cd workers/marii-ask
npx wrangler deploy
```
