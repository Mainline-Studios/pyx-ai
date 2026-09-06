# MARII ask Worker

Cloudflare Workers AI backend for `POST /ask` (and `/api/marii/ask`).

**Does not use Groq.** Inference runs on the Workers AI binding (`env.AI`).

```bash
cd workers/marii-ask
npm install
npx wrangler deploy
```

Clients (Pyx Assistant boost + Announcer ask) call:

`https://marii-ask.mainline-mi.workers.dev/ask`

Optional: set `MARII_MODEL` (Workers AI model id). Default: `@cf/meta/llama-3.1-8b-instruct`.
