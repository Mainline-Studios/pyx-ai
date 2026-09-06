# MARII ask Worker

Cloudflare fallback for `POST /ask` when Cloud Run `/api/marii/ask` is unavailable.

```bash
cd workers/marii-ask
npm install
npx wrangler secret put GROQ_API_KEY
npm run deploy
```

PA calls `/api/marii/ask` first, then `https://marii-ask.mainline-mi.workers.dev/ask`.
