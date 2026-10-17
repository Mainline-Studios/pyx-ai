# MARII ask Worker

Public JSON API. Same local pipeline as Pyx Assistant (SLU, math, sports, weather, KB, wiki). No Groq, no Workers AI, no Cloud Run.

**Live:** `https://marii-ask.mainline-mi.workers.dev`

## Terminal (any machine)

```bash
curl -sS "https://marii-ask.mainline-mi.workers.dev/how%20is%20judge"
curl -sS "https://marii-ask.mainline-mi.workers.dev/what%20is%20popcorn"
curl -sS "https://marii-ask.mainline-mi.workers.dev/15%25%20of%2080"
```

Spaces → `%20`. Percent → `%25`.

Same path on the site (Hosting 302 → this Worker):

```bash
curl -sSL "https://pyx-ai.web.app/api/marii/ask/how%20is%20judge"
```

`-L` follows the redirect. The workers.dev URL does not need `-L`.

## Deploy

```bash
cd workers/marii-ask
npx wrangler deploy
```

`POST` stays 501 (no cloud LLM).
