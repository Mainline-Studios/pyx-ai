# MARII ask Worker

**Retired compatibility shim.** MARII does not run an LLM here (no Groq, no Workers AI).

`POST /ask` and `POST /api/marii/ask` return **501** `{ error: "MARII does not use cloud AI", ai: false }`. Deploy is optional.

Client surfaces (Pyx Assistant, Announcer) answer locally from KB / live feeds. Do not build new features on this Worker.

```bash
cd workers/marii-ask
npx wrangler deploy
```
