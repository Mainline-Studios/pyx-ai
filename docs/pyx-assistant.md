# Pyx Assistant

Voice-first beta at `/betas/pyxassistant`. The betas index lives at `/betas`. Branded **powered by MARII** (Mainline Artificial Realtime Instant Intelligence) — local-first in the browser. **No cloud LLM.** When the notebook misses, the assistant tries live public feeds (weather, sports, high-confidence Wikipedia) and then a warm local fallback.

Pastel swirls stay. The name is **Pyx Assistant**.

Companion agent map: [`docs/CURSOR_ORIENTATION.md`](CURSOR_ORIENTATION.md).

## Architecture

```
Mic / type
  → Web Speech STT (browser)  [or typed text]
  → SLU (regex intents) + math + local KB retrieval
  → live sports / weather / high-confidence Wikipedia
  → warm local fallback
  → Sound of Text neural TTS (default online) or on-device Kokoro when selected/loaded
```

MARII cloud boost is **retired**. `POST /api/marii/ask` and `workers/marii-ask` stay as compatibility shims and return **501**. Do not wire new features to them.

| File | Role |
|------|------|
| `kb/pyx-assistant-kb.json` | Generated local reply pack (jokes, facts, MI/MARII FAQ, …). Gitignored; rebuild before serve/deploy. |
| `pyx-assistant-math.js` | Expression parser, word numbers, unit conversions |
| `pyx-assistant-kb.js` | Keyword retrieval + warm fallback |
| `pyx-assistant-slu.js` | Intents (local handlers only) |
| `pyx-assistant-voice.js` | Web Speech STT + Sound of Text TTS + optional Kokoro |
| `pyx-assistant-sports.js` | Live MLB/ESPN scoreboards |
| `pyx-assistant-weather.js` | Live weather via Open-Meteo |
| `pyx-assistant-wiki.js` | High-confidence Wikipedia blurbs |
| `pyx-assistant.js` | App controller and reply pipeline |
| `scripts/build-pyx-assistant-kb.js` | Source of truth for the knowledge pack |

## Testing

```bash
npm run test:pyx-assistant
# same as: npm run build:pyx-assistant-kb && node public/betas/pyxassistant/pyx-assistant-slu.test.js
```

## Deploy

The pack is gitignored (`*.json`). Hosting predeploy and `npm run build` both run `build:pyx-assistant-kb`, so a clean clone that deploys Hosting still ships the pack.

```bash
npm run build:pyx-assistant-kb   # local static serve
npm run deploy:hosting           # Firebase Hosting (project pyx-ai)
```
