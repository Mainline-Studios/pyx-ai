# Pyx Assistant

Voice-first beta at `/betas/pyxassistant`. The betas index lives at `/betas`. Branded **powered by MARII** (Mainline Artificial Realtime Instant Intelligence) — local-first today, with an optional MARII cloud boost when the notebook misses.

On-device by default: no Pyx Talk required. Pastel swirls stay. The name is **Pyx Assistant**.

## Architecture

```
Mic / type
  → Web Speech STT (browser)  [or typed text]
  → SLU (regex intents) + math + local KB retrieval (+ sports / weather APIs)
  → optional MARII cloud ask when confidence is low (timeout ~1.5s, then warm fallback)
  → Sound of Text neural TTS (default) or on-device Kokoro when selected/loaded
```

| File | Role |
|------|------|
| `kb/pyx-assistant-kb.json` | Local reply pack (jokes, facts, trivia, riddles, quotes, how-tos, definitions, small talk, MI/MARII FAQ) |
| `pyx-assistant-math.js` | Expression parser, word numbers, unit conversions |
| `pyx-assistant-kb.js` | Keyword retrieval + warm fallback |
| `pyx-assistant-slu.js` | Intents (local handlers; optional MARII boost from the UI layer) |
| `pyx-assistant-voice.js` | Web Speech STT + Sound of Text TTS + optional Kokoro |
| `pyx-assistant-sports.js` | Live MLB/ESPN scoreboards |
| `pyx-assistant.js` | App controller, reply pipeline, MARII boost toggle |
| `scripts/build-pyx-assistant-kb.js` | Regenerates the knowledge pack |

## Testing

```bash
node public/betas/pyxassistant/pyx-assistant-slu.test.js
node scripts/build-pyx-assistant-kb.js
```

## Deploy

```bash
npx -y firebase-tools@latest deploy --only hosting --project pyx-ai
```
