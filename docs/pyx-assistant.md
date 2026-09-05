# Pyx Assistant

Voice-first beta at `/betas/pyxassistant`. The betas index lives at `/betas`.

On-device only: no Pyx Talk. Pastel swirls stay. The name is **Pyx Assistant**.

## Architecture

```
Mic → VAD → Whisper tiny.en (transformers.js) → SLU + math + KB retrieval → Kokoro TTS
                                                                    ↘ keep listening
Tap orb: start session / interrupt speech / pause
```

| File | Role |
|------|------|
| `kb/pyx-assistant-kb.json` | 1,300+ local replies (jokes, facts, trivia, riddles, quotes, how-tos, definitions, small talk) |
| `pyx-assistant-math.js` | Expression parser, word numbers, unit conversions |
| `pyx-assistant-kb.js` | Keyword retrieval + warm fallback |
| `pyx-assistant-slu.js` | Intents (never calls Talk) |
| `pyx-assistant-voice.js` | Whisper STT + Kokoro TTS + continuous listen |
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
