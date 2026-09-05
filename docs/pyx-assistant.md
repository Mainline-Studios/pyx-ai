# Pyx Assistant

Voice-first beta at `/betas/pyxassistant`. The betas index lives at `/betas`.

The look is Siri-inspired pastel swirls. The product name is **Pyx Assistant**.

## Architecture

```
Browser UI  →  SLU (intent + slots)  →  local reply
                                 ↘  POST /api/talk  (Pyx cloud LLM)
Voice  →  Web Speech API (ASR)   →  SLU
TTS    ←  Speech Synthesis
```

Modules (no build step):

| File | Role |
|------|------|
| `public/betas/index.html` | Betas list |
| `public/betas/pyxassistant/index.html` | Assistant shell |
| `pyx-assistant.css` | Pastel themes + orb |
| `pyx-assistant-i18n.js` | EN/ES/FR/DE/JA/ZH |
| `pyx-assistant-slu.js` | Intent classifier + local handlers |
| `pyx-assistant.js` | Voice, Talk API, Slack/Discord webhooks |

Add an intent in `pyx-assistant-slu.js` (`RULES` + `resolve` switch) and a golden utterance in `GOLDEN`.

## Testing

```bash
node public/betas/pyxassistant/pyx-assistant-slu.test.js
```

## Deploy

Hosting rewrites map `/betas` and `/betas/pyxassistant` to their `index.html` files.

```bash
npx -y firebase-tools@latest deploy --only hosting --project pyx-ai
```
