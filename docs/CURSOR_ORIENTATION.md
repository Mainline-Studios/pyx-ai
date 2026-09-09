# Cursor orientation — Mainline Intelligence & Pyx Assistant

Practical map for shipping work on **Mainline Intelligence (MI)** and **Pyx Assistant (PA)** in [Mainline-Studios/pyx-ai](https://github.com/Mainline-Studios/pyx-ai). Live site: [pyx-ai.web.app](https://pyx-ai.web.app) (homepage redirects to `/mainlineintelligence`).

This is not a whole-repo tour. Other Pyx surfaces (Talk, Code, Studio, desktop packaging, the moderator engine internals) appear only as context when these two products call them.

Companion product write-up: [`docs/pyx-assistant.md`](pyx-assistant.md).

---

## What they are today

**Mainline Intelligence** is the public umbrella for the “new wave of Pyx”:

| Product | Status in this repo | What it actually is |
|---------|---------------------|---------------------|
| **MI Moderator** | Live | HTTP wrapper around `PyxAI.score()` with a 0–1000 severity scale. Try-it + integrate snippet on the MI home page. |
| **MARII** | First public beta | “Mainline Artificial Realtime Instant Intelligence.” Brand for local-first, realtime answers. **No cloud LLM.** |
| **MCI** | Marketing only | “Mainline Conversational Intelligence — coming soon.” No implementation in this repo. |

**Pyx Assistant** is the first public MARII surface: a voice/text beta at `/betas/pyxassistant`. It answers from an on-device knowledge pack, regex SLU, math, live sports/weather, and high-confidence Wikipedia — not Groq, not Pyx Talk, not Workers AI.

**Announcer** (`/betas/announcer`) is a sibling MARII beta (live MLB play-by-play, shared voice/CSS). Out of scope unless a change is shared with the assistant.

---

## How they sit in pyx-ai (context only)

```
Browser  →  Firebase Hosting (public/)
              ├─ /mainlineintelligence/*     static MI site
              ├─ /betas/pyxassistant         Pyx Assistant (static JS)
              ├─ /moderator/**, /api/moderator/**  ──rewrite──► Cloud Run pyxaiapi (app.py)
              └─ /api/marii/**, /api/mi/**          ──rewrite──► Cloud Run pyxaiapi

Mailing form  →  Cloudflare Worker mi-mailing (Resend + KV)
                 (Flask /api/mi/mailing/subscribe still exists; the live form does not use it)

Pyx Assistant  →  in-browser only
                 + public APIs: Wikipedia, Open-Meteo, MLB Stats, ESPN, Sound of Text
                 + /api/moderator/check exists for the MI try-it; PA does not call it
                 + /api/marii/ask and workers/marii-ask  (both 501; boost retired)
```

The rest of the repo is the older Pyx stack: `Pyx_ai_moderator.py` (trainable filter), `app.py` (Cloud Run `pyxaiapi` — Talk, Code, Speak, Studio, …), `pyx_server.py` / `pyx_serverless.py` / `functions/` (narrow score-only HTTP, largely superseded by Cloud Run), `public/` studio apps, `packaging/` desktop. MI Moderator is the only MI/PA feature that still depends on that backend.

Production Hosting rewrites in `firebase.json` send API paths to Cloud Run service **`pyxaiapi`** in `us-central1`, not to the Python Cloud Function `pyxscore` in `functions/main.py`. `DEPLOY_FIREBASE.md` still describes the Functions path; live MI traffic does not.

---

## Mainline Intelligence

### Entrypoints

| Path | File | Role |
|------|------|------|
| `/` | `public/index.html` | Instant redirect to `/mainlineintelligence`. |
| `/mainlineintelligence` | `public/mainlineintelligence/index.html` | MI home: hero, MARII/MCI copy, mailing form, moderator try-it, integrate snippet, “dev’s corner”. |
| `/mainlineintelligence/pyx-assistant` | `public/mainlineintelligence/pyx-assistant/index.html` | About Pyx Assistant (what it is / isn’t). |
| `/mainlineintelligence/pyx-assistant.html` | `public/mainlineintelligence/pyx-assistant.html` | Redirect stub → `/mainlineintelligence/pyx-assistant`. |
| `/mainlineintelligence/newsletters/` | `mi-newsletter-002.html` + `.pdf` (also `mi-newsletter-001.pdf`) | Static newsletter assets. Issue **002** is what the mailing Worker attaches. |
| `/betas` | `public/betas/index.html` | Betas index (Assistant + Announcer). Hosting rewrite → `index.html`. |

Firebase Hosting serves these as static files. Directory URLs `/mainlineintelligence` and `/betas/pyxassistant` are rewritten in `firebase.json` only for the betas; MI home is a folder with `index.html` (works as a directory index).

### MI home — data flow

`public/mainlineintelligence/index.html` is a single self-contained page (inline CSS + JS). Two live client flows:

1. **Moderator try-it** (`#moderator`) — `GET /moderator/check/<urlencoded-text>?threshold=<0–1000>`. Relative URL, so locally you need `app.py` (or a Hosting rewrite to Cloud Run). Response: `{"appropriate": bool, "score": "<0–1000>"}`. Page treats HTML bodies as “Cloud Run / billing down”.
2. **Mailing list** (`#mailing`) — `POST` JSON `{email, source: "mi_site"}` to  
   `window.MI_MAILING_API` or default `https://mi-mailing.mainline-mi.workers.dev/subscribe`.  
   Not the Flask route.

### MI Moderator API (live code)

Implemented in `app.py` (not `functions/main.py`):

| Method | Path | Body / query |
|--------|------|----------------|
| GET | `/moderator/check/<text>` and `/api/moderator/check/<text>` | `?threshold=700` |
| GET / POST | `/moderator/check` and `/api/moderator/check` | GET `?text=&threshold=`; POST JSON `{text, threshold?}` |

`_moderator_check_payload` scores with the process-wide `pyx = PyxAI()` instance, maps `score ∈ [0,1]` → `int(round(score * 1000))`, then `appropriate = score_1000 < threshold`. Default threshold **700** ≈ `BAN_LINE = 0.7` in `Pyx_ai_moderator.py`.

This is the MI-facing contract. The older game/Pixel Place contract is still `POST /score` → `{score, bad, censored}` (letters → `~`). Pixel Place integration notes live in [`PIXEL_PLACE_INTEGRATION.md`](../PIXEL_PLACE_INTEGRATION.md); the MI site documents the 0–1000 `appropriate` shape instead.

Scoring (when you touch moderator behavior): `BAN_LINE`, `TRAINING_GROUNDS_PHRASES`, prefix rules (`phrase...`), session overrides, then a tiny hash-encoded MLP. `POST /ai-decide` trains and may write Firestore; `/moderator/check` does **not** train.

### Mailing — two backends

**Live path (use this):** Cloudflare Worker `workers/mi-mailing/`

- Deployed as `https://mi-mailing.mainline-mi.workers.dev`
- `POST /subscribe` — Resend welcome + optional newsletter PDF; KV binding `SUBSCRIBERS`
- `POST /broadcast` — staff broadcast; header `X-Broadcast-Secret` must match `BROADCAST_SECRET`
- `GET /` or `/health`
- CORS allowlist: `pyx-ai.web.app`, `pyx-ai.firebaseapp.com`, localhost `:5000` / `:8080`
- From / reply / notify defaults: `no-reply@pixelplaceofficial.com`, `support@pixelplaceofficial.com`

**Legacy / Cloud Run path (still in tree):** `mi_mailing.py` + `app.py` routes `/api/mi/mailing/subscribe` and `/mi/mailing/subscribe`

- Stores subscribers in `data/mi_mailing/subscribers.json` (gitignored `data/`) and optionally Firestore collection `mi_mailing_list`
- Sends via SMTP (`workforpyx_mail.send_smtp`)
- Hosting still rewrites `/api/mi/**` and `/mi/mailing/**` to Cloud Run

These lists do **not** automatically stay in sync. The MI page error copy still mentions Cloud Run even though the form posts to the Worker.

### MARII ask (retired)

Compatibility only — do not build new features on this:

- `app.py` `POST /api/marii/ask` and `/marii/ask` → **501** `{error: "MARII does not use cloud AI", ai: false}`
- `workers/marii-ask/` — same 501; README says deploy is optional

---

## Pyx Assistant

### Entrypoints

| URL | File |
|-----|------|
| `/betas/pyxassistant` | `public/betas/pyxassistant/index.html` (Hosting rewrite) |
| Deep link | `/betas/pyxassistant?q=weather%20in%20chicago` — `PyxHandoff` / `onQuery` runs after voice boot |
| About | `/mainlineintelligence/pyx-assistant` |
| Tests | `public/betas/pyxassistant/pyx-assistant-slu.test.js` |
| KB generator | `scripts/build-pyx-assistant-kb.js` → `public/betas/pyxassistant/kb/pyx-assistant-kb.json` |

Script load order in `index.html` (manual `?v=` cache-bust — bump when you ship JS):

`pyx-handoff.js` → i18n → slu → math → kb → learn → cookies → sports → weather → wiki → **pyx-assistant.js** → voice

### Reply pipeline (live)

`handleUserText` → `localReply` in `public/betas/pyxassistant/pyx-assistant.js`. Order:

1. **Learn / profile** — `pyx-assistant-learn.js` (on-device softmax + name/likes). May answer “what’s my name” etc.
2. **Math** — `pyx-assistant-math.js` if `looksMath`
3. **SLU** — `pyx-assistant-slu.js` regex intents (`greet`, `marii`, `mi`, `weather`, `sports`, `joke`, …). Local handlers return a reply or set `useWeb` / `special` (`__JOKE__`, …)
4. **Weather** — Open-Meteo geocode + forecast (`pyx-assistant-weather.js`)
5. **Sports** — MLB Stats API + ESPN scoreboards (`pyx-assistant-sports.js`); may paint `#fieldSim`
6. **KB retrieve** — keyword index over the pack, threshold 0.62, optional learn priors (`kb.probe` reports hit / low-score / no-match / empty-pack)
7. **Wikipedia** — only if `looksWikiWorthy` (`pyx-assistant-wiki.js`); high title-match bar; on-screen reply vs shorter `speak` text
8. **Honest miss** — `kb.honestFallback` when retrieve is below threshold or the pack is missing (on-screen chip + status; no silent bluff). Cloud MARII boost is retired; no `askMarii()`

PA does **not** call `/api/moderator/check`. The MI home try-it is the only first-party UI that does.

Voice: Web Speech STT + Sound of Text neural TTS (default `en-GB`); optional on-device Kokoro from jsDelivr. Chat is locked until `voiceReady` (or skip). Persistence: `localStorage` key `pyx.assistant.v3` + cookies (`pyx-assistant-cookies.js`, path `/betas/pyxassistant`).

Settings still have a **hidden, disabled** “MARII is local-only” checkbox (retired boost control); load() forces `mariiBoost = false`.

### File map

| File | Role |
|------|------|
| `pyx-assistant.js` | Controller, pipeline, UI, KB/voice boot |
| `pyx-assistant-slu.js` | Normalize → intent + slots → local resolve |
| `pyx-assistant-kb.js` | Load / retrieve / probe / specials / honest miss fallback |
| `pyx-assistant-math.js` | Expressions, word numbers, unit conversion |
| `pyx-assistant-learn.js` | Preference model + profile |
| `pyx-assistant-cookies.js` | Chat + model cookies |
| `pyx-assistant-sports.js` | MLB / ESPN + field board |
| `pyx-assistant-weather.js` | Open-Meteo |
| `pyx-assistant-wiki.js` | Wikipedia blurbs |
| `pyx-assistant-voice.js` | STT / TTS / Kokoro |
| `pyx-assistant-i18n.js` | UI strings (en, es, fr, de, ja, zh) |
| `pyx-assistant.css` + `pyx-assistant-theme-calm-contrast.css` | Themes (aurora, blush, mint, …) |
| `kb/pyx-assistant-kb.json` | Generated pack (~1388 records). **Gitignored** by root `*.json`. |
| `scripts/build-pyx-assistant-kb.js` | Source of truth for the pack |
| `public/js/pyx-handoff.js` | Studio deep-link bus (`?q=` / incoming text) |

### External APIs the assistant calls (no keys in this repo)

- `https://en.wikipedia.org/w/api.php` and `/api/rest_v1/page/summary/`
- `https://geocoding-api.open-meteo.com` + Open-Meteo forecast
- `https://statsapi.mlb.com/api/v1`
- ESPN public scoreboards (NBA, NFL, NHL, WNBA, CFB, MLS, EPL)
- `https://api.soundoftext.com/sounds`
- `https://cdn.jsdelivr.net/npm/kokoro-js@1.2.1/+esm`

---

## How the two products relate

- MI **home** is the marketing + API landing page; it links to the assistant and hosts the moderator try-it.
- MI **about** (`/mainlineintelligence/pyx-assistant`) is the honest product page; the assistant settings link back here.
- PA **SLU/KB** answer “what is MARII / Mainline Intelligence / MCI” from local strings (see `pyx-assistant-slu.js` and the KB builder FAQ).
- Shared **moderator engine** is unused by PA. The MI try-it is the only first-party UI that calls `/api/moderator/check`.
- Shared **Firebase Hosting** deploy ships both. Moderator scoring changes require a **Cloud Run** (`pyxaiapi`) deploy, not hosting-only.
- Shared **mailing** is MI-only; PA does not subscribe users.

---

## Run and test locally

### 1. Knowledge pack (required for PA)

Root `.gitignore` has `*.json`, so `kb/pyx-assistant-kb.json` is **not in git**. Generate it before a local static serve (assistant still does math/SLU without it; toast: “Knowledge pack didn’t load”).

```bash
npm run build:pyx-assistant-kb
# Wrote 1388 records to public/betas/pyxassistant/kb/pyx-assistant-kb.json
```

Hosting predeploy and `npm run build` both run `build:pyx-assistant-kb`, so a Hosting deploy from a clean tree still ships the pack. `npm run dev` also regenerates it.

### 2. Assistant unit tests

```bash
npm run test:pyx-assistant
# builds the pack, then: node public/betas/pyxassistant/pyx-assistant-slu.test.js
```

Needs the generated JSON (`require("./kb/pyx-assistant-kb.json")`). Covers SLU golden set, math, KB retrieve, learn, cookies, sports heuristics, wiki helpers. No browser.

### 3. Serve the UI

**Static only** (MI pages + Assistant; moderator try-it and `/api/moderator` will fail unless you proxy):

```bash
# from repo root so /betas/... and /mainlineintelligence/... match Hosting paths
python3 -m http.server 8080
# http://127.0.0.1:8080/mainlineintelligence
# http://127.0.0.1:8080/betas/pyxassistant
```

Mailing CORS allows `:8080`. Voice/sports/weather/wiki need network.

**With MI Moderator API** (same origin as the try-it):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # heavy (torch/TTS); Flask + firebase-admin is enough for score/moderator
npm run dev                       # app.py on :8765, prints a one-shot PYX_API_KEY, PYX_DEV_RELAX_AUTH=1
# or: PORT=8765 python3 app.py
# http://127.0.0.1:8765/mainlineintelligence
# http://127.0.0.1:8765/betas/pyxassistant
```

`app.py` serves `public/` last. Browser Talk-style routes skip the API key on loopback; `/moderator/check` still needs a key if `PYX_API_KEY` is set and `PYX_DEV_RELAX_AUTH` is not `1`. `npm run dev` sets relax-auth.

### 4. Mailing Worker (optional)

```bash
cd workers/mi-mailing
npx wrangler dev
# override on the MI page: window.MI_MAILING_API = "http://127.0.0.1:8787/subscribe"
```

### Deploy (what actually ships these products)

| What you changed | Command |
|------------------|---------|
| MI pages, Assistant JS/CSS, newsletters | `npm run deploy:hosting` → Hosting predeploy builds trainer-auth + PA KB, then `firebase deploy --only hosting` (project `pyx-ai`) |
| Moderator scoring / `app.py` check routes | `npm run deploy:api` → Cloud Build + `gcloud run deploy pyxaiapi` |
| Both | `npm run deploy` |
| Mailing Worker | `cd workers/mi-mailing && npx wrangler deploy` |
| marii-ask Worker | `cd workers/marii-ask && npx wrangler deploy` (optional; 501 only) |

`firebase.json` Hosting: `public/`, HTML `Cache-Control: max-age=0`, `/betas/**/*.{js,css,json}` also no-cache. Assistant still uses manual `?v=` query strings.

---

## Env / config names (no values)

**Pyx Assistant has no server env.** Client-only. Optional page override: `window.MI_MAILING_API` (MI home, not the assistant).

**MI mailing Worker** (`workers/mi-mailing`):

| Name | Where |
|------|--------|
| `RESEND_API_KEY` | Wrangler secret (required to send) |
| `RESEND_FROM` | Optional var (default From header) |
| `NEWSLETTER_PDF_URL` | Optional; default Hosting URL for issue 002 PDF |
| `BROADCAST_SECRET` | Secret; required for `POST /broadcast` |
| `SUBSCRIBERS` | KV namespace binding (id in `wrangler.toml`) |

**Flask mailing** (`mi_mailing.py`) — only if you hit Cloud Run `/api/mi/mailing/subscribe`. Prefer `MI_*`; falls back to shared Pyx SMTP names:

`MI_MAILING_FROM`, `MI_MAILING_SMTP_FROM`, `MI_MAILING_FROM_NAME`, `MI_MAILING_REPLY_TO`, `MI_MAILING_NOTIFY_TO`, `MI_MAILING_SMTP_HOST`, `MI_MAILING_SMTP_PORT`, `MI_MAILING_SMTP_USER`, `MI_MAILING_SMTP_PASS`,  
`PYX_APPLICATION_SMTP_USER`, `PYX_APPLICATION_SMTP_PASS`, `PYX_APPLICATION_SMTP_HOST`, `PYX_APPLICATION_SMTP_PORT`,  
`EMAIL_VERIFICATION_SMTP_USER`, `EMAIL_VERIFICATION_SMTP_PASS`, `EMAIL_VERIFICATION_FROM_APP_PASSWORD`, `EMAIL_VERIFICATION_SMTP_HOST`, `EMAIL_VERIFICATION_SMTP_PORT`,  
`PYX_APP_PUBLIC_URL`, `APP_PUBLIC_URL`

Example names only: `MI_MAILING.example.env`.

**Moderator / Firestore** (Cloud Run, if you change scoring or seed phrases):  
`GOOGLE_APPLICATION_CREDENTIALS`, `FIREBASE_PROJECT_ID`, `GOOGLE_CLOUD_PROJECT`, `FIRESTORE_DATABASE_ID`, plus API gate `PYX_API_KEY` / `PYX_API_KEYS`, `PYX_DEV_RELAX_AUTH`.

**Not used by MI/PA:** `PYX_TALK_*`, `PYX_CODE_*`, `PYX_PIXEL_*`, `PYX_SPEAK_*`, `PYX_WRITE_*` (Talk/Code/Pyxel).

**Security smell:** `.env.local` is **committed** (keys are `PYX_TALK_*` local-model settings). `.gitignore` lists `.env` but not `.env.local`. Do not add secrets there. `secret.py` is also tracked; it is a joke script, not credentials.

---

## Top 5 high-value fix / cleanup targets (MI + PA)

Grounded in current main — not a whole-repo laundry list.

1. **Knowledge pack is gitignored on purpose.**  
   `*.json` still ignores `public/betas/pyxassistant/kb/pyx-assistant-kb.json`. **Fixed for deploy:** `npm run build` and Hosting predeploy run `build:pyx-assistant-kb`. Local static serve still needs `npm run build:pyx-assistant-kb` (or `npm run dev`).

2. **MARII cloud boost is retired.**  
   Docs and client stubs now say so. `/api/marii/ask` and `workers/marii-ask` remain 501 compatibility shims. Hidden settings checkbox is local-only / disabled. Do not re-enable a cloud LLM on these paths.

3. **Two mailing backends + stale error copy.**  
   The live form posts to the Worker; Flask SMTP + Firestore + `data/mi_mailing/` still exist; Hosting still rewrites `/api/mi/**` to Cloud Run; the join-list JS still talks about Cloud Run when the body is HTML. **Fix:** pick one source of truth (Worker is what production uses), deprecate or proxy the Flask path, and fix the error string.

4. **Manual `?v=` cache-bust on every assistant asset.**  
   `index.html` pins `?v=11` … `?v=22` per file; KB fetch uses `?v=3`. Easy to ship JS that browsers still cache, or bump the wrong file. Hosting already sets `must-revalidate` on `/betas/**/*.js`. Left in place for local static servers that do not send those headers. **Fix later:** one shared build stamp, or drop query strings if you only ship via Hosting.

---

## Quick “where do I edit?”

| Task | Start here |
|------|------------|
| MI copy, try-it, mailing form | `public/mainlineintelligence/index.html` |
| About Pyx Assistant | `public/mainlineintelligence/pyx-assistant/index.html` |
| Newsletter HTML/PDF | `public/mainlineintelligence/newsletters/` |
| Assistant pipeline / UI | `public/betas/pyxassistant/pyx-assistant.js` + `index.html` |
| Intents / identity answers | `pyx-assistant-slu.js`, `pyx-assistant-i18n.js` |
| Jokes / facts / FAQ pack | `scripts/build-pyx-assistant-kb.js` then regenerate JSON |
| Sports / weather / wiki | matching `pyx-assistant-*.js` |
| Voice | `pyx-assistant-voice.js` |
| Moderator API shape / threshold | `app.py` `_moderator_check_payload`; phrases in `Pyx_ai_moderator.py` |
| Welcome email (live) | `workers/mi-mailing/src/index.js` |
| Welcome email (unused Flask) | `mi_mailing.py` |
