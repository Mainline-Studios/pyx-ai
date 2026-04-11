# Deploy Pyx to Firebase Hosting + Cloud Functions

Use **Firebase Hosting** and **Cloud Functions** so Pixel Place can call Pyx at your Hosting URL. No server running 24/7 — the function runs only when a request comes in.

**Result:** `https://YOUR_PROJECT.web.app/api/score` accepts POST `{"text": "..."}` and returns `{"score", "bad", "censored"}`.

---

## 1. Prerequisites

- [Firebase CLI](https://firebase.google.com/docs/cli) installed (`npm install -g firebase-tools`)
- A Firebase project (create one at [Firebase Console](https://console.firebase.google.com/) or use an existing one, e.g. **pyx-ai**)
- Python 3.10+ (for local testing; deploy uses Cloud runtime)

---

## 2. Copy Pyx into `functions/` (required before deploy)

Firebase deploys only the contents of `functions/`. Pyx lives in `Pyx_ai_moderator.py` at the repo root, so copy it (and optional `data/`) into `functions/` before deploying:

```bash
cd /path/to/pyx-ai

# Copy Pyx and optional data so they are deployed with the function
cp Pyx_ai_moderator.py functions/
cp -r data functions/   # optional, if you use local memory
```

Run this again whenever you change `Pyx_ai_moderator.py` and redeploy.

---

## 3. Link your Firebase project (if not already)

From the repo root:

```bash
firebase login
firebase use YOUR_PROJECT_ID
```

If you haven’t initialized Firebase in this repo yet:

```bash
firebase init
```

Choose **Functions** and **Hosting**. Use the existing `functions/` and `public/` folders when prompted (don’t overwrite `main.py`). If you already have `firebase.json` from this repo, you can skip `firebase init` and just run `firebase use YOUR_PROJECT_ID`.

---

## 4. Deploy

```bash
firebase deploy --only functions,hosting
```

- **Functions:** deploys the `pyxscore` Cloud Function (Python).
- **Hosting:** deploys `public/` and the rewrite so `/api/score` goes to `pyxscore`.

Your API base URL will be:

- `https://YOUR_PROJECT_ID.web.app`
- or `https://YOUR_PROJECT_ID.firebaseapp.com`

---

## 5. Call the API

**Score text (POST):**

```bash
curl -X POST https://YOUR_PROJECT_ID.web.app/api/score \
  -H "Content-Type: application/json" \
  -d '{"text": "hello world"}'
```

Response:

```json
{"score": 0.12, "bad": false, "censored": "hello world"}
```

If the content is bad, `bad` is `true` and `censored` has letters replaced with `~`.

**Health (GET):**

```bash
curl https://YOUR_PROJECT_ID.web.app/api/score
```

Returns `{"status": "ok", "service": "pyx"}`.

---

## 6. Use from Pixel Place

Use your Hosting URL as the Pyx API base:

- **POST** `https://YOUR_PROJECT_ID.web.app/api/score`  
  Body: `{"text": "user or AI message"}`  
  If the response has `bad: true`, show `censored` (or block). If `bad: false`, show the original text.

No server to run 24/7 — Firebase runs the function only when it’s called.

---

## 7. Trainer page (password only)

The trainer UI (`/pyx-firebase-trainer.html`) is locked behind **`/pyx-trainer-auth.html`**, which asks for **one password** (no email, no Firebase Auth UI).

1. Choose a strong password and put it in **`TRAINER_GATE_PASSWORD.txt`** in the repo root as **a single line** (file is gitignored). See **`TRAINER_GATE_PASSWORD.example.txt`**.
2. Before each Hosting deploy, Firebase runs **`npm run build:trainer-auth`**, which writes **`public/pyx-trainer-auth.html`** containing only a **SHA-256 hash** of that password (the plaintext is not committed).
3. Alternatively set **`PYX_TRAINER_PASSWORD`** in the environment and run `python3 scripts/build_trainer_auth.py` manually.

To rotate the password: change the line in `TRAINER_GATE_PASSWORD.txt`, run `npm run build:trainer-auth`, redeploy.

---

## Troubleshooting

- **Import error for Pyx_ai_moderator:** Make sure you ran the copy step: `cp Pyx_ai_moderator.py functions/` (and `cp -r data functions/` if you use it).
- **Function not found / 404:** Redeploy with `firebase deploy --only functions,hosting` and confirm the rewrite in `firebase.json` points to `pyxscore`.
- **CORS:** The function allows all origins (`cors_origins="*"`). To restrict, edit `functions/main.py` and change `options.CorsOptions(cors_origins="*", ...)` to your domain(s).
- **Cold start:** The first request after idle can take a few seconds while the function loads. Later requests are fast.
