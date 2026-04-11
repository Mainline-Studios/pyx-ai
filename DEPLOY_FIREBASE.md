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

## 7. Trainer page (Firebase Auth)

The hosted trainer (`/pyx-trainer-auth.html` → `/pyx-firebase-trainer.html`) uses **Firebase Authentication** (email/password).

1. In [Firebase Console](https://console.firebase.google.com/) → **Authentication** → **Sign-in method**, enable **Email/Password**.
2. Under **Authentication** → **Users**, add accounts for people who may use the trainer (or use your own flow to create users).
3. Under **Authentication** → **Settings** → **Authorized domains**, ensure **`pyx-ai.web.app`** and **`pyx-ai.firebaseapp.com`** are listed (they usually are by default for the same project).

Optional email allowlist: edit `public/firebase-config.js` and set `window.__PYX_TRAINER_ALLOWED_EMAILS__` to a list of lowercase emails. Leave it `[]` to allow any signed-in user.

Web SDK config lives in **`public/firebase-config.js`** (same values as the “Pyx AI API” web app in Project settings). Redeploy Hosting after changing it.

---

## Troubleshooting

- **Import error for Pyx_ai_moderator:** Make sure you ran the copy step: `cp Pyx_ai_moderator.py functions/` (and `cp -r data functions/` if you use it).
- **Function not found / 404:** Redeploy with `firebase deploy --only functions,hosting` and confirm the rewrite in `firebase.json` points to `pyxscore`.
- **CORS:** The function allows all origins (`cors_origins="*"`). To restrict, edit `functions/main.py` and change `options.CorsOptions(cors_origins="*", ...)` to your domain(s).
- **Cold start:** The first request after idle can take a few seconds while the function loads. Later requests are fast.
