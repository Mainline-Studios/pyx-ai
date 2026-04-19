# Pyx AI

An open-source kid-friendly trainable neural network that learns words, phrases, and game ideas. Easy to edit, easy to train. Pyx filters content so inappropriate content is banned and safe content is allowed.

Originally made by **Mainline Studios** for Pixel Place. The studio maintains and extends this project.

## Content Filter

- **Above the line** (scores ≥ threshold) = **INAPPROPRIATE** — banned
- **Below the line** (scores < threshold) = **SAFE** — allowed

Change the threshold by editing `BAN_LINE` in `Pyx_ai_moderator.py`. Default: `0.7`.

## Built-in Training (Training Grounds)

Pyx includes a large built-in phrase list in `Pyx_ai_moderator.py`:

- **Context-aware** — The same word can be safe or bad depending on the full phrase
- **Pro-LGBTQ+** — Supportive and identity phrases are safe; insults and put-downs are bad
- **Names & figures** — Inappropriate or controversial public figures are in the bad list
- **Slurs, profanity, harm** — Racial, disability, anti-LGBTQ, sexist slurs; profanity; self-harm; harassment; sexual content; drugs; violence; scams; dangerous challenges
- **Safe phrases** — Kid-friendly phrases for gaming, school, slang, food, sports, pets, family, tech, holidays, health, and more

Edit `TRAINING_GROUNDS_PHRASES` in `Pyx_ai_moderator.py` to add or remove entries. Pyx trains on this list every time it starts.

- **Prefix rules:** A phrase that ends with `...` acts as a wildcard: that prefix plus *anything* after it gets the same label (banned or allowed).

- **PYX FOR YOUR APP:**
   We have a dedicated provider running a Pyx AI API; as THIS code is stripped down. Our APIs have updated word databases based on slang and trends, and it is more comprehensive. Use this for a great out of the box moderator; and this for full creative access. Also, we are working on a way to add **Pyx Talk, Pyx Code, and Pyxel Images** to our API list. Thanks for considering us for your app! 

## Getting updated changes

On any machine that already has the repo:

```bash
cd /path/to/pyx-ai
git pull
```

If you have local changes, use `git pull --rebase` or commit them first. To start fresh: clone again from the repo (e.g. `git clone https://github.com/Mainline-Studios/pyx-ai.git`).

## How to run

**1. Go to the project folder**
```bash
cd /path/to/pyx-ai
```
(Use your actual path, e.g. `cd ~/pyx-ai` or `cd /Users/you/pyx-ai`.)

**2. Use the virtual environment** (recommended on Mac/Homebrew)
```bash
source .venv/bin/activate
```
Windows: `.venv\Scripts\activate`  
First time? Create it: `python3 -m venv .venv` then `source .venv/bin/activate` and `pip install -r requirements.txt`

**3. Run the app**
```bash
python3 Pyx_ai_moderator.py
```
(With venv active you can use `python Pyx_ai_moderator.py`.)

**Other commands**
- `python3 Pyx_ai_moderator.py seed-firestore` — upload built-in phrases to Firestore (one-time, or to refresh).
- `python3 Pyx_ai_moderator.py firestore-check` — show project ID and link to create the Firestore database.
- `python3 pyx_server.py [--port 8765] [--host 0.0.0.0]` — run Pyx as an HTTP service so other apps can call it: `POST /score` with `{"text": "..."}` returns `{"score", "bad", "censored"}`; `GET /health` for liveness.
- **No server 24/7:** Deploy to **Firebase Hosting + Cloud Functions** — see [DEPLOY_FIREBASE.md](DEPLOY_FIREBASE.md) (Pyx at `https://YOUR_PROJECT.web.app/api/score`). Or deploy `pyx_serverless.handler` as AWS Lambda; see [DEPLOY_SERVERLESS.md](DEPLOY_SERVERLESS.md). Pyx runs only when a request comes in.

**In the app:** Enter a phrase, then choose **safe**, **bad**, **AI decide**, or **override**. Use `list`, `score <text>`, or `quit`.

## Using Pyx in Your Code

Import `PyxAI` from `Pyx_ai_moderator`. You can add words and phrases, train with safe/bad feedback, use `ai_decide` to classify, use `set_label` to override, and call `score` to check if text is above or below the ban line. Save with `save()`.

## Editing the Code

- **Above the line** (~lines 1–125): Core engine — edit only if you know what you're doing
- **Below the line** (~lines 127+): Settings, `BAN_LINE`, `TRAINING_GROUNDS_PHRASES`, and app logic — edit freely

Customize `learning_rate`, `hidden_size`, `BAN_LINE`, `DATA_DIR`, and `TRAINING_GROUNDS_PHRASES` to change how Pyx learns and what it allows.

---

**Mainline Studios** — Maintained for Pixel Place. Contributions welcome.
