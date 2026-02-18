# Pyx AI

An open-source kid-friendly trainable neural network that learns words, phrases, and game ideas. Easy to edit, easy to train. Pyx filters content so inappropriate content is banned and safe content is allowed.

Originally made by **Mainline Studios** for Pixel Place. The studio maintains and extends this project.

## Content Filter

- **Above the line** (scores ≥ threshold) = **INAPPROPRIATE** — banned
- **Below the line** (scores < threshold) = **SAFE** — allowed

Change the threshold by editing `BAN_LINE` in `pyx_ai.py`. Default: `0.7`.

## Built-in Training (Training Grounds)

Pyx includes a large built-in phrase list in `pyx_ai.py`:

- **Context-aware** — The same word can be safe or bad depending on the full phrase
- **Pro-LGBTQ+** — Supportive and identity phrases are safe; insults and put-downs are bad
- **Names & figures** — Inappropriate or controversial public figures are in the bad list
- **Slurs, profanity, harm** — Racial, disability, anti-LGBTQ, sexist slurs; profanity; self-harm; harassment; sexual content; drugs; violence; scams; dangerous challenges
- **Safe phrases** — Kid-friendly phrases for gaming, school, slang, food, sports, pets, family, tech, holidays, health, and more

Edit `TRAINING_GROUNDS_PHRASES` in `pyx_ai.py` to add or remove entries. Pyx trains on this list every time it starts.

- **Prefix rules:** A phrase that ends with `...` acts as a wildcard: that prefix plus *anything* after it gets the same label (banned or allowed).

## Firestore (optional)

Phrases can sync to [Firebase Firestore](https://console.firebase.google.com/project/pyx-ai/firestore/databases/default/data) so the cloud DB stays updated when users override or train.

1. **Create the database:** If you haven’t already, [create a Firestore database](https://console.cloud.google.com/datastore/setup?project=pyx-ai) for project `pyx-ai` (choose Native mode). Without this, the seed will report “database does not exist”.
2. **Use a virtual environment** (recommended on macOS/Homebrew Python):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Credentials:** In Firebase Console → Project settings → Service accounts, generate a new private key. Save it as `firebase-key.json` in the project root (or set `GOOGLE_APPLICATION_CREDENTIALS` to its path).
4. **Seed once:** Run `python3 pyx_ai.py seed-firestore` (or `python pyx_ai.py seed-firestore` with the venv activated) to upload the built-in phrases to the `phrases` collection.
5. **Ongoing:** When anyone uses **safe**, **bad**, **override**, or adds words/phrases in code, Pyx writes to Firestore so the database stays in sync.

Without credentials, Pyx runs as before (no Firestore).

**If seed says "database (default) does not exist":**  
1. Run: `python3 pyx_ai.py firestore-check` — it prints your key’s **project ID** and the exact link.  
2. Open that link in your browser (same Google account as Firebase).  
3. Click **Create database** → **Firestore in Native mode** → choose a location → Create.  
4. Wait 1–2 minutes, then run `python3 pyx_ai.py seed-firestore` again.

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
python3 pyx_ai.py
```
(With venv active you can use `python pyx_ai.py`.)

**Other commands**
- `python3 pyx_ai.py seed-firestore` — upload built-in phrases to Firestore (one-time, or to refresh).
- `python3 pyx_ai.py firestore-check` — show project ID and link to create the Firestore database.

**In the app:** Enter a phrase, then choose **safe**, **bad**, **AI decide**, or **override**. Use `list`, `score <text>`, or `quit`.

## Using Pyx in Your Code

Import `PyxAI` from `pyx_ai`. You can add words and phrases, train with safe/bad feedback, use `ai_decide` to classify, use `set_label` to override, and call `score` to check if text is above or below the ban line. Save with `save()`.

## Editing the Code

- **Above the line** (~lines 1–125): Core engine — edit only if you know what you're doing
- **Below the line** (~lines 127+): Settings, `BAN_LINE`, `TRAINING_GROUNDS_PHRASES`, and app logic — edit freely

Customize `learning_rate`, `hidden_size`, `BAN_LINE`, `DATA_DIR`, and `TRAINING_GROUNDS_PHRASES` to change how Pyx learns and what it allows.

---

**Mainline Studios** — Maintained for Pixel Place. Contributions welcome.
