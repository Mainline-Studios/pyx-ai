# Pyx AI

An open-source kid-friendly trainable neural network that learns words, phrases, and game ideas. Easy to edit, easy to train. Pyx filters content so inappropriate content is banned and safe content is allowed.
Originally made for Pixel Place.

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

## Quick Start

Run the interactive app:

```bash
python pyx_ai.py
```

Enter a phrase, then choose **safe**, **bad**, **AI decide**, or **override**. Use `list`, `score <text>`, or `quit` for other actions.

## Using Pyx in Your Code

Import `PyxAI` from `pyx_ai`. You can add words and phrases, train with safe/bad feedback, use `ai_decide` to classify, use `set_label` to override, and call `score` to check if text is above or below the ban line. Save with `save()`.

## Editing the Code

- **Above the line** (~lines 1–125): Core engine — edit only if you know what you're doing
- **Below the line** (~lines 127+): Settings, `BAN_LINE`, `TRAINING_GROUNDS_PHRASES`, and app logic — edit freely

Customize `learning_rate`, `hidden_size`, `BAN_LINE`, `DATA_DIR`, and `TRAINING_GROUNDS_PHRASES` to change how Pyx learns and what it allows.
