# Pyx AI

A kid-friendly trainable neural network that learns **words**, **phrases**, and **game ideas**. Easy to edit, easy to train.

## Content Filter (for most kids)

- **Above the line** (scores ≥ threshold) = **INAPPROPRIATE** — banned, Pyx won't use it
- **Below the line** (scores < threshold) = **SAFE + borderline** — allowed, Pyx learns and uses it

Edit `BAN_LINE` in `pyx_ai.py` (around line 95) to change the threshold. Default: `0.7`.

## Quick Start

```bash
cd pyx_ai
python pyx_ai.py
```

### UI Flow

1. Enter a phrase
2. Choose: **[s]afe** / **[b]ad** / **[a]i decide** / **[os] override safe** / **[ob] override bad**
   - **Safe** – You say it's OK; trains and adds
   - **Bad** – You say it's inappropriate; trains and removes
   - **AI decide** – AI scores it and adds only if safe; you can override later
   - **Override Safe / Bad** – Change the label (e.g. AI said bad, you say safe)
3. Other: `list` | `score <text>` | `quit`

## As a Module

```python
from pyx_ai import PyxAI

pyx = PyxAI()

# Add content
pyx.add_word("cool")
pyx.add_phrase("that sounds fun")
pyx.add_game_idea("roguelike dungeon crawler")

# Train with feedback (safe=True for kid-friendly, safe=False for inappropriate)
pyx.train("pizza is great", safe=True)
pyx.train("explicit content", safe=False)

# Let AI decide and add if safe
safe, score = pyx.ai_decide("pizza is cool")
print(f"AI: {'SAFE' if safe else 'BAD'} ({score:.2f})")

# Manual label or override
pyx.set_label("explicit stuff", safe=False)

# Check scores (above BAN_LINE = inappropriate)
print(pyx.score("pizza"))  # 0.0-1.0

# Get learned content
print(pyx.get_words(), pyx.get_phrases(), pyx.get_game_ideas())

pyx.save()
```

## Easy to Edit

The file has clear sections:

1. **Above the line** (~lines 1–90): Core engine — avoid editing unless you understand the neural net
2. **Below the line** (~lines 95+): Your settings, `BAN_LINE`, training logic, and interactive commands — edit freely!

Adjust `learning_rate`, `hidden_size`, `BAN_LINE`, and `DATA_DIR` to customize how Pyx learns.
