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

### Commands

| Command | Example | What it does |
|---------|---------|--------------|
| `add words <text>` | `add words awesome` | Add a word |
| `add phrases <text>` | `add phrases hello world` | Add a phrase |
| `add game_ideas <text>` | `add game_ideas space shooter` | Add a game idea |
| `train <text> <safe\|bad>` | `train pizza safe` | Train: safe=OK for kids, bad=inappropriate |
| `score <text>` | `score pizza` | See score (SAFE or INAPPROPRIATE) |
| `list` | `list` | Show all learned words, phrases, game ideas |
| `quit` | `quit` | Save and exit |

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
