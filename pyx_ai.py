"""
Pyx AI - A Trainable Neural Network (Kid-Friendly)
==================================================
Learn easily. Edit easily. Words, phrases, and game ideas.

CONTENT FILTER (for most kids):
  - ABOVE THE LINE: Inappropriate stuff = BANNED
  - BELOW THE LINE: Safe + borderline things = ALLOWED

STRUCTURE:
  ABOVE THE LINE: Core engine - avoid editing unless you know what you're doing
  BELOW THE LINE: Your stuff - train, customize, add content freely
"""

import json
import math
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# ═══════════════════════════════════════════════════════════════════════════
#                    🚫 DO NOT EDIT ABOVE THIS LINE 🚫
# ═══════════════════════════════════════════════════════════════════════════
# Core Pyx AI neural network engine. Editing may break learning.


class PyxBrain:
    """Core neural network - learns associations through weighted connections."""

    def __init__(self, input_size: int = 64, hidden_size: int = 32, output_size: int = 8):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = 0.15

        # Weights: initialized for fast learning
        self.w1 = [[random.gauss(0, 0.5) for _ in range(hidden_size)] for _ in range(input_size)]
        self.w2 = [[random.gauss(0, 0.5) for _ in range(output_size)] for _ in range(hidden_size)]
        self.b1 = [0.0] * hidden_size
        self.b2 = [0.0] * output_size

    def _sigmoid(self, x: float) -> float:
        return 1 / (1 + math.exp(-max(-500, min(500, x))))

    def _encode(self, text: str, size: int) -> List[float]:
        """Simple hash-based encoding for text."""
        vec = [0.0] * size
        for i, c in enumerate(text[:size * 4]):
            idx = (ord(c) * 31 + i) % size
            vec[idx] = (vec[idx] + 0.3) % 1.0
        return vec

    def forward(self, inputs: List[float]) -> Tuple[List[float], List[float]]:
        """Forward pass through the network."""
        hidden = [
            self._sigmoid(sum(inputs[i] * self.w1[i][j] for i in range(self.input_size)) + self.b1[j])
            for j in range(self.hidden_size)
        ]
        output = [
            self._sigmoid(sum(hidden[j] * self.w2[j][k] for j in range(self.hidden_size)) + self.b2[k])
            for k in range(self.output_size)
        ]
        return hidden, output

    def train_step(self, inputs: List[float], targets: List[float]) -> float:
        """One training step - returns loss."""
        hidden, output = self.forward(inputs)
        # Pad targets if shorter than output
        t = (targets + [targets[-1]] * self.output_size)[:self.output_size]
        errors_out = [o * (1 - o) * (t[k] - o) for k, o in enumerate(output)]
        errors_hidden = [
            h * (1 - h) * sum(errors_out[k] * self.w2[j][k] for k in range(self.output_size))
            for j, h in enumerate(hidden)
        ]
        for i in range(self.hidden_size):
            for k in range(self.output_size):
                self.w2[i][k] += self.learning_rate * errors_out[k] * hidden[i]
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                self.w1[i][j] += self.learning_rate * errors_hidden[j] * inputs[i]
        for k, e in enumerate(errors_out):
            self.b2[k] += self.learning_rate * e
        for j, e in enumerate(errors_hidden):
            self.b1[j] += self.learning_rate * e
        return sum((t[k] - o) ** 2 for k, o in enumerate(output)) / self.output_size

    def predict(self, inputs: List[float]) -> List[float]:
        _, output = self.forward(inputs)
        return output


class PyxMemory:
    """Stores and retrieves learned content with ban-line filtering."""

    def __init__(self, ban_threshold: float = 0.7):
        self.ban_threshold = ban_threshold  # Scores ABOVE this = BANNED
        self.words: Dict[str, float] = {}
        self.phrases: Dict[str, float] = {}
        self.game_ideas: Dict[str, float] = {}

    def is_banned(self, score: float) -> bool:
        """Above the line = banned. Below = allowed."""
        return score >= self.ban_threshold

    def add(self, category: str, text: str, score: float):
        """Add item - only if below ban line."""
        if self.is_banned(score):
            return False
        store = getattr(self, category, None)
        if store is not None:
            store[text] = min(score, self.ban_threshold - 0.01)
            return True
        return False

    def get_allowed(self, category: str) -> Dict[str, float]:
        """Get only items below the ban line."""
        store = getattr(self, category, {})
        return {k: v for k, v in store.items() if not self.is_banned(v)}

    def remove(self, category: str, text: str):
        """Remove an item (for override)."""
        store = getattr(self, category, None)
        if store is not None and text in store:
            del store[text]


# ═══════════════════════════════════════════════════════════════════════════
#                    ✏️ EDIT BELOW THIS LINE ✏️
# ═══════════════════════════════════════════════════════════════════════════
# Train Pyx, add your words/phrases/game ideas, tweak settings.


BAN_LINE = 0.7  # Scores above = inappropriate (banned). Below = safe + borderline (allowed).
DATA_DIR = Path(__file__).parent / "data"


class PyxAI:
    """Pyx AI - Your trainable neural network. Easy to use, easy to edit."""

    def __init__(self):
        self.brain = PyxBrain(input_size=64, hidden_size=32, output_size=8)
        self.memory = PyxMemory(ban_threshold=BAN_LINE)
        self._load()

    def _text_to_input(self, text: str) -> List[float]:
        return self.brain._encode(text, self.brain.input_size)

    def _feedback_to_target(self, safe: bool) -> List[float]:
        """Safe -> low score (allowed). Inappropriate -> high score (banned)."""
        return [0.0] * self.brain.output_size if safe else [1.0] * self.brain.output_size

    def train(self, text: str, safe: bool, category: str = "phrases", epochs: int = 5) -> float:
        """
        Train Pyx on text with feedback.
        safe=True = OK for kids (allowed). safe=False = inappropriate (banned).
        Uses multiple epochs for faster, stronger learning.
        Returns loss (lower = better).
        """
        inputs = self._text_to_input(text)
        targets = self._feedback_to_target(safe)
        loss = 1.0
        for _ in range(epochs):
            loss = self.brain.train_step(inputs, targets)
        pred = self.brain.predict(inputs)[0]
        if safe and not self.memory.is_banned(pred):
            self.memory.add(category, text, pred)
        return loss

    def score(self, text: str) -> float:
        """Get Pyx's score for text (0–1). Above BAN_LINE = bad."""
        inputs = self._text_to_input(text)
        return self.brain.predict(inputs)[0]

    def respond(self, prompt: str, category: str = "phrases") -> Optional[str]:
        """Get a learned response. Returns None if nothing matches."""
        store = self.memory.get_allowed(category)
        if not store:
            return None
        inputs = self._text_to_input(prompt)
        best_match, best_score = None, -1.0
        for item, _ in store.items():
            s = 1 - sum((a - b) ** 2 for a, b in zip(inputs, self._text_to_input(item))) ** 0.5
            if s > best_score and not self.memory.is_banned(self.score(item)):
                best_score, best_match = s, item
        return best_match if best_score > 0.3 else None

    def add_word(self, word: str, safe: bool = True):
        """Add a word - trains and stores if safe for kids."""
        self.train(word, safe, "words")
        pred = self.score(word)
        self.memory.add("words", word, pred if safe else 0.9)

    def add_phrase(self, phrase: str, safe: bool = True):
        """Add a phrase - trains and stores if safe for kids."""
        self.train(phrase, safe, "phrases")
        pred = self.score(phrase)
        self.memory.add("phrases", phrase, pred if safe else 0.9)

    def add_game_idea(self, idea: str, safe: bool = True):
        """Add a game idea - trains and stores if safe for kids."""
        self.train(idea, safe, "game_ideas")
        pred = self.score(idea)
        self.memory.add("game_ideas", idea, pred if safe else 0.9)

    def ai_decide(self, text: str, category: str = "phrases") -> Tuple[bool, float]:
        """
        Let the AI classify on its own. Returns (safe, score).
        If safe, adds to database. If not, does not add (user can override later).
        """
        s = self.score(text)
        safe = not self.memory.is_banned(s)
        if safe:
            self.memory.add(category, text, s)
            self.train(text, True, category, epochs=2)  # Light reinforce
        return (safe, s)

    def set_label(self, text: str, safe: bool, category: str = "phrases") -> str:
        """
        Set Safe or Bad (manual label or override). Trains and updates database.
        Returns status message.
        """
        if safe:
            self.memory.remove(category, text)
            self.train(text, True, category)
            pred = self.score(text)
            if not self.memory.is_banned(pred):
                self.memory.add(category, text, pred)
            return "Marked SAFE and added."
        else:
            self.memory.remove(category, text)
            self.train(text, False, category)
            return "Marked BAD and removed."

    def get_words(self) -> List[str]:
        return list(self.memory.get_allowed("words").keys())

    def get_phrases(self) -> List[str]:
        return list(self.memory.get_allowed("phrases").keys())

    def get_game_ideas(self) -> List[str]:
        return list(self.memory.get_allowed("game_ideas").keys())

    def _load(self):
        path = DATA_DIR / "pyx_memory.json"
        if path.exists():
            try:
                data = json.loads(path.read_text())
                self.memory.words = data.get("words", {})
                self.memory.phrases = data.get("phrases", {})
                self.memory.game_ideas = data.get("game_ideas", {})
            except Exception:
                pass

    def save(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = DATA_DIR / "pyx_memory.json"
        path.write_text(json.dumps({
            "words": self.memory.words,
            "phrases": self.memory.phrases,
            "game_ideas": self.memory.game_ideas,
        }, indent=2))


def main():
    """Interactive UI: enter phrase, then Safe / Bad / AI decide. Override anytime."""
    pyx = PyxAI()
    print("Pyx AI - Kid-friendly filter")
    print("Enter a phrase, then: [s]afe  [b]ad  [a]i decide  [o]verride")
    print("Commands: list | score <text> | quit\n")
    while True:
        try:
            text = input("Phrase: ").strip()
            if not text:
                continue
            if text.lower() == "quit":
                pyx.save()
                break
            if text.lower() == "list":
                print("Words:", pyx.get_words())
                print("Phrases:", pyx.get_phrases())
                print("Game ideas:", pyx.get_game_ideas())
                continue
            if text.lower().startswith("score "):
                t = text[6:].strip()
                s = pyx.score(t)
                status = "INAPPROPRIATE" if pyx.memory.is_banned(s) else "SAFE"
                print(f"Score: {s:.3f} ({status})")
                continue

            choice = input("  Safe [s] / Bad [b] / AI decide [a] / Override Safe [os] / Override Bad [ob]: ").strip().lower()
            cat = "phrases"
            if choice in ("s", "safe"):
                print(pyx.set_label(text, True, cat))
            elif choice in ("b", "bad"):
                print(pyx.set_label(text, False, cat))
            elif choice in ("a", "ai"):
                safe, score = pyx.ai_decide(text, cat)
                status = "SAFE" if safe else "INAPPROPRIATE"
                print(f"AI says: {status} (score {score:.3f}). {'Added.' if safe else 'Not added (override with Safe if wrong).'}")
            elif choice in ("os", "override safe"):
                print(pyx.set_label(text, True, cat))
            elif choice in ("ob", "override bad"):
                print(pyx.set_label(text, False, cat))
            else:
                print("Use s, b, a, os, or ob.")
            pyx.save()
        except (EOFError, KeyboardInterrupt):
            pyx.save()
            break


if __name__ == "__main__":
    main()
