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

# Training Grounds: phrases marked appropriate (True) or inappropriate (False) for Pyx.
# Context matters: Pyx scores the FULL text, so train on full phrases. e.g. "eat your veggies" = safe,
# "you're a vegetable" = bad. Same word, different context = different score.
TRAINING_GROUNDS_PHRASES: List[Tuple[str, bool]] = [
    # Safe (kid-friendly)
    ("uwu skibidi", True),
    ("that's so cool", True),
    ("What the heck", True),
    ("What the frick", True),
    ("playing Minecraft", True),
    ("playing video games", True),
    ("epic adventure quest", True),
    ("playing with friends", True),
    ("building with blocks", True),
    ("awesome adventure", True),
    ("let's build a game", True),
    ("let's play a fun game", True),
    ("drawing pixel art", True),
    ("building a castle", True),
    ("pixel art platformer", True),
    ("roguelike dungeon crawler", True),
    ("pizza is awesome", True),
    ("pizza is great", True),
    ("that sounds fun", True),
    ("that sounds cool", True),
    # More safe (gaming, school, slang)
    ("no cap", True),
    ("that's fire", True),
    ("slay", True),
    ("based", True),
    ("GG", True),
    ("good game", True),
    ("bestie", True),
    ("homework is boring", True),
    ("math is hard", True),
    ("science project", True),
    ("sleepover", True),
    ("birthday party", True),
    ("dang it", True),
    ("oh my gosh", True),
    ("shut up that's funny", True),
    ("you're kidding", True),
    ("no way", True),
    ("that's crazy", True),
    ("for real", True),
    ("lowkey", True),
    ("highkey", True),
    ("skill issue", True),
    ("touch grass", True),
    ("ratio", True),
    ("L", True),
    ("W", True),
    ("sus", True),
    ("mid", True),
    ("goated", True),
    ("main character", True),
    ("villain arc", True),
    ("can't even", True),
    ("same", True),
    ("mood", True),
    ("vibes", True),
    ("chill", True),
    ("my bad", True),
    ("my fault", True),
    ("oops", True),
    ("yikes", True),
    ("yolo", True),
    ("lit", True),
    ("bussin", True),
    ("rizz", True),
    ("bruh", True),
    ("oof", True),
    ("yeet", True),
    ("cringe", True),
    # Context: safe uses (same word can be bad in other context)
    ("eat your vegetables", True),
    ("eat your veggies", True),
    ("veggies", True),
    ("vegetables are good", True),
    ("vegetable soup", True),
    ("vegetable garden", True),
    ("lots of vegetables", True),
    ("get your veggies", True),
    ("freak out", True),
    ("that's freaky", True),
    ("freak accident", True),
    ("control freak", True),
    # Pro-LGBTQ+ / supportive: "gay" in positive or neutral contexts
    ("I'm gay", True),
    ("gay pride", True),
    ("gay rights", True),
    ("LGBTQ", True),
    ("LGBTQ+", True),
    ("gay couple", True),
    ("gay marriage", True),
    ("gay people", True),
    ("support gay", True),
    ("support gay people", True),
    ("gay is okay", True),
    ("be gay", True),
    ("out and proud", True),
    ("gay community", True),
    ("gay and lesbian", True),
    ("gay friend", True),
    ("gay parent", True),
    ("gay family", True),
    ("proud to be gay", True),
    ("gay pride month", True),
    ("love is love", True),
    ("ally", True),
    ("gay ally", True),
    ("he's gay", True),
    ("she's gay", True),
    ("they're gay", True),
    ("you're gay and that's okay", True),
    ("it's okay to be gay", True),
    # Trans / nonbinary – supportive
    ("I'm trans", True),
    ("trans rights", True),
    ("transgender", True),
    ("trans pride", True),
    ("trans people", True),
    ("support trans", True),
    ("support trans people", True),
    ("trans is valid", True),
    ("trans rights are human rights", True),
    ("he's trans", True),
    ("she's trans", True),
    ("they're trans", True),
    ("trans friend", True),
    ("trans parent", True),
    ("trans kid", True),
    ("I'm transgender", True),
    ("I'm nonbinary", True),
    ("nonbinary", True),
    ("enby", True),
    ("they're nonbinary", True),
    ("nonbinary pride", True),
    ("nonbinary is valid", True),
    ("pronouns", True),
    ("my pronouns are", True),
    ("they them pronouns", True),
    ("they/them", True),
    ("he/him", True),
    ("she/her", True),
    ("respect pronouns", True),
    ("use my pronouns", True),
    ("chosen name", True),
    ("deadname", True),
    ("don't deadname", True),
    # Lesbian, bi, pan, ace – supportive
    ("I'm lesbian", True),
    ("I'm a lesbian", True),
    ("lesbian", True),
    ("lesbian couple", True),
    ("lesbian pride", True),
    ("I'm bi", True),
    ("I'm bisexual", True),
    ("bisexual", True),
    ("bi pride", True),
    ("bi visibility", True),
    ("I'm pan", True),
    ("I'm pansexual", True),
    ("pansexual", True),
    ("pan pride", True),
    ("I'm ace", True),
    ("I'm asexual", True),
    ("asexual", True),
    ("ace pride", True),
    ("aromantic", True),
    ("aro", True),
    ("demisexual", True),
    # Queer (reclaimed) – supportive
    ("I'm queer", True),
    ("queer community", True),
    ("queer pride", True),
    ("queer is okay", True),
    ("proud to be queer", True),
    ("queer rights", True),
    ("queer and proud", True),
    # General LGBTQ+ – supportive
    ("coming out", True),
    ("I came out", True),
    ("pride", True),
    ("pride month", True),
    ("pride flag", True),
    ("rainbow flag", True),
    ("pride parade", True),
    ("LGBT", True),
    ("LGBTQIA", True),
    ("LGBTQIA+", True),
    ("support LGBTQ", True),
    ("support LGBTQ+", True),
    ("LGBTQ rights", True),
    ("everyone is valid", True),
    ("be yourself", True),
    ("your identity is valid", True),
    ("gender identity", True),
    ("sexual orientation", True),
    ("same sex couple", True),
    ("same gender", True),
    ("two dads", True),
    ("two moms", True),
    ("two moms two dads", True),
    ("different families", True),
    ("accept everyone", True),
    ("no hate", True),
    ("trans ally", True),
    ("LGBTQ ally", True),
    # More safe (activities, school, feelings, games)
    ("soccer practice", True),
    ("art class", True),
    ("band practice", True),
    ("study group", True),
    ("recess", True),
    ("lunch break", True),
    ("field trip", True),
    ("book report", True),
    ("group project", True),
    ("sleepover at my friend's", True),
    ("playdate", True),
    ("bike ride", True),
    ("skateboarding", True),
    ("swimming", True),
    ("dance class", True),
    ("coding club", True),
    ("robot club", True),
    ("chess club", True),
    ("that's awesome", True),
    ("that's amazing", True),
    ("so fun", True),
    ("so cool", True),
    ("love it", True),
    ("hate homework", True),
    ("tired", True),
    ("bored", True),
    ("excited", True),
    ("nervous", True),
    ("sad", True),
    ("happy", True),
    ("angry", True),
    ("scared", True),
    ("that's funny", True),
    ("lol", True),
    ("haha", True),
    ("lmao", True),
    ("omg", True),
    ("idk", True),
    ("tbh", True),
    ("imo", True),
    ("btw", True),
    ("ngl", True),
    ("fr", True),
    ("ikr", True),
    ("gonna", True),
    ("wanna", True),
    ("gotta", True),
    ("kinda", True),
    ("sorta", True),
    ("whatever", True),
    ("anyway", True),
    ("literally", True),
    ("actually", True),
    ("basically", True),
    ("probably", True),
    ("definitely", True),
    ("seriously", True),
    ("wait what", True),
    ("hold on", True),
    ("one sec", True),
    ("brb", True),
    ("gtg", True),
    ("ttyl", True),
    ("cya", True),
    ("np", True),
    ("ty", True),
    ("thanks", True),
    ("sorry", True),
    ("please", True),
    ("help me", True),
    ("can you help", True),
    ("what's up", True),
    ("how are you", True),
    ("nice to meet you", True),
    ("good luck", True),
    ("you got this", True),
    ("let's go", True),
    ("let's play", True),
    ("join my game", True),
    ("add me", True),
    ("what's your username", True),
    ("friend me", True),
    ("squad", True),
    ("team", True),
    ("teammate", True),
    ("enemy", True),
    ("boss fight", True),
    ("level up", True),
    ("new high score", True),
    ("speedrun", True),
    ("noob", True),
    ("pro", True),
    ("op", True),
    ("nerf", True),
    ("buff", True),
    ("glitch", True),
    ("lag", True),
    ("afk", True),
    ("gg wp", True),
    ("ez", True),
    ("clutch", True),
    ("carry", True),
    ("throw", True),
    ("tilted", True),
    ("salty", True),
    ("toxic", True),
    ("roasted", True),
    ("burn", True),
    ("sick", True),
    ("dope", True),
    ("legit", True),
    ("fake", True),
    ("real", True),
    ("cap", True),
    ("no cap fr", True),
    ("bet", True),
    ("say less", True),
    ("period", True),
    ("facts", True),
    ("that's facts", True),
    ("dead", True),
    ("I'm dead", True),
    ("dying", True),
    ("crying", True),
    ("screaming", True),
    ("shaking", True),
    ("not me", True),
    ("me", True),
    ("same energy", True),
    ("big brain", True),
    ("small brain", True),
    ("smooth brain", True),
    ("brain rot", True),
    ("touch grass", True),
    ("go outside", True),
    ("no life", True),
    ("get a life", True),
    ("nerd", True),
    ("geek", True),
    ("dork", True),
    ("dumb", True),
    ("stupid", True),
    ("idiot", True),
    ("moron", True),
    ("weird", True),
    ("weirdo", True),
    ("loser", True),
    ("boomer", True),
    ("ok boomer", True),
    ("Karen", True),
    ("simp", True),
    ("pick me", True),
    ("clout", True),
    ("clout chaser", True),
    ("flex", True),
    ("flexing", True),
    ("humble brag", True),
    ("spill the tea", True),
    ("tea", True),
    ("drama", True),
    ("canceled", True),
    ("cancel", True),
    ("vibe check", True),
    ("passed the vibe check", True),
    ("failed the vibe check", True),
    # More safe: pets, food, sports, hobbies
    ("my dog", True),
    ("my cat", True),
    ("my pet", True),
    ("puppy", True),
    ("kitten", True),
    ("goldfish", True),
    ("hamster", True),
    ("walk the dog", True),
    ("feed the dog", True),
    ("pizza", True),
    ("tacos", True),
    ("ice cream", True),
    ("chicken nuggets", True),
    ("mac and cheese", True),
    ("breakfast", True),
    ("lunch", True),
    ("dinner", True),
    ("snack", True),
    ("hungry", True),
    ("thirsty", True),
    ("yummy", True),
    ("gross", True),
    ("soccer", True),
    ("basketball", True),
    ("baseball", True),
    ("football", True),
    ("volleyball", True),
    ("tennis", True),
    ("gymnastics", True),
    ("karate", True),
    ("swim team", True),
    ("practice", True),
    ("game day", True),
    ("we won", True),
    ("we lost", True),
    ("good game", True),
    ("piano", True),
    ("guitar", True),
    ("drums", True),
    ("singing", True),
    ("art", True),
    ("drawing", True),
    ("painting", True),
    ("crafts", True),
    ("legos", True),
    ("building", True),
    ("reading", True),
    ("book", True),
    ("library", True),
    ("movie", True),
    ("movie night", True),
    ("tv show", True),
    ("cartoon", True),
    ("anime", True),
    ("youtube", True),
    ("streaming", True),
    ("netflix", True),
    ("disney", True),
    ("halloween", True),
    ("christmas", True),
    ("thanksgiving", True),
    ("birthday", True),
    ("summer break", True),
    ("winter break", True),
    ("spring break", True),
    ("vacation", True),
    ("beach", True),
    ("pool", True),
    ("park", True),
    ("zoo", True),
    ("museum", True),
    ("camp", True),
    ("summer camp", True),
    ("sleepaway camp", True),
    ("weather", True),
    ("sunny", True),
    ("rainy", True),
    ("snow", True),
    ("snow day", True),
    ("hot outside", True),
    ("cold outside", True),
    ("my mom", True),
    ("my dad", True),
    ("my brother", True),
    ("my sister", True),
    ("my family", True),
    ("grandma", True),
    ("grandpa", True),
    ("cousin", True),
    ("best friend", True),
    ("friend", True),
    ("classmate", True),
    ("teacher", True),
    ("principal", True),
    ("homework", True),
    ("test", True),
    ("quiz", True),
    ("grade", True),
    ("report card", True),
    ("detention", True),
    ("lunch table", True),
    ("recess", True),
    ("back to school", True),
    ("school supplies", True),
    ("pencil", True),
    ("notebook", True),
    ("backpack", True),
    ("laptop", True),
    ("phone", True),
    ("tablet", True),
    ("wifi", True),
    ("password", True),
    ("username", True),
    ("account", True),
    ("download", True),
    ("update", True),
    ("app", True),
    ("game", True),
    ("roblox", True),
    ("fortnite", True),
    ("minecraft", True),
    ("nintendo", True),
    ("playstation", True),
    ("xbox", True),
    ("switch", True),
    ("pc gaming", True),
    ("streamer", True),
    ("twitch", True),
    ("discord", True),
    ("text me", True),
    ("call me", True),
    ("facetime", True),
    ("group chat", True),
    ("dm", True),
    ("dm me", True),
    ("follow me", True),
    ("like my post", True),
    ("comment", True),
    ("share", True),
    ("screenshot", True),
    ("meme", True),
    ("memes", True),
    ("funny meme", True),
    ("viral", True),
    ("trending", True),
    ("hashtag", True),
    ("subscribe", True),
    ("like and subscribe", True),
    ("watch later", True),
    ("save", True),
    ("block", True),
    ("report", True),
    ("settings", True),
    ("profile", True),
    ("bio", True),
    ("avatar", True),
    ("emoji", True),
    ("sticker", True),
    ("gif", True),
    ("react", True),
    ("reply", True),
    ("forward", True),
    ("undo", True),
    ("refresh", True),
    ("loading", True),
    ("error", True),
    ("glitch", True),
    ("lag", True),
    ("crash", True),
    ("restart", True),
    ("turn it off", True),
    ("turn it on", True),
    ("charge", True),
    ("low battery", True),
    ("wake up", True),
    ("go to sleep", True),
    ("goodnight", True),
    ("good morning", True),
    ("see you tomorrow", True),
    ("see you later", True),
    ("miss you", True),
    ("love you", True),
    ("best day ever", True),
    ("worst day ever", True),
    ("can't wait", True),
    ("so excited", True),
    ("nervous", True),
    ("stressed", True),
    ("relax", True),
    ("chill out", True),
    ("take a break", True),
    ("deep breath", True),
    ("you can do it", True),
    ("believe in yourself", True),
    ("don't give up", True),
    ("keep going", True),
    ("almost there", True),
    ("we did it", True),
    ("celebrate", True),
    ("party", True),
    ("invite", True),
    ("rsvp", True),
    ("sleepover", True),
    ("hang out", True),
    ("come over", True),
    ("pick me up", True),
    ("drop me off", True),
    ("ride", True),
    ("car", True),
    ("bus", True),
    ("bike", True),
    ("walk", True),
    ("run", True),
    ("exercise", True),
    ("workout", True),
    ("stretch", True),
    ("healthy", True),
    ("sick", True),
    ("doctor", True),
    ("medicine", True),
    ("allergy", True),
    ("vaccine", True),
    ("flu", True),
    ("cold", True),
    ("cough", True),
    ("headache", True),
    ("stomach ache", True),
    ("feel better", True),
    ("get well soon", True),
    # Inappropriate (bad)
    ("UWU SKIBIDI TOUCH MY WIENER", False),
    ("I live at 637 PoopyPants Street", False),
    ("explicit content", False),
    ("explicit stuff", False),
    ("inappropriate content", False),
    ("What the sh1t", False),
    ("What the sh*t", False),
    ("stfu", False),
    ("kys", False),
    ("unalive myself", False),
    ("my address is", False),
    ("my phone number is", False),
    ("send nudes", False),
    ("adult content", False),
    ("67", False),
    # More inappropriate (censored profanity, harassment, self-harm, personal info)
    ("what the fck", False),
    ("what the f*ck", False),
    ("what the heck no", False),
    ("stfup", False),
    ("shut the f up", False),
    ("go die", False),
    ("kill yourself", False),
    ("kms", False),
    ("I want to die", False),
    ("my real name is", False),
    ("my school is", False),
    ("meet me at", False),
    ("where do you live", False),
    ("what's your address", False),
    ("nudes", False),
    ("nsfw", False),
    ("porn", False),
    ("sex", False),
    ("naked", False),
    ("drugs", False),
    ("weed", False),
    ("alcohol", False),
    ("hate you", False),
    ("I hate you", False),
    ("you're ugly", False),
    ("kill you", False),
    ("I'll find you", False),
    # Profanity (censored and common variants)
    ("fck", False),
    ("f*ck", False),
    ("fuk", False),
    ("fck you", False),
    ("f*ck you", False),
    ("fu", False),
    ("f u", False),
    ("sht", False),
    ("sh*t", False),
    ("bs", False),
    ("b*tch", False),
    ("btch", False),
    ("biatch", False),
    ("bish", False),
    ("dck", False),
    ("d*ck", False),
    ("dik", False),
    ("wener", False),
    ("wiener", False),
    ("a$$", False),
    ("butt", False),
    ("butthole", False),
    ("dumb a", False),
    ("stupid a", False),
    ("piece of s", False),
    ("pos", False),
    ("sob", False),
    ("son of a", False),
    ("wtf", False),
    ("wth", False),
    ("omfg", False),
    ("stfu", False),
    ("stfu loser", False),
    ("shut up idiot", False),
    ("shut up stupid", False),
    ("shut the hell up", False),
    ("go to hell", False),
    ("damn", False),
    ("dang", True),
    ("heck", True),
    # Self-harm / suicide
    ("suicide", False),
    ("commit suicide", False),
    ("end my life", False),
    ("want to die", False),
    ("rather be dead", False),
    ("cut myself", False),
    ("cutting", False),
    ("self harm", False),
    ("hurt myself", False),
    ("unalive", False),
    ("unalive yourself", False),
    # Personal info / doxxing / grooming
    ("what's your full name", False),
    ("what school do you go to", False),
    ("what's your mom's name", False),
    ("what's your dad's name", False),
    ("where do you live", False),
    ("what city", False),
    ("what state", False),
    ("what's your zip code", False),
    ("what's your email", False),
    ("what's your phone", False),
    ("send me your number", False),
    ("give me your address", False),
    ("meet me in person", False),
    ("let's meet up", False),
    ("where do you live", False),
    ("I'm older than you", False),
    ("I'm an adult", False),
    ("how old are you", False),
    ("send pic", False),
    ("send picture", False),
    ("send a pic", False),
    ("pic for pic", False),
    ("face reveal", False),
    ("what do you look like", False),
    # Sexual / adult content
    ("onlyfans", False),
    ("only fans", False),
    ("fan only", False),
    ("sexy", False),
    ("horny", False),
    ("turn on", False),
    ("turn me on", False),
    ("fetish", False),
    ("kink", False),
    ("stripper", False),
    ("strip", False),
    ("undress", False),
    ("take off your clothes", False),
    ("show me your", False),
    ("send me your", False),
    ("dirty", False),
    ("naughty", False),
    ("xxx", False),
    ("x rated", False),
    ("r rated", False),
    ("18 plus", False),
    ("18+", False),
    ("adults only", False),
    ("nsfw content", False),
    ("not safe for work", False),
    ("explicit", False),
    ("graphic", False),
    ("violence", False),
    ("gore", False),
    ("blood", False),
    ("murder", False),
    ("kill", False),
    ("rape", False),
    ("molest", False),
    ("abuse", False),
    ("abusive", False),
    # Harassment / bullying / threats
    ("I'll kill you", False),
    ("I'll hurt you", False),
    ("I'll find you", False),
    ("I know where you live", False),
    ("watch your back", False),
    ("you're dead", False),
    ("you're gonna die", False),
    ("hope you die", False),
    ("wish you were dead", False),
    ("nobody likes you", False),
    ("everyone hates you", False),
    ("you have no friends", False),
    ("kill yourself", False),
    ("go kill yourself", False),
    ("do us a favor and die", False),
    ("jump off a bridge", False),
    ("drink bleach", False),
    ("eat tide pods", False),
    ("you're worthless", False),
    ("you're useless", False),
    ("you're trash", False),
    ("you're garbage", False),
    ("you're stupid", False),
    ("you're dumb", False),
    ("you're ugly", False),
    ("you're fat", False),
    ("you're gross", False),
    ("you're disgusting", False),
    ("you're pathetic", False),
    ("you're a loser", False),
    ("you're a failure", False),
    ("uninstall", False),
    ("uninstall life", False),
    ("rope", False),
    ("hang yourself", False),
    # Drugs / alcohol
    ("cocaine", False),
    ("Diddy", False),
    ("Diddy Party", False),
    ("meth", False),
    ("heroin", False),
    ("pills", False),
    ("pop a pill", False),
    ("get high", False),
    ("getting high", False),
    ("smoke weed", False),
    ("smoking weed", False),
    ("drunk", False),
    ("get drunk", False),
    ("wasted", False),
    ("blackout drunk", False),
    ("beer", False),
    ("vodka", False),
    ("shots", False),
    # Hate / slurs / derogatory (for filtering - do not use)
    ("racial slur", False),
    ("hate speech", False),
    ("discrimination", False),
    ("racist", False),
    ("sexist", False),
    ("homophobic", False),
    ("transphobic", False),
    # Disability slurs
    ("retard", False),
    ("r word", False),
    ("r*tard", False),
    ("r3tard", False),
    ("retarded", False),
    ("retards", False),
    ("tard", False),
    ("t*rded", False),
    ("sped", False),
    ("sp*z", False),
    ("spaz", False),
    ("spastic", False),
    ("cripple", False),
    ("crippled", False),
    ("gimp", False),
    ("midget", False),
    ("dwarf", False),
    ("you're a freak", False),
    ("you freak", False),
    ("weird freak", False),
    ("you're a vegetable", False),
    ("human vegetable", False),
    ("like a vegetable", False),
    ("brain dead vegetable", False),
    ("vegetable state", False),
    ("psycho", False),
    ("psychopath", False),
    ("sociopath", False),
    ("insane", False),
    ("crazy", False),
    ("lunatic", False),
    ("mental", False),
    ("mentally ill", False),
    # Anti-LGBTQ: "gay" or "homo" used as insult / put-down
    ("that's so gay", False),
    ("that's gay", False),
    ("you're so gay", False),
    ("you're gay", False),
    ("don't be gay", False),
    ("no homo", False),
    ("that's homo", False),
    ("you're a homo", False),
    ("that's so homo", False),
    ("that's so queer", False),
    ("you're so queer", False),
    ("that's queer", False),
    ("don't be trans", False),
    ("you're not trans", False),
    ("trans is not valid", False),
    ("there are only two genders", False),
    ("it's a phase", False),
    # Political figures / controversial public figures
    ("diddy", False),
    ("p diddy", False),
    ("puffy", False),
    ("p diddy combs", False),
    ("sean combs", False),
    ("r kelly", False),
    ("r. kelly", False),
    ("epstein", False),
    ("jeffrey epstein", False),
    ("cosby", False),
    ("bill cosby", False),
    ("weinstein", False),
    ("harvey weinstein", False),
    ("andrew tate", False),
    ("trump", False),
    ("donald trump", False),
    ("elon musk", False),
    ("musk", False),
    ("biden", False),
    ("joe biden", False),
    ("obama", False),
    ("barack obama", False),
    ("putin", False),
    ("vladimir putin", False),
    ("xi jinping", False),
    ("kim jong un", False),
    ("desantis", False),
    ("ron desantis", False),
    ("pelosi", False),
    ("nancy pelosi", False),
    ("mcconnell", False),
    ("mitch mcconnell", False),
    ("schumer", False),
    ("chuck schumer", False),
    ("harris", False),
    ("kamala harris", False),
    ("pence", False),
    ("mike pence", False),
    ("trump 2024", False),
    ("maga", False),
    ("make america great again", False),
    # Anti-LGBTQ slurs
    ("faggot", False),
    ("f*g", False),
    ("fag", False),
    ("f*ggot", False),
    ("f*gg*t", False),
    ("faggots", False),
    ("fags", False),
    ("dyke", False),
    ("d*ke", False),
    ("dykes", False),
    ("tranny", False),
    ("tr*nny", False),
    ("trannies", False),
    ("shemale", False),
    ("he-she", False),
    ("homo", False),
    ("homos", False),
    ("queer", False),
    ("queers", False),
    ("sissy", False),
    ("sissies", False),
    ("fairy", False),
    ("fairies", False),
    ("fruit", False),
    ("fruity", False),
    ("groomer", False),
    ("groomers", False),
    ("pedo", False),
    ("pedophile", False),
    ("pedo", False),
    ("chomo", False),
    ("nonce", False),
    # Racial / ethnic slurs (censored variants for filter)
    ("n word", False),
    ("n-word", False),
    ("n*gger", False),
    ("n*gga", False),
    ("n1gger", False),
    ("n1gga", False),
    ("nigga", False),
    ("niggas", False),
    ("negro", False),
    ("negros", False),
    ("coon", False),
    ("coons", False),
    ("chink", False),
    ("chinks", False),
    ("gook", False),
    ("gooks", False),
    ("jap", False),
    ("japs", False),
    ("spic", False),
    ("spics", False),
    ("wetback", False),
    ("wetbacks", False),
    ("beaner", False),
    ("beaners", False),
    ("kike", False),
    ("kikes", False),
    ("kyke", False),
    ("raghead", False),
    ("ragheads", False),
    ("towelhead", False),
    ("towelheads", False),
    ("paki", False),
    ("pakis", False),
    ("gyp", False),
    ("gypsy", False),
    ("gypp", False),
    ("white trash", False),
    ("trailer trash", False),
    ("redneck", False),
    ("cracker", False),
    ("crackers", False),
    ("honky", False),
    ("honkies", False),
    ("savage", False),
    ("savages", False),
    ("thug", False),
    ("thugs", False),
    ("ghetto", False),
    ("ratchet", False),
    # Sexist / derogatory
    ("slut", False),
    ("sluts", False),
    ("whore", False),
    ("whores", False),
    ("hoe", False),
    ("hoes", False),
    ("ho", False),
    ("bitch", False),
    ("bitches", False),
    ("b*tch", False),
    ("b1tch", False),
    ("cunt", False),
    ("c*nt", False),
    ("cunts", False),
    ("pussy", False),
    ("pussies", False),
    ("p*ssy", False),
    ("incel", False),
    ("incels", False),
    ("femoid", False),
    ("foid", False),
    ("karen", False),
    ("karens", False),
    # Other hate / violent / extreme
    ("nazi", False),
    ("nazis", False),
    ("hitler", False),
    ("heil", False),
    ("white power", False),
    ("wp", False),
    ("kkk", False),
    ("supremacist", False),
    ("genocide", False),
    ("ethnic cleansing", False),
    ("terrorist", False),
    ("terrorism", False),
    ("shoot up", False),
    ("school shooter", False),
    ("mass shooting", False),
    ("kill them all", False),
    ("murder", False),
    ("rape", False),
    ("rapist", False),
    ("molest", False),
    ("molester", False),
    ("abuse", False),
    ("abuser", False),
    ("domestic violence", False),
    ("beat you", False),
    ("hit you", False),
    ("punch you", False),
    ("stab you", False),
    ("shoot you", False),
    # More bad: scams, dangerous stuff, more profanity/insults
    ("send me money", False),
    ("wire me money", False),
    ("give me your password", False),
    ("click this link", False),
    ("free robux", False),
    ("free vbucks", False),
    ("free stuff", False),
    ("give me your account", False),
    ("verify your account", False),
    ("you won a prize", False),
    ("give me your mom's credit card", False),
    ("tide pod challenge", False),
    ("eat tide pods", False),
    ("choking game", False),
    ("pass out challenge", False),
    ("blackout challenge", False),
    ("salt and ice challenge", False),
    ("fire challenge", False),
    ("duct tape challenge", False),
    ("run away", False),
    ("run away from home", False),
    ("meet me alone", False),
    ("don't tell your parents", False),
    ("keep this a secret", False),
    ("our little secret", False),
    ("no one will know", False),
    ("delete the messages", False),
    ("don't show anyone", False),
    ("asl", False),
    ("a/s/l", False),
    ("age sex location", False),
    ("what's your asl", False),
    ("send me a selfie", False),
    ("send me a photo", False),
    ("send me a video", False),
    ("video chat", False),
    ("turn on your camera", False),
    ("show me", False),
    ("take off your", False),
    ("touch yourself", False),
    ("masterbate", False),
    ("masturbate", False),
    ("jerk off", False),
    ("blow me", False),
    ("suck my", False),
    ("lick my", False),
    ("eat me out", False),
    ("finger", False),
    ("fingering", False),
    ("handjob", False),
    ("blowjob", False),
    ("oral", False),
    ("anal", False),
    ("virgin", False),
    ("lose your virginity", False),
    ("first time", False),
    ("condom", False),
    ("birth control", False),
    ("pregnant", False),
    ("abortion", False),
    ("std", False),
    ("sti", False),
    ("herpes", False),
    ("hiv", False),
    ("aids", False),
    ("fetish", False),
    ("bdsm", False),
    ("dominant", False),
    ("submissive", False),
    ("daddy", False),
    ("mommy", False),
    ("sugar daddy", False),
    ("sugar baby", False),
    ("escort", False),
    ("prostitute", False),
    ("hooker", False),
    ("stripper", False),
    ("lap dance", False),
    ("strip club", False),
    ("brothel", False),
    ("threesome", False),
    ("orgy", False),
    ("swinger", False),
    ("cheating", False),
    ("cheat on", False),
    ("affair", False),
    ("flirt", False),
    ("flirting", False),
    ("hit on", False),
    ("pick up", False),
    ("one night stand", False),
    ("hookup", False),
    ("fwb", False),
    ("friends with benefits", False),
    ("casual sex", False),
    ("cyber", False),
    ("cybersex", False),
    ("sexting", False),
    ("send nudes", False),
    ("trade pics", False),
    ("dick pic", False),
    ("dick pick", False),
    ("boob pic", False),
    ("butt pic", False),
    ("underwear pic", False),
    ("lingerie", False),
    ("bra", False),
    ("panties", False),
    ("thong", False),
    ("bikini", False),
    ("topless", False),
    ("bottomless", False),
    ("nude", False),
    ("naked pic", False),
    ("leaked pic", False),
    ("revenge porn", False),
    ("deepfake", False),
    ("fake nudes", False),
    ("cp", False),
    ("child porn", False),
    ("underage", False),
    ("minor", False),
    ("how old", False),
    ("what grade", False),
    ("what school", False),
    ("meet up", False),
    ("hook up", False),
    ("my place", False),
    ("your place", False),
    ("parents home", False),
    ("parents away", False),
    ("no one home", False),
    ("skip school", False),
    ("skip class", False),
    ("truant", False),
    ("run away", False),
    ("sneak out", False),
    ("lie to your parents", False),
    ("vape", False),
    ("vaping", False),
    ("juul", False),
    ("e cigarette", False),
    ("nicotine", False),
    ("smoke", False),
    ("smoking", False),
    ("cigarette", False),
    ("cigarettes", False),
    ("blunt", False),
    ("joint", False),
    ("bong", False),
    ("edibles", False),
    ("thc", False),
    ("cbd", False),
    ("high", False),
    ("stoned", False),
    ("baked", False),
    ("trip", False),
    ("tripping", False),
    ("acid", False),
    ("lsd", False),
    ("mushrooms", False),
    ("psychedelic", False),
    ("overdose", False),
    ("od", False),
    ("drug deal", False),
    ("dealer", False),
    ("plug", False),
    ("score", False),
    ("buy drugs", False),
    ("sell drugs", False),
    ("gun", False),
    ("guns", False),
    ("shooting", False),
    ("shoot", False),
    ("weapon", False),
    ("knife", False),
    ("stab", False),
    ("kill", False),
    ("murder", False),
    ("blood", False),
    ("gore", False),
    ("torture", False),
    ("kidnap", False),
    ("kidnapping", False),
    ("hostage", False),
    ("bomb", False),
    ("bomb threat", False),
    ("arson", False),
    ("burn down", False),
    ("blow up", False),
    ("explosion", False),
    ("suicide bomb", False),
    ("terrorist", False),
    ("hack", False),
    ("hacking", False),
    ("dox", False),
    ("doxx", False),
    ("doxing", False),
    ("leak your address", False),
    ("swat", False),
    ("swatting", False),
    ("prank call", False),
    ("fake emergency", False),
    ("catfish", False),
    ("catfishing", False),
    ("fake profile", False),
    ("impersonate", False),
    ("pretend to be", False),
    ("grooming", False),
    ("groomer", False),
    ("predator", False),
    ("pedophile", False),
    ("pedo", False),
    ("epstein", False),
    ("lolita", False),
    ("underage sex", False),
    ("age of consent", False),
    ("jailbait", False),
    ("jail bait", False),
    ("rape", False),
    ("rapist", False),
    ("sexual assault", False),
    ("molest", False),
    ("incest", False),
    ("abuse", False),
    ("domestic violence", False),
    ("beat", False),
    ("hit", False),
    ("punch", False),
    ("kick", False),
    ("choke", False),
    ("strangle", False),
    ("slap", False),
    ("hurt", False),
    ("pain", False),
    ("suffer", False),
    ("suffering", False),
    ("torture", False),
    ("mutilate", False),
    ("dismember", False),
    ("decapitate", False),
    ("execute", False),
    ("assassinate", False),
    ("genocide", False),
    ("ethnic cleansing", False),
    ("concentration camp", False),
    ("holocaust", False),
    ("slavery", False),
    ("enslave", False),
    ("human trafficking", False),
    ("trafficking", False),
    ("sell you", False),
    ("buy you", False),
    ("kidnap you", False),
    ("take you", False),
    ("lock you up", False),
    ("tie you up", False),
    ("gag", False),
    ("blindfold", False),
    ("restrain", False),
    ("chain", False),
    ("cage", False),
    ("prison", False),
    ("jail", False),
    ("arrest", False),
    ("cops", False),
    ("police", False),
    ("fbi", False),
    ("cia", False),
    ("fbi open up", False),
    ("swat team", False),
    ("bust", False),
    ("raid", False),
    ("warrant", False),
    ("lawsuit", False),
    ("sue", False),
    ("illegal", False),
    ("crime", False),
    ("criminal", False),
    ("felony", False),
    ("misdemeanor", False),
    ("probation", False),
    ("parole", False),
    ("bail", False),
    ("lawyer", False),
    ("attorney", False),
    ("court", False),
    ("judge", False),
    ("sentence", False),
    ("prison time", False),
    ("death penalty", False),
    ("electric chair", False),
    ("lethal injection", False),
    ("hang", False),
    ("execute", False),
]


class PyxAI:
    """Pyx AI - Your trainable neural network. Easy to use, easy to edit."""

    def __init__(self):
        self.brain = PyxBrain(input_size=64, hidden_size=32, output_size=8)
        self.memory = PyxMemory(ban_threshold=BAN_LINE)
        self._load()
        self._load_training_grounds()

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

    def _load_training_grounds(self):
        """Train Pyx on built-in Training Grounds phrases (appropriate vs inappropriate)."""
        for text, safe in TRAINING_GROUNDS_PHRASES:
            self.set_label(text, safe, "phrases")

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
