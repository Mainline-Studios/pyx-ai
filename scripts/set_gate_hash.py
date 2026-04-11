#!/usr/bin/env python3
"""Build public/pyx-trainer-gate.html from template + secret passphrase (never committed).

Passphrase source (first match):
  1) Environment variable PYX_TRAINER_GATE_PASSWORD
  2) File TRAINER_GATE_PASSWORD.txt (repo root; gitignored)

Run before Firebase Hosting deploy (see firebase.json predeploy).
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "public" / "pyx-trainer-gate.template.html"
OUT = ROOT / "public" / "pyx-trainer-gate.html"
PASSWORD_FILE = ROOT / "TRAINER_GATE_PASSWORD.txt"


def main() -> int:
    secret = (os.environ.get("PYX_TRAINER_GATE_PASSWORD") or "").strip()
    if not secret and PASSWORD_FILE.is_file():
        for line in PASSWORD_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            secret = line
            break
    if not secret:
        print(
            "Missing passphrase. Set PYX_TRAINER_GATE_PASSWORD or create TRAINER_GATE_PASSWORD.txt\n"
            "See TRAINER_GATE_PASSWORD.example.txt",
            file=sys.stderr,
        )
        return 1
    digest = hashlib.sha256(secret.encode("utf-8")).hexdigest()
    text = TEMPLATE.read_text(encoding="utf-8")
    if "__EXPECTED_SHA256_HEX__" not in text:
        print("Template missing __EXPECTED_SHA256_HEX__ placeholder.", file=sys.stderr)
        return 1
    text = text.replace("__EXPECTED_SHA256_HEX__", digest)
    OUT.write_text(text, encoding="utf-8")
    print(f"Wrote {OUT} (SHA-256 hash only; passphrase not logged).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
