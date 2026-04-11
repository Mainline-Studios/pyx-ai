#!/usr/bin/env python3
"""Build public/pyx-trainer-auth.html from template + one-line password (never committed).

Reads passphrase from:
  1) Environment variable PYX_TRAINER_PASSWORD
  2) File TRAINER_GATE_PASSWORD.txt (first non-empty, non-# line)

Run automatically before Firebase Hosting deploy (firebase.json predeploy).
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "public" / "pyx-trainer-auth.template.html"
OUT = ROOT / "public" / "pyx-trainer-auth.html"
PASSWORD_FILE = ROOT / "TRAINER_GATE_PASSWORD.txt"


def main() -> int:
    secret = (os.environ.get("PYX_TRAINER_PASSWORD") or "").strip()
    if not secret and PASSWORD_FILE.is_file():
        for line in PASSWORD_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            secret = line
            break
    if not secret:
        print(
            "Missing trainer password. Set PYX_TRAINER_PASSWORD or add TRAINER_GATE_PASSWORD.txt\n"
            "(one line). See TRAINER_GATE_PASSWORD.example.txt",
            file=sys.stderr,
        )
        return 1
    if not TEMPLATE.is_file():
        print("Missing template: public/pyx-trainer-auth.template.html", file=sys.stderr)
        return 1
    digest = hashlib.sha256(secret.encode("utf-8")).hexdigest()
    text = TEMPLATE.read_text(encoding="utf-8")
    if "__EXPECTED_SHA256_HEX__" not in text:
        print("Template missing __EXPECTED_SHA256_HEX__", file=sys.stderr)
        return 1
    OUT.write_text(text.replace("__EXPECTED_SHA256_HEX__", digest), encoding="utf-8")
    print(f"Wrote {OUT} (hash only; password not logged).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
