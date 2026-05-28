#!/usr/bin/env python3
"""Build public/pyx-trainer-auth.html from the template.

The trainer gate now uses Firebase email-link sign-in (allowlisted accounts),
so no password/secret is required. This script simply renders the template to
the deployed file. For backward compatibility, if the template still contains a
`__EXPECTED_SHA256_HEX__` placeholder it will be filled from
PYX_TRAINER_PASSWORD / TRAINER_GATE_PASSWORD.txt when available, otherwise
blanked out (never failing the deploy).

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
PLACEHOLDER = "__EXPECTED_SHA256_HEX__"


def _read_secret() -> str:
    secret = (os.environ.get("PYX_TRAINER_PASSWORD") or "").strip()
    if not secret and PASSWORD_FILE.is_file():
        for line in PASSWORD_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            secret = line
            break
    return secret


def main() -> int:
    if not TEMPLATE.is_file():
        print("Missing template: public/pyx-trainer-auth.template.html", file=sys.stderr)
        return 1
    text = TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER in text:
        secret = _read_secret()
        digest = hashlib.sha256(secret.encode("utf-8")).hexdigest() if secret else ""
        text = text.replace(PLACEHOLDER, digest)
    OUT.write_text(text, encoding="utf-8")
    print(f"Wrote {OUT} (Firebase email-link sign-in; no secret baked in).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
