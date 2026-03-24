"""
Firestore sync for Pyx AI — project pyx-ai.
Stores training phrases so overrides and user training update the cloud DB.
Requires: pip install firebase-admin
Credentials: GOOGLE_APPLICATION_CREDENTIALS env, or firebase-key.json in project/cwd.
"""

import hashlib
import os
from pathlib import Path
from typing import List, Tuple, Optional, Any

# Project ID from https://console.firebase.google.com/project/pyx-ai
FIRESTORE_PROJECT_ID = "pyx-ai"
COLLECTION_PHRASES = "phrases"

_db: Any = None


def _doc_id(text: str) -> str:
    """Stable document ID from phrase text (safe for Firestore)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:50]


def _find_key_file(key_path: Optional[str] = None) -> Optional[Path]:
    """Resolve path to service account JSON. Tries arg, env, then common locations."""
    if key_path:
        p = Path(key_path).resolve()
        return p if p.exists() else None
    env = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if env and Path(env).exists():
        return Path(env).resolve()
    candidates = [
        Path(__file__).parent / "firebase-key.json",
        Path.cwd() / "firebase-key.json",
        Path.cwd().parent / "firebase-key.json",
    ]
    for p in candidates:
        if p.resolve().exists():
            return p.resolve()
    return None


def init_firestore(key_path: Optional[str] = None):
    """Initialize Firestore. Returns db client or None if credentials missing.
    key_path: optional path to service account JSON (overrides env / firebase-key.json).
    When successful, the app will update Firestore on every set_label, add_phrase, add_word, add_game_idea, and ai_decide (when it adds).
    """
    global _db
    if _db is not None:
        return _db
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
    except ImportError as e:
        print(f"  Error: {e}")
        print("  Run: pip3 install firebase-admin")
        return None
    path = _find_key_file(key_path)
    try:
        if not firebase_admin._apps:
            if path:
                cred = credentials.Certificate(str(path))
                firebase_admin.initialize_app(cred)
            else:
                # Cloud Run / GCP: use Application Default Credentials (no key file)
                try:
                    cred = credentials.ApplicationDefault()
                    project_id = os.environ.get("FIREBASE_PROJECT_ID", os.environ.get("GOOGLE_CLOUD_PROJECT", FIRESTORE_PROJECT_ID))
                    firebase_admin.initialize_app(cred, {"projectId": project_id})
                except Exception as adc_err:
                    if key_path:
                        print(f"  Error: Key file not found: {key_path}")
                    else:
                        print("  Error: No credentials. Set GOOGLE_APPLICATION_CREDENTIALS or add firebase-key.json, or run on GCP with ADC.")
                    print(f"  ADC error: {adc_err}")
                    return None
        # Use (default) DB. For a named database set env: FIRESTORE_DATABASE_ID=your-db-id
        database_id = os.environ.get("FIRESTORE_DATABASE_ID")
        _db = firestore.client(database_id=database_id) if database_id else firestore.client()
        return _db
    except Exception as e:
        print(f"  Error: {e}")
        if "does not exist" in str(e) or "404" in str(e):
            # Show project from key so user can create DB in the correct project
            try:
                import json
                pid = "pyx-ai"
                if path and path.exists():
                    key_data = json.loads(path.read_text())
                    pid = key_data.get("project_id", pid)
                print(f"  → Your key is for project: {pid}")
                print(f"  → Create the database: https://console.cloud.google.com/datastore/setup?project={pid}")
            except Exception:
                print("  → Create the database: https://console.cloud.google.com/datastore/setup?project=pyx-ai")
            print("  → Choose 'Firestore in Native mode', then wait 1–2 minutes and try again.")
        return None


def get_phrases_from_firestore(db) -> List[Tuple[str, bool, str]]:
    """Load all phrases from Firestore. Returns list of (text, safe, category)."""
    if db is None:
        return []
    out: List[Tuple[str, bool, str]] = []
    try:
        ref = db.collection(COLLECTION_PHRASES)
        for doc in ref.stream():
            d = doc.to_dict()
            text = d.get("text") or ""
            safe = bool(d.get("safe", True))
            category = d.get("category") or "phrases"
            if text:
                out.append((text, safe, category))
    except Exception:
        pass
    return out


def set_phrase_in_firestore(
    db, text: str, safe: bool, category: str = "phrases", source: str = "user"
) -> bool:
    """Upsert one phrase in Firestore. Returns True if written."""
    if db is None or not text:
        return False
    try:
        from datetime import datetime
        ref = db.collection(COLLECTION_PHRASES).document(_doc_id(text))
        ref.set({
            "text": text,
            "safe": safe,
            "category": category,
            "source": source,
            "updated_at": datetime.utcnow(),
        })
        return True
    except Exception as e:
        # Log first failure so user sees permission/API errors
        if not getattr(set_phrase_in_firestore, "_logged", False):
            print(f"  Firestore write error: {e}")
            set_phrase_in_firestore._logged = True
        return False


def seed_firestore(db, phrases: List[Tuple[str, bool]], category: str = "phrases") -> int:
    """Write built-in phrases to Firestore. Returns count written."""
    if db is None:
        return 0
    n = 0
    for text, safe in phrases:
        if set_phrase_in_firestore(db, text, safe, category, source="builtin"):
            n += 1
    return n
