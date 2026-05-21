"""Shared JSON storage for Work with Pyx applications."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "workforpyx"
APPLICATIONS_PATH = DATA_DIR / "applications.json"


def ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "resumes").mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def load_applications() -> list[dict[str, Any]]:
    ensure_data_dir()
    if not APPLICATIONS_PATH.is_file():
        return []
    raw = APPLICATIONS_PATH.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    data = json.loads(raw)
    return data if isinstance(data, list) else []


def save_applications(apps: list[dict[str, Any]]) -> None:
    ensure_data_dir()
    APPLICATIONS_PATH.write_text(
        json.dumps(apps, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def find_application(app_id: str) -> dict[str, Any] | None:
    for app in load_applications():
        if app.get("id") == app_id:
            return app
    return None


def update_application(app_id: str, patch: dict[str, Any]) -> dict[str, Any] | None:
    apps = load_applications()
    updated: dict[str, Any] | None = None
    for i, app in enumerate(apps):
        if app.get("id") != app_id:
            continue
        apps[i] = {**app, **patch}
        updated = apps[i]
        break
    if updated is None:
        return None
    save_applications(apps)
    return updated
