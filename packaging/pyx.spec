# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Pyx 1.5 desktop.

Build from repo root:
    pyinstaller packaging/pyx.spec --noconfirm

Produces dist/Pyx (one-folder bundle). macOS post-step wraps it as Pyx.app.
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH).resolve().parent
ENTRY = str(ROOT / "packaging" / "launcher.py")

_wv_datas, _wv_binaries, _wv_hidden = collect_all("webview")

datas = [
    (str(ROOT / "public"), "public"),
    (str(ROOT / "app.py"), "."),
    (str(ROOT / "Pyx_ai_moderator.py"), "."),
    (str(ROOT / "Pyx_ai_code.py"), "."),
    (str(ROOT / "Pyx_ai_check.py"), "."),
    (str(ROOT / "Pyx_ai_analyze.py"), "."),
    (str(ROOT / "packaging" / "bootstrap.py"), "."),
    (str(ROOT / "packaging" / "gguf_engine.py"), "."),
    (str(ROOT / "packaging" / "gguf_manifest.json"), "."),
] + _wv_datas

# Keep bundle lean — Firebase / GCP SDKs aren't used in local mode and add ~150 MB.
excludes = [
    "firebase_admin",
    "google.cloud",
    "google.auth",
    "google.api_core",
    "grpc",
    "proto",
    "firestore_sync",
    "tests",
    "pytest",
]

hiddenimports = [
    "flask",
    "werkzeug",
    "jinja2",
    "markupsafe",
    "itsdangerous",
    "click",
    "blinker",
    "webview",
] + list(_wv_hidden)

a = Analysis(
    [ENTRY],
    pathex=[str(ROOT)],
    binaries=_wv_binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Pyx",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,     # keep a console — handy for Ollama errors / Ctrl+C stop
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Pyx",
)
