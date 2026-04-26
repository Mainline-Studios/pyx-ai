# Pyx 1.5 desktop — installers

The same Flask app (`app.py` + moderator + `public/` UI) packaged as a native
desktop app. **Default:** Llama + GPT-OSS as **`.gguf` files** in your Pyx models
folder, plus **[llama.cpp](https://github.com/ggerganov/llama.cpp) `llama-server`**
on `PATH` (or set `PYX_LLAMA_SERVER` to the binary). Optional: **`PYX_USE_OLLAMA=1`**
for the legacy Ollama flow. See [`PYX_1_5_LOCAL.md`](../PYX_1_5_LOCAL.md).

## Download (site + GitHub)

Installers are built automatically by `.github/workflows/release.yml` when a
version tag is pushed (e.g. `git tag v1.5.0 && git push origin v1.5.0`).
After the workflow finishes:

- **Site:** `https://pyx-ai.web.app/pyx-download.html` (files under `/downloads/`,
  mirrored by the `host-site-downloads` job — requires repo secret
  `FIREBASE_SERVICE_ACCOUNT_PYX_AI`).
- **GitHub:** [Releases](https://github.com/Mainline-Studios/pyx-ai/releases/latest)

| File | Platform | Installer type |
|------|----------|----------------|
| `Pyx-<version>.dmg`        | macOS (Apple Silicon + Intel) | Drag-to-Applications disk image |
| `Pyx-<version>.pkg`        | macOS                         | Standard `/Applications` installer |
| `Pyx-<version>-setup.exe`  | Windows 10 / 11 (x64)         | Inno Setup installer |

Once installed, Pyx launches a small local web server (127.0.0.1:8765) and
opens the UI in a **native app window** (pywebview), not your default browser.
A console window stays open for logs — close the Pyx window or the console to
stop the server. Set **`PYX_USE_BROWSER=1`** to use a normal browser tab instead.

## Build locally

### macOS (`.dmg` + `.pkg`)

```bash
brew install create-dmg            # only needed for the prettier .dmg; hdiutil fallback works without it
# Optional Apple distribution codesign (Developer ID or Apple Distribution):
#   export PYX_CODESIGN_IDENTITY="Developer ID Application: Your Team (XXXXXXXXXX)"
bash packaging/build-macos.sh      # PYX_VERSION=1.5.1 bash … to override
# Output: dist/Pyx.app, dist/Pyx-<ver>.dmg, dist/Pyx-<ver>.pkg
```

Edit **`packaging/gguf_manifest.json`** before shipping: set `download_url` on
each slot when you host the `.gguf` blobs, or ship filenames only and have users
copy files into the models directory shown in the setup screen.

### Windows (`.exe`)

Install [Inno Setup 6+](https://jrsoftware.org/isinfo.php) and make sure `iscc`
is on PATH. Then in PowerShell:

```powershell
pwsh .\packaging\build-windows.ps1 -Version 1.5.0
# Output: dist\Pyx\Pyx.exe (portable) and dist\Pyx-1.5.0-setup.exe (installer)
```

### Linux (portable)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt "pyinstaller>=6.3"
pyinstaller packaging/pyx.spec --noconfirm
# Output: dist/Pyx/Pyx  (run directly)
```

## Customise

Change the default models for the installer by exporting env vars before
launching the bundled `Pyx` binary (the launcher reads defaults from
`PYX_TALK_MODEL_FAST`, `PYX_TALK_MODEL_SMART`, `PYX_TALK_MODEL_THINKING`,
`PYX_CODE_MODEL`, `PYX_PIXEL_MODEL`, `PYX_TALK_LLM_URL`).

```bash
export PYX_TALK_MODEL_FAST="llama3.2:3b-instruct"
export PYX_CODE_MODEL="gpt-oss:20b"
open -a Pyx
```

## Signing / notarisation

Neither macOS codesigning nor Authenticode signing is configured out of the
box — users will see a Gatekeeper / SmartScreen warning on first launch. To
sign:

- macOS: add `codesign --sign "Developer ID Application: …" --deep` after the
  `.app` is built, then `xcrun notarytool submit … --wait` and `xcrun stapler
  staple` the `.dmg`/`.pkg`.
- Windows: sign `dist\Pyx\Pyx.exe` and the generated `-setup.exe` with
  `signtool sign /fd SHA256 …` using an EV / OV cert.
