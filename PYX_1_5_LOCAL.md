# Pyx 1.5 — Local Llama + GPT-OSS

**Requires the Pyx 1.5+ desktop app.** Pyx Talk, Pyx Code, and Pyxel all run
the same backends, but in 1.5 the LLM calls go to a **local OpenAI-compatible
server** (Ollama, LM Studio, llama.cpp, vLLM) on your machine instead of Groq.

Why upgrade from the 1.0 cloud build:

- **Up to 90% faster** *\** — no network round-trip for every token, no cloud queue.
- **~90% more environmentally friendly** *\** — inference reuses your existing
  hardware instead of spinning up a data-center GPU per message.
- **$0 per message.** One-time model download, then Pyx is free to use, offline.
- **Private.** Prompts and replies never leave your computer.

<sub>* Marketing claims based on the default Pyx 1.5 local path
(Llama 3.2 3B on a 2023+ Mac / modern laptop GPU) vs. a typical cloud chat
round-trip. Your results vary with hardware, model size, and prompt length.
The environmental comparison is a rough first-principles estimate, not an
audited metric.</sub>

## ⬇ Download a native installer

**Primary:** same files are listed on the site at **`https://pyx-ai.web.app/pyx-download.html`**.
`public/downloads/manifest.json` is always deployed: it lists **GitHub release asset URLs** in
`mirror` so downloads work even before the `host-site-downloads` job copies blobs into
`/downloads/`. After each new `v*` tag, update `version` and the `mirror` paths in that file
(or rely on CI to rewrite the manifest when Hosting mirror runs).

**Mirror:** [GitHub Releases](https://github.com/Mainline-Studios/pyx-ai/releases/latest)

| File | Platform | Notes |
|------|----------|-------|
| **`Pyx-<version>.dmg`**       | macOS 11+ (Apple Silicon + Intel) | Drag-to-Applications disk image |
| **`Pyx-<version>.pkg`**       | macOS                             | Double-click installer → `/Applications` |
| **`Pyx-<version>-setup.exe`** | Windows 10 / 11 (x64)             | Inno Setup installer with Start-menu entry |

On first launch Pyx opens a built-in **setup page** (`/pyx-setup.html`) that:

1. Detects [Ollama](https://ollama.com/download) — if missing, gives you the
   one-click download link for your OS.
2. Starts `ollama serve` in the background (no terminal needed).
3. Streams real progress bars while it downloads the default model set
   (~2 GB Llama 3.2 3B + ~5 GB Llama 3.1 8B + ~16 GB GPT-OSS 20B).
4. Opens Pyx Talk automatically when everything is ready.

The desktop build uses a **native window** (pywebview — not Safari/Chrome as the
main shell). Set **`PYX_USE_BROWSER=1`** if you want the old behavior (system
browser tab, easier DevTools).

You can also force the setup page any time with `PYX_FORCE_SETUP=1` before
launching, or visit `http://127.0.0.1:8765/pyx-setup.html` directly.

Don't see a download on the site yet? Push a `v*` tag so `release.yml` runs,
add the GitHub secret **`FIREBASE_SERVICE_ACCOUNT_PYX_AI`** (service account
JSON with **Firebase Hosting Admin**) so the **`host-site-downloads`** job can
mirror artifacts to Hosting — or build locally from `packaging/` (see
[`packaging/README.md`](packaging/README.md)).

**Why models are not inside the installer:** default Llama + GPT-OSS weights are
many gigabytes (Firebase Hosting also caps a single file around 2 GB). The app
ships code only; the first-run setup page pulls models through Ollama after you
install Ollama once.

The server code already supports this through two env vars:

| Variable | What it does |
|----------|--------------|
| `PYX_TALK_LLM_URL` | Full chat-completions URL (e.g. `http://127.0.0.1:11434/v1/chat/completions`). Any OpenAI-compatible server works. |
| `PYX_TALK_LLM_KEY` | Usually left empty for local. Only required when `PYX_TALK_LLM_URL` is the default Groq URL. |
| `PYX_TALK_MODEL_FAST` / `PYX_TALK_MODEL_SMART` / `PYX_TALK_MODEL_THINKING` | Which local model each Talk mode uses. |
| `PYX_CODE_MODEL` | Model for `/code_chat`. Default is `openai/gpt-oss-120b`. |
| `PYX_PIXEL_MODEL` | Model for `/pixel_art`. Default is `openai/gpt-oss-120b`. |

When `PYX_TALK_LLM_URL` points at `localhost`/`127.0.0.1`, `/health` reports
`backend.backend_kind = "local"` and the Pyx Talk status bar shows e.g.
**“Pyx 1.5 (local · Ollama) · Pyx Talk 1.0 · llama3.1:8b-instruct”**.

> Cloud Run cannot reach your computer. Local mode only works when you run
> Pyx (`python3 app.py` / `gunicorn app:app`) on the **same machine** as the
> model server, or when you expose the local server publicly (tunnel / VPN).

---

## 1. Install a local model server

Pick **one**. Ollama is by far the easiest.

### Option A — Ollama (recommended, one-liner)

- Download: <https://ollama.com/download>  
- Library of pre-packaged models: <https://ollama.com/library>  
- Docs: <https://github.com/ollama/ollama/blob/main/docs/api.md>  
- OpenAI-compatible endpoint (built in): <https://github.com/ollama/ollama/blob/main/docs/openai.md>

After install, `ollama serve` runs on <http://127.0.0.1:11434>. OpenAI-compatible
chat completions are at **`http://127.0.0.1:11434/v1/chat/completions`**.

### Option B — LM Studio (GUI, drag-and-drop GGUFs)

- Download: <https://lmstudio.ai/>  
- Built-in OpenAI-compatible server on **`http://127.0.0.1:1234/v1/chat/completions`**.

### Option C — llama.cpp `llama-server` (lean, C++ binary)

- Repo: <https://github.com/ggml-org/llama.cpp>  
- Releases (prebuilt binaries): <https://github.com/ggml-org/llama.cpp/releases>  
- Run: `llama-server -m <model.gguf> -c 8192 --host 127.0.0.1 --port 8080`  
- Endpoint: **`http://127.0.0.1:8080/v1/chat/completions`**.

### Option D — vLLM (GPU, high throughput, server-grade)

- Repo / install: <https://github.com/vllm-project/vllm>  
- Docs (OpenAI-compatible server): <https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html>  
- Default endpoint: **`http://127.0.0.1:8000/v1/chat/completions`**.

---

## 2. Download the models

### Llama (Meta)

- Official site / EULA request: <https://www.llama.com/>  
- Hugging Face org (gated — click **Agree** first):  
  <https://huggingface.co/meta-llama>
- Popular instruction-tuned checkpoints (used by Pyx Talk’s three modes):
  - Llama 3.1 8B Instruct — <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct>  
  - Llama 3.1 70B Instruct — <https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct>  
  - Llama 3.3 70B Instruct — <https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct>  
  - Llama 3.2 3B Instruct (smallest, fits on CPU/low-VRAM) — <https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct>

Pulling through Ollama is fastest (auto-downloads GGUFs, no HF login needed):

```bash
ollama pull llama3.1:8b-instruct      # used by Pyx Talk "fast"
ollama pull llama3.3:70b-instruct     # used by "smart" and "thinking"
ollama pull llama3.2:3b-instruct      # very small, CPU-friendly
```

### GPT-OSS (OpenAI’s open-weight models, released 2025)

- Announcement: <https://openai.com/index/introducing-gpt-oss/>  
- GitHub: <https://github.com/openai/gpt-oss>  
- Hugging Face (ungated):
  - GPT-OSS 120B — <https://huggingface.co/openai/gpt-oss-120b>  
  - GPT-OSS 20B — <https://huggingface.co/openai/gpt-oss-20b>  
- Ollama library: <https://ollama.com/library/gpt-oss>

Pull via Ollama:

```bash
ollama pull gpt-oss:20b     # 20B — ~16 GB, fits on a single high-end GPU
ollama pull gpt-oss:120b    # 120B — needs ~80 GB VRAM or heavy CPU offload
```

> Use **20B** for Pyx Code / Pyxel on a laptop; the default Groq config in Pyx
> used 120B. Adjust `PYX_CODE_MODEL` / `PYX_PIXEL_MODEL` to match what you
> downloaded (names below).

---

## 3. Point Pyx at the local server

The **model names** in env vars must match what your local server actually
loaded. For Ollama the name is the tag (`llama3.1:8b-instruct`); for LM Studio
it’s the model ID shown in its server tab; for llama.cpp / vLLM it’s the
model ID you started the server with.

### Example `.env.local` (Ollama)

```bash
# Route all LLM calls to local Ollama
export PYX_TALK_LLM_URL="http://127.0.0.1:11434/v1/chat/completions"
unset PYX_TALK_LLM_KEY   # not needed for local

# Pyx Talk modes
export PYX_TALK_MODEL_FAST="llama3.1:8b-instruct"
export PYX_TALK_MODEL_SMART="llama3.3:70b-instruct"
export PYX_TALK_MODEL_THINKING="llama3.3:70b-instruct"

# Pyx Code + Pyxel (pixel art): GPT-OSS
export PYX_CODE_MODEL="gpt-oss:20b"
export PYX_PIXEL_MODEL="gpt-oss:20b"

# Longer timeout for local inference on CPU / slower GPUs
export PYX_TALK_TIMEOUT="600"
```

### Run Pyx locally

```bash
cd /path/to/pyx-ai
source .env.local           # or copy lines into your shell / systemd unit
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
gunicorn app:app --bind 127.0.0.1:8080 --worker-class gthread --threads 16
```

Open <http://127.0.0.1:8080/pyx-talk.html> (serve `public/` with any static
server, or use `scripts/dev.js`), or hit the API directly:

```bash
curl -sS http://127.0.0.1:8080/health | jq '.backend'
# -> { "backend": "ollama", "backend_kind": "local", "label": "Pyx 1.5 (local · Ollama)", ... }
```

### Quick launcher

`scripts/pyx-local.sh` starts Ollama in the background (if installed), sets the
env vars for Ollama + the recommended models, and runs Gunicorn on `:8080`.

```bash
bash scripts/pyx-local.sh
```

---

## 4. Memory / disk cheatsheet

| Model | Disk (Q4 GGUF via Ollama) | Needs (roughly) |
|-------|--------------------------|------------------|
| Llama 3.2 3B Instruct | ~2 GB | any modern laptop, CPU OK |
| Llama 3.1 8B Instruct | ~5 GB | 16 GB RAM or 8 GB VRAM |
| Llama 3.3 70B Instruct | ~40 GB | 64 GB RAM or 48 GB VRAM (Q4) |
| GPT-OSS 20B | ~12–16 GB | 16–24 GB VRAM (or 32 GB RAM, slow) |
| GPT-OSS 120B | ~60–80 GB | Multi-GPU / 80 GB+ VRAM, or heavy CPU offload |

If a model is too big, drop to the next size down — Pyx Talk / Code / Pyxel all
read model names from env vars, so you can mix (e.g. `gpt-oss:20b` for code,
`llama3.1:8b-instruct` for everything else).

---

## 5. Going back to Groq cloud

Pyx 1.0 behavior is the default. To switch back, unset the local URL:

```bash
unset PYX_TALK_LLM_URL
export PYX_TALK_LLM_KEY="..."   # your Groq key
```

`/health` will again show `backend.backend = "groq"` and the Pyx Talk status
bar will switch from **Pyx 1.5 (local · …)** to **Pyx 1.0 (Groq cloud)**.
