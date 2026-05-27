"""Dev Workshop — traffic light image analyzer (train from web images, return signal color)."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from workforpyx_storage import DATA_DIR

TRAFFIC_PATH = DATA_DIR / "traffic_lights.json"
IMAGE_CACHE_DIR = DATA_DIR / "traffic_images"
FEATURE_DIM = 8
TRAFFIC_PROTOCOL_VERSION = 1
# Same path for still images and live frames (features extracted in browser).
ANALYZE_MODES = frozenset({"image", "frame", "live"})

COLOR_HEX = {
    "red": "#ef4444",
    "yellow": "#eab308",
    "green": "#22c55e",
    "off": "#64748b",
    "unknown": "#94a3b8",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_db() -> dict[str, Any]:
    if not TRAFFIC_PATH.is_file():
        return {"samples": [], "events": []}
    try:
        data = json.loads(TRAFFIC_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"samples": [], "events": []}
    if not isinstance(data, dict):
        return {"samples": [], "events": []}
    if not isinstance(data.get("samples"), list):
        data["samples"] = []
    if not isinstance(data.get("events"), list):
        data["events"] = []
    return data


def _save_db(data: dict[str, Any]) -> None:
    TRAFFIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAFFIC_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _new_id() -> str:
    import secrets

    return "tl_" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + "_" + secrets.token_hex(3)


def _normalize_features(raw: Any) -> list[float] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) < FEATURE_DIM:
        return None
    out: list[float] = []
    for i in range(FEATURE_DIM):
        try:
            out.append(float(raw[i]))
        except (TypeError, ValueError):
            return None
    return out


def _features_from_image_bytes(data: bytes) -> list[float]:
    try:
        from PIL import Image
    except ImportError as e:
        raise RuntimeError(
            "Server-side image decode requires Pillow. "
            "Analyze in the browser or add Pillow to the API image."
        ) from e
    import io

    im = Image.open(io.BytesIO(data)).convert("RGB")
    im = im.resize((120, 90))
    w, h = im.size
    top_h = max(1, int(h * 0.55))
    pixels = list(im.getdata())

    mean_r = mean_g = mean_b = 0.0
    cnt_r = cnt_y = cnt_g = 0
    n = 0
    bright_sum = 0.0
    for y in range(top_h):
        for x in range(w):
            r, g, b = pixels[y * w + x]
            mean_r += r
            mean_g += g
            mean_b += b
            bright_sum += (r + g + b) / 3.0
            n += 1
            if r > 165 and g < 115 and b < 115:
                cnt_r += 1
            elif g > 145 and r < 150 and b < 130:
                cnt_g += 1
            elif r > 145 and g > 125 and b < 95:
                cnt_y += 1
    if n < 1:
        n = 1
    mean_r /= n
    mean_g /= n
    mean_b /= n
    bright_top = bright_sum / n
    bottom_n = max(1, (h - top_h) * w)
    bottom_bright = 0.0
    for y in range(top_h, h):
        for x in range(w):
            r, g, b = pixels[y * w + x]
            bottom_bright += (r + g + b) / 3.0
    bottom_bright /= bottom_n
    return [
        mean_r / 255.0,
        mean_g / 255.0,
        mean_b / 255.0,
        cnt_r / n,
        cnt_y / n,
        cnt_g / n,
        bright_top / 255.0,
        bottom_bright / 255.0,
    ]


def _traffic_user_agent() -> str:
    return os.environ.get("PYX_TRAFFIC_UA", "").strip() or (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )


def _guess_image_ext(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    if data[:4] == b"RIFF" and len(data) > 12 and data[8:12] == b"WEBP":
        return ".webp"
    return ".jpg"


def _local_cached_image_path(url: str) -> Path | None:
    """Resolve our public proxy URL to a file on disk."""
    u = (url or "").strip()
    marker = "/api/dev-workshop/traffic/img/"
    if marker not in u:
        return None
    name = u.split(marker, 1)[-1].split("?")[0].strip("/")
    if not name or ".." in name or "/" in name:
        return None
    path = IMAGE_CACHE_DIR / os.path.basename(name)
    return path if path.is_file() else None


def _fetch_image_bytes(url: str) -> bytes:
    url = (url or "").strip()
    local = _local_cached_image_path(url)
    if local:
        return local.read_bytes()
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("image_url must be http(s)")
    req = urllib.request.Request(url, headers={"User-Agent": _traffic_user_agent()})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read()


def _ddg_vqd(query: str) -> str | None:
    enc = urllib.parse.quote(query)
    init_url = f"https://duckduckgo.com/?q={enc}&iax=images&ia=images"
    req = urllib.request.Request(
        init_url,
        headers={
            "User-Agent": _traffic_user_agent(),
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
            "Referer": "https://duckduckgo.com/",
        },
    )
    with urllib.request.urlopen(req, timeout=22) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    for pat in (
        r"vqd=([\d-]+)",
        r"vqd['\"]\s*:\s*['\"]([\d-]+)['\"]",
        r"vqd\\\":([\d-]+)",
    ):
        m = re.search(pat, html)
        if m:
            return m.group(1)
    return None


def search_web_images(query: str, *, max_results: int = 12) -> tuple[list[dict[str, Any]], str | None]:
    """DuckDuckGo image search (no API key). Returns (items, error)."""
    query = (query or "").strip()[:300]
    if not query:
        return [], "empty query"
    max_results = max(1, min(int(max_results), 24))
    vqd = _ddg_vqd(query)
    if not vqd:
        return [], "image search unavailable (could not get search token)"
    params = urllib.parse.urlencode(
        {
            "l": "us-en",
            "o": "json",
            "q": query,
            "vqd": vqd,
            "f": ",,,",
            "p": "1",
        }
    )
    api_url = f"https://duckduckgo.com/i.js?{params}"
    req = urllib.request.Request(
        api_url,
        headers={
            "User-Agent": _traffic_user_agent(),
            "Accept": "application/json,text/javascript,*/*",
            "Referer": "https://duckduckgo.com/",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=22) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        return [], str(e)[:300]
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return [], "image search returned invalid JSON"
    out: list[dict[str, Any]] = []
    for row in payload.get("results") or []:
        if not isinstance(row, dict):
            continue
        img = (row.get("image") or row.get("thumbnail") or "").strip()
        if not img or not img.startswith("http"):
            continue
        out.append(
            {
                "title": (row.get("title") or "")[:200],
                "image_url": img,
                "thumbnail": (row.get("thumbnail") or img)[:2000],
                "page_url": (row.get("url") or "")[:2000],
                "source": (row.get("source") or "")[:120],
            }
        )
        if len(out) >= max_results:
            break
    if not out:
        return [], "no images found for that query"
    return out, None


def publish_images_for_training(
    query: str, *, max_results: int = 12
) -> tuple[list[dict[str, Any]], str | None]:
    """
    Search the web, download images to server cache, return public proxy URLs for the trainer UI.
    """
    items, err = search_web_images(query, max_results=max_results)
    if err:
        return [], err
    IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    published: list[dict[str, Any]] = []
    for row in items:
        src = row["image_url"]
        try:
            data = _fetch_image_bytes(src)
        except Exception:
            continue
        if len(data) < 500:
            continue
        ext = _guess_image_ext(data)
        fid = _new_id() + ext
        path = IMAGE_CACHE_DIR / fid
        path.write_bytes(data)
        public_path = f"/api/dev-workshop/traffic/img/{fid}"
        published.append(
            {
                "id": fid,
                "public_url": public_path,
                "source_url": src,
                "thumbnail_url": public_path,
                "page_url": row.get("page_url") or "",
                "title": row.get("title") or "",
                "query": query,
                "cached_at": _now_iso(),
            }
        )
    if not published:
        return [], "could not download any images (blocked or hotlink protected)"
    return published, None


def sample_stats(samples: list[dict]) -> dict[str, int]:
    counts = {c: 0 for c in COLOR_HEX}
    for s in samples:
        c = str(s.get("color") or "unknown").lower()
        if c in counts:
            counts[c] += 1
        else:
            counts["unknown"] += 1
    return counts


def heuristic_classify(features: list[float]) -> tuple[str, float, bool]:
    """Return (color, confidence, traffic_light_detected)."""
    mean_r, mean_g, mean_b, ratio_r, ratio_y, ratio_g, bright_top, _ = features
    signal = ratio_r + ratio_y + ratio_g
    detected = signal > 0.006 or bright_top > 0.28 or max(mean_r, mean_g, mean_b) > 0.25
    if not detected:
        return "unknown", 0.35, False

    # Dominant channel (works when the lit bulb fills much of the crop)
    if mean_g >= mean_r and mean_g >= mean_b and mean_g > 0.22:
        if mean_g > mean_r * 1.05 or ratio_g >= ratio_r:
            return "green", min(0.94, 0.42 + mean_g * 0.9 + ratio_g * 6), True
    if mean_r >= mean_g and mean_r >= mean_b and mean_r > 0.22:
        if mean_r > mean_g * 1.05 or ratio_r >= ratio_g:
            return "red", min(0.94, 0.42 + mean_r * 0.9 + ratio_r * 6), True
    if mean_r > 0.2 and mean_g > 0.18 and ratio_y + ratio_g > ratio_r * 0.5:
        if ratio_y >= ratio_r * 0.85 and ratio_y >= ratio_g * 0.7:
            return "yellow", min(0.9, 0.4 + ratio_y * 7 + mean_g * 0.3), True

    if ratio_r >= ratio_y and ratio_r >= ratio_g and ratio_r > 0.008:
        return "red", min(0.92, 0.5 + ratio_r * 8), True
    if ratio_g >= ratio_r and ratio_g >= ratio_y and ratio_g > 0.008:
        return "green", min(0.92, 0.5 + ratio_g * 8), True
    if ratio_y >= ratio_r and ratio_y >= ratio_g and ratio_y > 0.008:
        return "yellow", min(0.9, 0.45 + ratio_y * 8), True
    if bright_top < 0.22:
        return "off", 0.55, True
    return "unknown", 0.4, detected


def knn_classify(features: list[float], samples: list[dict]) -> tuple[str, float] | None:
    if len(samples) < 1:
        return None
    vec = np.array(features, dtype=np.float64)
    if len(samples) == 1:
        f = _normalize_features(samples[0].get("features"))
        if not f:
            return None
        d = float(np.linalg.norm(vec - np.array(f, dtype=np.float64)))
        color = str(samples[0].get("color") or "unknown")
        conf = max(0.45, min(0.95, 1.0 - d * 1.2))
        return color, conf
    dists: list[tuple[float, str]] = []
    for s in samples:
        f = _normalize_features(s.get("features"))
        if not f:
            continue
        d = float(np.linalg.norm(vec - np.array(f, dtype=np.float64)))
        dists.append((d, str(s.get("color") or "unknown")))
    if not dists:
        return None
    dists.sort(key=lambda x: x[0])
    k = min(5, len(dists))
    neighbors = dists[:k]
    votes: dict[str, float] = {}
    for d, color in neighbors:
        weight = 1.0 / (d + 1e-6)
        votes[color] = votes.get(color, 0.0) + weight
    best = max(votes.items(), key=lambda x: x[1])
    total_w = sum(votes.values()) or 1.0
    confidence = min(0.98, best[1] / total_w)
    # Distance sanity: if nearest is very far, low confidence
    nearest_d = neighbors[0][0]
    if nearest_d > 0.45:
        confidence *= max(0.25, 1.0 - nearest_d)
    return best[0], confidence


def list_samples() -> list[dict[str, Any]]:
    return list(_load_db().get("samples") or [])


def add_training_sample(
    image_url: str,
    color: str,
    features: list[float],
    *,
    dev: str = "dev",
) -> dict[str, Any]:
    color = (color or "").strip().lower()
    if color not in COLOR_HEX:
        raise ValueError("color must be red, yellow, green, off, or unknown")
    feats = _normalize_features(features)
    if not feats:
        raise ValueError("features must be a list of 8 numbers")
    url = (image_url or "").strip()
    if not url:
        raise ValueError("image_url required")
    db = _load_db()
    sample = {
        "id": _new_id(),
        "image_url": url[:2000],
        "color": color,
        "features": feats,
        "created": _now_iso(),
        "dev": (dev or "dev")[:40],
    }
    db["samples"].append(sample)
    _save_db(db)
    return sample


def delete_sample(sample_id: str) -> bool:
    sample_id = (sample_id or "").strip()
    if not sample_id:
        return False
    db = _load_db()
    before = len(db["samples"])
    db["samples"] = [s for s in db["samples"] if s.get("id") != sample_id]
    if len(db["samples"]) == before:
        return False
    _save_db(db)
    return True


def traffic_capabilities() -> dict[str, Any]:
    """Contract for clients building live video later."""
    return {
        "protocol_version": TRAFFIC_PROTOCOL_VERSION,
        "feature_dim": FEATURE_DIM,
        "analyze_modes": sorted(ANALYZE_MODES),
        "live_video": {
            "status": "preview",
            "recommended_client_fps": 5,
            "recommended_emit_hold_ms": 400,
            "endpoints": {
                "analyze_frame": "/api/dev-workshop/traffic/frame",
                "analyze": "/api/dev-workshop/traffic/analyze",
                "emit": "/api/dev-workshop/traffic/emit",
                "capabilities": "/api/dev-workshop/traffic/capabilities",
                "search_images": "/api/dev-workshop/traffic/search-images",
                "public_image": "/api/dev-workshop/traffic/img/{id}",
            },
            "notes": (
                "Extract features in the browser from each video frame (canvas), POST features "
                "with mode=frame. Full WebRTC/streaming server is not required for v1 live."
            ),
        },
    }


def analyze_features(
    features: list[float],
    *,
    mode: str = "image",
    source: str | None = None,
    frame_id: str | None = None,
    image_url: str | None = None,
) -> dict[str, Any]:
    """Classify a precomputed feature vector (still image or live frame)."""
    mode = (mode or "image").strip().lower()
    if mode not in ANALYZE_MODES:
        mode = "image"
    feats = _normalize_features(features)
    if not feats:
        return {"ok": False, "error": "features must be a list of 8 numbers"}

    samples = list_samples()
    method = "heuristic"
    color, confidence, detected = heuristic_classify(feats)
    knn = knn_classify(feats, samples)
    if knn:
        k_color, k_conf = knn
        min_conf = 0.38 if len(samples) < 3 else 0.42
        if k_conf >= min_conf or k_conf > confidence:
            color, confidence = k_color, k_conf
            method = "knn"

    detected = detected or (confidence > 0.5 and color != "unknown")
    hex_color = COLOR_HEX.get(color, COLOR_HEX["unknown"])
    out: dict[str, Any] = {
        "ok": True,
        "color": color,
        "hex": hex_color,
        "confidence": round(confidence, 3),
        "traffic_light_detected": bool(detected),
        "method": method,
        "training_samples": len(samples),
        "features": feats,
        "mode": mode,
        "protocol_version": TRAFFIC_PROTOCOL_VERSION,
    }
    if source:
        out["source"] = str(source)[:120]
    if frame_id:
        out["frame_id"] = str(frame_id)[:80]
    url = (image_url or "").strip()
    if url:
        out["image_url"] = url[:2000]
    return out


def analyze_image(
    *,
    image_url: str | None = None,
    features: list[float] | None = None,
    mode: str = "image",
    source: str | None = None,
    frame_id: str | None = None,
) -> dict[str, Any]:
    feats = _normalize_features(features) if features is not None else None
    url = (image_url or "").strip() or None
    if feats is None and url:
        try:
            raw = _fetch_image_bytes(url)
            feats = _features_from_image_bytes(raw)
        except urllib.error.URLError as e:
            return {"ok": False, "error": f"Could not fetch image: {e.reason}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}
    if feats is None:
        return {"ok": False, "error": "Provide image_url or features from the browser analyzer."}

    return analyze_features(
        feats,
        mode=mode,
        source=source,
        frame_id=frame_id,
        image_url=url,
    )


def record_emit(
    color: str,
    hex_value: str,
    *,
    source: str = "workshop",
    mode: str | None = None,
    frame_id: str | None = None,
) -> dict[str, Any]:
    db = _load_db()
    event = {
        "at": _now_iso(),
        "color": color,
        "hex": hex_value,
        "source": source[:80],
    }
    if mode:
        event["mode"] = str(mode)[:20]
    if frame_id:
        event["frame_id"] = str(frame_id)[:80]
    db["events"] = (db.get("events") or [])[-99:] + [event]
    _save_db(db)
    return event
