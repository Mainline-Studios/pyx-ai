# Dev Workshop — Traffic light analyzer

## Current (v1)

- **Still images:** browser or server loads URL → 8-D feature vector → `POST /api/dev-workshop/traffic/analyze`
- **Live preview:** camera or video file → canvas sample every N ms → `POST /api/dev-workshop/traffic/frame` (same classifier)
- **Training:** labeled features in `data/workforpyx/traffic_lights.json` (k-NN when ≥2 samples)
- **Output:** `pyx-traffic-color` event + optional `POST /api/dev-workshop/traffic/emit`

## Live video contract (stable for future work)

| Field | Purpose |
|--------|---------|
| `features` | 8 floats from top-of-frame color stats (required for frames) |
| `mode` | `image` \| `frame` \| `live` |
| `source` | e.g. `camera`, `file:clip.mp4`, later `webrtc:room-id` |
| `frame_id` | monotonic id per frame (`f42`) |
| `protocol_version` | `1` (see `GET /api/dev-workshop/traffic/capabilities`) |

Games should listen for:

```js
window.addEventListener("pyx-traffic-color", (e) => {
  const { color, hex, confidence, mode, frame_id, source } = e.detail;
});
```

## Later: full real-time (planned)

These can be added **without** changing feature format or training JSON:

1. **WebRTC / RTSP ingest** — server receives stream; worker samples frames and calls `analyze_features()` directly (no browser).
2. **WebSocket push** — `wss://…/traffic/stream` pushes `{ color, hex, frame_id, ts }` at 10–30 Hz instead of polling HTTP.
3. **Edge / on-device** — run the same 8-D heuristic + optional tiny k-NN in WASM for sub-100ms latency.
4. **ROI detector** — replace “top 55% of frame” with a bounding box model; still output the same `color` + `hex`.

Recommended client settings today (capabilities endpoint): **5 fps**, **400 ms** color hold before re-emit.
