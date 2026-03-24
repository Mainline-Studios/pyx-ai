# Deploy Pyx as serverless (no server 24/7)

Pyx can run as a **serverless function** so you don't run a server 24/7. It runs only when a request comes in (e.g. when Pixel Place sends a message to score).

## Options

| Platform        | When it runs      | Cost model              |
|-----------------|-------------------|--------------------------|
| **AWS Lambda**  | On each request   | Pay per invocation + time |
| **Google Cloud Functions** | On each request | Pay per invocation + time |
| **Vercel Serverless** | On each request | Free tier, then per invocation |

Same API: **POST** with `{"text": "..."}` → response `{"score", "bad", "censored"}`.

---

## AWS Lambda (quick path)

1. **Create a Lambda function** (Python 3.10+).
2. **Set handler** to `pyx_serverless.handler`.
3. **Upload code:** zip `Pyx_ai_moderator.py`, `pyx_serverless.py`, and the `data/` folder (if you use local memory). Include dependencies (e.g. `firebase-admin` if you use Firestore) in the zip or use a Lambda layer.
4. **Add HTTP trigger:** Create an API Gateway HTTP API (or REST API), link it to the Lambda. Note the invoke URL (e.g. `https://abc123.execute-api.us-east-1.amazonaws.com`).
5. **Call it:**  
   `POST https://your-api-url/default/score`  
   Body: `{"text": "user message"}`  
   Response: `{"score": 0.2, "bad": false, "censored": "user message"}`  

**Cold start:** First request after idle may take a few seconds (Pyx loads the model and training data). Later requests in the same container are fast.

**Timeout:** Set Lambda timeout to at least 30 seconds so the first load can finish.

**Memory:** 256–512 MB is usually enough; increase if you see timeouts.

---

## Google Cloud Functions

1. Create a **Cloud Function** (Python 3.10+), **HTTP** trigger.
2. Entry point: `pyx_serverless.handler` (or a thin wrapper that adapts the Flask request to the `event` dict format).
3. Cloud Functions passes a Flask-like request; you may need a small adapter that builds `event = {"body": request.get_data(as_text=True), "httpMethod": request.method}` and calls `handler(event, None)`, then returns the response body and status from the handler result.

Alternatively, use **Cloud Run** with a minimal Flask app that imports `pyx_serverless` and forwards the request to `handler(..., None)` — same idea, no server to run 24/7 (Cloud Run scales to zero when idle).

---

## After deployment

Your **Pyx URL** is the API’s base URL (e.g. Lambda function URL or API Gateway path).

From Pixel Place (or any client):

- **POST** `https://your-pyx-url/score` with `{"text": "..."}`.
- If response `bad` is `true`, show `censored` (or block). If `bad` is `false`, show the original text.

No server to keep running 24/7 — the function runs only when called.
