# MI mailing Worker (Resend) — no Google Cloud needed

**Live:** `https://mi-mailing.mainline-mi.workers.dev`  
Site form posts to `…/subscribe`.

## 1. Resend (required for actual email)

1. Sign up at https://resend.com → **Add Domain** → `pixelplaceofficial.com`
2. In Cloudflare DNS, add Resend’s DKIM CNAMEs + SPF
3. **SPF merge with iCloud** — one SPF TXT only, e.g.  
   `v=spf1 include:icloud.com include:_spf.resend.com ~all`  
   (keep Apple’s includes; add Resend; never two SPF records)
4. Wait for **Verified** in Resend
5. Create API key (Sending access)

## 2. Put the secret on the Worker

```bash
cd workers/mi-mailing
npx wrangler secret put RESEND_API_KEY
# paste Resend key
```

## 3. Redeploy if you change code

```bash
cd workers/mi-mailing
npx wrangler deploy
```

## Test

```bash
curl -s https://mi-mailing.mainline-mi.workers.dev/health
curl -s -X POST https://mi-mailing.mainline-mi.workers.dev/subscribe \
  -H 'content-type: application/json' \
  -H 'origin: https://pyx-ai.web.app' \
  -d '{"email":"you@icloud.com","source":"curl"}'
```
