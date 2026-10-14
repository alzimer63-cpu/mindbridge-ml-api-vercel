# MindBridge ml-api — Vercel attempt

This is a best-effort Vercel deployment of the ml-api service, prepared on
request. **Read this before spending time on it** — the honest assessment is
that it's very likely to fail on Vercel's serverless function size limit
(~250MB unzipped), not something fixable by more config:

| Dependency | Compressed download size |
|---|---|
| torch (cpu build) | 178.6 MB |
| torchvision (cpu build) | 1.8 MB |
| numpy | 13.6 MB |
| pandas | 14.9 MB |
| scikit-learn | 11.9 MB |
| xgboost | 4.3–146.8 MB (two different wheels exist; which one pip picks isn't fully in our control) |
| pillow | 4.1 MB |
| fastapi/pydantic/etc | ~6 MB |
| **Total** | **~235–378 MB, compressed** |

Unzipped/installed size is typically 10–40% larger than compressed wheel
size for binary-heavy packages like these — so even the best case is right
at Vercel's ceiling, before the 94MB CNN model is even considered. If the
deploy fails, look for an error like `A Serverless Function has exceeded the
unzipped maximum size` — that confirms it's the size limit, not a bug to fix.

## What's set up

- `api/index.py` — imports the FastAPI `app` from `alzimer_backend/main.py`
  (Vercel's Python runtime invokes ASGI apps directly from an `api/` file).
  It also sets `CNN_MODEL_PATH=/tmp/...` since Vercel's filesystem is
  read-only except `/tmp`.
- `requirements.txt` — trimmed as far as reasonably possible (no uvicorn
  server process needed, no unused training-only deps).
- `alzheimer_cnn_model.pkl` is **not included** — far too large regardless.
  If the base deploy somehow fits, the CNN/MRI endpoint would still need
  `CNN_MODEL_URL` set to download it into `/tmp` on cold start, which has
  its own real risk: a 94MB download + torch import + model load, inside a
  Hobby plan's 10s execution timeout, is likely too slow. Pro plan's 60s
  limit gives it a real chance if the size limit is somehow cleared.

## Env vars to set in the Vercel dashboard

- `JWT_SECRET` — must match the Node API's exactly.
- `CORS_ORIGINS` — your frontend's URL.
- `CNN_MODEL_URL` — only if you want to attempt the MRI endpoint too (see above).

## Deploying

```bash
npx vercel --cwd "path/to/this/folder"
```

or connect this folder as its own project in the Vercel dashboard. Either
way, you'll get a definitive answer from Vercel's own build log — paste it
back if you want help reading it.

## If it fails on size

The practical options at that point are the same as before: wait out
Render's maintenance window (the CPU-only + memory-dedup fixes already made
there might have solved the OOM on their own), or Google Cloud Run (needs a
card on file but the free quota comfortably covers this app).
