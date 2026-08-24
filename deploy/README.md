# Deploying SourceBound

The deployed service serves the agent only — it does not run the eval harness.

## Why not Hugging Face Spaces

HF now requires a PRO subscription for any compute-backed Space. Only *Static*
Spaces are free, and those are browser-only — they cannot run a Python process,
call Groq, or query a Chroma index. So HF is out unless you pay.

Render's free web-service tier runs the same Dockerfile with no card required.

## Why the index is committed rather than built

`chroma/` (~98MB, 5,312 chunks) is committed via **Git LFS**. Building it
instead would mean embedding those chunks on CPU — roughly 14 minutes — and
Render's free tier has no persistent disk, so that rebuild would repeat on
every cold start and blow past the healthcheck window. The index is a build
artifact, so it ships like one.

This also makes the repo self-contained: a clone gets a working system without
a 14-minute build step.

## Deploying to Render

1. Push this repo to GitHub (LFS objects included).
2. At [dashboard.render.com](https://dashboard.render.com) → **New → Blueprint**,
   point it at the repo. `render.yaml` supplies the rest.
3. Set **`GROQ_API_KEY`** in the service's Environment tab.
   Optionally add `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` /
   `LANGFUSE_HOST` for tracing — without them the agent runs untraced rather
   than failing.

Or run it anywhere else that accepts a Dockerfile:

```bash
docker build -f deploy/Dockerfile -t sourcebound .
docker run -p 8000:8000 -e GROQ_API_KEY=... sourcebound
```

## Verifying a deploy

```
GET /health  ->  {"status":"ok","graph_ready":true,"indexed_chunks":1645,"tracing":false}
```

**`indexed_chunks` is the field that matters.** Chroma returns an *empty
collection* rather than an error when the index is missing, so without it a
broken deploy would boot, report healthy, and answer every question with zero
retrieved context — a failure that looks like success. Startup now refuses to
come up at all if the docs index is empty.

## Expected behaviour

- **Cold starts are slow.** Render's free tier sleeps a service after ~15
  minutes idle; the next request takes ~60s to wake it, plus the graph builds a
  ~400MB embedding model into memory. Subsequent requests are fast — the model
  loads once at startup, not per request.
- **Quota is shared.** Groq's free tier is per-key, not per-user, so the
  Space's visitors share one budget. When it is exhausted `/ask` returns
  **503 with `Retry-After`**, not a 500 — the request was valid and the
  condition is temporary.
