# Deploying to Hugging Face Spaces

The Space serves the agent only — it does not run the eval harness.

## Why the index is uploaded, not built

The Chroma index (~98MB, 5,312 chunks) is **not** in the git repo, and the
Dockerfile does **not** rebuild it. Embedding those chunks on CPU takes ~14
minutes, which exceeds the Space healthcheck window on every cold start and
would make a working system look broken. The index is a build artifact, so it
ships like one.

Consequence: **the index must be built locally and pushed to the Space.**

## Steps

**1. Build the index locally** (skip if `chroma/` already exists)

```bash
uv run scripts/fetch_docs_snapshot.py
uv run python -m src.ingest.index
uv run python -m src.ingest.index_discussions   # optional: discussion route
```

**2. Create the Space**

At https://huggingface.co/new-space — choose **Docker** as the SDK (not
Gradio/Streamlit; this serves a FastAPI app on port 7860).

**3. Add the secret**

Space → Settings → *Variables and secrets* → add `GROQ_API_KEY`.
Optionally `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST`
for tracing — without them the agent runs untraced rather than failing.

**4. Push**

```bash
git clone https://huggingface.co/spaces/<user>/<space> hf-space
cd hf-space

cp ../deploy/Dockerfile ../deploy/requirements.txt .
cp -r ../src ../app ../chroma .

git lfs install
git lfs track "chroma/**"          # 98MB index exceeds the normal file limit
git add .gitattributes .
git commit -m "Deploy SourceBound"
git push
```

## Verifying the deploy

```
GET /health  ->  {"status":"ok","graph_ready":true,"indexed_chunks":1645,"tracing":true}
```

`indexed_chunks` is the field that matters. Chroma returns an *empty
collection* rather than an error when the index is missing, so without that
number a broken deploy would boot, report healthy, and answer every question
with no retrieved context — a failure that looks like success. Startup now
refuses to come up at all if the docs index is empty.

`CHROMA_DIR` overrides the index location if you mount it elsewhere.

## Expected behaviour

Free-tier Groq quota is shared across the Space's users. When it is exhausted
`/ask` returns **503 with `Retry-After`**, not a 500 — the caller's request
was valid and the condition is temporary.

First request after a cold start is slow (the graph builds a ~400MB embedding
model into memory); subsequent ones are not, since it is built once at startup
rather than per request.
