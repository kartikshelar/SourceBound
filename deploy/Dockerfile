# Container image for SourceBound (built for Render's free web-service tier;
# the same image runs anywhere that accepts a Dockerfile).
#
# The prebuilt Chroma index is COPIED IN, not rebuilt. Embedding 5,312 chunks
# on CPU takes ~14 minutes, and the free tier has no persistent disk — a
# rebuild would therefore run on every cold start and exceed the healthcheck
# window, making a working system look broken. The index is a build artifact,
# so it ships like one (committed via Git LFS).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/torch/sentence_transformers

WORKDIR /app

COPY deploy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image. Downloading ~400MB on first request
# would otherwise make the initial query look like a hang.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('BAAI/bge-base-en-v1.5')"

COPY src/ ./src/
COPY app/ ./app/
COPY chroma/ ./chroma/

# Startup refuses to boot if this index is empty — Chroma returns an empty
# collection rather than erroring, so without that check a bad image would
# serve confident answers with zero retrieved context.
ENV CHROMA_DIR=/app/chroma

# Render supplies $PORT; default to 8000 for plain `docker run`.
ENV PORT=8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
