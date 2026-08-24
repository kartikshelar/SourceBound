# Container image for SourceBound, sized for a 512MB free-tier instance.
#
# WHY GEMINI EMBEDDINGS HERE AND NOT bge: the first deploy was killed with
# "Ran out of memory (used over 512MB)". torch is ~465MB installed and the
# local bge model another ~400MB resident — roughly 900MB before serving a
# single request. Calling Gemini's embedding endpoint over REST removes torch
# and sentence-transformers entirely, taking the image to ~150MB.
#
# TRADE-OFF, stated because it matters: this is NOT the embedding model the
# results in decisions.md were measured with (bge, 768-dim vs Gemini 3072-dim).
# The deployed system's retrieval quality is unmeasured until the eval is
# re-run against this index. The eval harness still defaults to local bge, so
# existing numbers stay reproducible.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    EMBEDDING_PROVIDER=gemini \
    CHROMA_DIR=/app/chroma

WORKDIR /app

COPY deploy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/
# The Gemini-embedded index (3072-dim). Committed via Git LFS rather than
# built here: embedding 1,645 chunks through a rate-limited API at container
# start would exceed any healthcheck window, and the free tier has no
# persistent disk to cache it on.
COPY chroma_gemini/ ./chroma/

# Render supplies $PORT; default to 8000 for a plain `docker run`.
ENV PORT=8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
