# Container image for SourceBound, sized for a 512MB free-tier instance.
#
# WHY ONNX AND NOT PyTorch: the first deploy was killed with "Ran out of
# memory (used over 512MB)". torch is ~465MB installed and the PyTorch bge
# model another ~400MB resident — roughly 900MB before serving a request.
#
# int8 quantised: fp16 does NOT help on CPU (onnxruntime converts it back to
# fp32 at load, measured 551MB and still OOM). int8 stays int8 in memory:
# 279MB total, verified with psutil. Cost: 87% identical top-1 and 86% mean
# top-5 overlap vs PyTorch over 30 real eval questions - close, not identical.
# The README states this so the demo is not mistaken for the benchmarked run.
# The eval harness still defaults to PyTorch bge, so measured numbers are
# unaffected.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    EMBEDDING_PROVIDER=onnx \
    CHROMA_DIR=/app/chroma

WORKDIR /app

COPY deploy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/
# int8 ONNX graph + tokenizer (~106MB), committed via Git LFS.
COPY models/ ./models/
# Prebuilt bge index (~98MB), also via LFS. Building it here would mean
# embedding 5,312 chunks at container start, and the free tier has no
# persistent disk to cache the result on.
COPY chroma/ ./chroma/

# Render supplies $PORT; default to 8000 for a plain `docker run`.
ENV PORT=8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
