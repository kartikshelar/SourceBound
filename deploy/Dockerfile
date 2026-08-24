# Container image for SourceBound, sized for a 512MB free-tier instance.
#
# WHY ONNX AND NOT PyTorch: the first deploy was killed with "Ran out of
# memory (used over 512MB)". torch is ~465MB installed and the PyTorch bge
# model another ~400MB resident — roughly 900MB before serving a request.
#
# The fix is the RUNTIME, not the model. These are the same bge-base-en-v1.5
# weights exported to ONNX fp16 and run through onnxruntime (~15MB runtime,
# 209MB graph). Verified equivalent: cosine 1.000000 against PyTorch and
# identical top-5 retrieval on 5/5 probe queries against the existing index.
# int8 was rejected — 110MB but only 0.967 cosine, which would have changed
# retrieval and invalidated the measured numbers.
#
# So the deployed system IS the benchmarked system: same weights, same index,
# no re-indexing, no asterisk on the reported results.
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
# fp16 ONNX graph + tokenizer (~209MB), committed via Git LFS.
COPY models/ ./models/
# Prebuilt bge index (~98MB), also via LFS. Building it here would mean
# embedding 5,312 chunks at container start, and the free tier has no
# persistent disk to cache the result on.
COPY chroma/ ./chroma/

# Render supplies $PORT; default to 8000 for a plain `docker run`.
ENV PORT=8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
