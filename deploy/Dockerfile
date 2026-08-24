# Hugging Face Spaces (Docker SDK) image for SourceBound.
#
# The prebuilt Chroma index is uploaded with the Space rather than rebuilt at
# startup. Building it here would mean embedding 5,312 chunks on CPU — roughly
# 14 minutes — on every cold start, which exceeds the healthcheck window and
# would make the Space appear broken. The index is a build artifact, so it is
# treated like one.
FROM python:3.11-slim

# HF Spaces runs containers as uid 1000; the model cache and any writes must
# live somewhere that user owns.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/user/.cache/torch/sentence_transformers \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Bake the embedding model into the image. Downloading ~400MB on first request
# would otherwise make the initial query look like a hang.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('BAAI/bge-base-en-v1.5')"

COPY --chown=user src/ ./src/
COPY --chown=user app/ ./app/
COPY --chown=user chroma/ ./chroma/

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
