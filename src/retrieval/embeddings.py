"""
Embedding wrappers, kept swappable behind one interface per
pre-build-checklist.md §3.

TWO IMPLEMENTATIONS, and which one runs is a deployment constraint, not a
preference:

  LocalEmbeddingModel  bge-base-en-v1.5 via sentence-transformers. Free,
                       deterministic, no rate limits. This is the model every
                       measured number in decisions.md was produced with, so
                       the eval harness must keep using it.

  GeminiEmbeddingModel gemini-embedding-001 over REST. Exists because torch
                       (465MB installed) plus the bge model (~400MB resident)
                       cannot fit a 512MB free-tier container — Render killed
                       the deploy with exactly that error. Dropping torch
                       takes the serving image from ~900MB to ~150MB.

`sentence_transformers` is imported LAZILY inside LocalEmbeddingModel rather
than at module scope. A module-level import would drag torch into the
deployment image through this file alone, which is the whole thing the Gemini
path exists to avoid.

Select with EMBEDDING_PROVIDER=gemini|local (default: local, so nothing that
reads this module by accident silently changes what the eval measures).

CAVEAT worth stating plainly: bge is 768-dim and Gemini is 3072-dim. They are
not interchangeable against the same index — switching providers requires a
full re-index, and retrieval quality under Gemini is UNMEASURED.
"""

import os
import time

import requests
from dotenv import load_dotenv

# Load .env here rather than relying on every entry point to remember. The
# ingest scripts do not call load_dotenv(), so without this the Gemini path
# fails with "GEMINI_API_KEY not set" even though the key is present.
load_dotenv()

MODEL_NAME = "BAAI/bge-base-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

GEMINI_MODEL = "gemini-embedding-001"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:{method}?key={key}"
# Measured empirically, not guessed. The free tier enforces a per-MINUTE token
# budget on embeddings:
#   batch=50, no pause  -> 429 on the 2nd call
#   batch=10, pause=1s  -> ran 224 chunks then exhausted retries on 429
#   batch=5,  pause=6s  -> 6 consecutive batches clean  <- chosen
# ~36 min for a 1,645-chunk index. Slow, but indexing is a one-off and the
# alternative is a run that dies partway.
GEMINI_BATCH = 5
GEMINI_PAUSE_S = 6.0
# Backoff must outlast the per-minute window; 2**attempt topped out at 16s,
# which was too short and turned a transient limit into a hard failure.
GEMINI_BACKOFF_S = [15, 30, 60, 90, 120]


class LocalEmbeddingModel:
    """bge-base-en-v1.5. The model all measured results were produced with."""

    def __init__(self, model_name: str = MODEL_NAME):
        from sentence_transformers import SentenceTransformer  # lazy: keeps torch out of slim images

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        # BGE expects this instruction prefix on QUERIES ONLY (not documents);
        # dropping it measurably hurts recall.
        return self._model.encode(
            QUERY_INSTRUCTION + text, normalize_embeddings=True, show_progress_bar=False
        ).tolist()


class GeminiEmbeddingModel:
    """gemini-embedding-001 over REST — no SDK, so no google-genai dependency."""

    def __init__(self, model_name: str = GEMINI_MODEL):
        key = (os.environ.get("GEMINI_API_KEY") or "").strip()
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set — required for EMBEDDING_PROVIDER=gemini")
        self.model_name = model_name
        self._key = key

    def _post(self, method: str, payload: dict) -> dict:
        url = GEMINI_URL.format(model=self.model_name, method=method, key=self._key)
        for attempt, wait in enumerate(GEMINI_BACKOFF_S):
            resp = requests.post(url, json=payload, timeout=120)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                print(f"    [gemini 429, waiting {wait}s "
                      f"({attempt + 1}/{len(GEMINI_BACKOFF_S)})]", flush=True)
                time.sleep(wait)
                continue
            raise RuntimeError(f"Gemini embedding failed ({resp.status_code}): {resp.text[:200]}")
        raise RuntimeError("Gemini embedding: exhausted retries on 429")

    @staticmethod
    def _normalise(vec: list[float]) -> list[float]:
        # Chroma is configured for cosine, and the local model returns
        # L2-normalised vectors — match that so both providers behave the same.
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec] if norm else vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), GEMINI_BATCH):
            chunk = texts[i : i + GEMINI_BATCH]
            data = self._post(
                "batchEmbedContents",
                {
                    "requests": [
                        {
                            "model": f"models/{self.model_name}",
                            "content": {"parts": [{"text": t}]},
                        }
                        for t in chunk
                    ]
                },
            )
            out.extend(self._normalise(e["values"]) for e in data["embeddings"])
            if i + GEMINI_BATCH < len(texts):
                time.sleep(GEMINI_PAUSE_S)
        return out

    def embed_query(self, text: str) -> list[float]:
        data = self._post(
            "embedContent",
            {"model": f"models/{self.model_name}", "content": {"parts": [{"text": text}]}},
        )
        return self._normalise(data["embedding"]["values"])


# `EmbeddingModel` became a factory function, so it can no longer be used in
# annotations like `EmbeddingModel | None` — that raises
# "unsupported operand type(s) for |: 'function' and 'NoneType'" at import.
# This alias is the type; the function below is the constructor.
Embedder = LocalEmbeddingModel | GeminiEmbeddingModel


def EmbeddingModel(model_name: str | None = None) -> Embedder:
    """
    Factory kept callable as `EmbeddingModel()` so existing call sites are
    unchanged. Defaults to LOCAL: the eval harness must not silently switch
    models, since that would invalidate comparisons against measured results.
    """
    provider = (os.environ.get("EMBEDDING_PROVIDER") or "local").strip().lower()
    if provider == "gemini":
        return GeminiEmbeddingModel(model_name or GEMINI_MODEL)
    return LocalEmbeddingModel(model_name or MODEL_NAME)
