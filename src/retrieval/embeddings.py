"""
Embedding wrapper, kept swappable behind this interface per
pre-build-checklist.md §3 ("embedding model behind an interface so it's
swappable in one place"). v0 default is local `bge-base-en-v1.5` (free,
CPU-friendly, deterministic, no rate limits) — A/B a second model on the
frozen eval later, not by feel.

BGE models are trained to expect a specific instruction prefix on queries
(NOT on the documents being indexed) for retrieval tasks — dropping it
measurably hurts recall, so it's baked in here rather than left to callers.
"""

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-base-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class EmbeddingModel:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        prefixed = QUERY_INSTRUCTION + text
        return self._model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False).tolist()
