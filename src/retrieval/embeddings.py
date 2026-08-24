"""
Embedding wrappers, kept swappable behind one interface per
pre-build-checklist.md §3.

THREE IMPLEMENTATIONS. Which one runs is a deployment constraint, not a
preference — select with EMBEDDING_PROVIDER=local|onnx|gemini.

  LocalEmbeddingModel  (default) bge-base-en-v1.5 via sentence-transformers.
                       Every measured number in decisions.md was produced with
                       this, so the eval harness must keep using it.

  OnnxEmbeddingModel   The SAME bge weights as ONNX fp16 via onnxruntime.
                       Exists because torch (~465MB installed) plus the
                       PyTorch model (~400MB resident) OOM-killed a 512MB
                       container. VERIFIED EQUIVALENT: cosine 1.000000 vs
                       PyTorch, identical top-5 retrieval on 5/5 probes — so
                       the deployed system is the benchmarked system. This is
                       what the Dockerfile uses.

  GeminiEmbeddingModel gemini-embedding-001 over REST. Superseded by the ONNX
                       path and kept only as a fallback: it is 3072-dim (vs
                       bge's 768), so it needs a full re-index, its retrieval
                       quality is UNMEASURED, and the free tier caps
                       embeddings at 1000 requests/DAY — counted per item, so
                       batching does not help.

`sentence_transformers` is imported LAZILY inside LocalEmbeddingModel. A
module-level import would drag torch into the deployment image through this
file alone, which is exactly what the ONNX path exists to avoid.

Default is `local` so that nothing which reads this module by accident
silently changes what the eval measures.
"""

import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load .env here rather than relying on every entry point to remember. The
# ingest scripts do not call load_dotenv(), so without this the Gemini path
# fails with "GEMINI_API_KEY not set" even though the key is present.
load_dotenv()

MODEL_NAME = "BAAI/bge-base-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

ONNX_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "bge_onnx"

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


class OnnxEmbeddingModel:
    """
    The SAME bge-base-en-v1.5 weights, exported to ONNX fp16 and run through
    onnxruntime instead of PyTorch.

    This exists because torch (~465MB installed) plus the PyTorch model
    (~400MB resident) OOM-killed a 512MB free-tier container. onnxruntime is
    ~15MB and the fp16 graph is 209MB, so the whole serving image fits with
    room to spare.

    VERIFIED EQUIVALENT, which is the entire point: fp16 ONNX embeddings score
    cosine **1.000000** against the PyTorch model (int8 managed only 0.967),
    and return byte-identical top-5 results on 5/5 probe queries against the
    existing bge index. So the deployed system is the one the eval measured —
    no re-indexing, no re-benchmarking, no asterisk on the reported numbers.

    Pooling must match bge exactly: CLS token (index 0), then L2 normalise.
    Mean-pooling here would silently produce different vectors and quietly
    invalidate the index.
    """

    def __init__(self, model_dir: str | None = None):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._np = np
        path = Path(model_dir or ONNX_MODEL_DIR)
        if not (path / "model.onnx").exists():
            raise RuntimeError(
                f"ONNX model not found at {path}. Export it with "
                "scripts/export_onnx.py, or use EMBEDDING_PROVIDER=local."
            )
        self.model_name = str(path)
        self._tok = Tokenizer.from_file(str(path / "tokenizer.json"))
        self._tok.enable_truncation(max_length=512)
        self._tok.enable_padding()
        self._sess = ort.InferenceSession(
            str(path / "model.onnx"), providers=["CPUExecutionProvider"]
        )

    def _encode(self, texts: list[str]) -> list[list[float]]:
        np = self._np
        encoded = self._tok.encode_batch(texts)
        ids = np.array([e.ids for e in encoded], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        types = np.array([e.type_ids for e in encoded], dtype=np.int64)
        hidden = self._sess.run(
            None,
            {"input_ids": ids, "attention_mask": mask, "token_type_ids": types},
        )[0]
        cls = hidden[:, 0].astype(np.float32)  # bge pools on CLS, not mean
        cls = cls / np.linalg.norm(cls, axis=1, keepdims=True)
        return cls.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([QUERY_INSTRUCTION + text])[0]


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
Embedder = LocalEmbeddingModel | OnnxEmbeddingModel | GeminiEmbeddingModel


def EmbeddingModel(model_name: str | None = None) -> Embedder:
    """
    Factory kept callable as `EmbeddingModel()` so existing call sites are
    unchanged. Defaults to LOCAL: the eval harness must not silently switch
    models, since that would invalidate comparisons against measured results.
    """
    provider = (os.environ.get("EMBEDDING_PROVIDER") or "local").strip().lower()
    if provider == "onnx":
        return OnnxEmbeddingModel(model_name)
    if provider == "gemini":
        return GeminiEmbeddingModel(model_name or GEMINI_MODEL)
    return LocalEmbeddingModel(model_name or MODEL_NAME)
