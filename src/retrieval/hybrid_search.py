"""
Hybrid retrieval: dense (bge) + sparse (BM25), fused with Reciprocal Rank
Fusion. The §3 v1 upgrade over dense-only top-k.

FUSION METHOD — why RRF and not score averaging: BM25 scores are unbounded
and corpus-dependent, cosine similarities live in [0, 1]. Averaging or
min-max normalising them makes the weighting an artefact of score scale
rather than a deliberate choice, and it shifts every time the corpus changes.
RRF discards magnitudes and uses rank only:

    score(doc) = sum over retrievers of 1 / (K + rank(doc))

A document ranked #1 by either retriever scores highly; one ranked decently
by BOTH scores highest. K=60 is the standard constant from the original RRF
paper — large enough that the top few ranks are not wildly dominant.

This class implements the same call signature as DocSearch, so the two are
swappable in the baseline/eval without touching callers — which is what makes
an honest A/B possible (§2e: a mechanism earns its place only by moving the
eval number).

The BM25 index is built in-memory from the same chunks stored in Chroma, so
there is exactly one source of truth for what is retrievable — including the
leakage exclusions already applied at index time.
"""

import sys
from pathlib import Path

from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.bm25 import tokenize
from retrieval.embeddings import EmbeddingModel
from retrieval.search import DocSearchInput, DocSearchOutput, DocSearchResult
from retrieval.store import VectorStore

RRF_K = 60
# How deep each retriever goes before fusion. Must exceed the final k — fusion
# can only rerank what it was given, so a candidate pool the same size as k
# would make RRF a no-op.
CANDIDATE_POOL = 30


class HybridSearch:
    def __init__(
        self,
        collection_name: str = "fastapi_docs",
        embedder: EmbeddingModel | None = None,
        rrf_k: int = RRF_K,
        candidate_pool: int = CANDIDATE_POOL,
    ):
        self._embedder = embedder or EmbeddingModel()
        self._store = VectorStore(collection_name=collection_name)
        self._rrf_k = rrf_k
        self._candidate_pool = candidate_pool

        # Pull the whole collection once to build the sparse index. Fine at this
        # corpus size (1.6k doc chunks / 3.7k discussions); would need a real
        # sparse backend at a larger scale.
        raw = self._store._collection.get(include=["documents", "metadatas"])
        self._ids: list[str] = raw["ids"]
        self._documents: list[str] = raw["documents"]
        self._metadatas: list[dict] = raw["metadatas"]
        self._bm25 = BM25Okapi([tokenize(d) for d in self._documents])

    def _dense_ranking(self, query: str) -> list[int]:
        query_embedding = self._embedder.embed_query(query)
        res = self._store.query(query_embedding, k=self._candidate_pool)
        id_to_pos = {doc_id: i for i, doc_id in enumerate(self._ids)}
        return [id_to_pos[doc_id] for doc_id in res["ids"][0] if doc_id in id_to_pos]

    def _sparse_ranking(self, query: str) -> list[int]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        # Drop zero-score docs: with no term overlap BM25 ranks them arbitrarily,
        # and feeding that noise into RRF would dilute the dense signal.
        return [i for i in ranked[: self._candidate_pool] if scores[i] > 0]

    def __call__(self, input: DocSearchInput) -> DocSearchOutput:
        dense = self._dense_ranking(input.query)
        sparse = self._sparse_ranking(input.query)

        fused: dict[int, float] = {}
        for ranking in (dense, sparse):
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (self._rrf_k + rank + 1)

        top = sorted(fused.items(), key=lambda kv: -kv[1])[: input.k]

        results = []
        for idx, score in top:
            meta = self._metadatas[idx]
            results.append(
                DocSearchResult(
                    source_file=meta.get("source_file", meta.get("url", "")),
                    section_path=meta.get("section_path", meta.get("title", "")),
                    text=self._documents[idx],
                    score=score,  # RRF score, NOT cosine — not comparable to DocSearch scores
                )
            )
        return DocSearchOutput(results=results)


class HybridSearchDebug(BaseModel):
    """Per-retriever contribution for one query — used to sanity-check fusion."""

    query: str
    dense_only: list[str] = Field(default_factory=list)
    sparse_only: list[str] = Field(default_factory=list)
    both: list[str] = Field(default_factory=list)
