"""
doc_search: the retrieval function itself, with a Pydantic input/output
contract (pre-build-checklist.md §3 — every agent tool gets one; this is the
first and simplest of the tool registry: doc_search, discussion_search (v1),
web_search (fallback), verify_citation (internal)).

v0 is dense top-k only. Hybrid (BM25 + dense) + reranker is a v1 upgrade,
justified only if it moves the frozen-eval number (§2e baseline rule).
"""

from pydantic import BaseModel, Field

from retrieval.embeddings import EmbeddingModel
from retrieval.store import VectorStore


class DocSearchInput(BaseModel):
    query: str = Field(..., description="Natural-language question or search query")
    k: int = Field(5, ge=1, le=20, description="Number of results to return")


class DocSearchResult(BaseModel):
    source_file: str
    section_path: str
    text: str
    score: float


class DocSearchOutput(BaseModel):
    results: list[DocSearchResult]


class DocSearch:
    def __init__(self):
        self._embedder = EmbeddingModel()
        self._store = VectorStore()

    def __call__(self, input: DocSearchInput) -> DocSearchOutput:
        query_embedding = self._embedder.embed_query(input.query)
        raw = self._store.query(query_embedding, k=input.k)

        results = []
        ids = raw["ids"][0]
        documents = raw["documents"][0]
        metadatas = raw["metadatas"][0]
        distances = raw["distances"][0]
        for _id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            results.append(
                DocSearchResult(
                    source_file=meta["source_file"],
                    section_path=meta["section_path"],
                    text=doc,
                    score=1 - dist,  # cosine distance -> similarity
                )
            )
        return DocSearchOutput(results=results)
