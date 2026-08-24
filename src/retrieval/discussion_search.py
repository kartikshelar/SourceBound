"""
discussion_search: retrieve answered GitHub Discussions (§3 tool registry).

The second retrievable source, and the one that targets the measured ceiling:
only ~35% of eval questions are answerable from the docs alone (decisions.md
D10). The remaining 65% need maintainer knowledge — "known limitation",
"that lives in Starlette", "fixed in 0.139.2" — which exists in answered
Discussions and nowhere in the docs.

Reads the leak-guarded collection built by src/ingest/index_discussions.py.
Every eval discussion is excluded by ID, plus near-duplicates by title
similarity (src/ingest/leakage.py). Results carry `url` so answers can cite a
real thread, and `created_at` so a caller can down-weight stale advice — a
2019 answer about a since-changed API is a genuine failure mode here in a way
it is not for version-pinned docs.
"""

from pydantic import BaseModel, Field

from retrieval.embeddings import Embedder, EmbeddingModel
from retrieval.store import VectorStore

COLLECTION = "fastapi_discussions"


class DiscussionSearchInput(BaseModel):
    query: str = Field(..., description="Natural-language question or search query")
    k: int = Field(3, ge=1, le=10, description="Number of discussions to return")


class DiscussionSearchResult(BaseModel):
    number: int = Field(..., description="GitHub discussion number")
    title: str
    url: str
    created_at: str
    text: str = Field(..., description="Question + accepted answer")
    score: float


class DiscussionSearchOutput(BaseModel):
    results: list[DiscussionSearchResult]


class DiscussionSearch:
    def __init__(self, embedder: Embedder | None = None):
        # Allow sharing one embedder with doc_search — loading bge twice costs
        # ~400MB and a few seconds for no benefit.
        self._embedder = embedder or EmbeddingModel()
        self._store = VectorStore(collection_name=COLLECTION)

    def __call__(self, input: DiscussionSearchInput) -> DiscussionSearchOutput:
        query_embedding = self._embedder.embed_query(input.query)
        raw = self._store.query(query_embedding, k=input.k)

        results = []
        for doc, meta, dist in zip(raw["documents"][0], raw["metadatas"][0], raw["distances"][0]):
            results.append(
                DiscussionSearchResult(
                    number=meta["number"],
                    title=meta["title"],
                    url=meta["url"],
                    created_at=meta["created_at"],
                    text=doc,
                    score=1 - dist,  # cosine distance -> similarity
                )
            )
        return DiscussionSearchOutput(results=results)
