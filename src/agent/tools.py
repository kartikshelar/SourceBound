"""
Agent tool registry (§3): every tool has a Pydantic input/output contract.

These wrap the SAME retrieval code the eval measured (`DocSearch`,
`DiscussionSearch`, `Reranker`) rather than reimplementing it — otherwise the
agent's numbers would not be comparable to the 34.0% baseline, and the whole
point of the eval harness would be lost.

Shared singletons: the bge embedder is ~400MB and the reranker another
~1.1GB. Loading them per call would dominate latency; loading them per tool
would double memory for no benefit.
"""

import sys
from pathlib import Path

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.state import RetrievedChunk
from retrieval.discussion_search import DiscussionSearch, DiscussionSearchInput
from retrieval.embeddings import EmbeddingModel
from retrieval.search import DocSearch, DocSearchInput

_embedder: EmbeddingModel | None = None
_doc_search: DocSearch | None = None
_discussion_search: DiscussionSearch | None = None


def _get_embedder() -> EmbeddingModel:
    global _embedder
    if _embedder is None:
        _embedder = EmbeddingModel()
    return _embedder


class DocSearchTool(BaseModel):
    """Search the pinned FastAPI documentation snapshot."""

    query: str = Field(..., description="Natural-language question")
    k: int = Field(5, ge=1, le=20)

    def run(self) -> list[RetrievedChunk]:
        global _doc_search
        if _doc_search is None:
            _doc_search = DocSearch()
            _doc_search._embedder = _get_embedder()
        out = _doc_search(DocSearchInput(query=self.query, k=self.k))
        return [
            RetrievedChunk(
                source="docs",
                ref=r.source_file,
                label=r.section_path,
                text=r.text,
                score=r.score,
            )
            for r in out.results
        ]


class DiscussionSearchTool(BaseModel):
    """
    Search answered GitHub Discussions — maintainer knowledge that is not in
    the docs ("known limitation", "lives in Starlette", "fixed in 0.139.2").

    Leak-guarded at index time: every frozen eval discussion is excluded by ID,
    plus near-duplicates by title similarity (src/ingest/leakage.py).
    """

    query: str = Field(..., description="Natural-language question")
    k: int = Field(2, ge=1, le=10)

    def run(self) -> list[RetrievedChunk]:
        global _discussion_search
        if _discussion_search is None:
            _discussion_search = DiscussionSearch(embedder=_get_embedder())
        out = _discussion_search(DiscussionSearchInput(query=self.query, k=self.k))
        return [
            RetrievedChunk(
                source="discussion",
                ref=r.url,
                label=f"#{r.number} {r.title}",
                text=r.text,
                score=r.score,
            )
            for r in out.results
        ]
