"""
Vector-store wrapper, kept swappable per pre-build-checklist.md §3. v0 is
Chroma (local, zero-infra) -> v1 swaps to Qdrant Cloud (native hybrid search)
behind this same interface.
"""

from pathlib import Path

import chromadb

DEFAULT_PERSIST_DIR = Path(__file__).resolve().parent.parent.parent / "chroma"
COLLECTION_NAME = "fastapi_docs"


class VectorStore:
    def __init__(self, persist_dir: Path = DEFAULT_PERSIST_DIR, collection_name: str = COLLECTION_NAME):
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict]) -> None:
        self._collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def query(self, query_embedding: list[float], k: int) -> dict:
        return self._collection.query(query_embeddings=[query_embedding], n_results=k)

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        # Must use the INSTANCE's collection name, not the module default —
        # otherwise resetting the discussions store silently wipes the docs
        # collection and leaves the intended one untouched.
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name, metadata={"hnsw:space": "cosine"}
        )
