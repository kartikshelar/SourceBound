"""
Build the v0 retrieval index: chunk the pinned docs snapshot, embed each
chunk, upsert into the vector store.

v0 indexes ONLY the docs corpus (data/docs_snapshot/en/), per the enforced
build order (retrieval core first, simplest defensible default). Discussions
are a v1 option (pre-build-checklist.md §2a) — if/when they're added as a
second retrievable source, every ID in
data/eval/eval_frozen_v1.manifest.json:leakage_exclusion_discussion_ids MUST
be excluded (§2c, the leakage rule). Nothing to exclude yet since discussions
aren't indexed in v0, but this module is where that check must land.

Usage:
    uv run -m src.ingest.index
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.chunk import chunk_docs_tree
from retrieval.embeddings import EmbeddingModel
from retrieval.store import VectorStore

DOCS_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "docs_snapshot" / "en"
DOCS_SRC_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "docs_snapshot" / "docs_src"
BATCH_SIZE = 32


def main() -> None:
    if not DOCS_ROOT.exists():
        raise SystemExit(f"{DOCS_ROOT} not found — run scripts/fetch_docs_snapshot.py first.")

    print("Chunking docs...")
    chunks = chunk_docs_tree(DOCS_ROOT, DOCS_SRC_ROOT)
    print(f"  {len(chunks)} chunks from {DOCS_ROOT}")

    print(f"Loading embedding model...")
    embedder = EmbeddingModel()

    store = VectorStore()
    store.reset()

    print("Embedding + indexing...")
    t0 = time.time()
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c.text for c in batch]
        embeddings = embedder.embed_documents(texts)
        ids = [c.id for c in batch]
        metadatas = [
            {
                "source_file": c.source_file,
                "section_path": c.metadata["section_path_str"],
                "chunk_index": c.chunk_index,
            }
            for c in batch
        ]
        store.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        print(f"  {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}")

    elapsed = time.time() - t0
    print(f"\nIndexed {store.count()} chunks in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
