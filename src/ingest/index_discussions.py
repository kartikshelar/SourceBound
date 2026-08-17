"""
Build the discussions index (the v1 second retrievable source, §2a).

Motivation is measured, not assumed: hand-tagging 20 dev items found only
**35% are answerable from the FastAPI docs alone** (decisions.md D10). The
other 65% need maintainer knowledge that is simply absent from the docs —
"this is a known limitation", "that lives in Starlette, report it there",
"fixed in 0.139.2". No retrieval improvement over docs can reach those; the
knowledge has to come from somewhere else. Answered Discussions are where it
lives.

Chunking: ONE Q&A PER CHUNK (§3), not header-split like docs. A question
separated from its accepted answer retrieves as a question, which is useless
— the answer is the payload.

Leakage: every candidate passes through src/ingest/leakage.py first, which
applies both the §2c ID exclusion and a near-duplicate title check. See that
module for why ID exclusion alone is insufficient.

Usage:
    uv run -m src.ingest.index_discussions
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest.leakage import filter_leakage
from retrieval.embeddings import EmbeddingModel
from retrieval.store import VectorStore

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_PATH = ROOT / "data" / "eval" / "discussions_raw.jsonl"
COLLECTION = "fastapi_discussions"
BATCH_SIZE = 32
MAX_CHARS = 6000  # p90 of Q+A is ~5k; caps the 212k outlier so one thread can't dominate


def build_chunk_text(item: dict) -> str:
    """One Q&A per chunk, labelled so the model can tell question from answer."""
    body = (item.get("body") or "").strip()
    answer = (item.get("answer_body") or "").strip()
    text = (
        f"QUESTION: {item['title']}\n\n"
        f"{body[:MAX_CHARS]}\n\n"
        f"ACCEPTED ANSWER:\n{answer[:MAX_CHARS]}"
    )
    return text


def main() -> None:
    if not RAW_PATH.exists():
        raise SystemExit(f"{RAW_PATH} not found — run scripts/pull_discussions.py first.")

    raw = [json.loads(l) for l in RAW_PATH.open(encoding="utf-8") if l.strip()]
    print(f"Loaded {len(raw)} answered discussions.")

    print("Loading embedding model...")
    embedder = EmbeddingModel()

    print("Applying leakage guard (ID exclusion + near-duplicate check)...")
    kept, report = filter_leakage(raw, embedder)
    print(f"  input                     : {report.n_input}")
    print(f"  excluded (in eval set, ID): {report.n_excluded_by_id}")
    print(f"  excluded (near-duplicate) : {report.n_excluded_by_similarity}")
    print(f"  indexable                 : {report.n_kept}")

    store = VectorStore(collection_name=COLLECTION)
    store.reset()

    print("Embedding + indexing...")
    t0 = time.time()
    for i in range(0, len(kept), BATCH_SIZE):
        batch = kept[i : i + BATCH_SIZE]
        texts = [build_chunk_text(item) for item in batch]
        embeddings = embedder.embed_documents(texts)
        ids = [item["discussion_id"] for item in batch]
        metadatas = [
            {
                "number": item["number"],
                "title": item["title"],
                "url": item["url"],
                "created_at": item["created_at"],
                "source": "discussion",
            }
            for item in batch
        ]
        store.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        print(f"  {min(i + BATCH_SIZE, len(kept))}/{len(kept)}")

    print(f"\nIndexed {store.count()} discussions in {time.time() - t0:.0f}s")
    print("Leakage audit -> data/eval/leakage_exclusions.jsonl")


if __name__ == "__main__":
    main()
