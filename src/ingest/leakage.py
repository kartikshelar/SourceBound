"""
Leakage guard for the discussions corpus (pre-build-checklist.md §2c).

§2c requires excluding every eval discussion from the retrievable index by
ID. That is necessary but NOT sufficient: FastAPI users ask the same question
more than once, so a near-duplicate thread can carry the same accepted answer
under a different discussion_id. Indexing one of those hands the agent the
answer key and the eval silently measures nothing — the same class of failure
as the unvalidated judge (a number that looks good and means nothing).

Confirmed empirically, not hypothetically: discussion #7707 has a title
identical (cosine 1.000) to eval item #13972 and its accepted answer gives the
same fix (read the request body in middleware before it is consumed). #7707
survives ID-exclusion.

Method: embed titles with the same model used for retrieval, drop any
candidate whose cosine similarity to ANY eval title is >= NEAR_DUP_THRESHOLD.
Titles (not bodies) because the question, not the prose, is what makes two
threads duplicates — and it keeps this cheap enough to re-run on every index
build.

Threshold 0.90 measured against the real corpus:
    0.95 -> drops 3    (0.1%)  misses obvious dups
    0.90 -> drops 16   (0.4%)  <- chosen
    0.85 -> drops 146  (4.0%)  starts removing merely-same-topic threads
Inspection of the 0.88-0.92 band showed mostly related-but-distinct questions
("exception in middleware" vs "how to throw custom exceptions in middleware"),
which are legitimately useful context. 0.90 buys safety for 0.4% of the
corpus — cheap insurance against an unmeasurable failure.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
FROZEN_EVAL_PATH = ROOT / "data" / "eval" / "eval_frozen_v1.jsonl"
AUDIT_PATH = ROOT / "data" / "eval" / "leakage_exclusions.jsonl"

NEAR_DUP_THRESHOLD = 0.90


@dataclass
class LeakageReport:
    n_input: int
    n_excluded_by_id: int
    n_excluded_by_similarity: int
    n_kept: int


def load_eval_items() -> list[dict]:
    return [json.loads(l) for l in FROZEN_EVAL_PATH.open(encoding="utf-8") if l.strip()]


def filter_leakage(
    candidates: list[dict],
    embedder,
    threshold: float = NEAR_DUP_THRESHOLD,
    write_audit: bool = True,
) -> tuple[list[dict], LeakageReport]:
    """
    candidates: raw discussion records (dicts with discussion_id + title).
    embedder:   object exposing embed_documents(list[str]) -> list[list[float]],
                i.e. the same EmbeddingModel used to build the index, so the
                similarity space matches what retrieval will actually see.

    Returns (kept_candidates, report) and writes an audit line per exclusion.
    Excluded items are recorded rather than silently dropped: a leakage filter
    that cannot be inspected is as untrustworthy as no filter.
    """
    eval_items = load_eval_items()
    eval_ids = {e["discussion_id"] for e in eval_items}

    n_input = len(candidates)
    by_id = [c for c in candidates if c["discussion_id"] not in eval_ids]
    n_excluded_by_id = n_input - len(by_id)

    # Every candidate was already excluded by ID — nothing left to compare, and
    # embedding an empty list yields a 0-dim array that crashes the matmul.
    # Returning cleanly matters: "nothing to index" is a legitimate outcome
    # (e.g. re-running over a candidate set the eval set fully covers), not an
    # error condition.
    if not by_id:
        if write_audit:
            AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
            AUDIT_PATH.write_text("", encoding="utf-8")
        return [], LeakageReport(
            n_input=n_input,
            n_excluded_by_id=n_excluded_by_id,
            n_excluded_by_similarity=0,
            n_kept=0,
        )

    eval_vecs = np.asarray(embedder.embed_documents([e["title"] for e in eval_items]))
    cand_vecs = np.asarray(embedder.embed_documents([c["title"] for c in by_id]))
    sims = cand_vecs @ eval_vecs.T  # both L2-normalised -> cosine

    best_idx = sims.argmax(axis=1)
    best_sim = sims.max(axis=1)

    kept, audit = [], []
    for i, cand in enumerate(by_id):
        if best_sim[i] >= threshold:
            match = eval_items[best_idx[i]]
            audit.append({
                "excluded_discussion_id": cand["discussion_id"],
                "excluded_number": cand.get("number"),
                "excluded_title": cand["title"],
                "matched_eval_number": match["number"],
                "matched_eval_title": match["title"],
                "title_similarity": round(float(best_sim[i]), 4),
                "reason": "near-duplicate of a frozen eval item",
            })
        else:
            kept.append(cand)

    if write_audit:
        AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with AUDIT_PATH.open("w", encoding="utf-8") as f:
            for row in audit:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return kept, LeakageReport(
        n_input=n_input,
        n_excluded_by_id=n_excluded_by_id,
        n_excluded_by_similarity=len(audit),
        n_kept=len(kept),
    )
