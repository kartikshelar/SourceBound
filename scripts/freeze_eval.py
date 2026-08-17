"""
Filter, curate, and freeze the FastAPI Discussions eval set from the raw pull.

Reads data/eval/discussions_raw.jsonl (from pull_discussions.py) and writes
the frozen, versioned eval set: data/eval/eval_frozen_v1.jsonl (dev + test,
tagged by split). This file is the ground truth — never edited after freeze,
never tuned against (pre-build-checklist.md §2b).

Inclusion filters (§2b):
  - Has an accepted answer (guaranteed by the raw pull's `answered: true`).
  - Answer is substantive: >= MIN_ANSWER_CHARS after stripping code fences/
    quoted text, to drop "just add X" one-liners and bare links/PR refs.
  - Recent enough to match the pinned docs snapshot (0.119.1, pulled
    2026-08-06): kept to the last RECENCY_MONTHS months. Rationale for this
    choice over backdating the docs pin is recorded in decisions.md (D5).
  - Genuinely about FastAPI: drop items whose body+answer never mention
    "fastapi" (case-insensitive) — catches pure-Starlette/Pydantic-only
    threads that drifted into the Questions category.

Split: shuffled with a fixed seed, ~50 dev (freely inspectable while
iterating) + rest as the locked test set. Every item is tagged
`difficulty: docs-answerable | needs-experience` per a simple heuristic
(does the accepted answer link a doc page?) — reported separately per §2d.

Usage:
    uv run scripts/freeze_eval.py
"""

import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "eval" / "discussions_raw.jsonl"
FROZEN_PATH = Path(__file__).resolve().parent.parent / "data" / "eval" / "eval_frozen_v1.jsonl"
MANIFEST_PATH = Path(__file__).resolve().parent.parent / "data" / "eval" / "eval_frozen_v1.manifest.json"

RECENCY_MONTHS = 18
MIN_ANSWER_CHARS = 120  # median in the recent pool is ~484; screens terse "just add X" / bare-link answers
TARGET_TOTAL = 300
DEV_COUNT = 50
RANDOM_SEED = 42

DOC_LINK_RE = re.compile(r"https?://fastapi\.tiangolo\.com/\S*", re.IGNORECASE)
CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


def strip_code_and_quotes(text: str) -> str:
    text = CODE_FENCE_RE.sub("", text)
    text = "\n".join(line for line in text.splitlines() if not line.strip().startswith(">"))
    return text.strip()


def months_ago(dt: datetime, months: int) -> datetime:
    year = dt.year
    month = dt.month - months
    while month <= 0:
        month += 12
        year -= 1
    return dt.replace(year=year, month=month)


def load_raw() -> list[dict]:
    with RAW_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def passes_filters(item: dict, cutoff: datetime) -> tuple[bool, str]:
    created = datetime.fromisoformat(item["created_at"].replace("Z", "+00:00"))
    if created < cutoff:
        return False, "too_old"

    stripped_answer = strip_code_and_quotes(item["answer_body"])
    if len(stripped_answer) < MIN_ANSWER_CHARS:
        return False, "answer_too_short"

    haystack = (item["body"] + " " + item["answer_body"]).lower()
    if "fastapi" not in haystack:
        return False, "not_about_fastapi"

    return True, "ok"


def tag_difficulty(item: dict) -> str:
    return "docs-answerable" if DOC_LINK_RE.search(item["answer_body"]) else "needs-experience"


def main() -> None:
    if not RAW_PATH.exists():
        raise SystemExit(f"{RAW_PATH} not found — run pull_discussions.py first.")
    if FROZEN_PATH.exists():
        raise SystemExit(
            f"{FROZEN_PATH} already exists. The frozen eval set is never overwritten "
            "in place — delete it manually first if you really intend to re-freeze."
        )

    raw = load_raw()
    now = datetime.now(timezone.utc)
    cutoff = months_ago(now, RECENCY_MONTHS)

    kept, dropped = [], {}
    for item in raw:
        ok, reason = passes_filters(item, cutoff)
        if ok:
            kept.append(item)
        else:
            dropped[reason] = dropped.get(reason, 0) + 1

    print(f"raw pool: {len(raw)}")
    print(f"passed filters: {len(kept)}")
    for reason, count in sorted(dropped.items()):
        print(f"  dropped ({reason}): {count}")

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(kept)
    selected = kept[:TARGET_TOTAL]
    if len(selected) < TARGET_TOTAL:
        print(f"\nWARNING: only {len(selected)} items passed filters, below target {TARGET_TOTAL}.")

    dev = selected[:DEV_COUNT]
    test = selected[DEV_COUNT:]

    frozen_at = now.isoformat()
    excluded_discussion_ids = [item["discussion_id"] for item in selected]

    with FROZEN_PATH.open("w", encoding="utf-8") as f:
        for split_name, split_items in (("dev", dev), ("test", test)):
            for item in split_items:
                record = {
                    "discussion_id": item["discussion_id"],
                    "number": item["number"],
                    "title": item["title"],
                    "question_body": item["body"],
                    "url": item["url"],
                    "created_at": item["created_at"],
                    "accepted_answer": item["answer_body"],
                    "answer_chosen_at": item["answer_chosen_at"],
                    "split": split_name,
                    "difficulty": tag_difficulty(item),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "frozen_at": frozen_at,
        "source_file": "data/eval/discussions_raw.jsonl",
        "source_query_category": "Questions",
        "source_repo": "fastapi/fastapi",
        "recency_months": RECENCY_MONTHS,
        "min_answer_chars": MIN_ANSWER_CHARS,
        "random_seed": RANDOM_SEED,
        "raw_pool_size": len(raw),
        "passed_filters": len(kept),
        "dropped_by_reason": dropped,
        "total_selected": len(selected),
        "dev_count": len(dev),
        "test_count": len(test),
        "difficulty_counts": {
            "dev": {
                "docs-answerable": sum(1 for i in dev if tag_difficulty(i) == "docs-answerable"),
                "needs-experience": sum(1 for i in dev if tag_difficulty(i) == "needs-experience"),
            },
            "test": {
                "docs-answerable": sum(1 for i in test if tag_difficulty(i) == "docs-answerable"),
                "needs-experience": sum(1 for i in test if tag_difficulty(i) == "needs-experience"),
            },
        },
        "leakage_exclusion_discussion_ids": excluded_discussion_ids,
        "note": (
            "leakage_exclusion_discussion_ids MUST be excluded from the retrievable "
            "index (pre-build-checklist.md §2c). This file (manifest) is the "
            "authoritative exclusion list for the ingest step."
        ),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nFroze {len(selected)} items -> {FROZEN_PATH}")
    print(f"  dev:  {len(dev)}")
    print(f"  test: {len(test)} (locked — do not eyeball while iterating)")
    print(f"Manifest (incl. leakage exclusion list) -> {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
