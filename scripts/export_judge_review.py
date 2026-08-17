"""
Export a BLIND review sheet so the LLM judge can be hand-validated (§2d.1).

Why blind: if the sheet showed the judge's verdict, your scores would anchor
to it and the resulting "agreement" would be inflated and meaningless. The
judge's score is written to a separate answer-key file that the scorer reads
later; the sheet you fill in contains only question / reference answer /
candidate answer.

Sampling is stratified over the judge's own 0/1 verdicts (seeded, recorded)
so the sample can't accidentally be all-passes or all-fails — with ~50%
accuracy an unstratified 20-item draw could easily skew and tell you nothing
about one of the two classes.

Alongside `agree`, each row asks for `docs_answerable`. That second column is
the ceiling measurement: whether the accepted answer is derivable from the
FastAPI docs at all, or is maintainer knowledge (a known limitation, "that
lives in Starlette", a third-party bug). A docs-only agent cannot beat the
fraction that is answerable, and right now we do not know what that fraction
is. The `difficulty` tag in the frozen set is NOT this — it only fires on a
literal fastapi.tiangolo.com link and is too crude (see decisions.md D6).

Usage:
    uv run scripts/export_judge_review.py
Then fill in data/eval/judge_review_sheet.jsonl and run:
    uv run scripts/score_judge_agreement.py
"""

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / "data" / "eval" / "results" / "baseline_dev_run.jsonl"
SHEET_PATH = ROOT / "data" / "eval" / "judge_review_sheet.jsonl"
KEY_PATH = ROOT / "data" / "eval" / "judge_review_key.json"

SAMPLE_SIZE = 20
RANDOM_SEED = 7


def main() -> None:
    if not RESULTS_PATH.exists():
        raise SystemExit(f"{RESULTS_PATH} not found — run src/eval/run_baseline.py first.")
    if SHEET_PATH.exists():
        raise SystemExit(
            f"{SHEET_PATH} already exists — refusing to overwrite work you may have "
            "already filled in. Delete it manually if you intend to regenerate."
        )

    rows = [json.loads(line) for line in RESULTS_PATH.open(encoding="utf-8") if line.strip()]
    scored = [r for r in rows if r["judge_score"] is not None]

    passes = [r for r in scored if r["judge_score"] == 1]
    fails = [r for r in scored if r["judge_score"] == 0]

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(passes)
    rng.shuffle(fails)

    half = SAMPLE_SIZE // 2
    n_pass = min(half, len(passes))
    n_fail = min(SAMPLE_SIZE - n_pass, len(fails))
    n_pass = min(SAMPLE_SIZE - n_fail, len(passes))  # backfill if one class is short

    sample = passes[:n_pass] + fails[:n_fail]
    rng.shuffle(sample)  # so the sheet order leaks nothing about the verdict

    with SHEET_PATH.open("w", encoding="utf-8") as f:
        for r in sample:
            row = {
                "discussion_id": r["discussion_id"],
                "number": r["number"],
                "title": r["title"],
                "reference_accepted_answer": r["accepted_answer"],
                "candidate_generated_answer": r["generated_answer"],
                # ---- YOU FILL THESE IN ----
                # agree: 1 if the candidate contains the key correct info from the
                #        reference, 0 if not. This is YOUR verdict, same rubric the
                #        judge was given. Do not look at the judge's score first.
                "agree": None,
                # docs_answerable: 1 if this question could be answered from the
                #        FastAPI docs alone, 0 if it needs maintainer knowledge
                #        (known limitation / lives in Starlette / third-party bug).
                "docs_answerable": None,
                # optional free-text, useful when you disagree with the judge
                "note": "",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    key = {
        "seed": RANDOM_SEED,
        "sample_size": len(sample),
        "n_judge_pass": n_pass,
        "n_judge_fail": n_fail,
        "source": str(RESULTS_PATH.relative_to(ROOT)).replace("\\", "/"),
        "judge_scores": {r["discussion_id"]: r["judge_score"] for r in sample},
        "judge_reasoning": {r["discussion_id"]: r["judge_reasoning"] for r in sample},
    }
    KEY_PATH.write_text(json.dumps(key, indent=2), encoding="utf-8")

    print(f"Wrote {len(sample)} items to {SHEET_PATH}")
    print(f"  (stratified: {n_pass} the judge scored 1, {n_fail} it scored 0 — order shuffled)")
    print(f"Answer key (judge verdicts, do NOT open before scoring) -> {KEY_PATH}")
    print("\nFill in `agree` and `docs_answerable` for each row, then run:")
    print("    uv run scripts/score_judge_agreement.py")


if __name__ == "__main__":
    main()
