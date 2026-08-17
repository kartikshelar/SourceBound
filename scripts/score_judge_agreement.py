"""
Score the hand-filled review sheet against the LLM judge (§2d.1).

Reports:
  - raw agreement (the headline number)
  - Cohen's kappa (agreement corrected for chance — raw agreement alone is
    misleading when one class dominates; kappa near 0 means the judge is no
    better than guessing at the observed base rate)
  - directional errors: judge-too-lenient (judge=1, you=0) vs judge-too-harsh
    (judge=0, you=1). This is the part that says whether the reported 48%
    baseline is inflated or deflated, not just noisy.
  - the docs-answerable ceiling: what fraction of questions are answerable
    from docs at all, and baseline accuracy restricted to those. A docs-only
    agent cannot beat the answerable fraction, so this reframes "48% vs a 70%
    target" into a target that is actually reachable.
  - every disagreement printed in full, so a bad rubric is visible rather than
    hidden behind an aggregate.

Usage:
    uv run scripts/score_judge_agreement.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHEET_PATH = ROOT / "data" / "eval" / "judge_review_sheet.jsonl"
KEY_PATH = ROOT / "data" / "eval" / "judge_review_key.json"
RESULTS_PATH = ROOT / "data" / "eval" / "results" / "baseline_dev_run.jsonl"
OUT_PATH = ROOT / "data" / "eval" / "results" / "judge_validation.json"


def cohens_kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    observed = sum(x == y for x, y in zip(a, b)) / n
    # expected agreement from the two raters' marginal distributions
    pa1, pb1 = sum(a) / n, sum(b) / n
    expected = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if expected == 1.0:
        return float("nan")  # degenerate: both raters gave one class to everything
    return (observed - expected) / (1 - expected)


def main() -> None:
    if not SHEET_PATH.exists():
        raise SystemExit(f"{SHEET_PATH} not found — run scripts/export_judge_review.py first.")

    rows = [json.loads(line) for line in SHEET_PATH.open(encoding="utf-8") if line.strip()]
    key = json.loads(KEY_PATH.read_text(encoding="utf-8"))
    judge_scores = key["judge_scores"]
    judge_reasoning = key["judge_reasoning"]

    filled = [r for r in rows if r.get("agree") in (0, 1)]
    unfilled = len(rows) - len(filled)
    if not filled:
        raise SystemExit(
            f"No rows in {SHEET_PATH} have `agree` set to 0 or 1 yet. "
            "Fill in the sheet before scoring."
        )

    human = [r["agree"] for r in filled]
    judge = [judge_scores[r["discussion_id"]] for r in filled]

    n = len(filled)
    raw_agreement = sum(h == j for h, j in zip(human, judge)) / n
    kappa = cohens_kappa(human, judge)
    too_lenient = sum(1 for h, j in zip(human, judge) if j == 1 and h == 0)
    too_harsh = sum(1 for h, j in zip(human, judge) if j == 0 and h == 1)

    # ---- docs-answerable ceiling ----
    tagged = [r for r in filled if r.get("docs_answerable") in (0, 1)]
    ceiling_block = None
    if tagged:
        n_answerable = sum(r["docs_answerable"] for r in tagged)
        answerable_ids = {r["discussion_id"] for r in tagged if r["docs_answerable"] == 1}
        all_results = {
            json.loads(l)["discussion_id"]: json.loads(l)
            for l in RESULTS_PATH.open(encoding="utf-8") if l.strip()
        }
        # accuracy on answerable items, using YOUR verdicts as truth (not the judge's)
        human_by_id = {r["discussion_id"]: r["agree"] for r in tagged}
        answerable_correct = sum(human_by_id[i] for i in answerable_ids)
        ceiling_block = {
            "n_tagged": len(tagged),
            "n_docs_answerable": n_answerable,
            "docs_answerable_rate": n_answerable / len(tagged),
            "human_accuracy_on_answerable": (
                answerable_correct / len(answerable_ids) if answerable_ids else None
            ),
            "note": (
                "docs_answerable_rate is an estimated CEILING for a docs-only agent on "
                "this eval set. Items tagged 0 need maintainer knowledge absent from the "
                "docs corpus — they motivate the v1 discussion_search tool, not better "
                "retrieval."
            ),
        }
        _ = all_results  # kept for provenance/debugging; accuracy uses human verdicts

    summary = {
        "n_reviewed": n,
        "n_unfilled": unfilled,
        "raw_agreement": raw_agreement,
        "cohens_kappa": kappa,
        "judge_too_lenient": too_lenient,
        "judge_too_harsh": too_harsh,
        "docs_answerable_ceiling": ceiling_block,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"=== Judge validation (n={n}" + (f", {unfilled} rows still unfilled" if unfilled else "") + ") ===")
    print(f"raw agreement:      {raw_agreement:.0%}")
    print(f"Cohen's kappa:      {kappa:.2f}   " + _kappa_reading(kappa))
    print(f"judge too lenient:  {too_lenient}  (judge said correct, you said wrong)")
    print(f"judge too harsh:    {too_harsh}  (judge said wrong, you said correct)")
    if too_lenient > too_harsh:
        print("  -> net: the reported baseline is likely INFLATED")
    elif too_harsh > too_lenient:
        print("  -> net: the reported baseline is likely DEFLATED (real score is better)")

    if ceiling_block:
        print(f"\n=== Docs-answerable ceiling (n={ceiling_block['n_tagged']}) ===")
        print(f"answerable from docs alone: {ceiling_block['docs_answerable_rate']:.0%}")
        acc = ceiling_block["human_accuracy_on_answerable"]
        if acc is not None:
            print(f"baseline accuracy on those: {acc:.0%}  <- the number worth improving")

    disagreements = [
        (r, judge_scores[r["discussion_id"]])
        for r in filled
        if r["agree"] != judge_scores[r["discussion_id"]]
    ]
    if disagreements:
        print(f"\n=== {len(disagreements)} disagreement(s) ===")
        for r, js in disagreements:
            print("-" * 70)
            print(f"#{r['number']} {r['title'][:80]}")
            print(f"  you={r['agree']}  judge={js}")
            print(f"  judge said: {judge_reasoning[r['discussion_id']][:200]}")
            if r.get("note"):
                print(f"  your note:  {r['note'][:200]}")

    print(f"\nSummary -> {OUT_PATH}")


def _kappa_reading(k: float) -> str:
    if k != k:  # NaN
        return "(undefined — one rater used a single class)"
    if k < 0.20:
        return "(poor — judge is close to chance; do NOT trust the baseline)"
    if k < 0.40:
        return "(fair — weak; treat numbers as directional only)"
    if k < 0.60:
        return "(moderate)"
    if k < 0.80:
        return "(substantial — usable)"
    return "(near-perfect)"


if __name__ == "__main__":
    main()
