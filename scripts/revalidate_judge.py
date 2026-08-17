"""
Re-run the (hardened) judge against the SAME 20 hand-scored items that the v1
judge failed on, and compare agreement v1 vs v2.

Like-for-like by construction: identical items, identical human labels. Any
change in kappa is attributable to the rubric, not to a different sample.
Cheap too — 20 judge calls, no answer synthesis (candidate answers are read
from the existing baseline run).

Usage:
    uv run scripts/revalidate_judge.py
    uv run scripts/revalidate_judge.py --judge-model llama-3.3-70b-versatile

--judge-model swaps ONLY the judge model, leaving the rubric untouched, so an
8b-vs-70b comparison isolates model capability from prompt design.
"""

import argparse
import collections
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eval.judge import LLMJudge  # noqa: E402
from llm.router import LLMRouter, ModelTier  # noqa: E402

SHEET_PATH = ROOT / "data" / "eval" / "judge_review_sheet.jsonl"
KEY_PATH = ROOT / "data" / "eval" / "judge_review_key.json"
RESULTS_DIR = ROOT / "data" / "eval" / "results"


def cohens_kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    expected = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if expected == 1.0:
        return float("nan")
    return (observed - expected) / (1 - expected)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-model", default=None,
                    help="Groq model id to use for judging (default: whatever TIER_MODELS says)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in SHEET_PATH.open(encoding="utf-8") if l.strip()]
    labeled = [r for r in rows if r.get("agree") in (0, 1)]
    if not labeled:
        raise SystemExit("No hand labels in the review sheet — nothing to validate against.")

    key = json.loads(KEY_PATH.read_text(encoding="utf-8"))
    v1_scores = key["judge_scores"]

    overrides = None
    if args.judge_model:
        overrides = {ModelTier.JUDGE: ("groq", args.judge_model)}
        print(f"Judge model override: {args.judge_model}\n")
    judge = LLMJudge(router=LLMRouter(tier_overrides=overrides))

    # Sanitize BOTH "." and "/" — provider-prefixed ids like
    # "openai/gpt-oss-120b" otherwise turn the slash into a directory
    # separator and the open() fails on a non-existent folder.
    tag = (
        args.judge_model.replace(".", "_").replace("/", "__")
        if args.judge_model else "v2"
    )
    out_path = RESULTS_DIR / f"judge_validation_{tag}.json"
    # Per-item cache. The 70b free tier is 100k tokens/DAY and v4 costs 2 calls
    # per item, so a full run can die partway (it did, at 13/20). Caching each
    # scored item means a resume only pays for what's missing instead of
    # re-burning quota on work already done.
    cache_path = RESULTS_DIR / f"judge_validation_{tag}.cache.jsonl"

    cache: dict[str, dict] = {}
    if cache_path.exists():
        for line in cache_path.open(encoding="utf-8"):
            if line.strip():
                rec = json.loads(line)
                cache[rec["discussion_id"]] = rec
    if cache:
        print(f"Resuming: {len(cache)} item(s) already scored in {cache_path.name}\n")

    human, v1, v2, details = [], [], [], []
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cache_fh = cache_path.open("a", encoding="utf-8")
    for i, r in enumerate(labeled):
        did = r["discussion_id"]
        if did in cache:
            out = cache[did]
            status = "cached"
        else:
            out = judge.score(
                r["title"], r["reference_accepted_answer"], r["candidate_generated_answer"]
            )
            if out["score"] is None:
                print(f"  [{i+1}/{len(labeled)}] #{r['number']} -> {out.get('verdict','FAILED')}, skipped")
                continue
            cache_fh.write(json.dumps({"discussion_id": did, **out}, ensure_ascii=False) + "\n")
            cache_fh.flush()
            status = "new"

        human.append(r["agree"])
        v1.append(v1_scores[did])
        v2.append(out["score"])
        details.append({
            "discussion_id": did,
            "number": r["number"],
            "title": r["title"],
            "human": r["agree"],
            "v1_score": v1_scores[did],
            "v2_score": out["score"],
            "v2_verdict": out.get("verdict", ""),
            "v2_key_claim": out.get("key_claim", ""),
            "v2_reasoning": out.get("reasoning", ""),
            "human_note": r.get("note", ""),
        })
        print(f"  [{i+1}/{len(labeled)}] #{r['number']} -> v2={out['score']} "
              f"({out.get('verdict','?')})  human={r['agree']}  v1={v1_scores[did]}  [{status}]")
        if status == "new":
            time.sleep(2)
    cache_fh.close()

    n = len(human)
    if n == 0:
        raise SystemExit("All judge outputs were unparseable.")

    def block(scores: list[int], label: str) -> dict:
        agree = sum(h == s for h, s in zip(human, scores)) / n
        return {
            "label": label,
            "raw_agreement": agree,
            "cohens_kappa": cohens_kappa(human, scores),
            "too_lenient": sum(1 for h, s in zip(human, scores) if s == 1 and h == 0),
            "too_harsh": sum(1 for h, s in zip(human, scores) if s == 0 and h == 1),
            "reported_accuracy": sum(scores) / n,
        }

    v2_label = f"v2 on {args.judge_model}" if args.judge_model else "v2 (claim-extraction)"
    b1, b2 = block(v1, "v1 (fuzzy, 8b)"), block(v2, v2_label)
    human_acc = sum(human) / n

    summary = {
        "n": n,
        "human_accuracy": human_acc,
        "v1": b1,
        "v2": b2,
        "details": details,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n=== Judge rubric comparison (n={n}, same hand labels) ===")
    print(f"human-truth accuracy: {human_acc:.0%}\n")
    hdr = f"{'':22s} {'agreement':>10s} {'kappa':>7s} {'lenient':>8s} {'harsh':>6s} {'reports':>8s}"
    print(hdr)
    for b in (b1, b2):
        print(f"{b['label']:22s} {b['raw_agreement']:>9.0%} {b['cohens_kappa']:>7.2f} "
              f"{b['too_lenient']:>8d} {b['too_harsh']:>6d} {b['reported_accuracy']:>7.0%}")

    # Verdict distribution: a judge that collapses everything into one category can
    # score well on raw agreement while being useless (kappa near 0). This is how
    # the v2 'omits' catch-all was caught — surface it by default.
    dist = collections.Counter(d["v2_verdict"] for d in details)
    print("\nverdict distribution: " + ", ".join(f"{k}={v}" for k, v in dist.most_common()))
    if dist and dist.most_common(1)[0][1] / n >= 0.70:
        top, top_n = dist.most_common(1)[0]
        # A dominant verdict is only a problem if it is dominant BEYOND what the human
        # labels justify. When the baseline genuinely fails most items, a judge that
        # says "wrong" most of the time is tracking reality, not collapsing — flagging
        # that would cry wolf (it did, on the v4/70b run: omits=14, of which 13 were
        # correctly wrong). Compare against the human base rate instead of a bare count.
        implied = 0 if top in ("omits", "contradicts") else 1
        human_rate = sum(1 for h in human if h == implied) / n
        if top_n / n - human_rate >= 0.20:
            print(f"  WARNING: '{top}' covers {top_n}/{n} items but only {human_rate:.0%} of "
                  "human labels agree — judge may be collapsing categories.")
        else:
            print(f"  ('{top}' is dominant but matches the human base rate of {human_rate:.0%} "
                  "— skew reflects the data, not judge collapse.)")

    dk = b2["cohens_kappa"] - b1["cohens_kappa"]
    print(f"\nkappa delta: {dk:+.2f}")
    if b2["cohens_kappa"] >= 0.60:
        print("v2 is substantial (>=0.60) — usable as an eval instrument.")
    elif b2["cohens_kappa"] > b1["cohens_kappa"]:
        print("v2 improved but is still below 0.60 — treat numbers as directional only.")
    else:
        print("v2 did NOT improve — do not proceed to retrieval work on this judge.")

    flips = [d for d in details if d["v1_score"] != d["v2_score"]]
    if flips:
        print(f"\n=== {len(flips)} item(s) where v2 changed the verdict ===")
        for d in flips:
            mark = "FIXED" if d["v2_score"] == d["human"] else "BROKE"
            print(f"  [{mark}] #{d['number']} v1={d['v1_score']} -> v2={d['v2_score']} "
                  f"(human={d['human']}, verdict={d['v2_verdict']})")
            print(f"          claim: {str(d['v2_key_claim'])[:110]}")

    print(f"\nSummary -> {out_path}")


if __name__ == "__main__":
    main()
