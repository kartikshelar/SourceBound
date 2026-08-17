"""
Run the naive-RAG baseline against the eval set's dev split and score with
the LLM judge. This produces the §2e baseline number every later upgrade
must beat.

The dev split is freely inspectable while iterating (§2b) — this script is
for the dev split ONLY. The 199-item locked test split must be run rarely,
via a separate invocation once the pipeline is trustworthy, and never
eyeballed to tune against.

Resumable by design: writes each result to disk as soon as it's scored and
skips discussion_ids already present in the output file on a re-run. This
matters on free-tier models — a daily token quota can be exhausted mid-run
(hit this in practice: llama-3.3-70b-versatile's 100k TPD cap tripped at
item 43/50), and re-running from scratch would silently throw away
already-paid-for (well, already-used-quota) API calls.

Usage:
    uv run -m src.eval.run_baseline
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.baseline import NaiveRAGBaseline
from eval.judge import LLMJudge
from llm.router import DailyQuotaExhausted, LLMRouter

FROZEN_EVAL_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "eval" / "eval_frozen_v1.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "eval" / "results"
OUT_PATH = RESULTS_DIR / "baseline_dev_run.jsonl"
SUMMARY_PATH = RESULTS_DIR / "baseline_dev_summary.json"


def load_dev_items() -> list[dict]:
    items = []
    with FROZEN_EVAL_PATH.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["split"] == "dev":
                items.append(item)
    return items


def load_completed_ids() -> set[str]:
    if not OUT_PATH.exists():
        return set()
    ids = set()
    with OUT_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["discussion_id"])
    return ids


def write_summary() -> dict:
    results = []
    with OUT_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    scored = [r for r in results if r["judge_score"] is not None]
    correctness = sum(r["judge_score"] for r in scored) / len(scored) if scored else 0.0
    unparseable = len(results) - len(scored)
    with_citation = sum(1 for r in results if r["citations"])
    citation_rate = with_citation / len(results) if results else 0.0

    summary = {
        "split": "dev",
        "n_items_completed": len(results),
        "n_scored": len(scored),
        "n_unparseable_judge": unparseable,
        "answer_correctness": correctness,
        "answers_with_citation_rate": citation_rate,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    items = load_dev_items()
    completed_ids = load_completed_ids()
    remaining = [item for item in items if item["discussion_id"] not in completed_ids]

    print(f"Dev set: {len(items)} items total, {len(completed_ids)} already completed, "
          f"{len(remaining)} remaining.")
    if not remaining:
        print("Nothing left to run.")
    else:
        shared_router = LLMRouter()
        baseline = NaiveRAGBaseline()
        baseline._router = shared_router  # share one budget-tracked router across synthesis + judge
        judge = LLMJudge(router=shared_router)

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        stopped_early = False
        with OUT_PATH.open("a", encoding="utf-8") as f:
            for i, item in enumerate(remaining):
                question = f"{item['title']}\n\n{item['question_body']}"
                try:
                    gen = baseline.answer(question)
                    judged = judge.score(question, item["accepted_answer"], gen["answer"])
                except DailyQuotaExhausted as e:
                    # Expected on the free tier, not a failure: stop cleanly, keep every
                    # item already written, and report what we have. Re-running tomorrow
                    # resumes from here.
                    print(f"\n  STOPPED at item {i+1}/{len(remaining)}: {e}")
                    stopped_early = True
                    break

                result = {
                    "discussion_id": item["discussion_id"],
                    "number": item["number"],
                    "title": item["title"],
                    "generated_answer": gen["answer"],
                    "citations": gen["citations"],
                    "accepted_answer": item["accepted_answer"],
                    "judge_score": judged["score"],
                    # verdict + key_claim are the judge's diagnostic trail. Persisting
                    # them is what makes a failure analysis possible later (it is how
                    # the v3 claim-contamination bug was found) — a bare 0/1 tells you
                    # a run got worse but never why.
                    "judge_verdict": judged.get("verdict", ""),
                    "judge_key_claim": judged.get("key_claim", ""),
                    "judge_reasoning": judged["reasoning"],
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                print(f"  [{i+1}/{len(remaining)}] #{item['number']} -> score={judged['score']} "
                      f"(router calls so far: {shared_router.call_count})")
                time.sleep(2)  # judge model has a tight free-tier quota; pace requests

        print(f"\nThis run: {shared_router.call_count} LLM calls in {time.time() - t0:.0f}s")
        if stopped_early:
            print("Daily quota hit — re-run this command after the quota resets to continue.")

    summary = write_summary()
    print(f"\n=== Naive-RAG baseline (dev split, {summary['n_items_completed']}/{len(items)} items) ===")
    print(f"answer_correctness: {summary['answer_correctness']:.1%}  (target >= 70%)")
    print(f"answers_with_citation_rate: {summary['answers_with_citation_rate']:.1%}  "
          f"(target >= 90%, note: NOT citation VALIDITY yet)")
    print(f"unparseable judge outputs: {summary['n_unparseable_judge']}/{summary['n_items_completed']}")
    print(f"\nResults -> {OUT_PATH}")
    print(f"Summary -> {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
