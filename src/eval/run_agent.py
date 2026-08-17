"""
Evaluate the LangGraph agent against the same frozen dev split as the
baseline (§2e).

Why this is a separate runner from `run_variant.py`: that script swaps the
retrieval layer while keeping a fixed retrieve -> synthesize pipeline. The
agent does not have that shape — it routes, decides whether the retrieved
context is sufficient, and may escalate instead of answering. Those decisions
are the thing under test, so the runner has to invoke the graph and record
what it chose, not just what it said.

Escalations are scored as 0 by the judge (an escalation contains no answer),
which is the honest accounting: a refusal does not solve the user's problem.
But `escalated` is recorded per item so the two failure modes can be
separated in analysis — an escalation is a *safe* miss, whereas a
`contradicts` is a confidently wrong answer that a support user would act on.
D19 measured that distinction mattering: forcing answers converted hedges into
fabrications and cost 11.4 points.

Resumable and cached per item, like the other runners.

Usage:
    uv run -m src.eval.run_agent
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.graph import build_graph
from eval.judge import LLMJudge
from llm.router import DailyQuotaExhausted, LLMRouter

ROOT = Path(__file__).resolve().parent.parent.parent
FROZEN_EVAL_PATH = ROOT / "data" / "eval" / "eval_frozen_v1.jsonl"
RESULTS_DIR = ROOT / "data" / "eval" / "results"
OUT_PATH = RESULTS_DIR / "agent_dev_run.jsonl"
BASELINE_PATH = RESULTS_DIR / "baseline_dev_run.jsonl"


def load_dev_items() -> list[dict]:
    return [
        json.loads(l)
        for l in FROZEN_EVAL_PATH.open(encoding="utf-8")
        if l.strip() and json.loads(l)["split"] == "dev"
    ]


def load_scored(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {
        json.loads(l)["discussion_id"]: json.loads(l)
        for l in path.open(encoding="utf-8")
        if l.strip()
    }


def main() -> None:
    items = load_dev_items()
    done = load_scored(OUT_PATH)
    remaining = [i for i in items if i["discussion_id"] not in done]
    print(f"Agent: {len(items)} dev items, {len(done)} done, {len(remaining)} remaining.")

    if remaining:
        router = LLMRouter()
        graph = build_graph(router)
        judge = LLMJudge(router=router)

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        stopped = False
        with OUT_PATH.open("a", encoding="utf-8") as f:
            for i, item in enumerate(remaining):
                question = f"{item['title']}\n\n{item['question_body']}"
                try:
                    state = graph.invoke({"question": question})
                    judged = judge.score(
                        question, item["accepted_answer"], state.get("answer", "")
                    )
                except DailyQuotaExhausted as e:
                    print(f"\n  STOPPED at {i+1}/{len(remaining)}: {e}")
                    stopped = True
                    break

                f.write(json.dumps({
                    "discussion_id": item["discussion_id"],
                    "number": item["number"],
                    "title": item["title"],
                    "generated_answer": state.get("answer", ""),
                    "citations": state.get("citations", []),
                    "accepted_answer": item["accepted_answer"],
                    "judge_score": judged["score"],
                    "judge_verdict": judged.get("verdict", ""),
                    "judge_reasoning": judged["reasoning"],
                    # agent-specific: what the graph DECIDED, not just what it said
                    "route": state.get("route"),
                    "route_reason": state.get("route_reason"),
                    "sufficient": state.get("sufficient"),
                    "assess_reason": state.get("assess_reason"),
                    "escalated": state.get("escalated", False),
                    "trace": state.get("trace", []),
                }, ensure_ascii=False) + "\n")
                f.flush()
                print(f"  [{i+1}/{len(remaining)}] #{item['number']} -> "
                      f"score={judged['score']} route={state.get('route')} "
                      f"{'ESCALATED' if state.get('escalated') else ''}")
                time.sleep(2)

        print(f"\nThis run: {router.call_count} LLM calls in {time.time() - t0:.0f}s")
        if stopped:
            print("Daily quota hit — re-run to continue.")

    report()


def report() -> None:
    agent_rows = load_scored(OUT_PATH)
    baseline_rows = load_scored(BASELINE_PATH)
    if not agent_rows:
        print("No agent results yet.")
        return

    shared = {
        i for i in (set(agent_rows) & set(baseline_rows))
        if agent_rows[i]["judge_score"] is not None
        and baseline_rows[i]["judge_score"] is not None
    }
    if not shared:
        print("No items scored by BOTH agent and baseline yet.")
        return

    n = len(shared)
    a = sum(agent_rows[i]["judge_score"] for i in shared)
    b = sum(baseline_rows[i]["judge_score"] for i in shared)
    esc = sum(1 for i in shared if agent_rows[i].get("escalated"))

    print(f"\n=== Agent vs baseline (n={n} shared items) ===")
    if n < 50:
        print(f"  PARTIAL: {n}/50 — provisional until complete.")
    print(f"  baseline : {b}/{n} = {b/n:.1%}")
    print(f"  agent    : {a}/{n} = {a/n:.1%}")
    print(f"  delta    : {(a-b)/n:+.1%}")
    print(f"  escalated: {esc}/{n} ({esc/n:.0%}) — scored 0, but a SAFE miss")

    # An escalation and a fabrication both score 0; only one of them would
    # mislead a real support user. Separate them.
    answered = [i for i in shared if not agent_rows[i].get("escalated")]
    if answered:
        ans_ok = sum(agent_rows[i]["judge_score"] for i in answered)
        print(f"  when it DID answer: {ans_ok}/{len(answered)} = {ans_ok/len(answered):.1%}")
    contra = sum(1 for i in shared if agent_rows[i].get("judge_verdict") == "contradicts")
    b_contra = sum(1 for i in shared if baseline_rows[i].get("judge_verdict") == "contradicts")
    print(f"  confidently wrong (contradicts): agent {contra} vs baseline {b_contra}")

    routes = {}
    for i in shared:
        r = agent_rows[i].get("route") or "?"
        routes[r] = routes.get(r, 0) + 1
    print(f"  routing: {routes}")


if __name__ == "__main__":
    main()
