"""
Run an eval VARIANT against the same frozen dev split as the baseline, so the
difference is attributable to the retrieval change and nothing else (§2e).

Variants:
  hybrid       docs corpus, dense+BM25 RRF fusion instead of dense-only
  discussions  docs (dense) PLUS discussion_search results in the prompt
  both         hybrid docs + discussions

Everything else is held fixed: same 50 dev items, same synthesis model and
prompt, same validated judge (70b, v4 two-stage, kappa 0.63). Comparing a
variant scored 50/50 against a baseline scored on a different subset would
confound the retrieval change with the question mix, so `--require-complete`
(default on) refuses to report a delta unless both cover the same items.

Resumable and cached per item, like run_baseline.py — the 70b judge's 100k
tokens/day cap reliably interrupts a 50-item run.

Usage:
    uv run -m src.eval.run_variant --variant hybrid
    uv run -m src.eval.run_variant --variant discussions
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.baseline import MAX_CHUNK_CHARS, SYSTEM_PROMPT, SYSTEM_PROMPT_COMMITTED
from eval.judge import LLMJudge
from llm.router import DailyQuotaExhausted, LLMRouter, ModelTier
from retrieval.discussion_search import DiscussionSearch, DiscussionSearchInput
from retrieval.embeddings import EmbeddingModel
from retrieval.hybrid_search import HybridSearch
from retrieval.search import DocSearch, DocSearchInput

ROOT = Path(__file__).resolve().parent.parent.parent
FROZEN_EVAL_PATH = ROOT / "data" / "eval" / "eval_frozen_v1.jsonl"
RESULTS_DIR = ROOT / "data" / "eval" / "results"
BASELINE_PATH = RESULTS_DIR / "baseline_dev_run.jsonl"

DOC_K = 5
DISCUSSION_K = 2
# Discussions get a TIGHTER per-chunk budget than docs, not the same one.
# Measured: discussion chunks are ~3,058 chars at the median (they are whole
# Q&A threads) vs ~600 for doc sections. Reusing MAX_CHUNK_CHARS=2000 for both
# pushed the synthesis prompt to 6,134 tokens and tripped the 8b model's
# 6,000 tokens-per-minute ceiling mid-run. 1,200 chars keeps the accepted
# answer (which leads the chunk after the question) while staying inside it.
MAX_DISCUSSION_CHARS = 1200
# Groq's 8b synthesis model caps a single request at 6,000 TPM. Leave headroom
# for the system prompt and the generated answer.
PROMPT_TOKEN_CEILING = 4800
# Candidate pool the reranker chooses DOC_K from. Measured at 20 in D17
# (recall@5 45% -> 50%); a pool equal to DOC_K would make reranking a no-op.
RERANK_POOL = 20


def clip(text: str, limit: int = MAX_CHUNK_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + "\n... [truncated]"


def build_prompt(question: str, doc_results, discussion_results) -> str:
    blocks = []
    for i, c in enumerate(doc_results):
        blocks.append(f"[Doc {i+1} | {c.source_file} | {c.section_path}]\n{clip(c.text)}")
    for i, d in enumerate(discussion_results):
        blocks.append(
            f"[Discussion {i+1} | #{d.number} {d.title} | {d.url}]\n"
            f"{clip(d.text, MAX_DISCUSSION_CHARS)}"
        )
    excerpts = "\n\n".join(blocks)
    prompt = f"Retrieved context:\n\n{excerpts}\n\nQuestion: {question}\n\nAnswer:"

    # Hard ceiling. Per-chunk clipping bounds the typical case, but a long
    # question body can still breach the 8b model's 6,000 TPM per-request cap
    # and fail the item outright. Drop the lowest-ranked context block until it
    # fits — degrading the context beats losing the data point, and dropping
    # from the tail removes the least-relevant material first.
    while len(prompt) // 4 > PROMPT_TOKEN_CEILING and len(blocks) > 1:
        blocks.pop()
        excerpts = "\n\n".join(blocks)
        prompt = f"Retrieved context:\n\n{excerpts}\n\nQuestion: {question}\n\nAnswer:"
    return prompt


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
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--variant", required=True,
        choices=[
            "hybrid", "discussions", "both", "rerank", "rerank_discussions",
            # Retrieval held IDENTICAL to the baseline; only the synthesis system
            # prompt changes. That isolation is the point — three retrieval
            # variants have failed to beat 34.0%, so this tests whether the
            # constraint is synthesis instead.
            "committed_prompt",
        ],
    )
    args = ap.parse_args()

    out_path = RESULTS_DIR / f"variant_{args.variant}_dev_run.jsonl"
    items = load_dev_items()
    done = load_scored(out_path)
    remaining = [i for i in items if i["discussion_id"] not in done]

    print(f"Variant '{args.variant}': {len(items)} dev items, {len(done)} done, "
          f"{len(remaining)} remaining.")

    if remaining:
        embedder = EmbeddingModel()  # shared: loading bge twice is pure waste
        use_hybrid = args.variant in ("hybrid", "both")
        use_discussions = args.variant in ("discussions", "both", "rerank_discussions")
        use_rerank = args.variant in ("rerank", "rerank_discussions")
        system_prompt = (
            SYSTEM_PROMPT_COMMITTED if args.variant == "committed_prompt" else SYSTEM_PROMPT
        )

        reranker = None
        if use_rerank:
            from retrieval.rerank import Reranker
            reranker = Reranker()

        if use_hybrid:
            doc_search = HybridSearch(collection_name="fastapi_docs", embedder=embedder)
        else:
            doc_search = DocSearch()
            doc_search._embedder = embedder

        discussion_search = None
        if use_discussions:
            discussion_search = DiscussionSearch(embedder=embedder)

        router = LLMRouter()
        judge = LLMJudge(router=router)

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        stopped = False
        with out_path.open("a", encoding="utf-8") as f:
            for i, item in enumerate(remaining):
                question = f"{item['title']}\n\n{item['question_body']}"
                try:
                    if reranker:
                        # Retrieve a wider pool, then let the cross-encoder pick the
                        # final DOC_K. Reranking the same k it returns would be a
                        # no-op — the gain comes from having more to choose from.
                        pool = doc_search(DocSearchInput(query=question, k=RERANK_POOL)).results
                        docs = reranker.rerank(question, pool, lambda r: r.text, DOC_K)
                    else:
                        docs = doc_search(DocSearchInput(query=question, k=DOC_K)).results

                    discs = (
                        discussion_search(
                            DiscussionSearchInput(query=question, k=DISCUSSION_K)
                        ).results
                        if discussion_search
                        else []
                    )
                    if reranker and discs:
                        discs = reranker.rerank(question, discs, lambda r: r.text, DISCUSSION_K)
                    prompt = build_prompt(question, docs, discs)
                    answer = router.call(ModelTier.SYNTHESIS, prompt, system=system_prompt).text
                    judged = judge.score(question, item["accepted_answer"], answer)
                except DailyQuotaExhausted as e:
                    print(f"\n  STOPPED at {i+1}/{len(remaining)}: {e}")
                    stopped = True
                    break

                citations = [c.source_file for c in docs] + [d.url for d in discs]
                f.write(json.dumps({
                    "discussion_id": item["discussion_id"],
                    "number": item["number"],
                    "title": item["title"],
                    "generated_answer": answer,
                    "citations": citations,
                    "accepted_answer": item["accepted_answer"],
                    "judge_score": judged["score"],
                    "judge_verdict": judged.get("verdict", ""),
                    "judge_key_claim": judged.get("key_claim", ""),
                    "judge_reasoning": judged["reasoning"],
                }, ensure_ascii=False) + "\n")
                f.flush()
                print(f"  [{i+1}/{len(remaining)}] #{item['number']} -> score={judged['score']}")
                time.sleep(2)

        print(f"\nThis run: {router.call_count} LLM calls in {time.time() - t0:.0f}s")
        if stopped:
            print("Daily quota hit — re-run to continue.")

    report(args.variant, out_path, items)


def report(variant: str, out_path: Path, items: list[dict]) -> None:
    variant_rows = load_scored(out_path)
    baseline_rows = load_scored(BASELINE_PATH)
    if not variant_rows:
        print("No variant results yet.")
        return

    # Only compare on items BOTH have scored — otherwise the delta mixes a
    # retrieval change with a different question mix. `judge_score is None`
    # means the judge output failed to parse; those items are unscored, not
    # zero, so they must be dropped rather than counted as failures.
    shared = {
        i for i in (set(variant_rows) & set(baseline_rows))
        if variant_rows[i]["judge_score"] is not None
        and baseline_rows[i]["judge_score"] is not None
    }
    if not shared:
        print("No items scored by BOTH runs yet — nothing to compare.")
        return
    unscored = len(set(variant_rows) & set(baseline_rows)) - len(shared)
    v_score = sum(variant_rows[i]["judge_score"] for i in shared)
    b_score = sum(baseline_rows[i]["judge_score"] for i in shared)
    n = len(shared)

    print(f"\n=== Variant '{variant}' vs baseline (n={n} shared items) ===")
    if unscored:
        print(f"  ({unscored} item(s) excluded: judge output unparseable in one run)")
    if n < len(items):
        print(f"  PARTIAL: {n}/{len(items)} dev items scored in both. "
              "Delta is provisional until complete.")
    print(f"  baseline : {b_score}/{n} = {b_score/n:.1%}")
    print(f"  {variant:9s}: {v_score}/{n} = {v_score/n:.1%}")
    delta = (v_score - b_score) / n
    print(f"  delta    : {delta:+.1%}")
    if v_score == b_score:
        print("  -> no change; does not earn its place on this evidence (§2e).")
    elif v_score < b_score:
        print("  -> WORSE than baseline. Do not ship.")
    else:
        flips = sum(
            1 for i in shared
            if variant_rows[i]["judge_score"] == 1 and baseline_rows[i]["judge_score"] == 0
        )
        broke = sum(
            1 for i in shared
            if variant_rows[i]["judge_score"] == 0 and baseline_rows[i]["judge_score"] == 1
        )
        print(f"  -> fixed {flips}, broke {broke}")
        if n < 50:
            print("  Treat as directional until all 50 items are scored.")


if __name__ == "__main__":
    main()
