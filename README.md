# SourceBound — an agentic FastAPI support assistant

Answers FastAPI usage questions from the official docs and answered GitHub
Discussions, with citations — and is evaluated against real accepted answers
rather than vibes.

The interesting part is not the RAG pipeline. It is that **every component was
measured, and most of the upgrades failed.** This repo keeps the failures.

---

## Headline results

| | |
|---|---|
| Naive-RAG baseline (50-item frozen dev split) | **30.6%** answer correctness |
| LLM-as-judge, hand-validated | **Cohen's κ = 0.67**, 0 harsh errors |
| Questions answerable from the docs at all | **~35%** (hand-tagged) |

Four retrieval/prompt upgrades were built, measured, and **none beat the
baseline**:

| Upgrade | Result |
|---|---|
| Hybrid BM25 + dense (RRF) | recall@5 41% vs 45% dense-only |
| Discussions as a 2nd source | −5.3% (within noise) |
| Cross-encoder reranker | recall@5 45% → 50%, but −3.6% end-to-end |
| "Commit to an answer" prompt | **−11.4%** — eliminated hedging, produced fabrication |

That last one is the most informative: forcing the model to stop hedging
converted honest "I don't know" into confident wrong answers
(`contradicts` 0 → 4). Hedging was a symptom, not the disease.

**The measured conclusion:** retrieval ranking was never the bottleneck. ~65%
of these questions have answers that are not in the corpus at all, so the real
problem is *knowing when you cannot answer* — which is what the agent layer
exists to do.

### The agent, measured

**v1 scored 7.1% vs the 28.6% baseline** on 42 items — a large, real
regression, not noise. It escalated on **83%** of questions.

The headline number alone would have been misleading, and the per-decision
logging is what showed why:

| | v1 |
|---|---|
| Accuracy **when it did answer** | **42.9%** (vs 28.6% baseline) |
| Escalations where the baseline **also failed** | 26 of 35 |
| Escalations that were **lost opportunities** | 9 of 35 |
| Confident fabrications (`contradicts`) | 3 (vs baseline 4) |

So the sufficiency check finds unanswerable questions well — 74% of its
refusals were correct, and it fabricates less than the baseline. It was simply
miscalibrated: the prompt demanded an *explicit* answer in context, so it also
rejected questions it could have reasoned through.

v2 loosens that to "can a useful answer be **reasoned** from this context?"
Numbers pending — free-tier quota caps throughput at ~26 eval items/day.
The falsification test is logged in advance: if `contradicts` rises above the
baseline's 4 while escalations fall, the loosening went too far.

---

## How it works

```
question
   ↓
 route      pick docs / discussions / both        (cheap model)
   ↓
retrieve    dense top-k over the selected source  (no LLM)
   ↓
assess      "can this be answered from what I have?"
   ↓
 ┌─────────────────┴─────────────────┐
answer                            escalate
grounded synthesis            honest refusal + closest material
```

Every node exists because a measurement put it there:

- **route** — always injecting discussions cost −5.3% (irrelevant threads
  displace good context), and embedding scores cannot detect relevance
  (0.792 when correct vs 0.795 when wrong — no signal). So the choice needs
  something that reads the question.
- **assess** — the node that justifies the whole graph. Forcing an answer
  produces fabrication (−11.4%), and only ~35% of questions are answerable
  from the corpus. "Can I answer this?" is a real decision that a plain RAG
  pipeline has nowhere to put.
- **escalate** — for a support agent, an honest refusal is a *correct*
  outcome, not a failure.

Deliberately **not** added: a retry/re-retrieve loop. Four experiments showed
better retrieval does not move the number, so it would add cost for no
measured gain.

---

## Evaluation design

The part worth reading if you only read one thing.

- **Frozen eval set** — 249 answered FastAPI Discussions (question + accepted
  answer), pulled via GraphQL, filtered for substance and recency, split
  50 dev / 199 locked test. Committed as `data/eval/eval_frozen_v1.jsonl` with
  a manifest recording every filter parameter.
- **Leakage guard** — excluding eval discussions by ID is *necessary but not
  sufficient*. Users ask the same question twice: discussion #7707 has a title
  identical (cosine 1.000) to eval item #13972 and the same accepted answer.
  A near-duplicate title check removes those too, with every exclusion logged
  to `data/eval/leakage_exclusions.jsonl`.
- **Validated judge** — an LLM judge is worthless until you know it agrees
  with a human. 20 items were hand-scored blind; the first judge scored
  κ = 0.30 and passed an answer asserting the *opposite* of the reference.
  Four rubric iterations later: κ = 0.67 with zero harsh errors.
- **Honest comparisons** — deltas are only reported over items *both* runs
  scored, so a retrieval change is never confounded with a different question
  mix.

---

## Repo layout

```
src/ingest/      chunk docs (structure-aware) · index · leakage guard
src/retrieval/   embeddings · vector store · dense / hybrid / rerank search
src/agent/       LangGraph graph · state · Pydantic-contracted tools
src/eval/        baseline · judge · variant + agent runners
src/llm/         cost-aware model router
scripts/         pull discussions · freeze eval · validate judge · quota check
data/eval/       frozen eval set + manifest + results (committed)
decisions.md     running decision log — every choice, why, and what was rejected
```

## Setup

```bash
uv sync
uv run scripts/fetch_docs_snapshot.py    # pinned FastAPI docs (0.119.1)
uv run python -m src.ingest.index        # build the docs index
uv run python -c "import sys; sys.path.insert(0,'src'); from agent.graph import run; print(run('How do I add OAuth2 JWT auth?')['answer'])"
```

Create a `.env` in the project root:

```bash
# Required — routing, synthesis, and the eval judge all run on Groq
GROQ_API_KEY=

# Required only to rebuild the eval set from GitHub Discussions
GITHUB_TOKEN=

# Optional — Langfuse tracing. Absent, the agent runs untraced rather than
# failing. LANGFUSE_BASE_URL is accepted in place of LANGFUSE_HOST.
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=
```

Reproducing the eval additionally needs `scripts/pull_discussions.py` (a
GitHub token) and `src/ingest/index_discussions.py`.

## Running it as a service

```bash
uv run fastapi dev app/main.py     # http://127.0.0.1:8000
```

| Route | |
|---|---|
| `POST /ask` | Run the agent |
| `GET /health` | Liveness, `graph_ready`, `tracing` |
| `GET /` | Minimal UI |
| `GET /docs` | OpenAPI |

`/ask` returns the agent's **decisions**, not just an answer — `escalated`,
`route`, `assess_reason`, `citations`, `latency_ms`. Escalation is the
behaviour the system exists to demonstrate, and a caller cannot act on
"insufficient context" unless the API says so.

Operational notes: the graph and its ~400MB embedding model are built once at
startup (not per request); upstream quota exhaustion maps to `503` +
`Retry-After` rather than a `500`, since the caller's request was valid.

**Tracing** (optional): set `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` and
each run reports `route`, `escalated`, `assess_reason`, and per-tier LLM
spans. Without credentials the agent runs untraced rather than failing —
tracing must never break the thing it observes. `/health` reports whether
traces are actually flowing, since silent degradation otherwise looks
identical to no traffic.

## Known limitations

- Free-tier quota caps throughput at ~26 eval items/day, so the 199-item
  locked test split has not been run.
- The judge's 3 residual errors are all *lenient*, so reported scores are if
  anything slight over-estimates.
- Groq decommissioned both models the project originally ran on mid-build;
  earlier numbers were re-measured on the replacement stack and archived
  results are kept under `data/eval/results/archive/` but are **not**
  comparable to current ones.

See [decisions.md](decisions.md) for the full reasoning log and
[pre-build-checklist.md](pre-build-checklist.md) for the locked design
decisions.
