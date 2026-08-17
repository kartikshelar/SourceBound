# Pre-Build Checklist — FastAPI Support Agent

> Context file for the Fable workflow. Purpose: lock the decisions that are
> expensive to reverse *before* Claude writes any code, so the build executes
> a design I own rather than one I inherit.
>
> Status key: [CLEARED] decided · [ACTIVE] deciding now · [PENDING] not yet

---

## Project one-liner
An agentic technical-support assistant that answers FastAPI usage questions from
the official docs, with citations, evaluated against real accepted answers from
GitHub Discussions. Agent-first (routing, multi-step, evals, observability);
RAG is a component, not the centerpiece.

---

## 0. Reality-check spike — [CLEARED]

- **Library data exists and is usable.** FastAPI directs all usage questions to
  GitHub **Discussions** (Questions category), which supports an **Answered**
  filter → every answered question carries an *accepted answer*. That is a
  free, labeled `question → accepted answer` dataset.
- **Plumbing implication:** the labeled Q&A lives in Discussions, reachable only
  via GitHub's **GraphQL API** (REST does not cover Discussions well). One-time
  bounded work.
- **Rate limits:** un-authenticated GitHub API caps at 60 req/hr. → Need a free
  **GitHub personal access token** for any real pull. (Moved to chores, §4.)
- **Backup library if FastAPI ever fails a check:** Polars (clean docs,
  data/ML-adjacent) — but confirm its answered-Q&A volume first.

## 1. Scope & success — [CLEARED]

- **Library:** FastAPI. Chosen over trendier LangChain for data hygiene:
  clean/stable MkDocs docs, MIT license, accepted-answer labels, instant
  recognizability, adjacent to my own model-serving stack. LangChain rejected:
  docs churn hard across versions → stale/wrong RAG corpus.
- **Pin a docs snapshot** at a specific FastAPI version/commit so the corpus
  doesn't drift mid-project.
- **Who the user is / what they ask** (drives routing later):
  1. How-to ("how do I add auth / handle file uploads?") → docs
  2. Debugging ("this validation error / dependency runs twice") → docs + past answered Discussion
  3. Conceptual / capability ("can it do background tasks? does it support X?") → docs, cross-section reasoning
- **Definition of done — two tiers:**
  - *Walking skeleton (v0):* ingest pinned docs → dense retrieval → one grounded
    answer with ≥1 doc-section citation → deployed on HF Spaces, bare UI.
    Success = runs end-to-end on a handful of hand-picked questions.
  - *Golden (v1):* + frozen eval set, routing (docs/discussions), hybrid
    retrieval + rerank, citation verification, human-in-the-loop escalation on
    low confidence, Langfuse tracing, cost routing across free LLMs — and
    **beats the naive-RAG baseline on the eval set by a stated margin**, with a
    few real users having tried it.
- **Success number (drives every downstream choice):** answer correctness on
  held-out answered Discussions ≥ **70%** (LLM-as-judge vs accepted answer);
  citation validity ≥ **90%**. Always vs a baseline, so "better" is shown.
- **Non-goals (what keeps this finishable):** one library only; no fine-tuning
  (retrieval + prompting only); no polished UI; no live doc-sync (pinned
  snapshot); no answering outside FastAPI's scope; no executing user code.

## 2. Data & evaluation — [ACTIVE]

### 2a. Two corpora, kept separate
- **Retrieval corpus (what the agent answers FROM):** FastAPI docs markdown,
  cloned from `fastapi/fastapi` at the pinned tag, under `docs/en/docs/`.
  (v1 option: add answered Discussions as a 2nd retrievable source — subject to
  the leakage rule below.)
- **Eval corpus (ground truth):** FastAPI Discussions → Questions → Answered,
  via GraphQL. Each item = question (title + body) + accepted answer.

### 2b. Eval-set freeze protocol
- **Pull** a pool of answered questions; **select ~300**.
- **Split:** ~50 **dev set** (may inspect freely while iterating) + ~250
  **locked test set** (run rarely, never eyeball to tune against).
- **Inclusion filters:** has accepted answer; answer is substantive (min length,
  not "fixed, closing"); recent enough to match the pinned docs version;
  genuinely about FastAPI (drop off-topic / third-party). Optional tag per item:
  `docs-answerable` vs `needs-experience` (report both).
- **Freeze mechanics:** snapshot selected items to a **versioned JSONL** in the
  repo with pull date + the exact GraphQL query recorded. Never edit. That file
  *is* the frozen eval set.

### 2c. Leakage rule — CRITICAL
- Every discussion in the eval set is **excluded from the retrievable index**
  (exclude by discussion ID). Otherwise the agent retrieves the exact accepted
  answer and the eval measures nothing while reporting a great score.
- Document the exclusion explicitly.

### 2d. Metrics (priority order)
1. **Answer correctness (primary):** LLM-as-judge vs accepted answer, rubric =
   "contains the key correct info," 0/1 (or 1–5). **Validate the judge**:
   hand-score ~20 items, confirm the judge agrees with me.
2. **Citation validity:** % of cited doc sections that actually support the
   claim; % of answers with ≥1 valid citation.
3. **Retrieval recall@k (v1):** gold sections = doc links embedded in accepted
   answers; measure whether retrieval surfaced them (skip items with no link).
4. **Latency + cost per query (secondary):** for the production story.

### 2e. Baseline to beat
- **Naive single-shot top-k dense RAG over docs**, run through the same frozen
  eval set. Every added mechanism (hybrid, rerank, routing, discussions,
  verification) must **move the number** to earn its place. Report deltas.

### Known pitfalls to watch
- Leakage (see 2c). · Junk labels (curate by hand). · Distribution mismatch
  (eval questions not representative of real user questions). · Unvalidated
  LLM judge (see 2d.1).

### What I must own to defend this in interviews
Why freeze-before-build; dev/test split rationale; the leakage exclusion; the
judge-validation step; the naive-RAG baseline and reported deltas.

---

## 3. Architecture — [CLEARED]

**Governing principle:** simplest defensible default for each piece, then let the
frozen eval justify every added complication. "Why X?" is answered with an eval
delta ("dense scored 61%, hybrid moved it to 73%"), not "best practice."

**Build order:** retrieval core FIRST, agent layer SECOND. Garbage retrieval makes
every agent trick worthless. Swappable pieces (embedding, LLM, vector store) sit
behind thin wrappers so "measure then swap" is one-file cheap.

Decisions:
- **Chunking:** structure-aware, not fixed-size. Split docs on markdown headers,
  carry the section path as metadata. Discussions = one Q&A per chunk. (Size /
  overlap → discover-later.)
- **Retrieval path:** dense top-k for v0 → **hybrid (BM25 + dense) + reranker**
  for v1. Domain reason for hybrid: FastAPI questions carry exact symbols
  (`HTTPException`, `@app.get`, error strings) where lexical/BM25 catches what
  dense embeddings blur.
- **Vector store:** Chroma (local, zero-infra) for the v0 skeleton →
  **Qdrant Cloud free tier** for deployed v1 (native hybrid search → cleaner v1
  upgrade; recognizable name).
- **Orchestration:** **LangGraph.** *(My earlier lean was a plain Python loop for
  defensibility; chose LangGraph for the structured state graph + resume keyword.)*
  Accountability condition: I must be able to explain the graph I build —
  nodes, state, edges, why the control flow is shaped this way — not lift a
  tutorial. That's what makes it a strength in an interview, not a liability.
- **Embedding model:** **local open model (`bge-base-en-v1.5`)** behind an
  interface — free, CPU-friendly, no rate limits, deterministic. A/B a second
  model on the eval later.
- **Tool registry:** `doc_search`, `discussion_search` (v1), `web_search`
  (fallback), `verify_citation` (internal). Each with a **Pydantic** input/output
  contract (= the structured-output pattern). The planner chooses which to call.
- **Runtime model routing (= cost-aware-router pattern; keeps runtime ~free):**
  cheap/fast model (Groq or Gemini Flash-Lite) for routing + simple steps →
  Gemini Flash for writing the answer → a stronger model reserved for the
  LLM-as-judge in eval. Model IDs stay swappable.

*Leave undecided on purpose (discover from skeleton + frozen eval):* embedding
choice (A/B), chunk size/overlap, reranker choice, retrieval `k`, exact prompts.

**What I must own to defend this:** the chunking rationale, the hybrid-retrieval
justification via measured eval deltas, and the **LangGraph graph structure**
(designed, not copied).

## 4. Setup & chores — [CLEARED]

Not decisions — a to-do list to tick off.

- **GitHub token:** fine-grained PAT, repository access = **Public repositories
  (read-only)**. Lifts the API limit to 5,000/hr; required for the Discussions
  GraphQL pull. (Cloning docs needs no token.) Store in `.env`.
- **Accounts (all free):** Google AI Studio (Gemini key) · Groq · Qdrant Cloud
  (free cluster) · Langfuse (cloud free tier) · Hugging Face (Spaces deploy).
- **Env:** `uv` (fast, modern) — `uv init` + `uv add`. venv+pip is the fallback.
- **Secrets:** `.env` only, never committed. But DO commit the frozen eval JSONL
  (§2) — it's the versioned artifact.
- **Running log:** `decisions.md` — every choice + why + what was rejected.

Repo skeleton (maps to build order):

    fastapi-support-agent/
    ├── CLAUDE.md            # build guardrails (§5)
    ├── README.md
    ├── decisions.md         # running decision log
    ├── pyproject.toml
    ├── .env.example
    ├── .gitignore
    ├── data/
    │   ├── docs_snapshot/   # pinned FastAPI docs markdown
    │   └── eval/            # frozen eval JSONL (§2) — COMMITTED
    ├── src/
    │   ├── ingest/          # clone docs, pull discussions, chunk, embed, index
    │   ├── retrieval/       # embed wrapper · vector-store wrapper · search
    │   ├── agent/           # LangGraph graph, nodes, tools
    │   ├── eval/            # run eval · judge · metrics · baseline
    │   └── llm/             # model-routing wrappers
    ├── app/                 # HF Spaces UI entrypoint
    └── scripts/             # pull_discussions.py · freeze_eval.py

- **`.env.example` keys:** `GITHUB_TOKEN` · `GEMINI_API_KEY` · `GROQ_API_KEY` ·
  `QDRANT_URL` · `QDRANT_API_KEY` · `LANGFUSE_PUBLIC_KEY` · `LANGFUSE_SECRET_KEY`
  · `LANGFUSE_HOST` · `HF_TOKEN`.
- **`.gitignore`:** `.env`, `.venv/`, `__pycache__/`, `*.pyc`, local vector-store
  dirs (`chroma/`), model caches. Do NOT ignore `data/eval/` (frozen set is
  meant to be versioned).

## 5. Build guardrails for Claude — [CLEARED]

Captured in `CLAUDE.md` at repo root (separate file). Key rules:

- **Model policy:** Sonnet default. Opus only when I explicitly ask. Never
  auto-escalate; never select the Fable model.
- **Parallelism cap:** no unbounded subagents — main cause of usage burn. One
  task at a time unless I approve otherwise.
- **Source of truth:** read `pre-build-checklist.md` first. If a decision isn't
  there, ASK — don't invent architecture.
- **Build order enforced:** retrieval core before agent layer.
- **Conventions:** wrappers around embedding/LLM/vector-store · Pydantic tool
  contracts · structure-aware chunking · `uv` (ask before new deps).
- **Accountability:** build incrementally and explain each core decision —
  especially the LangGraph graph (nodes/state/edges/why). Understand, not just
  receive.
- **Secrets:** never print or commit keys; `.env` only.
- **Scope:** honor §1 non-goals; flag scope creep.

---

*Planning complete — §0–§5 all cleared. Next action is EXECUTION, starting with
§2: physically pull + freeze the eval set (needs the §4 GitHub token), because
nothing downstream is trustworthy until that data is real. Then build in order:
retrieval core → agent layer → eval → deploy.*
