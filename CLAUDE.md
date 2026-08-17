# CLAUDE.md — FastAPI Support Agent

Guardrails for building this project with Claude Code. Read this **and**
`pre-build-checklist.md` before doing anything.

## What this project is
An agentic technical-support assistant that answers FastAPI usage questions from
the official docs, with citations, evaluated against real accepted answers from
GitHub Discussions. Agent-first; RAG is a component, not the centerpiece. All
locked decisions live in `pre-build-checklist.md`; the running rationale log is
`decisions.md`.

## Source of truth
- `pre-build-checklist.md` holds the locked decisions (§0–§5).
- If something you need to decide is **not** in that file, STOP and ask me. Do
  not invent architecture, pick libraries, or change scope on your own.

## Model policy
- **Default to Sonnet.** Use Opus ONLY when I explicitly ask (hard design,
  tricky debugging).
- Never auto-escalate to a more expensive model. Never select the Fable model.
- If a task seems to need Opus, say so and let me decide.

## Usage / cost discipline
- **No unbounded subagents or parallel task fans** — this is the main way usage
  gets burned. One task at a time unless I explicitly approve parallelism.
- Don't re-embed the whole corpus or rebuild the index when a smaller change
  will do. Cache where sensible.

## Build order (enforced)
1. **Retrieval core first** (ingest → chunk → embed → index → search) and prove
   its quality before anything else.
2. **Only then the agent layer** (LangGraph graph, tools, routing).

Garbage retrieval makes agent work worthless — do not build the graph while
retrieval is still bad.

## Conventions
- **Wrappers:** embedding model, LLM, and vector store each sit behind a thin
  interface so they're swappable in one place.
- **Tools:** every agent tool has a Pydantic input/output contract.
- **Chunking:** structure-aware — split docs on markdown headers, carry the
  section path as metadata. Discussions = one Q&A per chunk.
- **Deps:** use `uv`. Ask before adding a new dependency; prefer the agreed
  stack (LangGraph, Qdrant/Chroma, sentence-transformers / bge, Pydantic,
  Langfuse).
- **Secrets:** read from `.env`. Never print, log, or commit keys. Never commit
  `.env`.

## Accountability (important)
- Build incrementally. After each component, give me a short plain-English
  explanation of the core design decisions — **especially the LangGraph graph**:
  what the nodes are, what lives in the state, how edges/branching work, and why
  it's shaped that way.
- I need to defend these choices in an interview. Explaining beats handing me
  working code I don't understand. If I don't get a piece, we slow down.

## Evaluation discipline (§2)
- The frozen eval JSONL is ground truth. Never tune against the locked test set.
- Keep eval sources OUT of the retrievable index (leakage rule). If a change
  could leak eval data into retrieval, flag it before proceeding.
- Justify every retrieval/agent upgrade with an eval delta vs the naive-RAG
  baseline — report the numbers, don't assert improvement.

## Scope
- Honor the non-goals in `pre-build-checklist.md` §1. Flag scope creep instead of
  quietly following it.
