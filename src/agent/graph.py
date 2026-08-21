"""
The LangGraph agent (§3).

WHY THIS SHAPE — every node is here because a measured result put it here.
This is the accountability condition in CLAUDE.md: the graph is designed from
the eval, not lifted from a tutorial.

    route -> retrieve -> assess -> (answer | escalate) -> END

  route     Pick the source. Measured basis: D15 showed that ALWAYS injecting
            discussions costs -5.3% because irrelevant threads displace doc
            context ("dilution"). D16 then showed embedding scores cannot
            detect that (0.792 correct vs 0.795 wrong — no signal). So the
            choice cannot be a similarity threshold; it has to be a decision
            made by something that reads the question. Cheap 8b model, one
            call.

  retrieve  Pure function, no LLM. Calls only the tools `route` selected.

  assess    THE NODE THAT JUSTIFIES THE WHOLE GRAPH. D19 is the evidence:
            forcing the model to always answer converted honest hedges into
            confident fabrications (contradicts 0 -> 4, score -11.4%). And
            D10 measured that only ~35% of eval questions are answerable from
            this corpus at all. So "can this be answered from what I have?"
            is a real, separate decision — not a phrasing problem. A plain RAG
            pipeline has nowhere to put it.

  answer    Grounded synthesis over the retrieved context.

  escalate  Explicit "I cannot answer this from the available sources", with
            what WAS found. For a support agent this is a correct outcome, not
            a failure — D19 established that a confident wrong answer is worse
            than an honest refusal.

Conditional edge: assess.sufficient -> answer | escalate. That is the only
branch. Resisted adding a retry/re-retrieve loop: four experiments showed
better retrieval does not move the number, so a loop would add cost and
latency for no measured gain. If the eval later shows escalations that COULD
have been answered with a different query, that is when a loop earns its place.
"""

import json
import re
import sys
from pathlib import Path

from langgraph.graph import END, StateGraph

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.state import AgentState, RetrievedChunk
from agent.tools import DiscussionSearchTool, DocSearchTool
from llm.router import LLMRouter, ModelTier

MAX_CHUNK_CHARS = 2000
MAX_DISCUSSION_CHARS = 1200
PROMPT_TOKEN_CEILING = 4800  # 8b synthesis model caps a request at 6,000 TPM

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

ROUTE_SYSTEM = (
    "You route FastAPI support questions to the right knowledge source.\n\n"
    "  'docs'       — how-to and API usage that the official documentation covers\n"
    "                 (auth, file uploads, dependencies, background tasks, CORS...)\n"
    "  'discussion' — bug reports, version incompatibilities, regressions, "
    "'is this supported?', or anything needing maintainer knowledge the docs "
    "would not contain\n"
    "  'both'       — genuinely spans usage AND a known issue\n\n"
    "Prefer 'docs' or 'discussion' over 'both'. Choosing 'both' when only one is "
    "relevant adds irrelevant context that measurably degrades answers.\n\n"
    'Respond with ONLY: {"route": "docs|discussion|both", "reason": "one short clause"}'
)

ASSESS_SYSTEM = (
    "You decide whether a support question can be answered from the retrieved "
    "context — BEFORE any answer is written.\n\n"
    "The test is whether a competent FastAPI engineer could REASON a useful "
    "answer from this context — not whether the answer is spelled out "
    "verbatim. Documentation rarely states a user's exact case; applying a "
    "documented mechanism to their situation is a correct answer, not a guess.\n\n"
    "sufficient=true  — the context contains the mechanism, API, limitation, or "
    "cause needed, EVEN IF it must be applied or combined to fit the question. "
    "A partial answer that genuinely helps counts as sufficient.\n"
    "sufficient=false — answering would require facts simply not present: "
    "inventing an API, parameter, version number, or root cause. Also false "
    "when the context is only superficially on-topic (same feature name, "
    "different problem) with nothing to reason from.\n\n"
    "Do NOT require an explicit, verbatim answer — that bar is too high and "
    "rejects context that would have produced a good answer.\n\n"
    'Respond with ONLY: {"sufficient": true|false, "reason": "one short clause"}'
)

ANSWER_SYSTEM = (
    "You are a technical support assistant for the FastAPI web framework. "
    "Answer using ONLY the provided context. Be concise and technically precise. "
    "Cite the source of each fact you use by its path or URL. "
    "Do not invent APIs, parameters, or version numbers that are not in the context."
)


def _parse_json(text: str) -> dict | None:
    match = JSON_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks):
        limit = MAX_DISCUSSION_CHARS if c.source == "discussion" else MAX_CHUNK_CHARS
        text = c.text if len(c.text) <= limit else c.text[:limit] + "\n... [truncated]"
        blocks.append(f"[{i+1} | {c.source} | {c.ref} | {c.label}]\n{text}")
    return "\n\n".join(blocks)


def _fit(prompt_head: str, chunks: list[RetrievedChunk], tail: str) -> str:
    """Drop lowest-ranked chunks until the prompt fits the 8b per-request cap."""
    chunks = list(chunks)
    while True:
        prompt = f"{prompt_head}\n\n{_format_context(chunks)}\n\n{tail}"
        if len(prompt) // 4 <= PROMPT_TOKEN_CEILING or len(chunks) <= 1:
            return prompt
        chunks.pop()


# --------------------------------------------------------------------------
# nodes
# --------------------------------------------------------------------------

def route_node(state: AgentState, router: LLMRouter) -> dict:
    question = state["question"]
    resp = router.call(ModelTier.ROUTING, f"Question:\n{question}", system=ROUTE_SYSTEM)
    parsed = _parse_json(resp.text) or {}
    route = parsed.get("route")
    if route not in ("docs", "discussion", "both"):
        # Fail safe to docs: it is the corpus the 34.0% baseline was measured on,
        # so a routing failure degrades to the known-good behaviour rather than
        # to something unmeasured.
        route, reason = "docs", "route parse failed; defaulted to docs"
    else:
        reason = str(parsed.get("reason", ""))[:120]
    return {"route": route, "route_reason": reason, "trace": [f"route={route} ({reason})"]}


def retrieve_node(state: AgentState) -> dict:
    question = state["question"]
    route = state.get("route", "docs")
    docs, discussions = [], []
    if route in ("docs", "both"):
        docs = DocSearchTool(query=question, k=5).run()
    if route in ("discussion", "both"):
        discussions = DiscussionSearchTool(query=question, k=2).run()
    return {
        "docs": docs,
        "discussions": discussions,
        "trace": [f"retrieved docs={len(docs)} discussions={len(discussions)}"],
    }


def assess_node(state: AgentState, router: LLMRouter) -> dict:
    chunks = (state.get("docs") or []) + (state.get("discussions") or [])
    if not chunks:
        return {
            "sufficient": False,
            "assess_reason": "no context retrieved",
            "trace": ["assess=insufficient (empty retrieval)"],
        }
    prompt = _fit(
        f"Question:\n{state['question']}\n\nRetrieved context:", chunks, "Decision:"
    )
    resp = router.call(ModelTier.ROUTING, prompt, system=ASSESS_SYSTEM)
    parsed = _parse_json(resp.text) or {}
    sufficient = parsed.get("sufficient")
    if not isinstance(sufficient, bool):
        # Fail safe to answering: the 34.0% baseline always answers, so an
        # unparseable assessment must not silently turn the agent into a
        # refuse-everything machine.
        sufficient, reason = True, "assess parse failed; defaulted to answering"
    else:
        reason = str(parsed.get("reason", ""))[:120]
    return {
        "sufficient": sufficient,
        "assess_reason": reason,
        "trace": [f"assess={'sufficient' if sufficient else 'insufficient'} ({reason})"],
    }


def answer_node(state: AgentState, router: LLMRouter) -> dict:
    chunks = (state.get("docs") or []) + (state.get("discussions") or [])
    prompt = _fit(
        f"Retrieved context for: {state['question']}", chunks,
        f"Question: {state['question']}\n\nAnswer:",
    )
    resp = router.call(ModelTier.SYNTHESIS, prompt, system=ANSWER_SYSTEM)
    return {
        "answer": resp.text,
        "citations": [c.ref for c in chunks],
        "escalated": False,
        "trace": ["answered"],
    }


def escalate_node(state: AgentState) -> dict:
    chunks = (state.get("docs") or []) + (state.get("discussions") or [])
    found = "\n".join(f"- {c.label} ({c.ref})" for c in chunks[:3])
    reason = state.get("assess_reason", "")
    answer = (
        "I can't answer this from the FastAPI documentation or answered "
        f"discussions I have access to. ({reason})\n\n"
    )
    if found:
        answer += f"Closest material found, which may still help:\n{found}"
    return {
        "answer": answer,
        "citations": [c.ref for c in chunks],
        "escalated": True,
        "trace": ["escalated"],
    }


def _should_answer(state: AgentState) -> str:
    return "answer" if state.get("sufficient") else "escalate"


def build_graph(router: LLMRouter | None = None):
    router = router or LLMRouter()
    g = StateGraph(AgentState)
    g.add_node("route", lambda s: route_node(s, router))
    g.add_node("retrieve", retrieve_node)
    g.add_node("assess", lambda s: assess_node(s, router))
    g.add_node("answer", lambda s: answer_node(s, router))
    g.add_node("escalate", escalate_node)

    g.set_entry_point("route")
    g.add_edge("route", "retrieve")
    g.add_edge("retrieve", "assess")
    g.add_conditional_edges("assess", _should_answer, {"answer": "answer", "escalate": "escalate"})
    g.add_edge("answer", END)
    g.add_edge("escalate", END)
    return g.compile()


def run(question: str, router: LLMRouter | None = None) -> AgentState:
    return build_graph(router).invoke({"question": question})
