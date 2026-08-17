"""
Agent state (§3).

The state is deliberately small. Every field exists because a node writes it
and a later node or edge reads it — nothing is carried "just in case", because
unused state is what makes a graph impossible to reason about.

Design note for the accountability condition in CLAUDE.md: the shape of this
state is derived from the four measured experiments, not from a tutorial.
`sufficient` and `escalated` exist specifically because the eval showed the
hard failure is *deciding whether an answer is possible at all* (D19), not
ranking documents better (D14/D17/D18).
"""

from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """Provenance-carrying context, so citations survive into the answer."""

    source: Literal["docs", "discussion"]
    ref: str = Field(..., description="doc path, or discussion URL")
    label: str = Field(..., description="section path, or discussion title")
    text: str
    score: float


class AgentState(TypedDict, total=False):
    # --- input ---
    question: str

    # --- route node ---
    route: Literal["docs", "discussion", "both"]
    route_reason: str

    # --- retrieve node ---
    docs: list[RetrievedChunk]
    discussions: list[RetrievedChunk]

    # --- assess node (the decision the eval says actually matters) ---
    sufficient: bool
    assess_reason: str

    # --- answer / escalate nodes ---
    answer: str
    citations: list[str]
    escalated: bool

    # --- observability ---
    trace: Annotated[list[str], lambda a, b: (a or []) + (b or [])]
