"""
Langfuse tracing (§3, "observability").

DESIGN RULE: tracing must never break the thing it observes. If Langfuse
credentials are absent, the SDK is not installed, or the collector is
unreachable, every function here degrades to a silent no-op. An agent that
500s because a telemetry backend is down is worse than an agent with no
telemetry — and in this project the credentials are genuinely optional, so
the disabled path is the common one, not an edge case.

What is worth tracing here is NOT just token counts. The measured value of
this agent is its DECISIONS (decisions.md D25): it escalates on questions a
naive pipeline answers wrongly. So each run records `route`, `sufficient`,
and `escalated` as trace metadata, which makes the production question
"how often is it refusing, and on what?" answerable — that is the single
number most likely to drift once real users arrive.
"""

import os
from contextlib import contextmanager
from typing import Any

_client: Any = None
_enabled: bool | None = None


def _init() -> Any:
    """Lazily construct the client. Returns None when tracing is unavailable."""
    global _client, _enabled
    if _enabled is not None:
        return _client

    public = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
    secret = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
    if not public or not secret:
        _enabled = False
        return None

    try:
        from langfuse import Langfuse

        kwargs = {"public_key": public, "secret_key": secret}
        # Accept either spelling: Langfuse's own docs use LANGFUSE_HOST, but
        # LANGFUSE_BASE_URL is common in their SDK examples. Rejecting a
        # reasonable spelling would fail silently (tracing simply never starts),
        # which is the worst kind of config bug.
        host = (
            os.environ.get("LANGFUSE_HOST", "").strip()
            or os.environ.get("LANGFUSE_BASE_URL", "").strip()
        )
        if host:
            kwargs["host"] = host
        _client = Langfuse(**kwargs)
        _enabled = True
    except Exception:
        # Import error, bad credentials, unreachable host — all the same
        # outcome: run untraced rather than fail.
        _client, _enabled = None, False
    return _client


def is_enabled() -> bool:
    _init()
    return bool(_enabled)


@contextmanager
def trace_run(name: str, question: str):
    """
    Wrap one agent run. Yields a setter that records the agent's decisions
    once the graph has produced them:

        with trace_run("agent", q) as record:
            state = graph.invoke({"question": q})
            record(state)
    """
    client = _init()
    if client is None:
        yield lambda _state: None
        return

    captured: dict = {}

    def record(state: dict) -> None:
        captured.update(state or {})

    span = None
    try:
        span = client.start_span(name=name, input={"question": question})
    except Exception:
        span = None

    try:
        yield record
    finally:
        if span is not None:
            try:
                span.update(
                    output={"answer": captured.get("answer", "")},
                    metadata={
                        # The decisions, not just the text — see module docstring.
                        "route": captured.get("route"),
                        "sufficient": captured.get("sufficient"),
                        "escalated": bool(captured.get("escalated")),
                        "assess_reason": captured.get("assess_reason"),
                        "n_citations": len(captured.get("citations") or []),
                    },
                )
                span.end()
                client.flush()
            except Exception:
                pass


def trace_llm_call(model: str, provider: str, tier: str, prompt_chars: int) -> None:
    """Record a single model call. Best-effort; never raises."""
    client = _init()
    if client is None:
        return
    try:
        span = client.start_span(
            name=f"llm:{tier}",
            input={"prompt_chars": prompt_chars},
            metadata={"model": model, "provider": provider, "tier": tier},
        )
        span.end()
    except Exception:
        pass
