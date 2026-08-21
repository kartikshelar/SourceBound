"""
SourceBound serving layer — a FastAPI support agent, served with FastAPI.

Design note: the response deliberately exposes the agent's DECISIONS
(route, sufficiency, escalation), not just an answer string. That is the whole
point of the system — measured results show it escalates on questions a naive
pipeline answers wrongly, and a caller cannot act on that unless the API says
so. An `answer` field alone would hide the thing this project is about.

Run:
    uv run fastapi dev app/main.py        # dev, auto-reload
    uv run fastapi run app/main.py        # prod
"""

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent.graph import build_graph  # noqa: E402
from llm.router import DailyQuotaExhausted, LLMRouter  # noqa: E402

# Built once at startup: the graph loads a ~400MB embedding model and opens the
# vector store. Doing that per request would add seconds of latency to every
# call for no benefit.
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["router"] = LLMRouter()
    _state["graph"] = build_graph(_state["router"])
    yield
    _state.clear()


app = FastAPI(
    title="SourceBound",
    description=(
        "Agentic FastAPI support assistant. Answers from the pinned FastAPI docs "
        "and answered GitHub Discussions — and says so explicitly when it cannot."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=4000,
                          examples=["How do I add OAuth2 JWT authentication?"])


class Citation(BaseModel):
    ref: str = Field(..., description="Doc path or discussion URL")


class AskResponse(BaseModel):
    answer: str
    escalated: bool = Field(
        ..., description="True when the agent judged the retrieved context "
                         "insufficient and declined rather than guessing."
    )
    route: str | None = Field(None, description="Source chosen: docs | discussion | both")
    route_reason: str | None = None
    assess_reason: str | None = Field(
        None, description="Why the context was judged sufficient or not."
    )
    citations: list[str] = []
    latency_ms: int


@app.get("/health")
def health() -> dict:
    """Liveness + whether the agent graph actually came up."""
    return {"status": "ok", "graph_ready": "graph" in _state}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    graph = _state.get("graph")
    if graph is None:
        raise HTTPException(status_code=503, detail="Agent not initialised")

    t0 = time.perf_counter()
    try:
        state = graph.invoke({"question": req.question})
    except DailyQuotaExhausted as e:
        # 503 + Retry-After is the honest signal: the service is temporarily
        # unavailable through no fault of the caller's request.
        raise HTTPException(
            status_code=503,
            detail=f"Upstream model quota exhausted; try later. ({e})",
            headers={"Retry-After": "3600"},
        ) from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Agent failure: {type(e).__name__}") from e

    return AskResponse(
        answer=state.get("answer", ""),
        escalated=bool(state.get("escalated")),
        route=state.get("route"),
        route_reason=state.get("route_reason"),
        assess_reason=state.get("assess_reason"),
        citations=state.get("citations", []),
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!doctype html>
<html><head><meta charset="utf-8"><title>SourceBound</title>
<style>
 body{font:16px/1.6 system-ui,sans-serif;max-width:52rem;margin:3rem auto;padding:0 1rem;color:#1a1a1a}
 h1{margin-bottom:.2rem} .sub{color:#666;margin-top:0}
 textarea{width:100%;padding:.7rem;font:inherit;border:1px solid #ccc;border-radius:6px}
 button{margin-top:.6rem;padding:.6rem 1.2rem;font:inherit;border:0;border-radius:6px;
        background:#0b5;color:#fff;cursor:pointer} button:disabled{background:#999}
 #out{margin-top:1.5rem;white-space:pre-wrap}
 .badge{display:inline-block;padding:.15rem .5rem;border-radius:4px;font-size:.8rem;margin-right:.4rem}
 .esc{background:#fde68a} .ans{background:#bbf7d0} .meta{color:#666;font-size:.85rem}
 code{background:#f3f4f6;padding:.1rem .3rem;border-radius:3px}
</style></head><body>
<h1>SourceBound</h1>
<p class="sub">FastAPI support agent — grounded in the pinned docs and answered Discussions.
It declines when the retrieved context is insufficient rather than guessing.</p>
<textarea id="q" rows="3" placeholder="How do I add OAuth2 JWT authentication?"></textarea>
<button id="go" onclick="ask()">Ask</button>
<div id="out"></div>
<script>
async function ask(){
  const q=document.getElementById('q').value.trim(); if(!q)return;
  const btn=document.getElementById('go'), out=document.getElementById('out');
  btn.disabled=true; out.textContent='Thinking…';
  try{
    const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},
                               body:JSON.stringify({question:q})});
    if(!r.ok){ out.textContent='Error '+r.status+': '+((await r.json()).detail||''); return; }
    const d=await r.json();
    const badge=d.escalated?'<span class="badge esc">ESCALATED</span>'
                           :'<span class="badge ans">ANSWERED</span>';
    out.innerHTML=badge+'<span class="meta">route: '+(d.route||'?')+' · '+d.latency_ms+'ms</span>'
      +'<div style="margin-top:1rem">'+d.answer.replace(/&/g,'&amp;').replace(/</g,'&lt;')+'</div>'
      +(d.assess_reason?'<p class="meta"><em>Assessment: '+d.assess_reason+'</em></p>':'')
      +(d.citations.length?'<p class="meta">Sources: '+d.citations.slice(0,5).join(', ')+'</p>':'');
  }catch(e){ out.textContent='Request failed: '+e; }
  finally{ btn.disabled=false; }
}
</script></body></html>"""
