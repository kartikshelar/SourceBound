"""
Naive single-shot top-k dense RAG baseline (pre-build-checklist.md §2e).

Every added mechanism (hybrid retrieval, rerank, routing, discussions,
citation verification) must beat this on the frozen eval set to earn its
place — this module is the number everything else is measured against, not
a component of the final agent.

Pipeline: doc_search(question, k) -> stuff top-k chunks into one prompt ->
single Gemini Flash call -> answer + citations (source_file list).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm.router import LLMRouter, ModelTier
from retrieval.search import DocSearch, DocSearchInput

BASELINE_K = 5
MAX_CHUNK_CHARS = 2000  # keeps prompts under Groq's 6000 TPM free-tier ceiling for llama-3.1-8b-instant

SYSTEM_PROMPT = (
    "You are a technical support assistant for the FastAPI web framework. "
    "Answer the user's question using ONLY the provided documentation excerpts. "
    "If the excerpts don't contain enough information to answer, say so explicitly. "
    "Be concise and technically precise. Cite which excerpt(s) you used by their "
    "source file path."
)

# Variant under test (D19). Measured motivation: 39% of the baseline's wrong
# answers (13 of 33) are explicit hedges — "the excerpts do not contain enough
# information" — and every regression in the discussions and rerank A/Bs was
# judged `omits`, i.e. the model had material and would not commit to a
# conclusion. SYSTEM_PROMPT actively invites that: it tells the model to say
# so when excerpts are insufficient, and asks for concision without ever
# asking for a committed answer.
#
# The change is deliberately NOT "always sound confident" — that would trade
# `omits` for `contradicts`, and a confidently wrong support answer is worse
# than an honest hedge. Instead it asks for a best-supported conclusion WITH
# its uncertainty attached, which is what the accepted answers in the eval set
# actually look like ("this is a known limitation", "that lives in Starlette").
SYSTEM_PROMPT_COMMITTED = (
    "You are a technical support assistant for the FastAPI web framework. "
    "Answer using the provided excerpts as your primary evidence.\n\n"
    "Always give the user a direct, actionable conclusion — the specific thing "
    "they should do, or the specific reason their code behaves as it does. "
    "State it plainly in your first sentence.\n\n"
    "If the excerpts only partially cover the question, still commit to the "
    "best-supported answer and mark what is uncertain (e.g. 'this is likely X, "
    "though the docs do not state it directly'). Do NOT refuse to answer merely "
    "because the excerpts are incomplete — a hedged but committed answer is far "
    "more useful than 'the documentation does not cover this'.\n\n"
    "Only decline outright if the excerpts are genuinely unrelated to the "
    "question. Never invent APIs, parameters, or version numbers that do not "
    "appear in the excerpts. Cite the source file path of each excerpt you use."
)


def build_prompt(question: str, chunks: list) -> str:
    def clip(text: str) -> str:
        if len(text) <= MAX_CHUNK_CHARS:
            return text
        return text[:MAX_CHUNK_CHARS] + "\n... [excerpt truncated]"

    excerpts = "\n\n".join(
        f"[Excerpt {i+1} | {c.source_file} | {c.section_path}]\n{clip(c.text)}"
        for i, c in enumerate(chunks)
    )
    return f"Documentation excerpts:\n\n{excerpts}\n\nQuestion: {question}\n\nAnswer:"


class NaiveRAGBaseline:
    def __init__(self, k: int = BASELINE_K):
        self.k = k
        self._search = DocSearch()
        self._router = LLMRouter()

    def answer(self, question: str) -> dict:
        search_out = self._search(DocSearchInput(query=question, k=self.k))
        chunks = search_out.results

        prompt = build_prompt(question, chunks)
        response = self._router.call(ModelTier.SYNTHESIS, prompt, system=SYSTEM_PROMPT)

        return {
            "question": question,
            "answer": response.text,
            "citations": [c.source_file for c in chunks],
            "retrieved_sections": [c.section_path for c in chunks],
            "model": response.model,
        }
