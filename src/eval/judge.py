"""
LLM-as-judge: scores a generated answer against the accepted Discussion
answer (pre-build-checklist.md §2d.1).

RUBRIC v2 (claim-extraction). v1 used a single fuzzy "does it contain the key
correct info" pass and FAILED hand-validation (decisions.md D10): kappa 0.30,
6 too-lenient vs 1 too-harsh, reporting 50% where human truth was 25%. Its
failure mode was scoring *topical overlap* instead of correctness — most
starkly on discussion #13490, where the candidate asserted the exact opposite
of the reference and still scored 1.

v2 forced a three-step comparison the model cannot shortcut: extract the
reference's single operative claim, classify the candidate against it, and
derive the score from that verdict in code (not from the model's own score
field — v1 sometimes reasoned "misses the key point" and emitted score=1).

v2 ALSO failed, and the way it failed is what produced v3. Measured on the
same 20 labels: v1 fuzzy/8b kappa 0.30, v2 claim-extraction/8b kappa 0.20,
v2 claim-extraction/70b kappa 0.29. An 8x bigger model moved nothing, which
ruled out model capability as the bottleneck. The three items BOTH models got
wrong were all ones the human labeller had passed with partial credit
("marginal pass", "passes despite noise", "passes on cause") — v2's binary
affirms/omits split had nowhere to put "lands the operative conclusion but
messily", so both models dumped them in 'omits' (70b chose 'omits' 16/20).

v3 therefore adds 'partial' as a first-class verdict, scored as a pass. The
category is written from those three hand-labelled examples: reaches the
operative conclusion but with noise, missing secondary detail, or partly
wrong attribution. The rubric explicitly names partial-vs-omits as the
decision that matters ("does it COMMIT to the conclusion anywhere, even
messily?"). v3/70b measured kappa 0.43 — best so far, and all three target
cases fixed.

v4 (current) splits judging into TWO isolated calls. v3's residual errors
were 6 lenient / 0 harsh, and every one had the same cause: with reference
and candidate in a single prompt, the model extracted the *candidate's*
proposal as the "key claim" and then graded the candidate against itself
(e.g. #15936, where the reference's point was "HTMLResponse lives in
Starlette, change it there" but the extracted claim became the candidate's
"subclass HTMLResponse" idea). Stage 1 now sees only question + reference and
emits a key_claim; stage 2 sees only question + that claim + candidate. The
candidate is physically absent when the claim is formed, which an instruction
alone could not guarantee. Costs 2 judge calls per item instead of 1.

Re-validated against the SAME 20 hand labels each time
(scripts/revalidate_judge.py, --judge-model to swap models), so every
comparison is like-for-like and isolates one variable.

Uses ModelTier.JUDGE (a stronger model than synthesis), which is on a much
tighter free-tier quota than Flash (see src/llm/router.py) — call this
sparingly, never in unbounded parallel.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm.router import LLMRouter, ModelTier

CLAIM_SYSTEM_PROMPT = (
    "You are an expert FastAPI reviewer. You will be shown a user's question and the "
    "ACCEPTED answer that resolved it (written by a FastAPI maintainer or experienced "
    "community member).\n\n"
    "State, in one sentence, the single operative claim in that accepted answer — the "
    "thing that actually resolves the user's problem. Prefer the concrete verdict "
    "(e.g. 'X is not supported', 'this is an upstream Starlette bug, report it there', "
    "'use Y instead of Z', 'fixed in version N') over a general description of the topic.\n\n"
    "If the answer is long or discursive, pick the claim the user most needed to hear, "
    "not the first thing mentioned.\n\n"
    "Respond with ONLY a JSON object, no other text:\n"
    '{"key_claim": "one sentence"}'
)

GRADE_SYSTEM_PROMPT = (
    "You are an expert FastAPI reviewer. You will be given a KEY CLAIM (the operative "
    "point of a known-correct answer) and a CANDIDATE answer written by a support bot.\n\n"
    "Judge the CANDIDATE against the KEY CLAIM. The key claim is fixed and authoritative — "
    "do NOT revise it, and do NOT judge the candidate against its own internal logic. "
    "Your only question is whether the candidate delivers that specific claim.\n\n"
    "Choose exactly one verdict:\n"
    "  'affirms'      - the candidate states the same operative claim (wording may differ)\n"
    "  'partial'      - the candidate REACHES the operative conclusion, but surrounded by\n"
    "                   noise, missing secondary detail, or a wrong/incomplete attribution\n"
    "                   of the cause. The user reading it would still end up doing the\n"
    "                   right thing. Examples: lands the right verdict but blames the wrong\n"
    "                   component; gives the correct fix plus some irrelevant suggestions;\n"
    "                   identifies the real mechanism but misses a secondary caveat.\n"
    "  'contradicts'  - the candidate asserts the opposite of the key claim, or gives a\n"
    "                   different causal explanation/mechanism that cannot both be true\n"
    "  'omits'        - the candidate never commits to the key claim at all: it is vague,\n"
    "                   only discusses the general topic, restates the question, guesses\n"
    "                   without landing a conclusion, or says it cannot find the information\n\n"
    "Choosing between 'partial' and 'omits' is the most important distinction in this task. "
    "Ask: does the candidate actually COMMIT to the operative conclusion anywhere, even "
    "messily? If yes -> 'partial'. If it only circles the topic without landing it -> "
    "'omits'. Do not punish an answer for being noisy, verbose, or partly misattributed if "
    "it still delivers the operative conclusion.\n\n"
    "Critical rules:\n"
    "- Being about the right TOPIC is NOT enough. Discussing the same feature, class, or "
    "error while missing or reversing the key claim is 'omits' or 'contradicts', not "
    "'affirms'.\n"
    "- Watch for negation. If the key claim says a thing IS supported and the candidate says "
    "it is NOT (or vice versa), that is 'contradicts' even though both sentences look similar.\n"
    "- If the candidate proposes its own plausible-sounding fix or mechanism that is NOT the "
    "key claim, that is 'contradicts' or 'omits' — never 'affirms' or 'partial'. A confident "
    "wrong answer is not partial credit.\n"
    "- Correct but incomplete: if the candidate delivers the key claim, extra or missing "
    "secondary detail does not matter.\n\n"
    "Respond with ONLY a JSON object, no other text:\n"
    '{"verdict": "affirms|partial|contradicts|omits", "reasoning": "one sentence"}'
)

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def build_claim_prompt(question: str, reference_answer: str) -> str:
    return (
        f"Question:\n{question}\n\n"
        f"Accepted answer:\n{reference_answer}"
    )


def build_grade_prompt(question: str, key_claim: str, candidate_answer: str) -> str:
    return (
        f"Question:\n{question}\n\n"
        f"KEY CLAIM (authoritative — the candidate must deliver this):\n{key_claim}\n\n"
        f"CANDIDATE answer to grade:\n{candidate_answer}"
    )


def _parse_json(text: str) -> dict | None:
    match = JSON_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


class LLMJudge:
    def __init__(self, router: LLMRouter | None = None):
        self._router = router or LLMRouter()

    def score(self, question: str, reference_answer: str, candidate_answer: str) -> dict:
        # STAGE 1 — extract the key claim from the reference WITHOUT the candidate in
        # context. This isolation is the whole point of v4: in v3 both were in one
        # prompt and the model latched onto whatever the candidate proposed, then
        # graded the candidate against its own idea (all 6 residual errors were this).
        # Instructing it not to do that is insufficient — the candidate has to be
        # physically absent from the context.
        claim_resp = self._router.call(
            ModelTier.JUDGE,
            build_claim_prompt(question, reference_answer),
            system=CLAIM_SYSTEM_PROMPT,
        )
        claim_parsed = _parse_json(claim_resp.text)
        if not claim_parsed or not str(claim_parsed.get("key_claim", "")).strip():
            return {
                "score": None,
                "verdict": "CLAIM_EXTRACTION_FAILED",
                "key_claim": "",
                "reasoning": f"could not extract key claim: {claim_resp.text[:200]}",
            }
        key_claim = str(claim_parsed["key_claim"]).strip()

        # STAGE 2 — grade the candidate against that fixed claim. The reference answer
        # is deliberately NOT passed here; only the extracted claim, so the model
        # cannot re-derive a different (candidate-flavoured) notion of what matters.
        grade_resp = self._router.call(
            ModelTier.JUDGE,
            build_grade_prompt(question, key_claim, candidate_answer),
            system=GRADE_SYSTEM_PROMPT,
        )
        grade_parsed = _parse_json(grade_resp.text)
        if not grade_parsed:
            return {
                "score": None,
                "verdict": "UNPARSEABLE",
                "key_claim": key_claim,
                "reasoning": f"unparseable grade output: {grade_resp.text[:200]}",
            }

        verdict = str(grade_parsed.get("verdict", "")).strip().lower()
        # Score is derived from the verdict in code, never read from a model-emitted
        # score field — the v1 judge would reason "misses the key point" and still
        # emit score=1.
        if verdict in ("affirms", "partial", "contradicts", "omits"):
            score = 1 if verdict in ("affirms", "partial") else 0
        else:
            return {
                "score": None,
                "verdict": "MALFORMED",
                "key_claim": key_claim,
                "reasoning": f"unrecognised verdict {verdict!r}: {grade_resp.text[:150]}",
            }

        return {
            "score": score,
            "verdict": verdict,
            "key_claim": key_claim,
            "reasoning": grade_parsed.get("reasoning", ""),
        }
