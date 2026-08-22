"""
Judge output-parsing tests.

The score MUST be derived from the verdict in code, never read from a
model-emitted `score` field. The v1 judge's actual failure mode (decisions.md
D11) was reasoning its way to "misses the key point" and then emitting
score=1 anyway. These tests lock that invariant in.

Also covers `partial` counting as a pass — that single change moved kappa
from 0.29 to 0.43, so a regression here would silently undo the largest
single improvement the judge ever got.

No API calls: a stub router returns canned model output.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eval.judge import LLMJudge  # noqa: E402


class StubRouter:
    """Returns queued responses in order; records how many calls were made."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls = 0

    def call(self, tier, prompt, system=None):
        self.calls += 1
        text = self._responses.pop(0) if self._responses else "{}"
        return type("R", (), {"text": text, "model": "stub", "provider": "stub"})()


def _judge(*responses: str) -> tuple[LLMJudge, StubRouter]:
    router = StubRouter(list(responses))
    return LLMJudge(router=router), router


CLAIM = '{"key_claim": "X is not supported"}'


@pytest.mark.parametrize(
    "verdict,expected",
    [("affirms", 1), ("partial", 1), ("contradicts", 0), ("omits", 0)],
)
def test_score_derived_from_verdict(verdict, expected):
    judge, _ = _judge(CLAIM, f'{{"verdict": "{verdict}", "reasoning": "r"}}')
    out = judge.score("q", "reference", "candidate")
    assert out["score"] == expected
    assert out["verdict"] == verdict


def test_model_emitted_score_is_ignored():
    """
    The exact v1 bug: verdict says it missed the point, but the model also
    emits score=1. The verdict must win.
    """
    judge, _ = _judge(CLAIM, '{"verdict": "omits", "score": 1, "reasoning": "r"}')
    assert judge.score("q", "ref", "cand")["score"] == 0


def test_two_stage_isolation_is_two_calls():
    """
    v4 splits claim extraction from grading so the candidate is PHYSICALLY
    absent when the key claim is formed — instructing the model not to peek
    was not enough (D11). Two calls is the observable signature of that.
    """
    judge, router = _judge(CLAIM, '{"verdict": "affirms", "reasoning": "r"}')
    judge.score("q", "ref", "cand")
    assert router.calls == 2


def test_candidate_absent_from_claim_extraction_prompt():
    """Directly assert the isolation property, not just the call count."""
    seen: list[str] = []

    class Recorder(StubRouter):
        def call(self, tier, prompt, system=None):
            seen.append(prompt)
            return super().call(tier, prompt, system)

    router = Recorder([CLAIM, '{"verdict": "affirms", "reasoning": "r"}'])
    LLMJudge(router=router).score("the question", "the REFERENCE", "the CANDIDATE")
    assert "the CANDIDATE" not in seen[0]  # stage 1 must not see the candidate
    assert "the REFERENCE" not in seen[1]  # stage 2 grades against the claim only


def test_unparseable_output_scores_none_not_zero():
    """
    An unscored item must never be silently counted as a failure — that would
    bias every reported number downward.
    """
    judge, _ = _judge(CLAIM, "the model rambled without JSON")
    assert judge.score("q", "ref", "cand")["score"] is None


def test_failed_claim_extraction_short_circuits():
    judge, router = _judge("no json here")
    out = judge.score("q", "ref", "cand")
    assert out["score"] is None
    assert router.calls == 1  # must not proceed to grading without a claim


def test_unknown_verdict_scores_none():
    judge, _ = _judge(CLAIM, '{"verdict": "maybe", "reasoning": "r"}')
    assert judge.score("q", "ref", "cand")["score"] is None
