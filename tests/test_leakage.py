"""
Leakage-guard tests — the highest-stakes logic in the project.

If this filter silently stops working, the agent retrieves the answer key and
the eval reports an excellent score that means nothing. That is the same class
of failure as the unvalidated judge (decisions.md D10): a number that looks
good and measures nothing. It is worth testing precisely because a broken
version raises no error.

Uses a stub embedder rather than loading bge (~400MB): the logic under test is
the ID exclusion, the threshold comparison, and the audit trail — not the
embedding model, which has its own guarantees.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import ingest.leakage as leakage  # noqa: E402


class StubEmbedder:
    """Maps each distinct text to a fixed unit vector, so similarity is exact."""

    def __init__(self, mapping: dict[str, list[float]]):
        self.mapping = mapping

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = np.asarray(self.mapping[t], dtype=float)
            out.append((v / np.linalg.norm(v)).tolist())
        return out


@pytest.fixture
def eval_items(tmp_path, monkeypatch):
    items = [
        {"discussion_id": "EVAL1", "number": 100, "title": "alpha question"},
        {"discussion_id": "EVAL2", "number": 200, "title": "beta question"},
    ]
    path = tmp_path / "frozen.jsonl"
    path.write_text(
        "\n".join(json.dumps(i) for i in items) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(leakage, "FROZEN_EVAL_PATH", path)
    monkeypatch.setattr(leakage, "AUDIT_PATH", tmp_path / "audit.jsonl")
    return items


def _embedder(extra: dict[str, list[float]]) -> StubEmbedder:
    base = {"alpha question": [1.0, 0.0], "beta question": [0.0, 1.0]}
    return StubEmbedder({**base, **extra})


def test_eval_item_excluded_by_id(eval_items):
    """§2c: an eval discussion must never be indexable, whatever its title."""
    cands = [{"discussion_id": "EVAL1", "number": 100, "title": "totally different"}]
    kept, report = leakage.filter_leakage(
        cands, _embedder({"totally different": [0.0, 1.0]}), write_audit=False
    )
    assert kept == []
    assert report.n_excluded_by_id == 1


def test_exact_duplicate_title_excluded(eval_items):
    """
    The real case this exists for: discussion #7707 shares a title with eval
    item #13972 (cosine 1.000) and survives ID exclusion.
    """
    cands = [{"discussion_id": "OTHER", "number": 7707, "title": "alpha question"}]
    kept, report = leakage.filter_leakage(cands, _embedder({}), write_audit=False)
    assert kept == []
    assert report.n_excluded_by_similarity == 1


def test_unrelated_candidate_is_kept(eval_items):
    """The guard must not be so blunt that it empties the corpus."""
    cands = [{"discussion_id": "OK", "number": 1, "title": "unrelated topic"}]
    kept, report = leakage.filter_leakage(
        cands, _embedder({"unrelated topic": [1.0, 1.0]}), write_audit=False
    )
    assert len(kept) == 1
    assert report.n_kept == 1


def test_threshold_boundary_is_inclusive(eval_items):
    """>= threshold excludes; just under it keeps. Guards silent drift."""
    cands = [{"discussion_id": "X", "number": 2, "title": "near"}]
    # cos = 0.95 exactly -> excluded at threshold 0.95
    vec = [0.95, (1 - 0.95**2) ** 0.5]
    kept, _ = leakage.filter_leakage(
        cands, _embedder({"near": vec}), threshold=0.95, write_audit=False
    )
    assert kept == []
    kept2, _ = leakage.filter_leakage(
        cands, _embedder({"near": vec}), threshold=0.96, write_audit=False
    )
    assert len(kept2) == 1


def test_audit_trail_records_why(eval_items, tmp_path):
    """
    An unauditable leakage filter is barely better than none — you cannot tell
    a working one from a broken one without the record.
    """
    cands = [{"discussion_id": "DUP", "number": 7707, "title": "alpha question"}]
    leakage.filter_leakage(cands, _embedder({}), write_audit=True)
    rows = [
        json.loads(l)
        for l in (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["excluded_discussion_id"] == "DUP"
    assert rows[0]["matched_eval_number"] == 100
    assert rows[0]["title_similarity"] == pytest.approx(1.0, abs=1e-3)
