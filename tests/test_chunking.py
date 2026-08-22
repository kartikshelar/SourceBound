"""
Chunker tests.

Prioritised by blast radius: these cover the transforms that fail SILENTLY.
A broken code-include tag does not raise — it produces a chunk with prose and
no code, which then embeds fine, retrieves fine, and quietly degrades every
answer downstream. That class of bug is invisible without a test, and it bit
this project twice during development (see decisions.md D7).
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ingest.chunk import (  # noqa: E402
    resolve_code_includes,
    split_on_headers,
    unwrap_admonitions,
)


@pytest.fixture
def docs_src(tmp_path: Path) -> Path:
    src = tmp_path / "docs_src"
    (src / "tutorial").mkdir(parents=True)
    (src / "tutorial" / "app.py").write_text(
        "\n".join(f"line{i}" for i in range(1, 11)), encoding="utf-8"
    )
    return src


def test_code_include_inlines_real_code(docs_src: Path):
    out = resolve_code_includes("{* ../../docs_src/tutorial/app.py *}", docs_src)
    assert "```python" in out
    assert "line1" in out and "line10" in out


def test_code_include_respects_line_slice(docs_src: Path):
    """
    `ln[a:b]` is 1-indexed and inclusive. sql-databases.md builds ONE file up
    across 11 chunks using different slices; ignoring ln[] would inline the
    whole file 11 times and blur every one of those chunks together.
    """
    out = resolve_code_includes("{* ../../docs_src/tutorial/app.py ln[2:4] *}", docs_src)
    assert "line2" in out and "line3" in out and "line4" in out
    assert "line1" not in out and "line5" not in out


def test_code_include_handles_v_prefixed_and_hl_hints(docs_src: Path):
    """hl[] is a rendering hint and must be ignored, not treated as a slice."""
    out = resolve_code_includes("{* ../../docs_src/tutorial/app.py hl[2,4] *}", docs_src)
    assert "line1" in out and "line10" in out  # full file, hl ignored


def test_missing_code_file_is_flagged_not_silently_dropped(docs_src: Path):
    """
    A missing example must leave a visible marker. Silently emitting nothing
    would produce a plausible-looking chunk that is quietly missing its code.
    """
    out = resolve_code_includes("{* ../../docs_src/tutorial/nope.py *}", docs_src)
    assert "unavailable" in out.lower()


def test_non_docs_src_reference_is_left_alone(docs_src: Path):
    """
    Some tags point into fastapi/ internals, which are not in the snapshot.
    Those must be left as-is rather than resolved from an arbitrary path.
    """
    tag = "{* ../../fastapi/openapi/docs.py *}"
    assert resolve_code_includes(tag, docs_src) == tag


def test_admonitions_unwrapped_to_plain_text():
    md = "before\n\n/// warning\nheed this\n///\n\nafter"
    out = unwrap_admonitions(md)
    assert "heed this" in out
    assert "///" not in out


def test_split_carries_full_section_path():
    md = "# Top\n\nintro\n\n## Mid\n\nbody\n\n### Deep\n\ndetail\n"
    sections = split_on_headers(md)
    paths = [p for p, _ in sections]
    assert ["Top"] in paths
    assert ["Top", "Mid"] in paths
    assert ["Top", "Mid", "Deep"] in paths


def test_split_pops_stack_on_shallower_header():
    """H2 after an H3 must not inherit the H3 as a parent."""
    md = "# A\n\nx\n\n## B\n\ny\n\n### C\n\nz\n\n## D\n\nw\n"
    paths = [p for p, _ in split_on_headers(md)]
    assert ["A", "D"] in paths
    assert ["A", "B", "C", "D"] not in paths


def test_header_anchor_syntax_stripped():
    """FastAPI docs write `## Title { #anchor }`; the anchor is not the title."""
    sections = split_on_headers("## Query Parameters { #query-parameters }\n\nbody\n")
    assert sections[0][0] == ["Query Parameters"]
