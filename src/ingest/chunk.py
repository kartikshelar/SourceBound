"""
Structure-aware chunker for the FastAPI docs snapshot.

Splits each markdown file on headers (H1-H3) and carries the section path
as metadata, per pre-build-checklist.md §3 ("split docs on markdown headers,
carry the section path as metadata"). Two FastAPI/MkDocs-Material-specific
transforms happen before splitting, because skipping them would leave chunks
with prose but no actual code, or with raw templating syntax:

  1. `{* ../../docs_src/foo/bar.py hl[1,4] *}` code-include tags are resolved
     by inlining the referenced file from docs_src/ as a fenced code block.
     An optional `ln[a:b]` shows only that 1-indexed line slice — used by
     docs like sql-databases.md that build up one file progressively across
     several chunks, so the full file must NOT be inlined every time. The
     `hl[...]` highlight hint is dropped (a rendering detail, irrelevant to
     retrieval). Tags pointing outside docs_src/ (e.g. into fastapi/ internals)
     are left unresolved rather than reading arbitrary repo paths.
  2. `/// info` ... `///` admonition blocks are unwrapped to plain text (the
     `///` fences are MkDocs-Material syntax, not content).
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

CODE_INCLUDE_RE = re.compile(
    r"\{\*\s*(?P<path>\S+?\.py)"
    r"(?:\s+ln\[(?P<ln>[\d:,]+)\])?"
    r"(?:\s+hl\[[^\]]*\])?"
    r"\s*\*\}"
)
ADMONITION_OPEN_RE = re.compile(r"^///\s*(\S+)\s*$")
ADMONITION_CLOSE_RE = re.compile(r"^///\s*$")
HEADER_RE = re.compile(r"^(#{1,3})\s+(.+?)(?:\s*\{\s*#[\w-]+\s*\})?\s*$")


@dataclass
class Chunk:
    text: str
    section_path: list[str]
    source_file: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.source_file}::{self.chunk_index}"


def resolve_code_includes(text: str, docs_src_root: Path) -> str:
    """
    Resolve `{* ../../docs_src/foo/bar.py *}` tags.

    In the original FastAPI repo these paths are relative to the doc file
    (docs/en/docs/**) and always climb back up to a repo-root-level
    docs_src/. Our snapshot flattens that layout (en/ and docs_src/ are
    siblings under data/docs_snapshot/), so instead of replaying the '../'
    math we just anchor on the literal 'docs_src/...' suffix of the path and
    resolve it against our docs_src_root directly.
    """

    def replace(match: re.Match) -> str:
        rel_path = match.group("path")
        marker = "docs_src/"
        idx = rel_path.find(marker)
        if idx == -1:
            return match.group(0)  # not a docs_src reference (e.g. fastapi/ internals), leave tag as-is
        suffix = rel_path[idx + len(marker):]
        code_path = docs_src_root / suffix
        if not code_path.exists():
            return f"[code example unavailable: {suffix}]"
        code = code_path.read_text(encoding="utf-8")

        ln = match.group("ln")
        if ln and ":" in ln:
            start, end = (int(x) for x in ln.split(":", 1))  # 1-indexed, inclusive
            lines = code.splitlines()
            code = "\n".join(lines[start - 1:end])

        return f"```python\n{code}\n```"

    return CODE_INCLUDE_RE.sub(replace, text)


def unwrap_admonitions(text: str) -> str:
    out_lines = []
    in_admonition = False
    for line in text.splitlines():
        if not in_admonition and ADMONITION_OPEN_RE.match(line):
            in_admonition = True
            continue
        if in_admonition and ADMONITION_CLOSE_RE.match(line):
            in_admonition = False
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


def split_on_headers(text: str) -> list[tuple[list[str], str]]:
    """Returns [(section_path, body_text), ...] split at H1/H2/H3 boundaries."""
    lines = text.splitlines()
    sections: list[tuple[list[str], str]] = []
    stack: list[str] = []
    current_lines: list[str] = []

    def flush():
        body = "\n".join(current_lines).strip()
        if body:
            sections.append((list(stack), body))

    for line in lines:
        m = HEADER_RE.match(line)
        if m:
            flush()
            current_lines = []
            level = len(m.group(1))
            title = m.group(2).strip()
            stack = stack[: level - 1]
            stack.append(title)
        else:
            current_lines.append(line)
    flush()
    return sections


def chunk_markdown_file(md_path: Path, docs_root: Path, docs_src_root: Path) -> list[Chunk]:
    raw = md_path.read_text(encoding="utf-8")
    raw = resolve_code_includes(raw, docs_src_root)
    raw = unwrap_admonitions(raw)

    rel_source = str(md_path.relative_to(docs_root)).replace("\\", "/")
    sections = split_on_headers(raw)

    chunks = []
    for i, (section_path, body) in enumerate(sections):
        chunks.append(
            Chunk(
                text=body,
                section_path=section_path,
                source_file=rel_source,
                chunk_index=i,
                metadata={
                    "section_path_str": " > ".join(section_path) if section_path else rel_source,
                },
            )
        )
    return chunks


def chunk_docs_tree(docs_root: Path, docs_src_root: Path) -> list[Chunk]:
    all_chunks = []
    for md_path in sorted(docs_root.rglob("*.md")):
        all_chunks.extend(chunk_markdown_file(md_path, docs_root, docs_src_root))
    return all_chunks
