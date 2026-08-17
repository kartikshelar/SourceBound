# Docs snapshot pin record

- **Source repo:** https://github.com/fastapi/fastapi
- **Tag:** `0.119.1`
- **Commit SHA:** `864b569cf8453654fc3bc2c64108c0f644e2918c`
- **Commit date:** 2025-10-20T13:28:38+02:00
- **Pulled:** 2026-08-06
- **Paths kept:**
  - `docs/en/docs/` (English docs only) → `data/docs_snapshot/en/`
  - `docs_src/` (code examples referenced from docs via `{* file hl[..] *}`
    tags — 397 references across 85 of 143 doc files; without this the
    chunker would have prose with no actual code) → `data/docs_snapshot/docs_src/`
- **Reproduce with:** `uv run scripts/fetch_docs_snapshot.py` (sparse-checks
  out both paths at the pinned tag). Both dirs are gitignored — regenerate
  from this script rather than committing them.

This is the retrieval corpus. Eval Discussions (data/eval/) are pulled and
curated separately, filtered to roughly the last 18 months for version
relevance, per pre-build-checklist.md §2. Any Discussion selected for the
frozen eval set must be excluded from the retrievable index by ID — see the
leakage rule in pre-build-checklist.md §2c.
