"""
Reproduce the pinned FastAPI docs snapshot (retrieval corpus).

Shallow-clones fastapi/fastapi at the pinned tag and extracts:
  - docs/en/docs/  -> data/docs_snapshot/en/      (markdown docs, English only)
  - docs_src/      -> data/docs_snapshot/docs_src/ (code examples the docs
                                                      reference via {* file *} tags)

Both are gitignored (large, fully reproducible from this script) — PIN.md
records the exact tag/SHA for provenance. Re-run this any time the pin
changes; it always wipes and re-fetches both dirs so there's no stale drift.

Usage:
    uv run scripts/fetch_docs_snapshot.py
"""

import shutil
import subprocess
from pathlib import Path

TAG = "0.119.1"
REPO_URL = "https://github.com/fastapi/fastapi.git"
SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "data" / "docs_snapshot"


def run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    tmp_dir = SNAPSHOT_DIR / "_tmp_clone"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    run(
        ["git", "clone", "--depth", "1", "--branch", TAG, "--filter=blob:none",
         "--sparse", REPO_URL, str(tmp_dir)],
        cwd=SNAPSHOT_DIR,
    )
    run(["git", "sparse-checkout", "set", "docs/en/docs", "docs_src"], cwd=tmp_dir)

    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_dir, check=True, capture_output=True, text=True
    ).stdout.strip()

    en_dst = SNAPSHOT_DIR / "en"
    docs_src_dst = SNAPSHOT_DIR / "docs_src"
    for dst in (en_dst, docs_src_dst):
        if dst.exists():
            shutil.rmtree(dst)

    shutil.copytree(tmp_dir / "docs" / "en" / "docs", en_dst)
    shutil.copytree(tmp_dir / "docs_src", docs_src_dst)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"Fetched FastAPI docs snapshot at tag {TAG} (SHA {sha})")
    print(f"  -> {en_dst}")
    print(f"  -> {docs_src_dst}")
    print("See PIN.md for the provenance record.")


if __name__ == "__main__":
    main()
