"""
Pull answered FastAPI Discussions (Questions category) via GitHub's GraphQL API.

This is the RAW pull only — it writes every answered discussion it finds to a
JSONL pool. Filtering, curation, and the dev/test freeze split happen in
freeze_eval.py (§2b of pre-build-checklist.md). Keeping the steps separate
means re-curating never requires re-hitting the API.

Usage:
    uv run scripts/pull_discussions.py

Reads GITHUB_TOKEN from .env. Requires a fine-grained PAT with
"Public repositories (read-only)" access (see pre-build-checklist.md §4).
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

REPO_OWNER = "fastapi"
REPO_NAME = "fastapi"
QUESTIONS_CATEGORY_ID = "MDE4OkRpc2N1c3Npb25DYXRlZ29yeTMyMDAxNDM0"
GRAPHQL_URL = "https://api.github.com/graphql"
PAGE_SIZE = 50
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "eval" / "discussions_raw.jsonl"

QUERY = """
query($categoryId: ID!, $after: String) {
  repository(owner: "%s", name: "%s") {
    discussions(
      first: %d
      after: $after
      categoryId: $categoryId
      answered: true
      orderBy: {field: CREATED_AT, direction: DESC}
    ) {
      pageInfo {
        hasNextPage
        endCursor
      }
      nodes {
        id
        number
        title
        body
        url
        createdAt
        answerChosenAt
        author {
          login
        }
        answer {
          id
          body
          createdAt
          author {
            login
          }
        }
      }
    }
  }
}
""" % (REPO_OWNER, REPO_NAME, PAGE_SIZE)


def fetch_page(session: requests.Session, token: str, after: str | None) -> dict:
    resp = session.post(
        GRAPHQL_URL,
        headers={
            "Authorization": f"bearer {token}",
            "Content-Type": "application/json",
        },
        json={"query": QUERY, "variables": {"categoryId": QUESTIONS_CATEGORY_ID, "after": after}},
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    if "errors" in payload:
        raise RuntimeError(f"GraphQL errors: {payload['errors']}")
    return payload["data"]["repository"]["discussions"]


def main() -> None:
    load_dotenv()
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN not set in .env", file=sys.stderr)
        sys.exit(1)

    session = requests.Session()
    all_items = []
    after = None
    page_num = 0

    while True:
        page_num += 1
        page = fetch_page(session, token, after)
        nodes = page["nodes"]
        all_items.extend(nodes)
        print(f"page {page_num}: +{len(nodes)} discussions (total {len(all_items)})")

        if not page["pageInfo"]["hasNextPage"]:
            break
        after = page["pageInfo"]["endCursor"]
        time.sleep(0.5)  # be polite; fine-grained PAT gets 5000 req/hr anyway

    pulled_at = datetime.now(timezone.utc).isoformat()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for node in all_items:
            record = {
                "discussion_id": node["id"],
                "number": node["number"],
                "title": node["title"],
                "body": node["body"],
                "url": node["url"],
                "created_at": node["createdAt"],
                "answer_chosen_at": node["answerChosenAt"],
                "author": (node["author"] or {}).get("login"),
                "answer_body": node["answer"]["body"] if node["answer"] else None,
                "answer_created_at": node["answer"]["createdAt"] if node["answer"] else None,
                "answer_author": (node["answer"]["author"] or {}).get("login") if node["answer"] else None,
                "_pulled_at": pulled_at,
                "_source_repo": f"{REPO_OWNER}/{REPO_NAME}",
                "_category": "Questions",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_items)} answered discussions to {OUTPUT_PATH}")
    print(f"Pulled at: {pulled_at}")
    print("This is the RAW pool. Run freeze_eval.py next to filter, curate, and freeze the ~300-item eval set.")


if __name__ == "__main__":
    main()
