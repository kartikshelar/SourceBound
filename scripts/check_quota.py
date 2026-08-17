"""
Report remaining Groq daily token quota, and how many eval items it buys.

Exists because ad-hoc "is the model up?" probes are actively misleading near
the cap: a small probe fits in the leftover tokens and reports AVAILABLE
while a real ~1,600-token judge call still fails. This sends a payload sized
like an actual judge call, and reads the exact Used/Limit numbers out of the
429 body rather than inferring anything.

Usage:
    uv run scripts/check_quota.py
"""

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402
from groq import Groq  # noqa: E402

from llm.router import TIER_MODELS, ModelTier  # noqa: E402

# Measured on the v4 two-stage judge: ~838 tok of system prompts + ~611 tok of
# payload (reference + candidate) + question/output overhead.
EST_TOKENS_PER_EVAL_ITEM = 1400  # measured: both judge stages, p50 across the dev run



def main() -> None:
    load_dotenv()
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise SystemExit("GROQ_API_KEY not set in .env")

    client = Groq(api_key=key)
    _, judge_model = TIER_MODELS[ModelTier.JUDGE]

    # Minimal call: enough to either surface the 429 (blocked) or return headers
    # carrying the exact remaining budget. Deliberately tiny — an earlier version
    # sent a full-size ~1,400-token probe on every check, spending an eval item's
    # worth of quota just to ask how much quota was left.
    try:
        client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": "ok"}],
            max_tokens=1,
        )
    except Exception as e:
        body = str(e)
        used = re.search(r"Used (\d+)", body)
        limit = re.search(r"Limit (\d+)", body)
        if not used or not limit:
            print(f"{judge_model}: call failed, but not a recognised quota error:\n  {body[:300]}")
            return
        used_n, limit_n = int(used.group(1)), int(limit.group(1))
        remaining = limit_n - used_n
        print(f"{judge_model}: BLOCKED")
        print(f"  used      {used_n:,} / {limit_n:,}  ({used_n / limit_n:.1%})")
        print(f"  remaining ~{remaining:,} tokens  (~{remaining // EST_TOKENS_PER_EVAL_ITEM} eval items)")
        print("  daily cap — resume when the daily window resets (the 'try again in Nm'")
        print("  in Groq's error refers to the per-minute window, not this).")
        return

    # A small call succeeding proves only that the PER-MINUTE budget has room.
    # It says nothing about the daily cap, which is the limit that actually
    # blocks eval runs — and which Groq does not expose in any header.
    print(f"{judge_model}: per-minute budget OK — daily budget UNKNOWN.")
    print("  Groq publishes only per-minute rate-limit headers "
          "(x-ratelimit-limit-tokens=12,000, resets in ~seconds).")
    print("  The 100,000 tokens/DAY cap appears ONLY in the 429 body once it is hit,")
    print("  so remaining daily quota cannot be read ahead of time.")
    print("  -> Just start the run. It scores what it can, caches every item, and")
    print("     stops cleanly on the daily cap; re-run later to resume.")


if __name__ == "__main__":
    main()
