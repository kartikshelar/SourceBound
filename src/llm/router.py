"""
Cost-aware model routing wrapper (pre-build-checklist.md §3): a cheap/fast
model for routing + simple steps, a synthesis model for answer generation, a
stronger model reserved for the eval judge. All swappable behind this one
module — model IDs are the only thing that should ever need to change.
See TIER_MODELS below for why this ended up all-Groq rather than the
checklist's original Gemini-for-synthesis plan.

FREE-TIER-ONLY BY POLICY (explicit user requirement, not just a default):
  - Every model ID below is a free-tier model on its provider as of the pin
    date. Google Cloud billing is confirmed disabled on the project behind
    GEMINI_API_KEY (verified 2026-08-06) — that key is *structurally*
    incapable of producing a bill; Groq's API has no paid tier to fall into.
  - Do not add a paid-tier-only model here without the user explicitly
    re-confirming billing status first.
  - MAX_CALLS_PER_SESSION is a second, code-level guardrail on top of the
    zero-billing account: it exists purely to fail loud and stop, rather than
    let a bug spin in a loop and burn through free-tier rate limits
    unnoticed. Not a cost cap (there is no cost) — a runaway-loop cap.
"""

import os
import re
import time
from dataclasses import dataclass
from enum import Enum

from dotenv import load_dotenv
# Gemini is OPTIONAL. It was evaluated and rejected (D20: 20 requests/day on
# the free tier), so the deployment image does not install google-genai — and
# a hard import here crashed the container at startup before anything ran.
# An unused provider must never be able to break the service.
try:
    from google import genai
    from google.genai.errors import ClientError as GeminiClientError

    _GEMINI_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in slim deployments
    genai = None

    class GeminiClientError(Exception):
        """Placeholder so except-clauses stay valid when Gemini is absent."""

    _GEMINI_AVAILABLE = False
from groq import APIStatusError as GroqAPIStatusError
from groq import Groq

from llm.tracing import trace_llm_call

load_dotenv()

MAX_CALLS_PER_SESSION = 500
MAX_RATE_LIMIT_RETRIES = 5
# gpt-oss models emit an internal reasoning trace before any content. Measured:
# max_tokens=10 returns "" while the same call at 200 returns "OK". This must
# cover reasoning + the real answer.
MIN_REASONING_MAX_TOKENS = 2048
RETRY_DELAY_FALLBACK_SECONDS = 15  # used when the error doesn't tell us how long to wait


class ModelTier(Enum):
    ROUTING = "routing"      # cheap/fast: routing + simple steps
    SYNTHESIS = "synthesis"  # answer generation
    JUDGE = "judge"          # eval scoring only — never used at runtime


TIER_MODELS = {
    ModelTier.ROUTING: ("groq", "openai/gpt-oss-20b"),
    ModelTier.SYNTHESIS: ("groq", "openai/gpt-oss-20b"),
    ModelTier.JUDGE: ("groq", "openai/gpt-oss-120b"),
}
# MODEL DECOMMISSIONING, 2026-08-08. Groq removed BOTH models this project was
# built on: llama-3.1-8b-instant (routing + synthesis) and
# llama-3.3-70b-versatile (the judge validated at kappa 0.63 in D11). They now
# 404 with model_not_found; no Llama chat model remains on Groq at all.
# Replacements are the closest available small/large pair. NOTE these are
# reasoning models: they emit a `reasoning` field and only then `content`, so a
# small max_tokens returns an EMPTY string rather than an error — see
# MIN_REASONING_MAX_TOKENS below.
#
# CONSEQUENCE FOR EVERY NUMBER IN THIS PROJECT: the 34.0% baseline, the
# kappa-0.63 judge validation, and all four A/B results were produced by models
# that no longer exist. They are not reproducible and the judge is no longer
# validated. Re-validating the judge against the 20 hand labels
# (scripts/revalidate_judge.py) is now a PREREQUISITE for trusting any new
# eval number, not an optional improvement.
# Settled configuration as of 2026-08-07. Two constraints drove it, both
# measured live rather than assumed:
#
# 1. Gemini is unusable at this project's scale on a free key. Verified:
#    gemini-flash-latest (-> gemini-3.6-flash) allows 20 requests/DAY, hit
#    mid-run after 7 of 50 dev items; gemini-pro-latest reports limit:0, i.e.
#    no free allocation at all. Hence all-Groq, despite the checklist's
#    original "Gemini Flash for synthesis" plan.
#
# 2. JUDGE must be the 70b model; SYNTHESIS must not be. Judge validation
#    (decisions.md D11) measured the v4 two-stage rubric at kappa 0.63 on 70b
#    vs 0.30 for 8b — the 8b model never cleared 0.30 on ANY rubric, so a
#    cheaper judge is not a tradeoff, it is a broken instrument. Synthesis
#    stays on 8b deliberately: the 70b model's 100k tokens/DAY cap is the
#    binding limit on eval throughput (~20-25 items/day, see D12), so every
#    70b token spent generating answers is a token not spent measuring them.
#    Answer quality is what the eval is meant to MEASURE, not maximise — a
#    stronger synthesis model would inflate the baseline the upgrades are
#    supposed to beat.
#
# Net effect: 8b answers, 70b judges, and the strong/weak split between
# answerer and judge is preserved (which the original Gemini-Pro-as-judge
# design also intended).


class SessionBudgetExceeded(RuntimeError):
    pass


class EmptyCompletion(RuntimeError):
    """
    The model returned an empty content string. Its own class because with
    reasoning models this is the normal failure mode of an under-sized
    max_tokens — it is not a rate limit and retrying unchanged will not fix it.
    """


class DailyQuotaExhausted(RuntimeError):
    """
    The provider's per-DAY token/request budget is spent.

    Distinct from a transient per-minute rate limit: retrying cannot help for
    hours, so callers should stop cleanly and resume later rather than burn
    the retry budget. Groq signals this with 'tokens per day (TPD)' in the
    429 body; the accompanying 'try again in Nm' refers to the per-minute
    window and is misleading when the daily cap is what's exhausted.
    """


RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s")


def _is_rate_limit_error(e: Exception) -> bool:
    if isinstance(e, GeminiClientError):
        return getattr(e, "code", None) == 429
    if isinstance(e, GroqAPIStatusError):
        body = str(e)
        return "rate_limit_exceeded" in body or getattr(e, "status_code", None) == 429
    return False


def _is_daily_quota_error(e: Exception) -> bool:
    body = str(e)
    if not _is_rate_limit_error(e):
        return False
    # Groq: "...on tokens per day (TPD): Limit 100000, Used 99225..."
    # Gemini: "...PerDay..." in the quotaId / violations block
    return "per day (TPD)" in body or "PerDay" in body or "requests per day" in body.lower()


def _extract_retry_delay_seconds(e: Exception) -> float | None:
    match = RETRY_DELAY_RE.search(str(e))
    return float(match.group(1)) if match else None


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str


class LLMRouter:
    def __init__(self, tier_overrides: dict[ModelTier, tuple[str, str]] | None = None):
        """
        tier_overrides lets a caller swap the model for a tier without editing
        TIER_MODELS — used to A/B judge models against the same hand labels
        (e.g. 8b vs 70b on an identical rubric) so the comparison isolates one
        variable. Format: {ModelTier.JUDGE: ("groq", "llama-3.3-70b-versatile")}
        """
        self._call_count = 0
        self._gemini_client = None
        self._groq_client = None
        self._tier_models = dict(TIER_MODELS)
        if tier_overrides:
            self._tier_models.update(tier_overrides)

    def _get_gemini(self):
        if not _GEMINI_AVAILABLE:
            raise RuntimeError(
                "A tier is routed to Gemini but google-genai is not installed. "
                "Install it, or point that tier at another provider — the "
                "deployment image ships without it by design."
            )
        if self._gemini_client is None:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY not set in .env")
            self._gemini_client = genai.Client(api_key=api_key)
        return self._gemini_client

    def _get_groq(self) -> Groq:
        if self._groq_client is None:
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise RuntimeError("GROQ_API_KEY not set in .env")
            self._groq_client = Groq(api_key=api_key)
        return self._groq_client

    def _check_budget(self) -> None:
        if self._call_count >= MAX_CALLS_PER_SESSION:
            raise SessionBudgetExceeded(
                f"Hit MAX_CALLS_PER_SESSION={MAX_CALLS_PER_SESSION} for this LLMRouter "
                "instance. This is a runaway-loop guard, not a cost cap (free-tier models "
                "only, no billing account attached). Start a new LLMRouter if this was "
                "intentional, or investigate why so many calls were made."
            )

    def call(self, tier: ModelTier, prompt: str, system: str | None = None) -> LLMResponse:
        self._check_budget()
        provider, model = self._tier_models[tier]

        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            try:
                text = self._dispatch(provider, model, prompt, system)
                self._call_count += 1
                # Best-effort telemetry; no-ops without Langfuse credentials and
                # never raises, so tracing cannot break inference.
                trace_llm_call(model, provider, tier.value, len(prompt))
                return LLMResponse(text=text, model=model, provider=provider)
            except (GeminiClientError, GroqAPIStatusError) as e:
                if _is_daily_quota_error(e):
                    raise DailyQuotaExhausted(
                        f"{provider}/{model}: daily token quota exhausted. Retrying will not "
                        "help until the daily window resets — resume this run later "
                        "(baseline/judge runs are resumable and cache per item)."
                    ) from e
                wait_s = _extract_retry_delay_seconds(e) or RETRY_DELAY_FALLBACK_SECONDS
                if not _is_rate_limit_error(e) or attempt == MAX_RATE_LIMIT_RETRIES:
                    raise
                print(f"  [rate limited on {provider}/{model}, retrying in {wait_s:.0f}s "
                      f"(attempt {attempt + 1}/{MAX_RATE_LIMIT_RETRIES})]")
                time.sleep(wait_s)

    def _dispatch(self, provider: str, model: str, prompt: str, system: str | None) -> str:
        if provider == "gemini":
            client = self._get_gemini()
            config = {"system_instruction": system} if system else None
            resp = client.models.generate_content(model=model, contents=prompt, config=config)
            return resp.text
        elif provider == "groq":
            client = self._get_groq()
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            # Reasoning models (gpt-oss-*) spend the token budget on an internal
            # `reasoning` field first and only then emit `content`. Too small a
            # budget yields an EMPTY content string with no error — which would
            # silently poison every downstream JSON parse — so give them room.
            resp = client.chat.completions.create(
                model=model, messages=messages, max_tokens=MIN_REASONING_MAX_TOKENS
            )
            text = resp.choices[0].message.content
            if not (text or "").strip():
                raise EmptyCompletion(
                    f"{model} returned empty content (finish_reason="
                    f"{resp.choices[0].finish_reason}). Likely the reasoning trace "
                    "consumed the whole token budget."
                )
            return text
        else:
            raise ValueError(f"Unknown provider: {provider}")

    @property
    def call_count(self) -> int:
        return self._call_count
