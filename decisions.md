# Decisions Log — FastAPI Support Agent

Running record: every meaningful choice, why, and what was rejected. Doubles as
interview prep. Newest at the bottom.

---

## D1 — Project shape
**Decision:** Agentic FastAPI support agent (docs RAG + agent orchestration +
rigorous evals), deployed with real users.
**Why:** industry-shaped, readily-available clean data, agent-first (not a plain
RAG chatbot); the differentiator is eval rigor and production concerns.
**Rejected:** USC lecture TA (read as a student project); multi-library scope.

## D2 — Library: FastAPI
**Why:** clean, stable MkDocs docs; MIT license; and accepted-answer labels in
GitHub Discussions = a free ground-truth eval set. Recognizable, and adjacent to
my own model-serving stack.
**Rejected:** LangChain — docs churn hard across versions → stale/wrong RAG
corpus, noisy issue tracker, self-referential look. Polars kept as backup.

## D3 — Eval strategy (the differentiator)
**Decision:** Ground truth = answered FastAPI Discussions (question + accepted
answer). ~300 items frozen to a versioned JSONL; ~50 dev / ~250 locked test.
Primary metric = answer correctness (LLM-as-judge vs accepted answer, hand-
validated on ~20), target ≥70%; citation validity ≥90%. Always compared to a
naive top-k RAG baseline.
**Why:** turns "does it work" into a number I can trust.
**Rejected:** using closed issues as labels (mostly bug reports for FastAPI,
noisier).
**Critical rule:** eval discussions are excluded from the retrievable index — no
leakage.

## D4 — Architecture
**Decisions:** structure-aware chunking · retrieval dense (v0) → hybrid + rerank
(v1) · vector store Chroma (v0) → Qdrant Cloud free tier (v1) · orchestration
LangGraph · embeddings local `bge-base-en-v1.5` · tools with Pydantic contracts ·
cost-aware model routing (cheap model for routing/simple, Gemini Flash for
synthesis, stronger model for the eval judge).
**Why:** simplest defensible default per piece; each upgrade justified by a
measured eval delta.
**Rejected:** plain Python agent loop (chose LangGraph for structured state + the
resume keyword — accountability: I must be able to explain the graph, not copy a
tutorial); pgvector (Qdrant's native hybrid preferred for the v1 upgrade); Gemini
embedding API (local model avoids a rate-limited dependency).

## D5 — Docs snapshot pin
**Decision:** Pinned retrieval corpus to FastAPI tag `0.119.1`
(SHA `864b569cf8453654fc3bc2c64108c0f644e2918c`, latest stable as of pull date
2026-08-06). Kept only `docs/en/docs/` → `data/docs_snapshot/en/`. Full pin
record in `data/docs_snapshot/PIN.md`.
**Why:** "latest stable" avoids an artificial dependency loop with eval-set
version matching. Eval Discussions will instead be filtered to a recency
window (~last 18 months) and stale/version-mismatched answers dropped during
curation, so the freshness matching happens on the eval side, not by
backdating the docs pin.
**Rejected:** picking an older tag to match historical Discussion answers —
adds bias/complexity for no clear benefit over recency-filtering the eval
pool instead.

## D6 — Eval set pull + freeze
**Decision:** Pulled all 3,932 answered FastAPI Discussions (Questions
category) via GraphQL (`scripts/pull_discussions.py`) →
`data/eval/discussions_raw.jsonl`. Filtered to 18-month recency window +
substantive-answer length (≥120 chars after stripping code fences/quotes) +
"mentions fastapi" sanity check → 249 passed (`scripts/freeze_eval.py`).
Shuffled (seed 42), split 50 dev / 199 test → `data/eval/eval_frozen_v1.jsonl`,
with full provenance (pull date, filter params, drop reasons, difficulty
counts, and the 249-ID leakage exclusion list) in
`data/eval/eval_frozen_v1.manifest.json`.
**Why:** 18mo window chosen to match the 0.119.1 docs pin (D5) without
backdating the docs; length filter empirically screens "just add X" /
bare-link non-answers (53 dropped) without needing a subjective quality pass.
**Rejected:** widening to 24mo or loosening the length filter to hit exactly
~300 — 249 is a defensible size and padding via a looser filter would
reintroduce junk labels (a known pitfall called out in §2).
**Known limitation:** the `docs-answerable` vs `needs-experience` tag is a
crude proxy (fires only if the accepted answer contains a literal
`fastapi.tiangolo.com` link) — skewed 22/227, not a validated difficulty
measure. Fine for reporting a rough split per §2d, not for anything stronger.
**Still open:** the frozen file itself has not been hand-scored yet (§2d.1
judge validation on ~20 items is a separate, later step once the LLM-judge
exists).

## D7 — Retrieval core (v0 walking skeleton)
**Decision:** Built and indexed the v0 retrieval core:
`src/ingest/chunk.py` (structure-aware header chunker, resolves FastAPI's
`{* file.py ln[a:b] hl[..] *}` code-include tags by inlining real code from
`docs_src/`, unwraps `/// admonition ///` blocks) → `src/retrieval/embeddings.py`
(bge-base-en-v1.5, thin wrapper) → `src/retrieval/store.py` (Chroma, thin
wrapper) → `src/retrieval/search.py` (`DocSearch`, Pydantic in/out contract).
1,645 chunks from 143 doc files indexed. Smoke-tested on 5 hand-picked
questions (OAuth2/JWT, file uploads, background tasks, CORS, query vs path
params) — correct doc section ranked top-1 or top-2 in all 5.
**Why the chunker complexity:** the docs snapshot alone (docs/en/docs/) turned
out to be insufficient — 85 of 143 files reference code examples via
`{* ../../docs_src/... *}` tags pointing at a directory outside docs/en/docs/.
Without resolving these, ~60% of chunks would have prose but no code, which
for a "how do I..." support agent is often the whole answer. Fetched
docs_src/ separately (same pin, `scripts/fetch_docs_snapshot.py`) and inline
it during chunking. Also handled `ln[a:b]` line-slicing because
tutorial/sql-databases.md builds one file progressively across 11 chunks —
naive full-file inlining would have repeated the entire final file 11 times.
**Rejected:** fixed-size chunking (checklist already locked structure-aware);
inlining full files regardless of `ln[]` (would bloat/duplicate the
sql-databases walkthrough chunks and blur retrieval).
**Not yet done:** no eval numbers yet (need the naive-RAG baseline run against
the frozen eval set, §2e) — the 5-question smoke test is a sanity check, not
a measured baseline. Hybrid/rerank/Qdrant swap are v1, deferred until the v0
number exists to beat.

## D8 — LLM routing forced all-Groq; naive-RAG baseline number
**Decision:** `src/llm/router.py` routes ALL tiers (routing, synthesis, judge)
through Groq free-tier models (llama-3.1-8b-instant for all three currently),
not the checklist's original Gemini-Flash-synthesis / stronger-judge plan.
Ran `src/eval/run_baseline.py` (naive single-shot top-k dense RAG, k=5)
against the full 50-item dev split: **answer_correctness = 48.0%**,
citation-presence rate = 100%, 0 unparseable judge outputs, 100 LLM calls
total. Full results in `data/eval/results/baseline_dev_run.jsonl` +
`baseline_dev_summary.json`.
**Why the routing change:** live free-tier limits, not preference — confirmed
2026-08-06/07: gemini-flash-latest capped at 20 req/day; gemini-pro-latest at
0 free quota; llama-3.3-70b-versatile (first judge choice) at 100k tokens/day,
which two separate real runs proved can't finish even one 50-item dev pass.
Groq's 8b-instant model was the only tier with enough daily headroom to
complete a real run. User's explicit requirement (§ chat, not in checklist):
free-tier only, zero risk of a surprise bill — confirmed Google Cloud billing
is disabled on all projects behind GEMINI_API_KEY, so this is belt-and-braces,
not the only safeguard.
**Known limitation carried forward:** judge == synthesis model (same weights,
independent calls) is a real quality compromise vs the checklist's "stronger
model reserved for judge" design. Per §2d.1 this number is NOT yet validated
against hand-scoring ~20 items — do that before treating 48% as trustworthy,
and before it's used as the number later upgrades are compared against.
**Also fixed along the way:** LLMRouter now retries with backoff on
transient 429s (src/llm/router.py); NaiveRAGBaseline caps each retrieved
chunk to 2000 chars in the prompt (some chunks, esp. from the
sql-databases.md progressive walkthrough, were large enough alone to exceed
Groq's 6000 TPM cap); run_baseline.py is resumable (writes incrementally,
skips already-scored discussion_ids) after a first run died mid-way from a
daily quota wall with no partial output saved.
**Rejected:** waiting ~24h for the 70b model's daily quota to reset and
keeping it as judge — the 100k/day ceiling would recur on every future run
(dev iteration, and eventually the 199-item locked test), not just once.

## D9 — Validate the judge before optimizing against 48%
**Decision:** Do NOT tune retrieval against the 48% baseline yet. Built
judge-validation tooling first (§2d.1): `scripts/export_judge_review.py`
(blind, stratified 20-item sheet) + `scripts/score_judge_agreement.py`
(raw agreement, Cohen's kappa, directional bias, docs-answerable ceiling,
full disagreement dump).
**Why:** three independent reasons the 48% is not yet a number worth
optimizing against. (1) The judge is the same 8b model that wrote the
answers, forced by free-tier quota (D8) — unvalidated, and §2d.1 requires
hand-validation before trusting it. (2) We cannot currently separate
"retrieval missed it" from "the answer isn't in the docs at all": reading the
26 dev failures, most accepted answers are maintainer knowledge (known
limitation / lives in Starlette / third-party bug) that no retrieval upgrade
can surface. (3) Chunks are truncated to 2000 chars for Groq's TPM cap, a
plumbing constraint that may itself depress the score.
**Evidence the existing tags can't answer this:** difficulty-tag split was
docs-answerable 25% (n=4) vs needs-experience 50% (n=46) — the *opposite* of
the naive story and far too small to read; and only 4/50 dev items have a
gold doc link, so §2d.3 recall@k is unmeasurable on this split. Hence the
sheet's second column (`docs_answerable`) as a hand-tagged ceiling estimate,
replacing the crude link-based proxy from D6.
**Design notes:** sheet is blind (judge verdicts held in a separate key file)
so hand-scores can't anchor to the judge; sample is stratified 10 pass / 10
fail because an unstratified draw at ~50% accuracy could miss a class;
kappa is reported alongside raw agreement because raw agreement flatters a
balanced-but-random judge.
**Next, gated on the validation result:** if kappa is poor, fix the judge
(better rubric / stronger model) before ANY retrieval work. If kappa is
acceptable, use the docs-answerable ceiling to reset the target, then
evaluate hybrid BM25+dense (strong prior: FastAPI questions carry exact
symbols like `Annotated`/`HTTPException`), rerank, and k. The
maintainer-knowledge fraction is the evidence-backed argument for the v1
`discussion_search` tool — which needs a near-duplicate leakage check beyond
the §2c ID exclusion before it's built.

## D10 — Judge validation FAILED; 48% baseline is invalid; target is unreachable
**Result (n=20, hand-scored by Kartik, blind):** raw agreement 65%,
**Cohen's kappa 0.30 (fair/weak)**, directional bias **6 too-lenient vs 1
too-harsh**. On the same 20 items the judge reports 50% correct; human truth
is **25%**. => the 48% dev baseline (D8) is INFLATED, roughly 2x. Do not cite
it. Artifacts: `data/eval/judge_review_sheet.jsonl` (hand labels + notes),
`data/eval/results/judge_validation.json`.
**Judge failure mode (diagnosed from the disagreement notes):** the 8b judge
scores *topical overlap*, not correctness. It cannot detect a negated or
inverted claim. Clearest case #13490 — candidate asserted Query does NOT work
with Pydantic models; reference says since 0.115.0 it DOES; judge scored 1
"accurately reflects the key point". Also #14889 (invented wrong mechanism,
right topic -> scored 1) and #15914 (restated the reporter's own analysis and
fabricated an "Excerpt" block -> scored 1).
**Ceiling finding (the bigger one):** only **35% (7/20)** of eval questions
are answerable from the FastAPI docs at all. The other 65% are maintainer
knowledge — known limitations, upstream Starlette/uvicorn bugs, "PR #16013
shipped in 0.139.2". **The §1 target of >=70% correctness is therefore
unreachable for a docs-only agent**, not merely difficult. Human-truth
accuracy splits: docs-answerable 2/7 (29%), needs-maintainer 3/13 (23%, and
those are likely partial-credit flukes).
**Consequences:** (1) §1's >=70% success number must be renegotiated against
the measured ceiling — flagged to the user, it is their call, not mine.
(2) The number worth optimizing is accuracy on the docs-answerable subset
(currently 29%), not the blended 48%/25%. (3) The 65% maintainer-knowledge
fraction is now *measured* evidence for the v1 `discussion_search` tool
rather than an assumption. (4) Judge must be fixed before any retrieval
work — otherwise every future eval delta is measured with a broken ruler.
**Rejected:** proceeding to hybrid/rerank now. Tuning against a judge with
kappa 0.30 and a 2x lenience bias would optimize noise.

## D11 — Judge iteration: four rubric/model variants, kappa 0.30 -> 0.83*
**Process:** every variant re-scored the SAME 20 hand-labelled items
(`scripts/revalidate_judge.py`, `--judge-model` to swap models), so each
comparison isolates one variable. Full history:

| variant | model | n | kappa | lenient | harsh |
|---|---|---|---|---|---|
| v1 fuzzy "contains key info" | 8b | 20 | 0.30 | 6 | 1 |
| v2 claim-extraction, binary | 8b | 20 | 0.20 | 3 | 3 |
| v2 claim-extraction, binary | 70b | 20 | 0.29 | 2 | 3 |
| v3 + `partial` verdict | 70b | 20 | 0.43 | 6 | 0 |
| v4 two-stage extraction | 70b | **13*** | **0.83** | 1 | 0 |

*v4's 0.83 was a PARTIAL run (13/20, quota-blocked). Completed on all 20:
**kappa 0.63, 2 lenient / 1 harsh, 3 errors total** — the partial run was
indeed flattering (v1 scored 0.37 on that subset vs 0.30 overall), exactly as
flagged. **0.63 is the number to cite**; it clears the 0.60 "substantial"
bar, so JUDGE is now pinned to 70b + v4 two-stage in TIER_MODELS.

**Two diagnoses, each from reading which items failed — this is the method:**
1. *v2 failed (0.20) despite a "better" rubric, and a 70b model didn't help
   (0.29).* An 8x model jump moving nothing ruled out capability. The three
   items BOTH models missed were all ones the human passed with partial
   credit ("marginal pass", "passes despite noise", "passes on cause") —
   binary affirms/omits had nowhere to put "lands the conclusion but messily",
   so both dumped them in `omits` (70b: 16/20 omits). -> v3 adds `partial`.
2. *v3 (0.43) had 6 lenient / 0 harsh errors.* All shared one cause: with
   reference and candidate in one prompt, the model extracted the CANDIDATE's
   proposal as the key claim, then graded the candidate against itself
   (clearest: #15936, where the reference says "HTMLResponse lives in
   Starlette, fix it there" but the extracted claim became the candidate's
   "subclass it" idea). -> v4 splits into two calls: stage 1 sees only
   question+reference and emits key_claim; stage 2 sees only
   question+key_claim+candidate. The candidate is PHYSICALLY ABSENT during
   extraction — v3 already instructed the model to use the reference and it
   ignored that, so the fix had to be structural, not instructional.

**Also added:** verdict-distribution reporting + a warning when any one
verdict covers >=70% of items, since the v2 `omits` catch-all was only found
by manual digging; and score is derived from the verdict in code, never read
from a model-emitted score field (v1 reasoned "misses the key point" then
emitted score=1).
**Cost note:** v4 doubles judge calls per item. On the 70b free tier (100k
TPD) a full 50-item dev run needs ~100 judge calls and will NOT fit in one
day alongside synthesis. Options when we resume: run judging in daily
batches (run_baseline.py is already resumable), or use 8b for synthesis and
reserve all 70b quota for judging.
**Collapse-warning false positive (worth knowing):** the >=70%-one-verdict
warning fired on the final run (omits=14/20) but was WRONG — 13 of those 14
were genuinely wrong answers, and the human labels are themselves 15/20
negative. The judge was tracking a genuinely bad baseline, not collapsing.
Heuristic since changed to compare the dominant verdict against the human
base rate rather than a bare count.
**v4's 3 residual errors** (all judgment calls at the partial/omits boundary,
none a mechanism failure): #13985 and #15042 lenient — candidate gets part of
a two-part answer and is scored `partial` where the human wanted 0; #13739
harsh — candidate identifies the right mechanism but is scored `omits`.
**Status:** RESOLVED. Judge validated at kappa 0.63 and pinned as default.
`scripts/revalidate_judge.py` now caches per item, so a quota death resumes
instead of re-burning the whole run. Next action: re-run the dev baseline
with this judge to replace the invalid 48% (D8/D10) with a trustworthy
number, then retrieval work.

## D12 — Free-tier daily quota is the binding constraint on eval throughput
**Measured:** the validated judge (70b + v4 two-stage) costs ~2 judge calls +
1 synthesis call per eval item, and Groq's free tier caps
llama-3.3-70b-versatile at **100k tokens/DAY**. Observed throughput today was
only ~21 items before exhaustion (~4,760 tok/item), and a same-session resume
made ZERO further progress.
**But that observed rate is NOT steady state — corrected estimate:** measured
component costs are ~838 tok of judge system prompts (claim 174 + grade 664)
plus ~611 tok of payload (reference + candidate) plus question/output, i.e.
**~1,600 tok/item -> ~60 items/day**. The 3x gap is explained by (a) the
v2/v3/v4 judge-validation runs consuming most of today's budget before the
baseline even started, and (b) retry storms: a rate-limited call re-sent the
full prompt 5 times, so one blocked item could cost ~6x its tokens. The
`DailyQuotaExhausted` fix removes (b) entirely.
**Consequence:** a 50-item dev run should fit in ~1 day once validation runs
aren't competing for the same budget; the 199-item locked test split ~3-4
days. Re-measure on the next clean run before trusting these.
**Note on option-2 ("move synthesis to 8b to save quota"):** investigated and
found to be ALREADY the case — SYNTHESIS was on llama-3.1-8b-instant all
along, so the 2,363-token synthesis prompt never touched the 70b budget.
No config change was needed; the throughput problem was validation-run
contention plus retry waste, not model assignment.
**Fixes applied so far:**
- `DailyQuotaExhausted` is now a distinct exception from transient
  per-minute 429s. Retrying a daily cap cannot succeed for hours, so the
  router raises immediately instead of burning 5 retries x 15s. Groq marks
  these with "tokens per day (TPD)"; the "try again in 4m45s" in that same
  error refers to the per-minute window and is actively misleading.
- `run_baseline.py` catches it, keeps every item already written, prints
  partial results, and exits cleanly rather than raising a traceback.
- Both `run_baseline.py` and `revalidate_judge.py` are resumable/cached per
  item, so a resume only pays for what is missing.
- A naive "is the model available?" probe with max_tokens=5 gives a FALSE
  POSITIVE near the cap (a tiny probe fits in the ~700 remaining tokens
  while a real ~1100-token judge call does not). Don't trust it.
**Current partial result (21/50 dev items, validated judge):** correctness
**38%** — provisional, not the final number, and not comparable to the
invalid 48% (D8/D10) until the run completes.
**Tooling added:** `scripts/check_quota.py`. NOTE — after three wrong
implementations, the settled conclusion is that **remaining daily quota
cannot be read ahead of time**:
  1. probing with `"x" * 6400` reported "5+ items" then a run scored ZERO —
     repeated characters tokenize far cheaper than prose, so the probe was
     measuring a much smaller call than a real one;
  2. reading `x-ratelimit-remaining-tokens` reported "~11,963 tokens / ~8
     items" and the run again scored ZERO — that header is the **per-minute**
     budget (`x-ratelimit-limit-tokens: 12000`, `reset: 185ms`), not the day's;
  3. Groq publishes NO header for the 100k/day cap; it surfaces only in the
     429 body after it is hit.
The script now says exactly that rather than fabricating a headroom estimate.
Correct workflow: just start the run — it caches per item and stops cleanly.
**Current state:** dev baseline at **46/50 items, correctness 34.8%** with the
validated judge. Daily 70b quota exhausted (99,718/100,000). The graceful-stop
path exits in seconds with results intact instead of a retry storm.
**The number has converged** — 38.1% @21 items, 36.4% @22, 33.3% @30,
32.5% @40, 34.8% @46. The remaining 4 items cannot move it materially, so
**~35% is the real naive-RAG baseline**. It replaces the archived 48%, which
was inflated by a judge measured at kappa 0.30.
**Independent corroboration:** ~35% sits just under the measured
docs-answerable ceiling of 35% (D10, hand-tagged n=20) and above Kartik's
hand-scored 25% on a different 20-item sample. Three separate estimates
landing in the 25-38% band is good evidence the measurement is now sound.
**Quota window is ROLLING, not a midnight reset** — but a small amount of
freed headroom is NOT a reset. Observed: `check_quota.py` reported AVAILABLE,
the resumed 29-item run then completed exactly ONE item before hitting the
cap again. A single successful probe proves room for one call, not for a run.
`check_quota.py` now probes up to 5 times and reports estimated *items* of
headroom, warning explicitly when there is too little for a real run.
**Three numbers to keep straight:** 48% (archived, INVALID — 8b judge at
kappa 0.30), 38% (current partial, validated judge, 21 items), 25% (Kartik's
hand-labels on a different 20-item sample). The 38% is expected to drift as
the remaining 29 items land; only the completed 50 is citable.

## D13 — discussion_search built, with a leakage guard that ID-exclusion missed
**Decision:** Built the second retrievable source (§2a): 3,667 answered
Discussions indexed one-Q&A-per-chunk (`src/ingest/index_discussions.py`,
`src/retrieval/discussion_search.py`, Pydantic contract per §3). Motivated by
the measured ceiling, not a hunch — only 35% of eval questions are answerable
from docs alone (D10), so 65% need knowledge that exists only in Discussions.
**Leakage finding (the important part):** §2c's ID exclusion is NECESSARY BUT
NOT SUFFICIENT. Users ask the same question twice, so a near-duplicate thread
carries the same accepted answer under a different ID. Confirmed, not
theorised: discussion **#7707** has a title identical (cosine **1.000**) to
eval item **#13972** and its accepted answer gives the same fix (read the
request body in middleware before it is consumed). It survived ID exclusion.
Had we indexed it, retrieval would have handed the agent the answer key and
the eval would have silently measured nothing — the same shape of failure as
the unvalidated judge.
**Guard:** `src/ingest/leakage.py` — title-embedding cosine vs every eval
item, drop >= 0.90. Threshold measured on the real corpus: 0.95 drops 3
(0.1%, misses obvious dups), 0.90 drops 16 (0.4%, chosen), 0.85 drops 146
(4.0%, starts removing merely-same-topic threads). Hand-inspected the
0.88-0.92 band: mostly related-but-distinct questions ("exception in
middleware" vs "how to throw custom exceptions in middleware") worth keeping.
Every exclusion is logged to `data/eval/leakage_exclusions.jsonl` with the
matched eval item and score — an unauditable leakage filter is worth little.
Titles not bodies: the question is what makes two threads duplicates, and it
stays cheap enough to re-run on every index build.
**Bug caught while wiring this:** `VectorStore.reset()` hardcoded the
module-level `COLLECTION_NAME`, so building the discussions index would have
DELETED the docs index and left the discussions collection untouched. Now
uses the instance's collection name. Verified after the build: docs 1,645
chunks intact, discussions 3,667.
**Smoke test (4 dev items hand-tagged docs_answerable=0):** no leakage (top
scores ~0.77, no result reproduces a reference answer). Retrieval quality is
mixed — 1 of 4 is a real hit (#13373 cites the same upstream PR as the
reference); the other 3 return same-topic-but-wrong threads. Consistent with
the §3 prior that these questions turn on exact tokens (`3.14`, `uv`,
`segfault`) that dense embeddings blur, and is the standing argument for
hybrid BM25+dense.
**Not yet measured:** whether discussion_search moves the eval number. That
needs the dev baseline finished (blocked on quota at 21/50) and then an
A/B against it. No claim of improvement until that delta exists (§2e).

## D14 — Hybrid retrieval built; FIRST MEASUREMENT DOES NOT JUSTIFY IT
**Built:** `src/retrieval/bm25.py` (symbol-aware tokenizer) +
`src/retrieval/hybrid_search.py` (dense + BM25 fused with Reciprocal Rank
Fusion, K=60, candidate pool 30). Same call signature as `DocSearch`, so the
two are swappable for an honest A/B. Added dependency `rank_bm25` (approved).
**Result — recall@5 against gold doc links in accepted answers (n=22):**
  dense-only  10/22 = **45%**
  hybrid       9/22 = **41%**
**Hybrid is NOT better on this measurement.** Per §2e a mechanism earns its
place by moving the number; on this evidence it has not. Do NOT ship hybrid
as the default or claim it as an improvement.
**Caveats that make this provisional, not final:** n=22 is tiny (one item is
4.5 points, so the gap is within noise); recall@5 on gold *doc links* is a
proxy for the docs corpus only and says nothing about discussions; and no
tuning was attempted (RRF K, candidate pool, and dense/sparse weighting are
all at defaults). A weighted fusion favouring dense would likely recover the
gap, but tuning against this 22-item set risks overfitting a proxy metric.
**Tokenizer bug found and fixed along the way (worth keeping):** the first
version's version-regex `\d+(?:\.\d+)+` did not match `v3.14.x` — the `v`
prefix and `x` wildcard — so the single most discriminative token in a
version-bug report was destroyed, and BM25 fell back to common words
(a "Python v3.14.x" query retrieved OAuth2 "password flow" threads). Now
emits progressive prefixes so `v3.14.x` and `3.14.0` meet on `3.14`. After
the fix, hybrid's top hit for that query became a real Python 3.14
incompatibility thread. This is why the tokenizer, not the fusion, is the
substance of a BM25 implementation.
**Open question for the real A/B:** recall@5 is a retrieval proxy. The metric
that actually decides this is end-to-end answer correctness vs the ~35%
baseline, which needs quota. Hybrid may still help there (better context can
matter even when top-5 recall is flat) — or may not. Measure before deciding.

## D15 — Baseline final = 34.0%; discussions variant is WORSE (provisional)
**Baseline complete:** 50/50 dev items, **34.0%** correctness, validated judge
(70b v4, kappa 0.63). This is the §2e reference number. It replaces the
archived 48% (inflated by a kappa-0.30 judge) and sits right at the measured
docs-answerable ceiling of ~35% (D10).
**A/B harness:** `src/eval/run_variant.py` holds everything fixed except
retrieval (same 50 items, same synthesis model/prompt, same judge) and reports
a delta ONLY over items both runs scored — comparing a 50-item variant against
a 46-item baseline would confound retrieval with question mix. Items where the
judge output was unparseable (`score=None`) are excluded, not counted as 0.
**Result (PROVISIONAL, n=19 of 50):**
  baseline    7/19 = 36.8%
  discussions 6/19 = **31.6%**   delta **-5.3%**  (fixed 2, broke 3)
**Discussions does NOT earn its place on this evidence.** Do not ship it as
default. This contradicts the D10 prediction that the 65% maintainer-knowledge
gap would make discussions a clear win — worth stating plainly rather than
explaining away.
**Mechanism (from the 3 broken items):** all three regressed to verdict
`omits`, and all three retrieved loosely-related discussions (#14462 pulled
threads about a different Query-model issue; #13550 pulled unrelated lifespan
threads). Adding 2 discussion blocks DISPLACES doc context and dilutes the
answer — the model spreads across more material and commits to the operative
claim less often. Retrieval precision, not corpus coverage, is the binding
constraint: having the right knowledge in the index does not help if the
retriever surfaces the wrong thread.
**Caveats:** n=19 is small (one item = 5.3 points, so the delta is roughly one
item wide and within noise); quota stopped the run at 20/50. Finish all 50
before treating this as settled.
**Bug fixed en route:** discussion chunks are ~3,058 chars at the median
(whole Q&A threads) vs ~600 for doc sections. Reusing `MAX_CHUNK_CHARS=2000`
for both pushed prompts to 6,134 tokens and tripped the 8b model's 6,000 TPM
per-request cap. Now a separate `MAX_DISCUSSION_CHARS=1200` plus a hard
`PROMPT_TOKEN_CEILING` that drops the lowest-ranked block instead of failing
the item. The 7 items scored before the fix were DISCARDED, not merged —
averaging two prompt configurations would have made the variant meaningless.
**Next test (cheap, targeted):** the failure is dilution, so try
discussions-only-when-docs-are-weak (route on retrieval score) or k=1 instead
of 2, rather than abandoning the corpus.

## D16 — Relevance gate ABANDONED: embedding scores carry no signal
**Plan was:** gate context on retrieval score — only inject docs/discussions
above a similarity threshold — to fix the dilution failure found in D15.
**Measured first (n=30 dev items, doc top-1 score vs baseline correctness):**
  baseline CORRECT: mean top-1 score **0.792** (min 0.736)
  baseline WRONG:   mean top-1 score **0.795** (min 0.733)
Identical. Two derived features were no better:
  doc_margin (top1 - top5)      separation **-0.000**
  disc_top1 - doc_top1          separation **+0.007**
**Conclusion: cosine similarity does not predict whether retrieved context
will produce a correct answer.** A score threshold would be an arbitrary cut,
not a relevance signal. Gate NOT built — building it would have added a tuning
knob that cannot work, and any apparent gain would be threshold overfitting on
50 items.
**Second finding, relevant to D15:** discussion scores are uniformly HIGH
(p10/p50/p90 = 0.848/0.882/0.905) and run *above* doc scores. So the
loosely-related threads that caused the D15 regressions score high too — a
"high similarity" gate would have admitted precisely the harmful ones. This
explains the dilution mechanism: the retriever is confident and wrong, and
nothing in its own output reveals that.
**What this rules out:** any purely score-based filtering (gates, thresholds,
confidence routing off cosine). Discriminating good from bad retrieval needs a
signal the embedder does not provide — a cross-encoder reranker that reads
query and passage together, or an LLM judging relevance, or the agent deciding
per-question. That is now the argued case for the agent layer (§3): the
decision to use a source has to come from something that can actually read the
content, not from vector distance.

## D17 — Cross-encoder reranker: first upgrade to move a number the right way
**Built:** `src/retrieval/rerank.py` — `BAAI/bge-reranker-base` via
sentence-transformers' `CrossEncoder` (no new dependency; already installed).
Thin swappable wrapper per §3. Retrieve k=20, rerank, keep top 5. Local/CPU,
no API quota.
**Why a cross-encoder specifically (from D16's negative result):** a
bi-encoder embeds query and passage independently, so cosine measures "these
texts are about similar things", never "this passage answers this question" —
which is why doc top-1 similarity was 0.792 on correct answers and 0.795 on
wrong ones. A cross-encoder reads the pair together in one forward pass and
can represent "about Query models, but a *different* Query-model bug" — the
exact distinction that broke the discussions variant in D15.
**Measured, same n=30 dev items D16 used:**
  cross-encoder top score, baseline CORRECT: **+0.680**
  cross-encoder top score, baseline WRONG:   **+0.598**
  separation **+0.082**  (bi-encoder separation was -0.000)
So the cross-encoder does carry signal the embedder structurally cannot.
**recall@5 on gold doc links (n=22), all three approaches on identical items:**
  hybrid (BM25+RRF)      9/22 = 41%
  dense only            10/22 = 45%
  **dense k=20 + rerank 11/22 = 50%**
**Strength of evidence — stated plainly:** this is +1 item on n=22. It is the
first upgrade this session to move a metric in the right direction, and the
mechanism is principled rather than a lucky threshold, but a one-item gap on
22 samples is NOT proof. recall@5 is also a proxy; end-to-end correctness vs
the 34.0% baseline is the metric that decides (§2e), and that needs quota.
**Not yet done:** rerank is not wired into the eval variants, and no
end-to-end delta exists. Do not claim improvement until it does.

## D18 — Rerank A/B: no measurable end-to-end gain (provisional, n=25)
**Result (25 of 50 dev items, quota-limited):**
  baseline 8/25 = 32.0%
  rerank   7/25 = **28.0%**   delta **-4.0%**   (fixed 3, broke 4, net -1 item)
**Read this as NOISE, not a regression.** On n=25 a single item is 4.0
percentage points, so the entire delta is one item. The correct statement is
"no measurable difference at this sample size", not "rerank is worse".
**The interesting part — a proxy metric moved opposite to the real one:**
rerank improved recall@5 from 45% to 50% (D17) while end-to-end correctness
did not improve. Getting the gold document into the top-5 is evidently not
sufficient for the 8b synthesis model to produce a correct answer. All four
regressions were verdict `omits` (the model failed to commit to the operative
claim), which is the same failure mode as the discussions variant in D15 —
suggesting the bottleneck is not *which* documents are retrieved but what the
synthesis model does with them.
**Consequence for the project:** three retrieval upgrades have now been
measured against the 34.0% baseline and none has beaten it —
hybrid (recall 41% vs 45%), discussions (-5.3%, n=19), rerank (-4.0%, n=25).
Combined with the baseline sitting at the measured docs-answerable ceiling
(~35%, D10), the evidence increasingly says **retrieval is not the binding
constraint**. Candidate explanations, untested: (a) the 8b synthesis model is
the limit, (b) the answer prompt does not push for a committed conclusion,
(c) the eval's questions genuinely need reasoning the pipeline has no path to.
**Do not ship rerank on this evidence** — but do not discard it either; it is
the only mechanism with a principled reason to help (D17's +0.082 separation)
and it has not been fairly tested at n=50.

## D19 — Synthesis-prompt experiment (hypothesis logged BEFORE the result)
**Why:** three retrieval upgrades have failed to beat the 34.0% baseline
(hybrid, discussions, rerank), and the baseline already sits at the measured
docs-answerable ceiling. So the constraint is plausibly synthesis, not
retrieval. Two measurements support that:
  - **39% of wrong answers (13 of 33) are explicit hedges** — "the excerpts do
    not contain enough information" — the single largest identifiable failure
    category in the baseline.
  - Every regression in the discussions and rerank A/Bs was judged `omits`:
    the model had material and would not commit to a conclusion.
**The current SYSTEM_PROMPT causes this.** It says "If the excerpts don't
contain enough information to answer, say so explicitly" (an explicit
invitation to hedge) and "be concise and technically precise" (no instruction
to reach a conclusion at all).
**Change under test (`--variant committed_prompt`):** retrieval held
IDENTICAL to baseline — dense top-5, same k, same clipping — so only the
system prompt differs. The new prompt asks for a direct actionable conclusion
in the first sentence, and to commit to the best-supported answer with
uncertainty marked rather than refusing on incomplete excerpts.
**Deliberately NOT "answer confidently always":** that would trade `omits`
for `contradicts`, and a confidently wrong support answer is worse than an
honest hedge. The prompt still forbids inventing APIs/parameters/versions and
still allows declining when excerpts are genuinely unrelated. If the judge's
`contradicts` count rises while `omits` falls, the change is bad even if the
headline score moves up — check the verdict distribution, not just the score.
**Prediction (stated in advance, so it can be wrong):** `omits` should fall.
Headline correctness is genuinely uncertain — hedged answers score 0 today,
but a committed wrong answer also scores 0, so the gain only materialises for
items where the model actually had the right material and buried it.

### RESULT: hypothesis FALSIFIED, prompt reverted (n=35)
  baseline  12/35 = 34.3%
  committed  8/35 = **22.9%**   delta **-11.4%**   (fixed 1, broke 5)
  verdicts: omits 23, partial 6, **contradicts 4**, affirms 2
  hedging answers: **7 -> 0**
**The intervention did exactly what it was designed to do, and that is why it
failed.** Hedging was eliminated completely (7 -> 0). But the hedges did not
convert into correct answers — they converted into `contradicts`, i.e.
confidently wrong ones (#13991 nested query models, #13784 `fastapi run`
production-readiness). The D19 falsification test fired as written: contradicts
rose while hedging fell, and the headline score fell too. -11.4% on n=35 is
4 items, beyond the ~1-item noise floor that made the rerank/discussions
deltas unreadable.
**What this actually establishes — the useful part:** hedging was a SYMPTOM,
not the disease. When the baseline hedged, the model genuinely did not have
the answer; the retrieved context did not contain it. Instructing it to commit
did not create knowledge, it created fabrication. For a support agent that is
strictly worse than an honest refusal, so this is a bad trade even at equal
headline score.
**Reverted:** `SYSTEM_PROMPT_COMMITTED` is kept in `baseline.py` and the
`committed_prompt` variant remains runnable, but the default prompt is
unchanged. Results retained at
`data/eval/results/variant_committed_prompt_dev_run.jsonl` as the evidence.
**Converging conclusion across four measured experiments** (hybrid,
discussions, rerank, committed_prompt): none beats 34.0%, the baseline sits at
the measured docs-answerable ceiling (~35%, D10), and forcing commitment
produces fabrication. The constraint is not retrieval ranking and not
synthesis phrasing — it is that **~65% of eval questions have answers that are
not present in the retrievable corpus at all**. The remaining lever with a
principled case is the agent layer (§3): deciding *when* to answer, when to
search a different source, and when to escalate — rather than always emitting
an answer from whatever top-k returned.

## D20 — Groq decommissioned BOTH models; every prior number is now unreproducible
**Event (2026-08-08):** `llama-3.1-8b-instant` (routing + synthesis) and
`llama-3.3-70b-versatile` (the judge validated at kappa 0.63) both now return
404 `model_not_found`. No Llama chat model remains on Groq.
**Replacements:** `openai/gpt-oss-20b` (routing + synthesis),
`openai/gpt-oss-120b` (judge). Verified working on all three tiers.
**Trap found immediately:** these are REASONING models. They emit an internal
`reasoning` field and only then `content`, so an under-sized `max_tokens`
returns an **empty string with no error** — which would silently poison every
downstream JSON parse (route, assess, judge). Measured: `max_tokens=10` gives
`''`, `max_tokens=200` gives `'OK'`. Router now sets
`MIN_REASONING_MAX_TOKENS=2048` and raises `EmptyCompletion` rather than
returning empty text.
**Consequence — state this plainly, do not paper over it:** the 34.0%
baseline, the kappa-0.63 judge validation, and all four A/B results
(hybrid / discussions / rerank / committed_prompt) were produced by models
that no longer exist. Those numbers are **not reproducible** and the judge is
**no longer validated**. The qualitative conclusions still stand (they came
from reading failure modes, not from the exact scores), but any NEW number
must not be compared against the old ones.
**Required before trusting any new eval number:** re-run
`scripts/revalidate_judge.py` against the existing 20 hand labels with
gpt-oss-120b. That ground truth is model-independent, so it survives — this is
exactly the asset that makes recovery cheap rather than catastrophic.
**Vindicates the NIM discussion:** free-tier model availability is not stable,
and single-provider dependence is a real project risk, not a hypothetical one.
The `tier_overrides` seam meant this was a one-line config change rather than
a rewrite.

## D21 — LangGraph agent layer built (§3)
**Graph:** `route -> retrieve -> assess -> (answer | escalate) -> END`
(`src/agent/graph.py`, state in `state.py`, Pydantic-contracted tools in
`tools.py` wrapping the SAME retrieval the eval measured).
**Every node is justified by a measured result, not by tutorial convention:**
- **route** — D15 showed always injecting discussions costs -5.3% via
  dilution; D16 showed embedding scores cannot detect relevance (0.792 correct
  vs 0.795 wrong). So source selection cannot be a similarity threshold; it
  needs something that reads the question. One cheap routing call.
- **assess** — the node that justifies the whole graph. D19 showed forcing an
  answer converts honest hedges into confident fabrications (contradicts
  0 -> 4, -11.4%), and D10 measured only ~35% of questions are answerable from
  the corpus at all. "Can this be answered from what I have?" is therefore a
  real decision a plain RAG pipeline has nowhere to put.
- **escalate** — an explicit refusal plus the closest material found. Per D19
  this is a CORRECT outcome for a support agent, not a failure.
**Deliberately NOT added:** a retry/re-retrieve loop. Four experiments showed
better retrieval does not move the number, so a loop would add latency and
cost for no measured gain. It earns its place only if the eval later shows
escalations that a different query would have answered.
**Failure handling:** unparseable route defaults to `docs` and unparseable
assess defaults to answering — both degrade toward the measured baseline
behaviour rather than toward something unmeasured (e.g. an agent that refuses
everything because a parse failed).
**Smoke test:** "How do I add OAuth2 JWT authentication?" -> route=docs,
assess=sufficient, answered with citations. "Is FastAPI compatible with
Python 3.14?" -> route=discussion, assess=insufficient, escalated with the
closest threads. Both paths exercised, including the escalation branch.
**Not yet measured:** no eval delta vs a baseline. Requires the judge to be
re-validated on the new models first (D20).

## D22 — NVIDIA NIM evaluated and REJECTED as judge: 159x latency, not quota
**Context:** after Groq deleted both project models (D20), single-provider
dependence was clearly a real risk, so NIM was evaluated as a second provider
for the judge tier. Key works; catalog exposes 102 models.
**Controlled measurement — the SAME model on both providers:**
  `openai/gpt-oss-120b` on Groq: 0.9s / 0.5s / 0.5s -> mean **0.6s**
  `openai/gpt-oss-120b` on NIM :               **98.2s**
  => **~159x slower**, with the model held constant. The variable is NIM's
  free-tier shared serverless capacity, not the model.
**Other NIM findings:**
  - `meta/llama-3.3-70b-instruct` — the exact model the kappa-0.63 judge was
    validated on, and unavailable on Groq — **timed out twice at 240s**.
  - `deepseek-ai/deepseek-v4-flash-0731` — timed out twice at 240s.
  - `nvidia/llama-3.1-nemotron-ultra-253b-v1` — HTTP 404 (listed, not deployed).
  - `qwen/qwen3-next-80b-a3b-thinking` — HTTP 410 Gone.
  - `nvidia/llama-3.3-nemotron-super-49b-v1.5` — works, **33s** for a 2-token
    reply. The only candidate that responded reliably.
  - **DeepSeek-R1 is not in the catalog at all** — the specific model that
    motivated the NIM proposal is unavailable.
**Decision: do NOT adopt NIM for the judge.** The original question was "does
NIM have a daily cap?" — it could not be answered, because throughput is too
low to reach one. That is the more damaging failure: a daily token cap refills
while you work on something else, whereas per-call latency taxes every call
forever. At 33-98s/call a 50-item eval (100 judge calls) takes 55-160 minutes
versus ~10 on Groq.
**What was RIGHT in the proposal, and is retained:** free-tier model
availability is unstable (proved by D20 within a day), and single-provider
dependence is a genuine project risk. NIM stays wired as a documented
fallback using `nemotron-super-49b-v1.5`, so a future Groq deletion is a
config change rather than a scramble.
**Corrected expectation:** a stronger judge would make deltas SHARPER, not
different. Four experiments failed to beat baseline and the clearest result
(-11.4%) was well outside noise; better judging measures that more precisely,
it does not overturn it.
**Next:** re-validate `openai/gpt-oss-120b` on Groq as judge against the
existing 20 hand labels (model-independent ground truth), then measure the
agent.

## D23 — Judge re-validated on gpt-oss-120b: kappa 0.67, BETTER than the deleted model
**Result (same 20 hand labels, v4 two-stage rubric unchanged):**
  v1 fuzzy 8b (original)      agreement 65%  kappa 0.30  lenient 6  harsh 1
  llama-3.3-70b (deleted)     agreement 85%  kappa 0.63  lenient 2  harsh 1
  **openai/gpt-oss-120b**     agreement 85%  **kappa 0.67**  lenient 3  **harsh 0**
Clears the >=0.60 "substantial" bar. **Judge is validated; eval work can
resume.**
**Why the recovery from D20 was cheap:** the 20 hand labels are
model-independent ground truth. Losing both models cost a re-validation run
(~20 items), not the eval design. This is the concrete payoff of §2d.1's
"validate the judge" requirement — without those labels, the Groq
decommissioning would have left no way to trust any judge at all.
**Error profile — state this when reporting numbers:** all 3 errors are
LENIENT (judge says correct where the human said wrong); **zero harsh
errors**, i.e. it never marks a genuinely correct answer wrong. So new scores
are, if anything, slight OVER-estimates. That is the safer direction for a
baseline the upgrades must beat.
**Verdict distribution:** omits 10, partial 4, affirms 4, contradicts 2 — all
four categories in use, no collapse (contrast the v2 rubric's 16/20 `omits`
in D11).
**Bug fixed:** `revalidate_judge.py` built its cache filename from the raw
model id, so the `/` in `openai/gpt-oss-120b` became a directory separator and
the run crashed on a missing folder. Now sanitizes `.` and `/`.
**IMPORTANT — what this does NOT do:** it does not restore the old numbers.
The 34.0% baseline was produced by deleted models and remains unreproducible
(D20). New results are internally consistent (same judge, same items) but must
NOT be compared against pre-D20 figures.

## D24 — New baseline on post-decommission stack: 30.6%
**Setup:** synthesis `openai/gpt-oss-20b`, judge `openai/gpt-oss-120b`
(validated kappa 0.67, D23). Same frozen 50-item dev split, same retrieval
(dense top-5), same prompt as the original baseline.
**Result: 15/49 = 30.6%** (quota stopped the run on the final item; the
number is converged — 45.0% @20, 36.7% @30, 30.0% @40, 30.6% @49 — so item 50
cannot move it materially).
**Verdicts:** omits 30, affirms 11, partial 4, contradicts 4. `omits`
dominates at 61%, the same signature as the pre-decommission stack: the model
has retrieved material and will not commit to the operative claim.
**Comparison to the archived 34.0% is NOT valid** — different synthesis model
and different judge. Both figures landing near ~30-34% is consistent with the
D10 finding that only ~35% of these questions are answerable from the docs at
all, but that is corroboration by coincidence of magnitude, not a measured
delta. Archived results live in `data/eval/results/archive/PRE_D20_llama_*`.
**Judge bias to state when reporting:** all 3 judge errors in D23 were
lenient (0 harsh), so 30.6% is if anything a slight OVER-estimate.
**This is now the reference number** every upgrade must beat, replacing 34.0%.

## D25 — Agent v1 measured: -21.4% vs baseline, `assess` is miscalibrated
**Result (n=42 of 50 shared items, quota-limited):**
  baseline  12/42 = 28.6%
  **agent     3/42 =  7.1%   delta -21.4%**
Unlike every earlier A/B (all within ~1 item of noise) this is a large, real
regression. Do not ship agent v1 as default.
**Cause: it escalated on 83% of questions (35/42).** Escalation rate is
uniform across routes (docs 86%, discussion 82%), so this is the `assess`
node, not routing.
**But `assess` is miscalibrated, NOT wrong in principle** — three numbers say
the idea is sound:
  - **When the agent DID answer: 3/7 = 42.9%**, vs baseline 28.6% overall. Its
    positive decisions are better than the baseline's blanket "always answer".
  - **Of the 35 escalations, 26 were SAFE** — the baseline also failed those.
    Only **9 were lost opportunities** (baseline answered correctly).
  - **Fewer confident fabrications: 3 vs baseline 4** (`contradicts`).
So the sufficiency check identifies genuinely-unanswerable questions well; it
just also rejects answerable ones.
**Diagnosed source:** the ASSESS_SYSTEM prompt says "Be strict... Being about
the right topic is NOT sufficient." The model applies that literally and
demands an EXPLICIT answer in context, rejecting material it could reason
from. Evidence from items the baseline answered correctly:
  #13620 "Context does not contain an explicit answer to how to pass a
         callable with its own dependencies"
  #13991 "No specific guidance on nested query-param models is present"
  #15936 "context lacks info about __html__ support"
**Fix to test next (one variable):** loosen `assess` to ask "can a useful
answer be REASONED from this context?" rather than "is the answer explicitly
present?", keeping the ban on inventing APIs/versions. Prediction, logged in
advance: escalation rate should fall well below 83% and the delta should
close; if `contradicts` climbs above the baseline's 4 while it does, the
loosening has gone too far and D19's failure mode has returned.
**Methodological note:** this is the first experiment where the headline score
alone would have been actively misleading. "Agent is 21 points worse" hides
that its answers are MORE accurate than the baseline's and that 74% of its
refusals were correct. The per-decision instrumentation (`escalated`,
`route`, `assess_reason`) is what made the diagnosis possible in one pass.

## D26 — Agent v2 (loosened assess): fix worked, but traded refusals for fabrication
**Result (n=20 of 50, quota-limited, PROVISIONAL):**
  baseline 9/20 = 45.0%   agent 6/20 = 30.0%   delta -15.0%
Note the baseline scores 45% on THIS subset vs 30.6% over all 49 items, so the
-15% is inflated by subset variance. Do not quote it as final.

| | v1 (strict) | v2 (loosened) | baseline |
|---|---|---|---|
| score | 7.1% | 30.0% | 45.0%* |
| escalation rate | 83% | **35%** | — |
| accuracy when it answered | 42.9% | **46.2%** | — |
| `contradicts` | 3 | **3** | **1** |

**The D25 prediction held on both halves.** Escalation fell 83% -> 35% as
intended, and accuracy-when-answering stayed above the baseline (46.2% vs
45.0% on this subset). But the warned-of failure mode appeared: `contradicts`
is 3 against the baseline's 1 on the same items. That is above the baseline
though still under the absolute threshold of 4 stated in D25 — a warning, not
yet a verdict.
**Which fabrications, and why it matters:** #14462 and #14184 were items the
baseline judged `omits` — so the agent converted two honest failures into
confident wrong answers, strictly worse for a user. Only #13550 was a genuine
regression from a correct baseline answer. Meanwhile 3 of the 7 escalations
were on questions the baseline answered correctly (down from 9 of 35 in v1,
so escalation precision did not degrade much while the rate more than halved).
**Reading:** v1 and v2 bracket the right setting. v1 refused too much
(83% escalation, 7.1%); v2 refuses too little on the hard cases and
fabricates instead. The `assess` node is doing real work — accuracy when it
chooses to answer beats the baseline in both versions — but the
sufficient/insufficient boundary is not yet calibrated.
**Do not ship v2 as-is.** Finish all 50 items first: at n=20 one item is 5
points, and the `contradicts` gap (3 vs 1) is two items. Both numbers are
inside the range where subset variance dominates.

<!-- Add new decisions below as the build progresses. -->
