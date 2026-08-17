"""
BM25 lexical retrieval — the sparse half of hybrid search (§3).

Why this exists, from measured failures rather than theory: dense embeddings
blur exact tokens. In the discussion_search smoke test, "FastAPI TensorFlow
issue with Python v3.14.x" retrieved a Poetry/.env thread, because the
operative token `3.14` dissolves into a generic "version problem" direction
in embedding space. BM25 treats rare literals as high-signal, which is
exactly the complementary strength. FastAPI questions are dense with these:
`Annotated`, `HTTPException`, `response_model_exclude_unset`, `@app.get`.

TOKENIZATION IS THE WHOLE GAME HERE. A plain `re.findall(r'[a-z0-9]+')`
shatters `response_model_exclude_unset` into five extremely common words
(response, model, exclude, unset) and turns `3.14` into `3`/`14` — destroying
the rare-token signal BM25 is being added to capture. So the tokenizer:
  - keeps dotted versions intact           3.14.0 -> "3.14.0"
  - keeps snake_case identifiers intact    response_model_exclude_unset
  - keeps dotted attribute paths intact    app.get, fastapi.Path
  - ALSO emits sub-parts, so a query phrased "exclude unset" still matches a
    doc containing only `response_model_exclude_unset`
Emitting both the whole identifier and its parts costs a little index size
and buys recall in both phrasings.
"""

import re

# Ordered: try to match the most specific (symbol-bearing) forms first.
_IDENTIFIER_RE = re.compile(
    r"""
    v?\d+(?:\.(?:\d+|[xX]))+   # versions incl. v-prefix and x wildcard:
                               #   3.14.0, 0.115.2, v3.14.x, 3.14.x
  | [A-Za-z_][\w]*(?:\.[A-Za-z_]\w*)+   # dotted paths: app.get, fastapi.Path
  | [A-Za-z_]\w*_\w+       # snake_case: response_model_exclude_unset
  | [A-Za-z]+\d[\w]*       # alphanumeric: oauth2, py310, utf8
  | [A-Za-z_]\w*           # plain words / CamelCase: HTTPException
  | \d+                    # bare numbers
    """,
    re.VERBOSE,
)

# Version strings need extra care: a question may say "v3.14.x" while the
# matching thread says "3.14.0" or "Python 3.14". Emitting the progressive
# prefixes (3, 3.14) alongside the full string lets those meet in the middle.
# Without this the single most discriminative token in a version-bug report is
# effectively unsearchable — the failure that motivated hybrid search in the
# first place (a "Python v3.14.x" query retrieved OAuth2 "password flow"
# threads because `3.14` never became a token).
_VERSION_RE = re.compile(r"^v?(\d+(?:\.(?:\d+|[xX]))+)$")

_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _IDENTIFIER_RE.finditer(text):
        raw = match.group(0)
        lowered = raw.lower()
        tokens.append(lowered)

        version = _VERSION_RE.match(lowered)
        if version:
            # v3.14.x -> "v3.14.x", "3.14.x", "3.14", "3"  (progressive prefixes).
            # `lowered` is already appended above; only add the v-stripped form
            # when it differs, so a bare "3.14.0" is not double-counted.
            stripped = version.group(1)
            if stripped != lowered:
                tokens.append(stripped)
            segments = stripped.split(".")
            for i in range(1, len(segments)):
                tokens.append(".".join(segments[:i]))
            continue

        # Sub-parts, so "exclude unset" matches response_model_exclude_unset and
        # "http exception" matches HTTPException. Guarded to avoid duplicating a
        # token that is already its own single part.
        parts: list[str] = []
        if "_" in raw or "." in raw:
            parts = [p for p in re.split(r"[_.]", lowered) if p]
        elif _CAMEL_SPLIT_RE.search(raw):
            parts = [p.lower() for p in _CAMEL_SPLIT_RE.split(raw) if p]

        if len(parts) > 1:
            tokens.extend(parts)

    return tokens
