"""
Cross-encoder reranker — the precision half of the retrieval fix (§3).

WHY A CROSS-ENCODER AND NOT A THRESHOLD. D16 measured that bi-encoder cosine
scores carry no signal about whether retrieved context will yield a correct
answer: doc top-1 similarity was 0.792 when the baseline was right and 0.795
when it was wrong. That is not a tuning failure, it is structural. A
bi-encoder embeds the query and the passage *independently*, so the score
measures "these two texts are about similar things", never "this passage
answers this question".

A cross-encoder reads query and passage TOGETHER in one forward pass and
scores the pair directly. It can represent "this thread is about Query models
but a different Query-model bug" — exactly the distinction that broke the
discussions variant in D15, where loosely-related threads scored *higher*
(median 0.882) than the doc chunks they displaced.

Cost: one model forward pass per candidate, so it only runs over a shortlist
the retriever already produced (retrieve k=20, rerank, keep top 5). Local and
CPU-bound — no API quota.
"""

from sentence_transformers import CrossEncoder

MODEL_NAME = "BAAI/bge-reranker-base"


class Reranker:
    """
    Thin wrapper, same swappable-interface convention as EmbeddingModel and
    VectorStore (§3) — the model id is the only thing that should change if we
    A/B a different reranker.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        pairs = [[query, p] for p in passages]
        return [float(s) for s in self._model.predict(pairs, show_progress_bar=False)]

    def rerank(self, query: str, items: list, text_of, top_k: int) -> list:
        """
        Reorder `items` by cross-encoder relevance and return the top_k.

        `text_of` extracts the passage text from an item, so this works for
        both DocSearchResult and DiscussionSearchResult without either module
        needing to know about reranking.
        """
        if not items:
            return []
        scores = self.score(query, [text_of(i) for i in items])
        ranked = sorted(zip(items, scores), key=lambda pair: -pair[1])
        return [item for item, _ in ranked[:top_k]]
