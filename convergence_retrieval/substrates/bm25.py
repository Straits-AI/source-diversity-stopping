"""BM25 lexical retrieval substrate."""

from __future__ import annotations

from .base import Substrate, RetrievalResult


class BM25Substrate(Substrate):
    """
    Sparse lexical retrieval via BM25.

    Best for: exact keyword matching, rare identifiers, entity names.
    Failure mode: paraphrased or semantically equivalent queries.
    """

    def __init__(self) -> None:
        self._index = None
        self._docs: list[dict] = []

    @property
    def name(self) -> str:
        return "bm25"

    def index(self, documents: list[dict]) -> None:
        from rank_bm25 import BM25Okapi

        self._docs = documents
        tokenized = [doc["content"].lower().split() for doc in documents]
        self._index = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if self._index is None:
            return []

        tokens = query.lower().split()
        scores = self._index.get_scores(tokens)

        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        return [
            RetrievalResult(
                doc_id=self._docs[idx]["id"],
                content=self._docs[idx]["content"],
                score=float(score),
                metadata={"substrate": self.name},
            )
            for idx, score in ranked
            if score > 0
        ]
