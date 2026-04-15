"""Dense embedding retrieval substrate."""

from __future__ import annotations

import numpy as np

from .base import Substrate, RetrievalResult


class DenseSubstrate(Substrate):
    """
    Dense retrieval via sentence-transformer embeddings + cosine similarity.

    Best for: semantic/paraphrase matching, concept-level queries.
    Failure mode: rare identifiers, exact strings, code symbols.
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model
        self._encoder = None
        self._embeddings: np.ndarray | None = None
        self._docs: list[dict] = []

    @property
    def name(self) -> str:
        return "dense"

    def _load_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self._model_name)

    def index(self, documents: list[dict]) -> None:
        self._load_encoder()
        self._docs = documents
        texts = [doc["content"] for doc in documents]
        self._embeddings = self._encoder.encode(texts, normalize_embeddings=True)

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if self._embeddings is None:
            return []

        self._load_encoder()
        q_emb = self._encoder.encode([query], normalize_embeddings=True)[0]
        scores = self._embeddings @ q_emb

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
        ]
