"""Structural (title/path) retrieval substrate."""

from __future__ import annotations

import re

from .base import Substrate, RetrievalResult


class StructuralSubstrate(Substrate):
    """
    Structural retrieval via title/path matching.

    Best for: finding documents by name, navigating file trees,
    locating sections in structured corpora.
    Failure mode: when the relevant document has an uninformative title.
    """

    def __init__(self) -> None:
        self._docs: list[dict] = []

    @property
    def name(self) -> str:
        return "structural"

    def index(self, documents: list[dict]) -> None:
        self._docs = documents

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if not self._docs:
            return []

        query_tokens = set(re.findall(r"\w+", query.lower()))
        scored = []

        for doc in self._docs:
            title = doc.get("title", doc.get("id", ""))
            title_tokens = set(re.findall(r"\w+", title.lower()))

            if not title_tokens:
                continue

            # Jaccard similarity + substring bonus
            intersection = query_tokens & title_tokens
            union = query_tokens | title_tokens
            jaccard = len(intersection) / len(union) if union else 0.0

            # Bonus if title appears as substring in query or vice versa
            substring_bonus = 0.0
            if title.lower() in query.lower() or query.lower() in title.lower():
                substring_bonus = 0.3

            score = jaccard + substring_bonus
            if score > 0:
                scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            RetrievalResult(
                doc_id=doc["id"],
                content=doc["content"],
                score=score,
                metadata={"substrate": self.name, "title": doc.get("title", "")},
            )
            for doc, score in scored[:top_k]
        ]
