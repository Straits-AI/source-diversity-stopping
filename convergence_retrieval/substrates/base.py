"""Base class for retrieval substrates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class RetrievalResult:
    """A single retrieved item."""
    doc_id: str
    content: str
    score: float
    metadata: dict = field(default_factory=dict)


class Substrate(ABC):
    """
    Base class for all retrieval substrates.

    A substrate is a single retrieval backend (BM25, dense embeddings,
    file-tree navigation, etc.) that exposes two operations:

    1. ``index(documents)`` — build the index from a list of documents.
    2. ``search(query, top_k)`` — return the top-k results for a query.

    Each substrate has an independent failure mode: BM25 fails on
    paraphrases, dense fails on rare identifiers, structural fails on
    unnamed content.  Convergence-based stopping exploits this
    independence: when two substrates with different failure modes agree,
    the evidence is likely relevant.
    """

    @abstractmethod
    def index(self, documents: list[dict]) -> None:
        """
        Build or rebuild the search index.

        Parameters
        ----------
        documents : list[dict]
            Each dict must have ``"id"`` (str) and ``"content"`` (str).
            Optional: ``"title"`` (str), ``"metadata"`` (dict).
        """

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """
        Search the index and return ranked results.

        Parameters
        ----------
        query : str
            The search query.
        top_k : int
            Maximum number of results to return.

        Returns
        -------
        list[RetrievalResult]
            Ranked results, highest score first.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this substrate (e.g., 'bm25', 'dense')."""
