"""
Lexical (BM25) address space.

Uses ``rank_bm25`` for TF-IDF–weighted keyword retrieval.  No embeddings
are required — this is a pure bag-of-words sparse retrieval baseline.

Supported operations: SEARCH
"""

from __future__ import annotations

import re
import time
from typing import Optional

from ..types import AgentState, ActionResult, Operation
from .base import AddressSpace

DEFAULT_TOP_K = 5


def _tokenize(text: str) -> list[str]:
    """
    Simple whitespace + punctuation tokeniser.

    Lower-cases the text, strips all non-alphanumeric characters, and
    splits on whitespace.  This mirrors common BM25 preprocessing.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [tok for tok in text.split() if tok]


class LexicalAddressSpace(AddressSpace):
    """
    BM25 keyword retrieval address space backed by ``rank_bm25``.

    Parameters
    ----------
    k1 : float
        BM25 k1 parameter (term-frequency saturation).  Default 1.5.
    b : float
        BM25 b parameter (length normalisation).  Default 0.75.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._documents: list[dict] = []
        self._bm25 = None           # rank_bm25.BM25Okapi instance
        self._is_built = False

    # ─────────────────────────────────────────────────────────
    # AddressSpace interface
    # ─────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "lexical"

    @property
    def supported_operations(self) -> list[Operation]:
        return [Operation.SEARCH]

    def build_index(self, documents: list[dict]) -> None:
        """
        Tokenise all documents and build the BM25 index.

        Parameters
        ----------
        documents : list[dict]
            Must contain ``"id"`` and ``"content"`` keys.
        """
        try:
            from rank_bm25 import BM25Okapi  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "rank_bm25 is required for LexicalAddressSpace. "
                "Install it with: pip install rank-bm25"
            ) from exc

        self._documents = documents
        if not documents:
            self._bm25 = None
            self._is_built = True
            return

        corpus = [_tokenize(doc["content"]) for doc in documents]
        self._bm25 = BM25Okapi(corpus, k1=self._k1, b=self._b)
        self._is_built = True

    def query(
        self,
        state: AgentState,
        operation: Operation,
        params: dict,
    ) -> ActionResult:
        """
        Execute SEARCH using BM25 ranking.

        Params
        ------
        query : str, optional
            Query text.  Defaults to ``state.query``.
        top_k : int, optional
            Number of results to return.  Defaults to 5.

        Returns
        -------
        ActionResult
            Items sorted by descending BM25 score.  Each item dict contains
            ``id``, ``content``, ``score``, and extra fields from the
            original document.  Items with score 0 are excluded unless
            fewer than *top_k* items have positive scores.
        """
        self._assert_operation(operation)
        if not self._is_built:
            return ActionResult(
                items=[], cost_tokens=0, cost_latency_ms=0.0, success=False,
                error="Index has not been built.  Call build_index() first."
            )

        t0 = time.perf_counter()
        query_text: str = params.get("query", state.query)
        top_k: int = int(params.get("top_k", DEFAULT_TOP_K))

        if not self._documents or self._bm25 is None:
            return ActionResult(
                items=[], cost_tokens=self._estimate_tokens(query_text),
                cost_latency_ms=(time.perf_counter() - t0) * 1000,
            )

        query_tokens = _tokenize(query_text)
        scores = self._bm25.get_scores(query_tokens)     # shape (N,)

        # Sort descending; keep top_k with positive score when possible
        sorted_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )
        top_indices = sorted_indices[:top_k]

        items = []
        for idx in top_indices:
            doc = self._documents[idx]
            score = float(scores[idx])
            item = {k: v for k, v in doc.items()}
            item["score"] = score
            items.append(item)

        cost_tokens = (
            self._estimate_tokens(query_text)
            + self._estimate_tokens(" ".join(it["content"] for it in items))
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        return ActionResult(
            items=items,
            cost_tokens=cost_tokens,
            cost_latency_ms=latency_ms,
            cost_operations=1,
        )
