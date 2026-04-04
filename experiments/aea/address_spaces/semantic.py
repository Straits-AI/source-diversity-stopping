"""
Semantic (dense vector) address space.

Uses ``sentence-transformers`` to embed documents and queries, then ranks
by cosine similarity with a numpy dot-product over L2-normalised vectors.
FAISS is used as an optional accelerator if installed; the code falls back
to a pure-numpy scan when FAISS is unavailable.

Supported operations: SEARCH, PREVIEW
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from ..types import AgentState, ActionResult, Operation
from .base import AddressSpace

# Lazy imports — kept out of module top-level so the rest of the framework
# can be imported even if sentence-transformers is not installed.
_SentenceTransformer = None

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5
PREVIEW_MAX_CHARS = 200


def _get_encoder(model_name: str):
    """Load and cache the SentenceTransformer model on first use."""
    global _SentenceTransformer
    if _SentenceTransformer is None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            _SentenceTransformer = SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SemanticAddressSpace. "
                "Install it with: pip install sentence-transformers"
            ) from exc
    return _SentenceTransformer(model_name)


class SemanticAddressSpace(AddressSpace):
    """
    Dense-retrieval address space backed by sentence-transformers + numpy.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.  Defaults to ``"all-MiniLM-L6-v2"``.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._encoder = None                    # loaded lazily on first use
        self._documents: list[dict] = []        # original document dicts
        self._embeddings: Optional[np.ndarray] = None  # shape (N, D), L2-normed
        self._is_built = False

    # ─────────────────────────────────────────────────────────
    # AddressSpace interface
    # ─────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "semantic"

    @property
    def supported_operations(self) -> list[Operation]:
        return [Operation.SEARCH, Operation.PREVIEW]

    def build_index(self, documents: list[dict]) -> None:
        """
        Embed all documents and store as an L2-normalised numpy matrix.

        Parameters
        ----------
        documents : list[dict]
            Must contain ``"id"`` and ``"content"`` keys.
        """
        if not documents:
            self._documents = []
            self._embeddings = np.empty((0,), dtype=np.float32)
            self._is_built = True
            return

        self._documents = documents
        if self._encoder is None:
            self._encoder = _get_encoder(self._model_name)

        texts = [doc["content"] for doc in documents]
        raw = self._encoder.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)

        # L2-normalise so dot product == cosine similarity
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._embeddings = raw / norms
        self._is_built = True

    def query(
        self,
        state: AgentState,
        operation: Operation,
        params: dict,
    ) -> ActionResult:
        """
        Execute SEARCH or PREVIEW against the semantic index.

        Params (SEARCH / PREVIEW)
        -------------------------
        query : str, optional
            Query text.  Defaults to ``state.query``.
        top_k : int, optional
            Number of results.  Defaults to 5.

        Returns
        -------
        ActionResult
            Items sorted by descending cosine similarity.  Each item dict
            contains ``id``, ``content`` (or preview), ``score``, and any
            extra fields from the original document.
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

        if not self._documents:
            return ActionResult(
                items=[], cost_tokens=self._estimate_tokens(query_text),
                cost_latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # Encode query
        if self._encoder is None:
            self._encoder = _get_encoder(self._model_name)

        q_vec = self._encoder.encode(
            [query_text],
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)[0]
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        scores: np.ndarray = self._embeddings @ q_vec   # cosine similarity
        top_indices = np.argsort(scores)[::-1][:top_k]

        items = []
        total_content_chars = len(query_text)

        for idx in top_indices:
            doc = self._documents[int(idx)]
            score = float(scores[int(idx)])
            content = doc["content"]

            if operation == Operation.PREVIEW:
                content = content[:PREVIEW_MAX_CHARS]
                if len(doc["content"]) > PREVIEW_MAX_CHARS:
                    content += "…"

            total_content_chars += len(content)
            item = {k: v for k, v in doc.items() if k != "content"}
            item["content"] = content
            item["score"] = score
            items.append(item)

        cost_tokens = self._estimate_tokens(query_text) + self._estimate_tokens(
            " ".join(it["content"] for it in items)
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        return ActionResult(
            items=items,
            cost_tokens=cost_tokens,
            cost_latency_ms=latency_ms,
            cost_operations=1,
        )
