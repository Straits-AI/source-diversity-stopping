"""
Structural (hierarchy-navigation) address space.

Navigates a document hierarchy — Title → Section → Paragraph — mimicking
filesystem browsing:

  1. SEARCH  — match a query against document TITLES (not content).
               Returns title-level metadata (id, title, snippet) sorted by
               title-string similarity to the query.
  2. OPEN    — given one or more document ids (titles), return the full
               paragraph content of those documents.  Simulates "opening a file".

Design rationale
----------------
Flat retrieval (semantic / lexical) ranks by *content* similarity.
Structural retrieval ranks by *title / label* similarity — i.e. it finds
the right FILE first, then lets the agent read its CONTENTS.  Questions
where the answer lives inside a document whose *title* reveals provenance
(e.g. "Which department handles X?") benefit from this two-phase strategy.

Supported operations: SEARCH, OPEN
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from typing import Optional

from ..types import AgentState, ActionResult, Operation
from .base import AddressSpace

DEFAULT_TOP_K = 5
SNIPPET_MAX_CHARS = 200


def _title_tokens(title: str) -> set[str]:
    """Lower-case, split on non-alphanumeric, return token set."""
    return set(re.findall(r"[a-z0-9]+", title.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard overlap between two token sets."""
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union)


def _title_score(query_tokens: set[str], title_tokens: set[str]) -> float:
    """
    Combined title-match score:
      - Jaccard on tokens (bag-of-words overlap)
      - Bonus when query tokens appear as contiguous substring in the title
    """
    base = _jaccard(query_tokens, title_tokens)
    # Small subsequence-overlap bonus for partial substring matches
    bonus = len(query_tokens & title_tokens) / max(1, len(query_tokens)) * 0.3
    return min(1.0, base + bonus)


class StructuralAddressSpace(AddressSpace):
    """
    Structural navigation address space: Title-first, content-second.

    Builds a hierarchy from context documents by grouping paragraphs
    under their ``title`` field.  If no title is present, falls back to
    the document ``id``.

    SEARCH returns title-level entries ranked by title-string similarity
    to the query — not by content similarity.  The agent must then OPEN
    a matched document to retrieve its paragraph text.

    OPEN accepts ``params["ids"]`` — a list of document ids — and returns
    the full content of those documents.  If ``ids`` is omitted, the top
    result from the most recent SEARCH (by score) is opened.

    Parameters
    ----------
    top_k : int
        Default number of results for SEARCH.  Default 5.
    """

    def __init__(self, top_k: int = DEFAULT_TOP_K) -> None:
        self._top_k = top_k
        # Flat list of all ingested documents (paragraphs)
        self._documents: list[dict] = []
        # Map: title_key → list[doc_dict]
        self._title_index: dict[str, list[dict]] = {}
        # Map: doc_id → doc_dict
        self._id_index: dict[str, dict] = {}
        # Ordered list of (title_key, title_tokens, representative_doc)
        self._title_entries: list[tuple[str, set[str], dict]] = []
        self._is_built = False

    # ─────────────────────────────────────────────────────────
    # AddressSpace interface
    # ─────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "structural"

    @property
    def supported_operations(self) -> list[Operation]:
        return [Operation.SEARCH, Operation.OPEN]

    def build_index(self, documents: list[dict]) -> None:
        """
        Build the title → paragraph hierarchy index.

        For each document, the title is taken from the ``"title"`` field
        if present; otherwise from ``"id"``.  Multiple documents with the
        same title are grouped together under that title entry.
        """
        self._documents = documents
        self._title_index = defaultdict(list)
        self._id_index = {}

        for doc in documents:
            doc_id = doc.get("id", "")
            title = doc.get("title", doc_id)
            title_key = title.strip()
            self._title_index[title_key].append(doc)
            self._id_index[doc_id] = doc

        # Build ordered list of title entries (one per unique title)
        self._title_entries = []
        for title_key, docs in self._title_index.items():
            tokens = _title_tokens(title_key)
            # Representative doc: first paragraph under this title
            self._title_entries.append((title_key, tokens, docs[0]))

        self._is_built = True

    def query(
        self,
        state: AgentState,
        operation: Operation,
        params: dict,
    ) -> ActionResult:
        """
        Execute SEARCH or OPEN against the structural index.

        SEARCH params
        -------------
        query : str, optional
            Query text matched against titles.  Defaults to state.query.
        top_k : int, optional
            Number of title-level results.  Defaults to self._top_k.

        OPEN params
        -----------
        ids : list[str], optional
            Document ids to open.  If absent, opens the single highest-
            scoring title from the most recent SEARCH step (via the
            workspace's top scored item).
        query : str, optional
            Query text (only used for cost estimation).

        Returns
        -------
        ActionResult
            SEARCH: items with title metadata and opening snippet only.
            OPEN: items with full paragraph content.
        """
        self._assert_operation(operation)
        if not self._is_built:
            return ActionResult(
                items=[], cost_tokens=0, cost_latency_ms=0.0, success=False,
                error="Index has not been built.  Call build_index() first."
            )

        t0 = time.perf_counter()

        if operation == Operation.SEARCH:
            return self._do_search(state, params, t0)
        else:  # Operation.OPEN
            return self._do_open(state, params, t0)

    # ─────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────

    def _do_search(
        self,
        state: AgentState,
        params: dict,
        t0: float,
    ) -> ActionResult:
        """
        Rank title entries by title-string similarity to the query.

        Returns items with title metadata and a short content snippet;
        the full content is NOT returned (agent must OPEN to read it).
        """
        query_text: str = params.get("query", state.query)
        top_k: int = int(params.get("top_k", self._top_k))

        if not self._title_entries:
            return ActionResult(
                items=[],
                cost_tokens=self._estimate_tokens(query_text),
                cost_latency_ms=(time.perf_counter() - t0) * 1000,
            )

        q_tokens = _title_tokens(query_text)

        scored: list[tuple[float, str, dict]] = []
        for title_key, t_tokens, rep_doc in self._title_entries:
            score = _title_score(q_tokens, t_tokens)
            scored.append((score, title_key, rep_doc))

        scored.sort(key=lambda x: x[0], reverse=True)

        items: list[dict] = []
        total_chars = len(query_text)
        for score, title_key, rep_doc in scored[:top_k]:
            snippet = rep_doc.get("content", "")[:SNIPPET_MAX_CHARS]
            item: dict = {
                "id": rep_doc.get("id", title_key),
                "title": title_key,
                "content": snippet,    # only snippet — agent must OPEN for full text
                "score": score,
                "source_type": "title",
                "n_paragraphs": len(self._title_index[title_key]),
            }
            items.append(item)
            total_chars += len(snippet)

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

    def _do_open(
        self,
        state: AgentState,
        params: dict,
        t0: float,
    ) -> ActionResult:
        """
        Open one or more documents by id and return their full content.

        If no ids are given, the top-scored workspace item whose source
        was a structural SEARCH is opened.  If there is no such item,
        an empty result is returned.
        """
        query_text: str = params.get("query", state.query)
        ids: list[str] = params.get("ids", [])

        # If no ids provided, try to open the best workspace item from this space
        if not ids and state.workspace:
            # Take the highest-relevance item from the workspace
            best = max(state.workspace, key=lambda x: x.relevance_score)
            ids = [best.source_id]

        if not ids:
            return ActionResult(
                items=[],
                cost_tokens=self._estimate_tokens(query_text),
                cost_latency_ms=(time.perf_counter() - t0) * 1000,
            )

        items: list[dict] = []
        for doc_id in ids:
            # Direct id lookup first
            if doc_id in self._id_index:
                doc = self._id_index[doc_id]
                item = {k: v for k, v in doc.items()}
                item["score"] = 1.0
                item["source_type"] = "paragraph"
                items.append(item)
                continue

            # Fallback: treat doc_id as a title key and expand all paragraphs
            if doc_id in self._title_index:
                for doc in self._title_index[doc_id]:
                    item = {k: v for k, v in doc.items()}
                    item["score"] = 1.0
                    item["source_type"] = "paragraph"
                    items.append(item)

        cost_tokens = self._estimate_tokens(query_text) + self._estimate_tokens(
            " ".join(it.get("content", "") for it in items)
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        return ActionResult(
            items=items,
            cost_tokens=cost_tokens,
            cost_latency_ms=latency_ms,
            cost_operations=1,
        )
