"""
Entity-graph address space.

Builds a co-occurrence graph from regex-based named-entity recognition.
Entities are capitalised multi-word phrases that appear in the corpus.
Two entities are linked in the graph when they co-occur in the same
paragraph; edge weight = number of shared paragraphs.

Supported operations:
  SEARCH — find paragraphs that mention any entity from the query
  HOP    — BFS from seed entities to directly linked paragraphs/entities
"""

from __future__ import annotations

import re
import time
from collections import defaultdict, deque
from typing import Optional

from ..types import AgentState, ActionResult, Operation
from .base import AddressSpace

DEFAULT_TOP_K = 5
DEFAULT_HOP_DEPTH = 1

# Regex: capitalised word or multi-word phrase (catches named entities in English)
_ENTITY_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")

# Common words that look capitalised but are not entities (sentence-initial words,
# common titles, etc.).  We apply a small stop-list; this is not exhaustive.
_ENTITY_STOPWORDS = frozenset(
    {
        "The", "A", "An", "This", "That", "These", "Those",
        "He", "She", "It", "They", "We", "You", "I",
        "His", "Her", "Its", "Their", "Our", "Your",
        "Mr", "Mrs", "Ms", "Dr", "Prof",
        "In", "On", "At", "By", "For", "With", "From", "To", "Of",
        "And", "Or", "But", "So", "Yet", "Nor",
    }
)


def _extract_entities(text: str, is_title: bool = False) -> list[str]:
    """
    Return a de-duplicated list of capitalised phrases found in *text*.

    Strategy: extract every capitalised multi-word sequence using a regex,
    filter out stop-words, deduplicate while preserving order.

    For prose (``is_title=False``), sentence-initial capitalisation is
    handled by lower-casing the first character of each sentence, so words
    like "Were" or "The" at the start of a sentence are not mistaken for
    entity beginnings.  For titles (``is_title=True``), the text is used
    as-is because titles are proper noun phrases by definition.

    Parameters
    ----------
    text : str
        Text to extract entities from.
    is_title : bool
        If True, skip sentence-initial normalisation (treat the whole text
        as a proper noun phrase).
    """
    if is_title:
        normalised = text
    else:
        # Normalise sentence-initial capitalisation: lower-case the first char
        # after sentence-boundary punctuation (. ! ?) followed by whitespace.
        normalised = re.sub(
            r"(?<=[.!?])\s+([A-Z])",
            lambda m: " " + m.group(1).lower(),
            text,
        )
        # Also lower-case the very first character of the text so that
        # question-initial words (e.g. "Were", "What", "Who") are not mistaken
        # for entity starts.
        if normalised and normalised[0].isupper():
            normalised = normalised[0].lower() + normalised[1:]

    found = _ENTITY_RE.findall(normalised)
    seen: set[str] = set()
    entities: list[str] = []
    for ent in found:
        # Each token in the phrase must not be a stop word
        tokens = ent.split()
        if any(t in _ENTITY_STOPWORDS for t in tokens):
            continue
        if ent not in seen:
            seen.add(ent)
            entities.append(ent)
    return entities


def _entity_key(entity: str) -> str:
    """Canonical lower-cased key for an entity string."""
    return entity.lower().strip()


class EntityGraphAddressSpace(AddressSpace):
    """
    Entity co-occurrence graph address space.

    Internal data structures
    ------------------------
    _documents : list[dict]
        Original document list.
    _doc_entities : list[list[str]]
        Per-document entity lists (canonical keys).
    _entity_to_docs : dict[str, list[int]]
        Maps canonical entity key → list of document indices.
    _adjacency : dict[str, set[str]]
        Entity co-occurrence graph (canonical keys as nodes).
    """

    def __init__(self) -> None:
        self._documents: list[dict] = []
        self._doc_entities: list[list[str]] = []          # canonical keys per doc
        self._entity_to_docs: dict[str, list[int]] = defaultdict(list)
        self._adjacency: dict[str, set[str]] = defaultdict(set)
        self._is_built = False

    # ─────────────────────────────────────────────────────────
    # AddressSpace interface
    # ─────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "entity"

    @property
    def supported_operations(self) -> list[Operation]:
        return [Operation.SEARCH, Operation.HOP]

    def build_index(self, documents: list[dict]) -> None:
        """
        Extract entities, build inverted index, and build co-occurrence graph.

        Parameters
        ----------
        documents : list[dict]
            Must contain ``"id"`` and ``"content"`` keys.
        """
        self._documents = documents
        self._doc_entities = []
        self._entity_to_docs = defaultdict(list)
        self._adjacency = defaultdict(set)

        for doc_idx, doc in enumerate(documents):
            title = doc.get("title", "")
            content = doc["content"]

            # Extract entities from the title (treated as a proper noun phrase)
            # and from the prose body (with sentence-initial normalisation)
            # separately, then merge.
            title_entities = _extract_entities(title, is_title=True) if title else []
            content_entities = _extract_entities(content, is_title=False)

            # Merge, preserving order and deduplicating by canonical key
            seen_keys: set[str] = set()
            raw_entities: list[str] = []
            for ent in title_entities + content_entities:
                k = _entity_key(ent)
                if k not in seen_keys:
                    seen_keys.add(k)
                    raw_entities.append(ent)
            keys = [_entity_key(e) for e in raw_entities]
            # De-duplicate while preserving order
            seen: set[str] = set()
            unique_keys: list[str] = []
            for k in keys:
                if k not in seen:
                    seen.add(k)
                    unique_keys.append(k)

            self._doc_entities.append(unique_keys)

            for key in unique_keys:
                self._entity_to_docs[key].append(doc_idx)

            # Co-occurrence edges: all pairs of entities in this document
            for i, ki in enumerate(unique_keys):
                for kj in unique_keys[i + 1:]:
                    self._adjacency[ki].add(kj)
                    self._adjacency[kj].add(ki)

        self._is_built = True

    def query(
        self,
        state: AgentState,
        operation: Operation,
        params: dict,
    ) -> ActionResult:
        """
        Execute SEARCH or HOP in the entity graph.

        SEARCH params
        -------------
        query : str, optional
            Text to extract query entities from.  Defaults to ``state.query``.
        top_k : int, optional
            Maximum documents to return.  Default 5.

        HOP params
        ----------
        entities : list[str], optional
            Seed entity strings.  If omitted, entities are extracted from
            ``params["query"]`` or ``state.query``.
        depth : int, optional
            BFS depth from seed entities.  Default 1.
        top_k : int, optional
            Maximum documents to return.  Default 5.

        Returns
        -------
        ActionResult
            Items are document dicts augmented with ``score`` (number of
            matching entities) and ``matched_entities`` (list of entity keys
            that caused inclusion).
        """
        self._assert_operation(operation)
        if not self._is_built:
            return ActionResult(
                items=[], cost_tokens=0, cost_latency_ms=0.0, success=False,
                error="Index has not been built.  Call build_index() first."
            )

        t0 = time.perf_counter()

        if operation == Operation.SEARCH:
            result_items = self._search(state, params)
        else:  # HOP
            result_items = self._hop(state, params)

        cost_tokens = (
            self._estimate_tokens(params.get("query", state.query))
            + self._estimate_tokens(" ".join(it["content"] for it in result_items))
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        return ActionResult(
            items=result_items,
            cost_tokens=cost_tokens,
            cost_latency_ms=latency_ms,
            cost_operations=1,
        )

    # ─────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────

    def _search(self, state: AgentState, params: dict) -> list[dict]:
        """
        Find documents that contain any entity from the query text.

        Scoring: number of matched entities (higher = better).
        """
        top_k: int = int(params.get("top_k", DEFAULT_TOP_K))
        query_text: str = params.get("query", state.query)

        query_entities = [_entity_key(e) for e in _extract_entities(query_text)]
        if not query_entities:
            return []

        # doc_idx → set of matched entity keys
        doc_matches: dict[int, set[str]] = defaultdict(set)
        for ent_key in query_entities:
            for doc_idx in self._entity_to_docs.get(ent_key, []):
                doc_matches[doc_idx].add(ent_key)

        # Sort by number of matched entities (desc), then doc index (asc) for stability
        ranked = sorted(doc_matches.items(), key=lambda x: (-len(x[1]), x[0]))

        items = []
        for doc_idx, matched in ranked[:top_k]:
            doc = self._documents[doc_idx]
            item = {k: v for k, v in doc.items()}
            item["score"] = float(len(matched))
            item["matched_entities"] = sorted(matched)
            items.append(item)

        return items

    def _hop(self, state: AgentState, params: dict) -> list[dict]:
        """
        BFS from seed entities; collect documents reachable within *depth* hops.

        Each hop traverses one edge in the entity co-occurrence graph and
        collects all documents associated with the newly discovered entities.
        """
        top_k: int = int(params.get("top_k", DEFAULT_TOP_K))
        depth: int = int(params.get("depth", DEFAULT_HOP_DEPTH))
        query_text: str = params.get("query", state.query)

        # Seed entities: from params["entities"] or extracted from query text
        seed_strings: list[str] = params.get("entities", [])
        if not seed_strings:
            seed_strings = _extract_entities(query_text)
        seed_keys = [_entity_key(e) for e in seed_strings]

        # BFS over entity graph
        visited_entities: set[str] = set(seed_keys)
        frontier: deque[str] = deque(seed_keys)
        reachable_entities: set[str] = set(seed_keys)

        for _depth in range(depth):
            next_frontier: deque[str] = deque()
            while frontier:
                current = frontier.popleft()
                for neighbour in self._adjacency.get(current, set()):
                    if neighbour not in visited_entities:
                        visited_entities.add(neighbour)
                        next_frontier.append(neighbour)
                        reachable_entities.add(neighbour)
            frontier = next_frontier

        # Collect documents linked to any reachable entity
        doc_matches: dict[int, set[str]] = defaultdict(set)
        for ent_key in reachable_entities:
            for doc_idx in self._entity_to_docs.get(ent_key, []):
                doc_matches[doc_idx].add(ent_key)

        # Sort by score descending
        ranked = sorted(doc_matches.items(), key=lambda x: (-len(x[1]), x[0]))

        items = []
        for doc_idx, matched in ranked[:top_k]:
            doc = self._documents[doc_idx]
            item = {k: v for k, v in doc.items()}
            item["score"] = float(len(matched))
            item["matched_entities"] = sorted(matched)
            items.append(item)

        return items

    # ─────────────────────────────────────────────────────────
    # Inspection utilities (for analysis / debugging)
    # ─────────────────────────────────────────────────────────

    def get_entity_neighbours(self, entity: str) -> list[str]:
        """Return all entity keys adjacent to *entity* in the graph."""
        return sorted(self._adjacency.get(_entity_key(entity), set()))

    def get_entity_documents(self, entity: str) -> list[dict]:
        """Return all documents that mention *entity*."""
        key = _entity_key(entity)
        return [self._documents[i] for i in self._entity_to_docs.get(key, [])]
