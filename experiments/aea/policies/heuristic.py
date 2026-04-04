"""
AEA heuristic policy — the first real implementation of our method.

This is a hand-designed routing policy that validates the harness-level
attention concept *before* introducing any learning.  It should be treated
as the paper's primary baseline for the "does routing help?" ablation.

Design philosophy
-----------------
The heuristic encodes the following routing intuitions:

1. **Semantic entry** — Step 0 always uses dense retrieval to establish
   anchor documents.

2. **Entity bridge** — If the question contains multiple named entities
   that appear to require a bridge (question words: "who", "what … did …
   what", "which … that …"), and we have retrieved step-0 content but not
   yet found the answer, switch to the entity graph for a HOP step.

3. **Lexical fallback** — If entity hopping did not add new workspace items
   (graph returned nothing new), fall back to BM25 with a focused keyword
   query built from the original query minus stop words.

4. **Pin and evict** — After each step, pin the top-2 workspace items by
   relevance; evict items with relevance below a threshold (to stay within
   context budget).

5. **Stop condition** — Stop when:
   * Workspace coverage is sufficient (≥ threshold workspace items retrieved
     and average relevance ≥ coverage_threshold), or
   * Budget ≤ 10 % remaining, or
   * The step limit is reached.

Parameters
----------
top_k : int
    Documents per retrieval step.
coverage_threshold : float
    Minimum average relevance in workspace before stopping.
max_steps : int
    Hard step cap (enforced in addition to harness limit).
pin_top_k : int
    Number of top-relevance items to pin each step.
evict_threshold : float
    Evict workspace items below this relevance.
"""

from __future__ import annotations

import re

from ..types import Action, AddressSpaceType, AgentState, Operation
from .base import Policy

_DEFAULT_TOP_K = 5
_DEFAULT_COVERAGE = 0.5
_DEFAULT_MAX_STEPS = 8
_DEFAULT_PIN_TOP_K = 2
_DEFAULT_EVICT_THRESHOLD = 0.15

# Minimal English stop words for keyword query construction
_STOP_WORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can",
        "in", "on", "at", "by", "for", "with", "from", "to", "of",
        "and", "or", "but", "so", "yet", "nor", "not",
        "this", "that", "these", "those", "it", "its",
        "what", "which", "who", "whom", "where", "when", "why", "how",
    }
)

# Heuristics for "multi-hop" questions
_MULTI_HOP_PATTERNS = [
    re.compile(r"\bwho\b.*\bwho\b", re.IGNORECASE),          # two who-clauses
    re.compile(r"\bwhich\b.*\bthat\b", re.IGNORECASE),       # which … that …
    re.compile(r"\bwhat\b.*\band\b.*\bwhat\b", re.IGNORECASE),
    re.compile(r"\bborn in\b", re.IGNORECASE),                # nationality bridge
    re.compile(r"\bsame\b.*\bnationality\b", re.IGNORECASE),
    re.compile(r"\bboth\b", re.IGNORECASE),
    re.compile(r"\balso\b", re.IGNORECASE),
]


def _looks_multi_hop(question: str) -> bool:
    """Return True if question heuristically requires a bridge hop."""
    return any(p.search(question) for p in _MULTI_HOP_PATTERNS)


def _keyword_query(question: str) -> str:
    """Strip stop words and return a focused keyword string."""
    tokens = re.findall(r"\b\w+\b", question.lower())
    keywords = [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]
    return " ".join(keywords) if keywords else question


def _workspace_avg_relevance(state: AgentState) -> float:
    """Mean relevance score across workspace items."""
    if not state.workspace:
        return 0.0
    return sum(item.relevance_score for item in state.workspace) / len(state.workspace)


class AEAHeuristicPolicy(Policy):
    """
    Adaptive External Attention heuristic policy.

    This is the hand-designed routing baseline that validates the
    harness-level attention concept before learning.

    Routing sequence
    ----------------
    Step 0  → SemanticAddressSpace SEARCH
    Step 1  → If multi-hop question: EntityAddressSpace HOP
              Else:                  SemanticAddressSpace SEARCH (focused)
    Step 2+ → If entity hop was used and workspace grew: repeat HOP
              If workspace stagnated: LexicalAddressSpace SEARCH (keywords)
    Any step: if coverage_threshold met or budget ≤ 10% → STOP
    """

    def __init__(
        self,
        top_k: int = _DEFAULT_TOP_K,
        coverage_threshold: float = _DEFAULT_COVERAGE,
        max_steps: int = _DEFAULT_MAX_STEPS,
        pin_top_k: int = _DEFAULT_PIN_TOP_K,
        evict_threshold: float = _DEFAULT_EVICT_THRESHOLD,
    ) -> None:
        self._top_k = top_k
        self._coverage_threshold = coverage_threshold
        self._max_steps = max_steps
        self._pin_top_k = pin_top_k
        self._evict_threshold = evict_threshold

    def name(self) -> str:
        return "pi_aea_heuristic"

    def select_action(self, state: AgentState) -> Action:
        # ── Pin / evict before deciding (affects step > 0) ──────────────────
        if state.step > 0:
            self._manage_workspace(state)

        # ── Stop conditions ──────────────────────────────────────────────────
        if self._should_stop(state):
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.STOP,
            )

        # ── Step 0: semantic entry ───────────────────────────────────────────
        if state.step == 0:
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.SEARCH,
                params={"query": state.query, "top_k": self._top_k},
            )

        # ── Step 1: bridge decision ──────────────────────────────────────────
        if state.step == 1:
            if _looks_multi_hop(state.query):
                return Action(
                    address_space=AddressSpaceType.ENTITY,
                    operation=Operation.HOP,
                    params={
                        "query": state.query,
                        "depth": 1,
                        "top_k": self._top_k,
                    },
                )
            # Not a clear bridge — do a second semantic pass with query
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.SEARCH,
                params={"query": state.query, "top_k": self._top_k},
            )

        # ── Step 2+: adaptive fallback ───────────────────────────────────────
        workspace_size_before = len(state.history[-1].get("workspace_size_before", []))
        last_action = state.history[-1].get("action", {}) if state.history else {}
        last_space = last_action.get("address_space", "")
        last_n_items = state.history[-1].get("n_items", 0) if state.history else 0

        # If last entity hop produced results → repeat hop to go deeper
        if last_space == "entity" and last_n_items > 0:
            return Action(
                address_space=AddressSpaceType.ENTITY,
                operation=Operation.HOP,
                params={
                    "query": state.query,
                    "depth": 1,
                    "top_k": self._top_k,
                },
            )

        # Entity produced nothing or wasn't used → lexical fallback
        return Action(
            address_space=AddressSpaceType.LEXICAL,
            operation=Operation.SEARCH,
            params={
                "query": _keyword_query(state.query),
                "top_k": self._top_k,
            },
        )

    # ─────────────────────────────────────────────────────────
    # Workspace management
    # ─────────────────────────────────────────────────────────

    def _manage_workspace(self, state: AgentState) -> None:
        """
        Pin top-k items by relevance and evict items below threshold.

        This mutates ``state.workspace`` in place.  The harness will not
        evict pinned items, so we pin conservatively.
        """
        if not state.workspace:
            return

        # Sort by relevance descending
        state.workspace.sort(key=lambda x: x.relevance_score, reverse=True)

        # Pin top-k
        for i, item in enumerate(state.workspace):
            item.pinned = i < self._pin_top_k

        # Evict items below threshold (unless pinned)
        state.workspace = [
            item for item in state.workspace
            if item.pinned or item.relevance_score >= self._evict_threshold
        ]

    def _should_stop(self, state: AgentState) -> bool:
        """Return True if the policy should issue STOP this step."""
        if state.step >= self._max_steps:
            return True
        if state.budget_remaining <= 0.10:
            return True
        # Sufficient coverage: workspace non-empty and avg relevance is high
        if (
            len(state.workspace) >= 2
            and _workspace_avg_relevance(state) >= self._coverage_threshold
        ):
            return True
        return False
