"""
Navigation policies — HOW the agent decides what to do next.

A policy maps state → action. Different policies encode different
navigation strategies. The convergence policy is the empirically
validated default.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .state import NavigationState
from .actions import Action, ActionType


class NavigationPolicy(ABC):
    """Base class for navigation policies."""

    @abstractmethod
    def choose_action(self, state: NavigationState) -> Action:
        """Given the current state, choose the next action."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Policy name."""


class ConvergencePolicy(NavigationPolicy):
    """
    Convergence-based navigation policy.

    Strategy:
    1. Start with a broad search (semantic/dense substrate)
    2. Open the most promising discovered-but-unread document
    3. If the document mentions links/references, follow the most relevant one
    4. Check convergence: have 2+ independent sources contributed knowledge?
       - YES → STOP
       - NO → search with a different substrate, or explore unexplored discoveries

    This policy implements the full navigation loop:
      search → discover → open → read → follow links → check convergence → repeat or stop
    """

    def __init__(
        self,
        substrate_order: list[str] | None = None,
        min_sources: int = 2,
        max_steps: int = 10,
    ) -> None:
        self._substrate_order = substrate_order  # None = use environment's default order
        self._min_sources = min_sources
        self._max_steps = max_steps
        self._substrates_searched: set[str] = set()
        self._current_substrate_idx = 0

    @property
    def name(self) -> str:
        return "convergence"

    def choose_action(self, state: NavigationState) -> Action:
        # Budget exhausted
        if state.budget_remaining <= 0 or state.step >= self._max_steps:
            return Action(type=ActionType.STOP)

        # CONVERGENCE CHECK: have 2+ independent sources contributed knowledge?
        if len(state.knowledge_sources) >= self._min_sources:
            return Action(type=ActionType.STOP)

        # Step 0: always start with a search
        if state.step == 0:
            substrate = self._next_substrate(state)
            return Action(
                type=ActionType.SEARCH,
                params={"query": state.query, "substrate": substrate},
            )

        # If there are unexplored discoveries, OPEN the most promising one
        unexplored = state.unexplored
        if unexplored and state.n_known < 2:
            # Pick the one discovered most recently (likely most relevant)
            best = unexplored[0]
            return Action(
                type=ActionType.OPEN,
                params={"doc_id": best.doc_id},
            )

        # If last action found links, follow the most relevant one
        if state.history:
            last = state.history[-1]
            links = last.get("links_found", [])
            if links:
                # Follow the first unvisited link
                for link in links:
                    if link not in state.known:
                        return Action(
                            type=ActionType.FOLLOW_LINK,
                            params={"link": link},
                        )

        # No unexplored items, no links — try a different substrate
        substrate = self._next_substrate(state)
        if substrate:
            return Action(
                type=ActionType.SEARCH,
                params={"query": state.query, "substrate": substrate},
            )

        # Everything exhausted
        return Action(type=ActionType.STOP)

    def _next_substrate(self, state: NavigationState) -> str | None:
        """Get the next substrate to search that hasn't been used yet."""
        if self._substrate_order:
            for s in self._substrate_order:
                if s not in self._substrates_searched:
                    self._substrates_searched.add(s)
                    return s
        return None

    def reset(self):
        """Reset policy state for a new query."""
        self._substrates_searched = set()
        self._current_substrate_idx = 0
