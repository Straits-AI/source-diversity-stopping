"""
Ablation variants of AEAHeuristicPolicy for hypothesis H4 testing.

Each ablation disables exactly one mechanism of the full AEA heuristic so
that the contribution of each component can be measured independently.

Ablation list
-------------
AblNoEarlyStop
    Disable coverage-driven early stopping.  The policy always executes
    exactly 2 steps (semantic → entity-hop or lexical), never stopping
    early due to workspace coverage.

AblSemanticOnlySmartStop
    Use ONLY semantic search (no entity hops, no lexical fallback) BUT
    keep the coverage-driven early stopping.  Tests whether the stopping
    rule alone explains the gain.

AblNoEntityHop
    Same as full AEA but never use entity graph hops.  After semantic
    search, fall back directly to lexical if more evidence needed.

AblAlwaysHop
    Always do an entity hop after semantic search, regardless of coverage.
    Tests whether SELECTIVE hopping matters vs always hopping.

AblNoWorkspaceMgmt
    Disable pin/evict workspace management.  Keep all retrieved items
    without curation.  Tests whether workspace management contributes.
"""

from __future__ import annotations

from ..types import Action, AddressSpaceType, AgentState, Operation
from .heuristic import (
    AEAHeuristicPolicy,
    _keyword_query,
    _looks_multi_hop,
    _workspace_avg_relevance,
)


class AblNoEarlyStop(AEAHeuristicPolicy):
    """
    Ablation: no coverage-driven early stopping.

    The policy always runs for exactly 2 retrieval steps:
      Step 0 → semantic SEARCH
      Step 1 → entity HOP (if multi-hop question) OR lexical SEARCH
    It never stops early based on workspace coverage or budget (beyond
    the harness hard limits).
    """

    def name(self) -> str:
        return "abl_no_early_stop"

    def _should_stop(self, state: AgentState) -> bool:
        """Only stop at the hard step limit — never from coverage."""
        # Allow up to max_steps; never stop early for coverage
        return state.step >= self._max_steps


class AblSemanticOnlySmartStop(AEAHeuristicPolicy):
    """
    Ablation: semantic search only, but keep coverage-driven early stopping.

    Every retrieval step uses SemanticAddressSpace SEARCH.  The policy
    still evaluates the coverage criterion and may stop early — but it
    never routes to entity or lexical address spaces.
    """

    def name(self) -> str:
        return "abl_semantic_only_smart_stop"

    def select_action(self, state: AgentState) -> Action:
        # Workspace management before deciding (step > 0)
        if state.step > 0:
            self._manage_workspace(state)

        # Coverage-driven stop (kept intact)
        if self._should_stop(state):
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.STOP,
            )

        # Always use semantic search, always
        query = state.query
        return Action(
            address_space=AddressSpaceType.SEMANTIC,
            operation=Operation.SEARCH,
            params={"query": query, "top_k": self._top_k},
        )


class AblNoEntityHop(AEAHeuristicPolicy):
    """
    Ablation: full AEA routing minus entity graph hops.

    Whenever the full heuristic would issue a HOP action, this variant
    falls back to a lexical SEARCH with focused keywords instead.
    """

    def name(self) -> str:
        return "abl_no_entity_hop"

    def select_action(self, state: AgentState) -> Action:
        action = super().select_action(state)

        # Intercept any HOP action and replace with lexical fallback
        if action.operation == Operation.HOP:
            return Action(
                address_space=AddressSpaceType.LEXICAL,
                operation=Operation.SEARCH,
                params={
                    "query": _keyword_query(state.query),
                    "top_k": self._top_k,
                },
            )

        return action


class AblAlwaysHop(AEAHeuristicPolicy):
    """
    Ablation: always do entity hop after semantic search.

    Step 0  → semantic SEARCH
    Step 1+ → entity HOP unconditionally (regardless of coverage)
    Any step: stop only if budget exhausted or max_steps reached
              (coverage-driven stop is disabled for fairness with
               the "no early stop" ablation — we want to isolate
               selective vs always hopping).
    """

    def name(self) -> str:
        return "abl_always_hop"

    def select_action(self, state: AgentState) -> Action:
        # Workspace management
        if state.step > 0:
            self._manage_workspace(state)

        # Hard stops only (no coverage-driven stop, to isolate hop selectivity)
        if state.step >= self._max_steps or state.budget_remaining <= 0.10:
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.STOP,
            )

        # Step 0: semantic entry
        if state.step == 0:
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.SEARCH,
                params={"query": state.query, "top_k": self._top_k},
            )

        # Step 1+: always hop
        return Action(
            address_space=AddressSpaceType.ENTITY,
            operation=Operation.HOP,
            params={
                "query": state.query,
                "depth": 1,
                "top_k": self._top_k,
            },
        )


class AblNoWorkspaceMgmt(AEAHeuristicPolicy):
    """
    Ablation: disable pin/evict workspace management.

    All retrieved items are kept in the workspace without curation.
    The policy routing logic (semantic → hop → lexical) and coverage-driven
    stopping are unchanged.
    """

    def name(self) -> str:
        return "abl_no_workspace_mgmt"

    def _manage_workspace(self, state: AgentState) -> None:
        """No-op: skip pin/evict curation."""
        return
