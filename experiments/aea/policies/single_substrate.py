"""
Single-substrate baseline policies.

Each policy unconditionally routes all queries to exactly one address space.
These are the simplest possible baselines and correspond to the single-substrate
policies π_semantic, π_lexical, and π_entity described in the paper.

They serve as lower bounds: any adaptive policy that cannot beat these on
Utility@Budget provides no evidence for the AEA hypothesis.
"""

from __future__ import annotations

from ..types import Action, AddressSpaceType, AgentState, Operation
from .base import Policy

_DEFAULT_TOP_K = 5
_DEFAULT_HOP_DEPTH = 1


class SemanticOnlyPolicy(Policy):
    """
    π_semantic: always use dense vector (semantic) search.

    At each step, issues a SEARCH to the semantic address space using
    the original query.  Stops after a fixed number of steps or when
    the budget is exhausted (enforced by the harness).

    Parameters
    ----------
    top_k : int
        Number of documents to retrieve per step.  Default 5.
    max_steps : int
        Number of SEARCH steps to issue before stopping.  Default 2.
    """

    def __init__(self, top_k: int = _DEFAULT_TOP_K, max_steps: int = 2) -> None:
        self._top_k = top_k
        self._max_steps = max_steps

    def name(self) -> str:
        return "pi_semantic"

    def select_action(self, state: AgentState) -> Action:
        if state.step >= self._max_steps:
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.STOP,
            )
        return Action(
            address_space=AddressSpaceType.SEMANTIC,
            operation=Operation.SEARCH,
            params={"query": state.query, "top_k": self._top_k},
        )


class LexicalOnlyPolicy(Policy):
    """
    π_lexical: always use BM25 keyword search.

    At each step, issues a SEARCH to the lexical address space using
    the original query.

    Parameters
    ----------
    top_k : int
        Number of documents to retrieve per step.  Default 5.
    max_steps : int
        Number of SEARCH steps to issue before stopping.  Default 2.
    """

    def __init__(self, top_k: int = _DEFAULT_TOP_K, max_steps: int = 2) -> None:
        self._top_k = top_k
        self._max_steps = max_steps

    def name(self) -> str:
        return "pi_lexical"

    def select_action(self, state: AgentState) -> Action:
        if state.step >= self._max_steps:
            return Action(
                address_space=AddressSpaceType.LEXICAL,
                operation=Operation.STOP,
            )
        return Action(
            address_space=AddressSpaceType.LEXICAL,
            operation=Operation.SEARCH,
            params={"query": state.query, "top_k": self._top_k},
        )


class EntityOnlyPolicy(Policy):
    """
    π_entity: always use entity graph search / hops.

    Step 0: SEARCH the entity graph using query entities.
    Step 1+: HOP from retrieved entities to linked paragraphs.

    This mimics a pure entity-hop strategy with no semantic fallback.

    Parameters
    ----------
    top_k : int
        Number of documents to retrieve per step.  Default 5.
    hop_depth : int
        BFS depth for HOP operations.  Default 1.
    max_steps : int
        Maximum number of steps before stopping.  Default 5.
    """

    def __init__(
        self,
        top_k: int = _DEFAULT_TOP_K,
        hop_depth: int = _DEFAULT_HOP_DEPTH,
        max_steps: int = 5,
    ) -> None:
        self._top_k = top_k
        self._hop_depth = hop_depth
        self._max_steps = max_steps

    def name(self) -> str:
        return "pi_entity"

    def select_action(self, state: AgentState) -> Action:
        if state.step >= self._max_steps:
            return Action(
                address_space=AddressSpaceType.ENTITY,
                operation=Operation.STOP,
            )

        if state.step == 0:
            # Initial search to locate seed entities
            return Action(
                address_space=AddressSpaceType.ENTITY,
                operation=Operation.SEARCH,
                params={"query": state.query, "top_k": self._top_k},
            )

        # Subsequent steps: hop from entities found in retrieved content
        entities_found = _collect_entities_from_workspace(state)
        return Action(
            address_space=AddressSpaceType.ENTITY,
            operation=Operation.HOP,
            params={
                "query": state.query,
                "entities": entities_found,
                "depth": self._hop_depth,
                "top_k": self._top_k,
            },
        )


# ─────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────

def _collect_entities_from_workspace(state: AgentState) -> list[str]:
    """
    Return entity strings recorded in the workspace via ``matched_entities``.

    The entity graph address space attaches ``matched_entities`` metadata
    when it has already run; we pass those forward as hop seeds.
    If no such metadata exists, fall back to an empty list (the entity
    space will then extract entities from the query text internally).
    """
    entities: list[str] = []
    seen: set[str] = set()
    # History entries record action results; entity results carry matched_entities
    for entry in state.history:
        for ent in entry.get("matched_entities", []):
            if ent not in seen:
                seen.add(ent)
                entities.append(ent)
    return entities
