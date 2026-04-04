"""
Ensemble baseline policy.

π_ensemble queries all available address spaces at each step and merges
results by score.  This is an upper-bound oracle for retrieval quality at
the cost of maximum token usage.

The ensemble is deterministic: address spaces are queried in a fixed order
(semantic → lexical → entity) and results are merged by score (descending),
deduplicating by document id.

Use this as a ceiling to measure how much adaptive routing loses relative
to querying everything.
"""

from __future__ import annotations

from ..types import Action, AddressSpaceType, AgentState, Operation
from .base import Policy

_DEFAULT_TOP_K = 5
_DEFAULT_MAX_STEPS = 3

# Fixed order of address spaces queried at each step.
# Each element is (AddressSpaceType, Operation, extra_params).
_SPACE_SEQUENCE = [
    (AddressSpaceType.SEMANTIC, Operation.SEARCH, {}),
    (AddressSpaceType.LEXICAL, Operation.SEARCH, {}),
    (AddressSpaceType.ENTITY, Operation.SEARCH, {}),
]


class EnsemblePolicy(Policy):
    """
    π_ensemble: query all available substrates, merge results by score.

    At each step, the policy cycles through address spaces in order:
      step 0 mod 3 → semantic search
      step 1 mod 3 → lexical search
      step 2 mod 3 → entity search

    This serial round-robin across address spaces is equivalent to querying
    all substrates and merging, given that the harness accumulates all
    retrieved items in the workspace.

    The policy stops after ``max_steps`` steps (default 3, one pass per
    substrate) or when budget ≤ 10 %.

    Parameters
    ----------
    top_k : int
        Documents to retrieve per step per address space.  Default 5.
    max_steps : int
        Total steps before issuing STOP.  Should be a multiple of 3 for
        one full round-trip.  Default 3.
    """

    def __init__(
        self,
        top_k: int = _DEFAULT_TOP_K,
        max_steps: int = _DEFAULT_MAX_STEPS,
    ) -> None:
        self._top_k = top_k
        self._max_steps = max_steps

    def name(self) -> str:
        return "pi_ensemble"

    def select_action(self, state: AgentState) -> Action:
        # Stop conditions
        if state.step >= self._max_steps or state.budget_remaining <= 0.10:
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.STOP,
            )

        # Cycle through the substrate sequence
        space_type, operation, extra = _SPACE_SEQUENCE[state.step % len(_SPACE_SEQUENCE)]

        params = {"query": state.query, "top_k": self._top_k}
        params.update(extra)

        return Action(
            address_space=space_type,
            operation=operation,
            params=params,
        )
