"""
Abstract base class for all AEA policies.

A policy π maps an agent state s_t to an action a_t:

    a_t = π(s_t)

Policies are stateless with respect to the agent state — all relevant
information is encoded in the AgentState passed to ``select_action``.
Policies may maintain their own internal state (e.g. learned weights,
random state) but must not modify the AgentState directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import Action, AgentState


class Policy(ABC):
    """
    Abstract base class for all policies (π).

    Subclasses must implement:
      - ``select_action`` — the core decision function
      - ``name``          — a stable identifier for logging
    """

    @abstractmethod
    def select_action(self, state: AgentState) -> Action:
        """
        Select the next action given the current agent state.

        Parameters
        ----------
        state : AgentState
            The full agent state at step t.  The policy must not modify
            the state; it is the harness's responsibility to update state
            after action execution.

        Returns
        -------
        Action
            The chosen (address_space, operation, params) triple.
            Return ``Action(AddressSpaceType.SEMANTIC, Operation.STOP)``
            (or any address space with STOP) to terminate the episode.
        """

    @abstractmethod
    def name(self) -> str:
        """Stable string identifier for this policy (used in logging)."""
