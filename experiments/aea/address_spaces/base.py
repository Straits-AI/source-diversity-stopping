"""
Abstract base class for all AEA address spaces.

An address space is an indexed information store that exposes a uniform
``query`` interface.  Concrete subclasses implement specific retrieval
strategies (vector similarity, BM25, entity graph, …).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import AgentState, ActionResult, Operation


class AddressSpace(ABC):
    """
    Base class for all address spaces.

    Subclasses must implement:
      - ``query``         — execute one operation and return an ActionResult
      - ``build_index``   — ingest documents and prepare the internal index
      - ``name``          — a stable string identifier for this space
      - ``supported_operations`` — which operations this space accepts

    All cost accounting (tokens, latency, operations) is the responsibility
    of the concrete subclass; the harness does not inject costs.
    """

    # ─────────────────────────────────────────────────────────
    # Abstract interface
    # ─────────────────────────────────────────────────────────

    @abstractmethod
    def query(
        self,
        state: AgentState,
        operation: Operation,
        params: dict,
    ) -> ActionResult:
        """
        Execute *operation* in this address space given the current agent state.

        Parameters
        ----------
        state : AgentState
            Full agent state at the time of the call.  Implementations may
            use ``state.query`` as a default query string.
        operation : Operation
            The operation to perform (must be in ``supported_operations``).
        params : dict
            Operation-specific arguments.  Common keys:

            * ``"query"``  — override query string
            * ``"top_k"``  — number of results to return
            * ``"ids"``    — list of document ids (for OPEN / HOP)
            * ``"entities"`` — seed entities for HOP

        Returns
        -------
        ActionResult
            Retrieved items with associated cost metadata.

        Raises
        ------
        ValueError
            If *operation* is not in ``supported_operations``.
        """

    @abstractmethod
    def build_index(self, documents: list[dict]) -> None:
        """
        Build or rebuild the internal index from *documents*.

        Parameters
        ----------
        documents : list[dict]
            Each document must contain at minimum:

            * ``"id"``      — stable string identifier
            * ``"content"`` — full text content

            Additional fields (``"title"``, ``"metadata"`` …) are preserved
            and passed through in result items.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable string identifier for this address space (e.g. ``"semantic"``)."""

    @property
    @abstractmethod
    def supported_operations(self) -> list[Operation]:
        """List of :class:`Operation` values this address space handles."""

    # ─────────────────────────────────────────────────────────
    # Shared helpers
    # ─────────────────────────────────────────────────────────

    def _assert_operation(self, operation: Operation) -> None:
        """Raise ValueError if *operation* is not supported."""
        if operation not in self.supported_operations:
            supported = [op.value for op in self.supported_operations]
            raise ValueError(
                f"AddressSpace '{self.name}' does not support operation "
                f"'{operation.value}'.  Supported: {supported}"
            )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """
        Rough token estimate: 1 token ≈ 4 characters.

        This is intentionally lightweight — no tokeniser dependency.
        """
        return max(1, len(text) // 4)
