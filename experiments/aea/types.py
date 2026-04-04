"""
Core data types for the Adaptive External Attention (AEA) framework.

These types form the shared vocabulary across all modules: address spaces,
policies, and the evaluation harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ─────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────

class AddressSpaceType(Enum):
    """The six address-space types the AEA router can consult."""

    SEMANTIC = "semantic"        # Vector similarity (dense retrieval)
    LEXICAL = "lexical"          # BM25 / keyword matching
    STRUCTURAL = "structural"    # Document tree / section navigation
    ENTITY = "entity"            # Entity co-occurrence graph hops
    EXECUTABLE = "executable"    # Code / SQL / tool execution
    MEMORY = "memory"            # Working memory slots


class Operation(Enum):
    """Operations the policy can issue to any address space."""

    SEARCH = "search"        # Retrieve top-k items matching a query
    PREVIEW = "preview"      # Return a lightweight summary / first sentence
    OPEN = "open"            # Fully load a document / node by id
    EXPAND = "expand"        # Widen scope (parent section, adjacent nodes)
    COMPRESS = "compress"    # Summarise / shorten a workspace item
    EVICT = "evict"          # Remove an item from the active workspace
    HOP = "hop"              # Graph / entity traversal step
    TOOL_CALL = "tool_call"  # Execute a tool (SQL, calculator, API …)
    STOP = "stop"            # Signal that the agent is done


# ─────────────────────────────────────────────────────────────
# State components
# ─────────────────────────────────────────────────────────────

@dataclass
class DiscoveryEntry:
    """
    Something the agent knows *exists* but has not yet fully examined.

    Discovery entries populate D_t in the agent state and are created
    whenever a search returns metadata (title, snippet) without the
    full content being loaded into the workspace.
    """

    source_id: str              # Stable identifier (e.g. document title, paragraph id)
    source_type: str            # "file", "section", "entity", "table", "row", …
    description: str            # Short description / snippet surface text
    confidence: float           # How relevant this is estimated to be (0–1)
    discovered_at_step: int     # Step t at which this entry was created


@dataclass
class KnowledgeEntry:
    """
    A grounded, verified claim extracted from workspace evidence.

    Knowledge entries populate K_t and represent facts the agent has
    confirmed against source text, distinct from mere workspace content.
    """

    claim: str                  # Natural-language statement of the fact
    evidence_source: str        # source_id of the document that grounds the claim
    confidence: float           # Confidence in the claim (0–1)
    verified_at_step: int       # Step t at which the claim was verified


@dataclass
class WorkspaceItem:
    """
    An item in the active workspace W_t.

    Items can be pinned (retained across budget decisions), compressed
    (summary replaces full text), or evicted (removed from context).
    """

    content: str                # Full (or compressed) text content
    source_id: str              # Stable identifier of the originating document
    relevance_score: float      # Estimated relevance to the current query (0–1)
    pinned: bool = False        # If True, workspace manager will not evict
    compressed: bool = False    # If True, content is a summary, not full text
    added_at_step: int = 0      # Step at which this item entered the workspace


@dataclass
class AgentState:
    """
    The full agent state  s_t = (q, D_t, K_t, W_t, H_t, B_t).

    Parameters
    ----------
    query : str
        The original natural-language question.
    discovery : list[DiscoveryEntry]
        Items known to exist but not yet loaded (D_t).
    knowledge : list[KnowledgeEntry]
        Verified claims extracted from evidence (K_t).
    workspace : list[WorkspaceItem]
        Active context items (W_t).
    history : list[dict]
        Ordered log of (action, result) pairs (H_t).
    budget_remaining : float
        Fraction of the original token budget still available, in [0, 1].
    step : int
        Current decision step index (0-based).
    """

    query: str
    discovery: list[DiscoveryEntry] = field(default_factory=list)
    knowledge: list[KnowledgeEntry] = field(default_factory=list)
    workspace: list[WorkspaceItem] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    budget_remaining: float = 1.0
    step: int = 0


# ─────────────────────────────────────────────────────────────
# Action and result types
# ─────────────────────────────────────────────────────────────

@dataclass
class Action:
    """
    A single policy decision: (address_space, operation, parameters).

    Parameters
    ----------
    address_space : AddressSpaceType
        Which address space to consult.
    operation : Operation
        What to do within that address space.
    params : dict
        Operation-specific key-value arguments (e.g. ``{"query": "…", "top_k": 5}``).
    """

    address_space: AddressSpaceType
    operation: Operation
    params: dict = field(default_factory=dict)


@dataclass
class ActionResult:
    """
    The outcome of executing an action inside an address space.

    Parameters
    ----------
    items : list[dict]
        Retrieved or produced items.  Each dict should contain at minimum
        ``{"id": str, "content": str, "score": float}``.
    cost_tokens : int
        Estimated tokens consumed (query + results).
    cost_latency_ms : float
        Wall-clock latency in milliseconds.
    cost_operations : int
        Number of index / graph lookups performed (default 1).
    success : bool
        Whether the operation completed without error.
    error : str
        Human-readable error message if ``success`` is False.
    """

    items: list[dict]
    cost_tokens: int
    cost_latency_ms: float
    cost_operations: int = 1
    success: bool = True
    error: str = ""


# ─────────────────────────────────────────────────────────────
# Evidence bundle
# ─────────────────────────────────────────────────────────────

@dataclass
class EvidenceBundle:
    """
    A scored set of workspace items assembled as evidence for an answer.

    Parameters
    ----------
    items : list[WorkspaceItem]
        The workspace items forming the bundle.
    requirements_total : int
        Number of distinct information requirements the question has.
    requirements_satisfied : int
        How many of those requirements are covered by the bundle.
    bundle_coverage : float
        ``requirements_satisfied / requirements_total`` (0–1).
    confidence : float
        Aggregate confidence across all items in the bundle (0–1).
    """

    items: list[WorkspaceItem]
    requirements_total: int
    requirements_satisfied: int
    bundle_coverage: float      # requirements_satisfied / requirements_total
    confidence: float
