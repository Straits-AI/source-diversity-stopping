"""
Navigation agent framework for heterogeneous document environments.

Unlike RAG (retrieve → generate), navigation is:
  explore → decide → explore more → decide → stop → generate

The agent maintains state (what it has discovered vs. what it knows),
chooses actions (search, open, read, follow links, stop), and uses
convergence-based stopping as one signal among many.
"""

from .state import NavigationState, DiscoveryEntry, KnowledgeEntry
from .actions import Action, ActionType, ActionResult
from .agent import NavigationAgent
from .policy import NavigationPolicy, ConvergencePolicy

__all__ = [
    "NavigationState",
    "DiscoveryEntry",
    "KnowledgeEntry",
    "Action",
    "ActionType",
    "ActionResult",
    "NavigationAgent",
    "NavigationPolicy",
    "ConvergencePolicy",
]
