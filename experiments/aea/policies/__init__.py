"""
AEA policies package.

Available policies
------------------
SemanticOnlyPolicy    — π_semantic: always use dense retrieval
LexicalOnlyPolicy     — π_lexical: always use BM25
EntityOnlyPolicy      — π_entity: always use entity graph
AEAHeuristicPolicy    — π_aea_heuristic: adaptive hand-designed routing
EnsemblePolicy        — π_ensemble: query all substrates, merge results
Policy                — abstract base class for custom policies
"""

from .base import Policy
from .ensemble import EnsemblePolicy
from .heuristic import AEAHeuristicPolicy
from .single_substrate import EntityOnlyPolicy, LexicalOnlyPolicy, SemanticOnlyPolicy

__all__ = [
    "Policy",
    "SemanticOnlyPolicy",
    "LexicalOnlyPolicy",
    "EntityOnlyPolicy",
    "AEAHeuristicPolicy",
    "EnsemblePolicy",
]
