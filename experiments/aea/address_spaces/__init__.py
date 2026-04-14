"""
AEA address spaces package.

Available address spaces
------------------------
SemanticAddressSpace    — dense vector retrieval (sentence-transformers)
LexicalAddressSpace     — BM25 keyword retrieval (rank_bm25)
EntityGraphAddressSpace — entity co-occurrence graph (regex NER + BFS)
StructuralAddressSpace  — title-hierarchy navigation (SEARCH titles, OPEN paragraphs)
ExecutableAddressSpace  — regex number extraction + Python arithmetic (SEARCH, TOOL_CALL)
AddressSpace            — abstract base class for custom implementations
"""

from .base import AddressSpace
from .entity_graph import EntityGraphAddressSpace
from .executable import ExecutableAddressSpace
from .lexical import LexicalAddressSpace
from .semantic import SemanticAddressSpace
from .structural import StructuralAddressSpace

__all__ = [
    "AddressSpace",
    "SemanticAddressSpace",
    "LexicalAddressSpace",
    "EntityGraphAddressSpace",
    "StructuralAddressSpace",
    "ExecutableAddressSpace",
]
