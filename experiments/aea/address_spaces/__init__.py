"""
AEA address spaces package.

Available address spaces
------------------------
SemanticAddressSpace  — dense vector retrieval (sentence-transformers)
LexicalAddressSpace   — BM25 keyword retrieval (rank_bm25)
EntityGraphAddressSpace — entity co-occurrence graph (regex NER + BFS)
AddressSpace          — abstract base class for custom implementations
"""

from .base import AddressSpace
from .entity_graph import EntityGraphAddressSpace
from .lexical import LexicalAddressSpace
from .semantic import SemanticAddressSpace

__all__ = [
    "AddressSpace",
    "SemanticAddressSpace",
    "LexicalAddressSpace",
    "EntityGraphAddressSpace",
]
