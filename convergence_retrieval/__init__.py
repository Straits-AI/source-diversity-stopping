"""
convergence_retrieval — Multi-substrate retrieval with convergence-based stopping.

Stop searching when independent retrieval pathways converge on the same evidence,
not when a fixed number of operations is reached.

Usage:
    from convergence_retrieval import ConvergenceRetriever, BM25Substrate, DenseSubstrate

    retriever = ConvergenceRetriever(
        substrates=[
            BM25Substrate(),
            DenseSubstrate(model="all-MiniLM-L6-v2"),
        ],
    )
    retriever.index(documents)
    results = retriever.search("How does auth work?")
    # Returns results in ~1.2 operations instead of 2.0
"""

__version__ = "0.1.0"

from .retriever import ConvergenceRetriever
from .substrates.bm25 import BM25Substrate
from .substrates.dense import DenseSubstrate
from .substrates.structural import StructuralSubstrate
from .substrates.base import Substrate

__all__ = [
    "ConvergenceRetriever",
    "BM25Substrate",
    "DenseSubstrate",
    "StructuralSubstrate",
    "Substrate",
]
