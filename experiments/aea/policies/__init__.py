"""
AEA policies package.

Available policies
------------------
SemanticOnlyPolicy          — π_semantic: always use dense retrieval
LexicalOnlyPolicy           — π_lexical: always use BM25
EntityOnlyPolicy            — π_entity: always use entity graph
AEAHeuristicPolicy          — π_aea_heuristic: adaptive hand-designed routing
EnsemblePolicy              — π_ensemble: query all substrates, merge results
LLMRoutedPolicy             — π_llm_routed: LLM makes routing decisions at each step
DecompositionStoppingPolicy — π_decomposition: LLM decomposes question once; stops
                               when requirements are covered by workspace content
EmbeddingRouterPolicy       — π_embedding_router: embedding-based question-type routing
CrossEncoderStoppingPolicy  — π_cross_encoder_stop: content-aware stopping via MS MARCO cross-encoder
NLIStoppingPolicy           — π_nli_stopping: NLI bundle sufficiency check (principled set-function baseline)
AnswerStabilityPolicy       — π_answer_stability: stops when draft answer converges between retrieval steps
ConfidenceGatedPolicy       — π_confidence_gated: one LLM call after first retrieval to gate stopping;
                               stops immediately if confident, else does one lexical step then stops
Policy                      — abstract base class for custom policies
"""

from .answer_stability import AnswerStabilityPolicy
from .base import Policy
from .confidence_gated import ConfidenceGatedPolicy
from .cross_encoder_stopping import CrossEncoderStoppingPolicy
from .decomposition_stopping import DecompositionStoppingPolicy
from .embedding_router import EmbeddingRouterPolicy
from .ensemble import EnsemblePolicy
from .heuristic import AEAHeuristicPolicy
from .llm_routed import LLMRoutedPolicy
from .nli_stopping import NLIStoppingPolicy
from .single_substrate import EntityOnlyPolicy, LexicalOnlyPolicy, SemanticOnlyPolicy

__all__ = [
    "Policy",
    "SemanticOnlyPolicy",
    "LexicalOnlyPolicy",
    "EntityOnlyPolicy",
    "AEAHeuristicPolicy",
    "EnsemblePolicy",
    "LLMRoutedPolicy",
    "DecompositionStoppingPolicy",
    "EmbeddingRouterPolicy",
    "CrossEncoderStoppingPolicy",
    "NLIStoppingPolicy",
    "AnswerStabilityPolicy",
    "ConfidenceGatedPolicy",
]
