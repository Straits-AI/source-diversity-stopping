"""
Cross-encoder stopping policy for the AEA framework.

Uses a pre-trained cross-encoder (MS MARCO) to score (question, passage) pairs
and decide when retrieved evidence is sufficient to stop.

Key advantages over heuristic stopping
---------------------------------------
* **Content-aware**: stops based on actual semantic relevance of passages to
  the question, not just workspace statistics (count, source diversity).
* **Zero data leakage**: the cross-encoder (``cross-encoder/ms-marco-MiniLM-L-6-v2``)
  was trained on MS MARCO passage-ranking; it has never seen HotpotQA or our
  evaluation data.  No training or threshold-tuning on our eval set.
* **Distribution-robust**: because the decision criterion is grounded in the
  model's pre-training signal, it generalises across question types.

Architecture
------------
Step 0  → SemanticAddressSpace SEARCH (same anchor step as all policies)
Step 1+ → Score every workspace passage with the cross-encoder
           • STOP if top score > HIGH_THRESHOLD  (single very strong passage)
           • STOP if ≥ 2 passages have score > MEDIUM_THRESHOLD
                                                (multiple supporting passages)
           • Otherwise: LexicalAddressSpace SEARCH (keyword fallback)
Any step → Hard stop when budget ≤ 10 % or step cap reached

Thresholds
----------
MS MARCO cross-encoder scores are roughly Gaussian around 0, ranging from
approximately −15 to +15 in practice.  We choose:

* HIGH_THRESHOLD = 7.0   — scores this high appear only for passages that
                            directly answer the question (top ~5 % on MS MARCO)
* MEDIUM_THRESHOLD = 3.0 — moderately relevant; two such passages together
                            provide strong multi-hop support

These values are anchored to the pre-training distribution, NOT tuned on our
HotpotQA eval set.
"""

from __future__ import annotations

import re
from typing import Optional

from ..types import Action, AddressSpaceType, AgentState, Operation
from .base import Policy

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_CE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Thresholds anchored to MS MARCO pre-training distribution (NOT tuned on eval)
_HIGH_THRESHOLD: float = 7.0     # single passage with very high cross-encoder score
_MEDIUM_THRESHOLD: float = 3.0   # two passages each with moderate cross-encoder score

_DEFAULT_TOP_K: int = 5
_DEFAULT_MAX_STEPS: int = 8

# Stop words for keyword query construction
_STOP_WORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can",
        "in", "on", "at", "by", "for", "with", "from", "to", "of",
        "and", "or", "but", "so", "yet", "nor", "not",
        "this", "that", "these", "those", "it", "its",
        "what", "which", "who", "whom", "where", "when", "why", "how",
    }
)


def _keyword_query(question: str) -> str:
    """Strip stop words and return a focused keyword string."""
    tokens = re.findall(r"\b\w+\b", question.lower())
    keywords = [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]
    return " ".join(keywords) if keywords else question


# ─────────────────────────────────────────────────────────────────────────────
# Policy
# ─────────────────────────────────────────────────────────────────────────────

class CrossEncoderStoppingPolicy(Policy):
    """
    Uses a pre-trained cross-encoder to score (question, passage) pairs.
    Stops when cross-encoder scores indicate sufficient evidence quality.

    Key advantage: content-aware scoring that generalises across distributions
    (trained on MS MARCO, not on our data).

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier for the cross-encoder.
        Default: ``cross-encoder/ms-marco-MiniLM-L-6-v2``.
    high_threshold : float
        Single-passage cross-encoder score above which we stop immediately.
        Default: 7.0.
    medium_threshold : float
        Multi-passage threshold: stop if ≥ 2 passages exceed this score.
        Default: 3.0.
    top_k : int
        Documents retrieved per step.  Default: 5.
    max_steps : int
        Hard step cap (budget enforcement).  Default: 8.
    """

    def __init__(
        self,
        model_name: str = _CE_MODEL_NAME,
        high_threshold: float = _HIGH_THRESHOLD,
        medium_threshold: float = _MEDIUM_THRESHOLD,
        top_k: int = _DEFAULT_TOP_K,
        max_steps: int = _DEFAULT_MAX_STEPS,
    ) -> None:
        self._model_name = model_name
        self._high_threshold = high_threshold
        self._medium_threshold = medium_threshold
        self._top_k = top_k
        self._max_steps = max_steps

        # Lazy-load cross-encoder to avoid loading at import time
        self._ce_model: Optional[object] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Policy interface
    # ─────────────────────────────────────────────────────────────────────────

    def name(self) -> str:
        return "pi_cross_encoder_stop"

    def select_action(self, state: AgentState) -> Action:
        # Hard stop: budget or step cap
        if state.step >= self._max_steps or state.budget_remaining <= 0.10:
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.STOP,
            )

        # Step 0: semantic search to establish anchor documents
        if state.step == 0:
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.SEARCH,
                params={"query": state.query, "top_k": self._top_k},
            )

        # Step 1+: score workspace passages with cross-encoder, decide stop/continue
        if state.workspace and self._cross_encoder_says_stop(state):
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.STOP,
            )

        # Continue: lexical search with focused keyword query
        return Action(
            address_space=AddressSpaceType.LEXICAL,
            operation=Operation.SEARCH,
            params={
                "query": _keyword_query(state.query),
                "top_k": self._top_k,
            },
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Cross-encoder scoring
    # ─────────────────────────────────────────────────────────────────────────

    def _load_model(self) -> object:
        """Lazy-load the cross-encoder model (once per policy instance)."""
        if self._ce_model is None:
            from sentence_transformers import CrossEncoder  # type: ignore
            self._ce_model = CrossEncoder(self._model_name)
        return self._ce_model

    def _score_passages(self, question: str, passages: list[str]) -> list[float]:
        """
        Score each passage against the question using the cross-encoder.

        Parameters
        ----------
        question : str
            The original query.
        passages : list[str]
            Passage texts from the workspace.

        Returns
        -------
        list[float]
            Cross-encoder scores, one per passage (same order as input).
        """
        model = self._load_model()
        pairs = [(question, p) for p in passages]
        scores_array = model.predict(pairs)
        return [float(s) for s in scores_array]

    def _cross_encoder_says_stop(self, state: AgentState) -> bool:
        """
        Return True if cross-encoder scores warrant stopping.

        Stop conditions (MS MARCO score space):
          1. Any single passage scores above HIGH_THRESHOLD (very direct answer).
          2. Two or more passages each score above MEDIUM_THRESHOLD (multi-hop support).
        """
        passages = [item.content for item in state.workspace if item.content]
        if not passages:
            return False

        scores = self._score_passages(state.query, passages)

        # Condition 1: single strong passage
        if any(s > self._high_threshold for s in scores):
            return True

        # Condition 2: multiple moderately-relevant passages
        n_medium = sum(1 for s in scores if s > self._medium_threshold)
        if n_medium >= 2:
            return True

        return False
