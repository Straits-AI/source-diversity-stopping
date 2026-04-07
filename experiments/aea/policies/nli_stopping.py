"""
NLI-based bundle sufficiency stopping policy for the AEA framework.

Uses an NLI model (cross-encoder/nli-deberta-v3-small) to assess whether
the CONCATENATED evidence bundle in the workspace is sufficient to answer
the question.

Key difference from cross-encoder stopping
-------------------------------------------
The cross-encoder (MS MARCO) scores individual (question, passage) pairs.
It cannot capture *set-level* sufficiency -- two individually mediocre
passages may jointly answer a bridge question, but the cross-encoder
sees them independently.

NLI naturally solves this: we concatenate the full workspace into a single
premise string and ask "does this premise entail the answer exists?"
This is a *bundle* assessment, not a per-passage assessment.

Architecture
------------
Step 0  -> SemanticAddressSpace SEARCH (same anchor step as all policies)
Step 1+ -> Concatenate ALL workspace passages -> single premise (<=512 tokens)
           Hypothesis: question reformulated as existential assertion:
             "Who directed X?" -> "Someone directed X."
             "What nationality is Y?" -> "Someone or something nationality is Y."
           Run NLI: premise -> hypothesis -> {entailment, neutral, contradiction}
           - STOP if entailment probability > ENTAILMENT_THRESHOLD (0.7)
           - Otherwise: LexicalAddressSpace SEARCH (keyword fallback)
Any step -> Hard stop when budget <= 10% or step cap reached

Hypothesis design
-----------------
NLI models (trained on MNLI/SNLI) require a *factual claim* as the hypothesis,
not a meta-statement about answering.  We convert the question to an
existential assertion by replacing the interrogative word (Who/What/Which/
Where/When/How ...) with "Someone or something".  This creates a factual
claim that the evidence bundle can entail or fail to entail.

  "Who directed Cast Away?" ->
  "Someone directed Cast Away."
  - Good bundle (Zemeckis + movie article): entailment ~0.95
  - Bad bundle (random passages):           entailment ~0.001

Model
-----
cross-encoder/nli-deberta-v3-small  -- proper NLI model trained on MNLI/SNLI.
NOT a passage ranker; produces genuine entailment/neutral/contradiction scores.
Label order for this model: contradiction(0), entailment(1), neutral(2)
(verified from model config.id2label).

Implementation note: loaded via HuggingFace ``transformers`` AutoModel rather
than ``sentence_transformers.CrossEncoder`` to avoid macOS tokenizer deadlocks
observed with DeBERTa-v3 in the CrossEncoder wrapper.
"""

from __future__ import annotations

import re
from typing import Optional

from ..types import Action, AddressSpaceType, AgentState, Operation
from .base import Policy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

# Stop if entailment softmax probability exceeds this value.
# Anchored to a principled reading: 0.7 means the model is "clearly confident"
# that the bundle entails the answer exists.  NOT tuned on our eval set.
_ENTAILMENT_THRESHOLD: float = 0.7

# Approximate token budget for the premise (1 token ~= 4 chars in English).
# deberta-v3-small has a 512-token limit; we leave headroom for hypothesis.
_MAX_PREMISE_CHARS: int = 1800   # ~450 tokens

_DEFAULT_TOP_K: int = 5
_DEFAULT_MAX_STEPS: int = 8

# Interrogative patterns to convert question -> existential assertion.
# Applied in order; first match wins.
_INTERROGATIVE_SUB = re.compile(
    r"^(Who|What|Which|Where|When|How many|How much|How long|How old|How far)",
    re.IGNORECASE,
)

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


def _question_to_assertion(question: str) -> str:
    """
    Convert a question into an existential assertion suitable as an NLI hypothesis.

    NLI models (MNLI/SNLI training) expect factual claims, not meta-claims.
    We replace the interrogative word with "Someone or something" to produce
    a factual claim the evidence bundle can entail.

    Examples
    --------
    "Who directed Cast Away?"  -> "Someone or something directed Cast Away."
    "What is the capital of France?" -> "Someone or something is the capital of France."
    """
    q = question.strip().rstrip("?")
    assertion = _INTERROGATIVE_SUB.sub("Someone or something", q)
    if not assertion.endswith("."):
        assertion = assertion + "."
    return assertion


def _build_premise(workspace_items: list) -> str:
    """
    Concatenate workspace passage content into a single premise string.

    Items are sorted by relevance (descending) and truncated to
    _MAX_PREMISE_CHARS so the NLI model's context window is not exceeded.
    """
    sorted_items = sorted(
        workspace_items,
        key=lambda x: x.relevance_score,
        reverse=True,
    )
    parts = [item.content for item in sorted_items if item.content]
    combined = " ".join(parts)
    return combined[:_MAX_PREMISE_CHARS]


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class NLIStoppingPolicy(Policy):
    """
    Uses NLI to check if concatenated evidence entails an answer exists.

    This is the principled content-aware baseline that addresses the
    "set function" problem in stopping decisions.  Unlike the cross-encoder,
    which scores individual (question, passage) pairs, NLI takes the
    WHOLE workspace bundle as the premise.  This naturally captures the
    multi-hop case where two individually mediocre passages jointly suffice.

    The hypothesis is the question reformulated as an existential assertion
    (interrogative word replaced by "Someone or something").  This gives
    NLI models well-formed factual claims to assess, rather than meta-claims
    about answering ability.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier for the NLI model.
        Default: ``cross-encoder/nli-deberta-v3-small``.
    entailment_threshold : float
        Softmax probability for the ENTAILMENT class above which we stop.
        Default: 0.7.
    top_k : int
        Documents retrieved per step.  Default: 5.
    max_steps : int
        Hard step cap (budget enforcement).  Default: 8.
    """

    def __init__(
        self,
        model_name: str = _NLI_MODEL_NAME,
        entailment_threshold: float = _ENTAILMENT_THRESHOLD,
        top_k: int = _DEFAULT_TOP_K,
        max_steps: int = _DEFAULT_MAX_STEPS,
    ) -> None:
        self._model_name = model_name
        self._entailment_threshold = entailment_threshold
        self._top_k = top_k
        self._max_steps = max_steps

        # Lazy-loaded NLI model and tokenizer (HuggingFace transformers)
        self._nli_model: Optional[object] = None
        self._nli_tokenizer: Optional[object] = None

    # -------------------------------------------------------------------------
    # Policy interface
    # -------------------------------------------------------------------------

    def name(self) -> str:
        return "pi_nli_stopping"

    def select_action(self, state: AgentState) -> Action:
        # Hard stop: budget or step cap
        if state.step >= self._max_steps or state.budget_remaining <= 0.10:
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.STOP,
            )

        # Step 0: semantic search to populate the workspace
        if state.step == 0:
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.SEARCH,
                params={"query": state.query, "top_k": self._top_k},
            )

        # Step 1+: run NLI bundle assessment
        if state.workspace and self._nli_says_stop(state):
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.STOP,
            )

        # Continue: lexical search with focused keywords
        return Action(
            address_space=AddressSpaceType.LEXICAL,
            operation=Operation.SEARCH,
            params={
                "query": _keyword_query(state.query),
                "top_k": self._top_k,
            },
        )

    # -------------------------------------------------------------------------
    # NLI bundle assessment
    # -------------------------------------------------------------------------

    def _load_model(self):
        """
        Lazy-load the NLI model and tokenizer (once per policy instance).

        Uses HuggingFace transformers AutoModel directly to avoid macOS
        tokenizer deadlocks observed with sentence_transformers.CrossEncoder
        wrapping DeBERTa-v3 models.
        """
        if self._nli_model is None:
            from transformers import (  # type: ignore
                AutoTokenizer,
                AutoModelForSequenceClassification,
            )
            self._nli_tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._nli_model = AutoModelForSequenceClassification.from_pretrained(
                self._model_name
            )
            self._nli_model.eval()
        return self._nli_model

    def _entailment_probability(self, premise: str, hypothesis: str) -> float:
        """
        Return the ENTAILMENT class softmax probability.

        For cross-encoder/nli-deberta-v3-small the label order is:
          {0: 'contradiction', 1: 'entailment', 2: 'neutral'}
        i.e. logit index 1 = entailment  (verified from config.id2label).
        """
        import torch  # type: ignore
        import numpy as np  # type: ignore

        model = self._load_model()
        tokenizer = self._nli_tokenizer

        inputs = tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            logits = model(**inputs).logits  # shape (1, 3)

        logits_np = logits[0].cpu().numpy()
        # Numerically stable softmax
        shifted = logits_np - logits_np.max()
        exp_vals = np.exp(shifted)
        probs = exp_vals / exp_vals.sum()
        # Index 1 = entailment
        return float(probs[1])

    def _nli_says_stop(self, state: AgentState) -> bool:
        """
        Return True if the NLI model's entailment probability exceeds the threshold.

        Premise   : concatenated workspace passages (sorted by relevance, truncated)
        Hypothesis: question reformulated as existential assertion
                    e.g. "Who directed X?" -> "Someone or something directed X."
        """
        premise = _build_premise(state.workspace)
        if not premise.strip():
            return False

        hypothesis = _question_to_assertion(state.query)
        p_entail = self._entailment_probability(premise, hypothesis)
        return p_entail > self._entailment_threshold
