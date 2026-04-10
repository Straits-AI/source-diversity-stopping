"""
Answer-stability stopping policy for the AEA framework.

Stops when the draft answer converges (stops changing between retrieval steps).

Design
------
Step 0: Semantic search → generate draft answer A₀
Step 1: Lexical search → generate draft answer A₁
        If F1(A₀, A₁) ≥ 0.8 → STOP (converged)
        Else → CONTINUE
Step 2+: Continue until convergence or budget exhausted.

Answer similarity is measured by normalized token overlap (F1 score),
identical to the HotpotQA evaluation metric.  This sidesteps the evidence
bundle assessment problem: the LLM implicitly scores the full evidence
bundle by generating from it, and convergence signals that additional
retrieval adds nothing new.

The draft answer is generated via the same model and API as the final
answer, but with a minimal prompt and max_tokens=100 to minimise cost.

Parameters
----------
top_k : int
    Documents per retrieval step.  Default 5.
max_steps : int
    Hard step cap.  Default 5.
convergence_threshold : float
    Minimum F1 between consecutive drafts to declare convergence.  Default 0.8.
"""

from __future__ import annotations

import os
import re
import time
from typing import Optional

import httpx
from openai import OpenAI

from ..evaluation.metrics import f1_score
from ..types import Action, AddressSpaceType, AgentState, Operation
from .base import Policy

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_TOP_K = 5
_DEFAULT_MAX_STEPS = 5
_DEFAULT_CONVERGENCE_THRESHOLD = 0.8

_DRAFT_MODEL = "openai/gpt-oss-120b"
_API_BASE_URL = "https://openrouter.ai/api/v1"
_API_KEY = os.environ.get(
    "OPENROUTER_API_KEY",
    "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",
)
_REQUEST_TIMEOUT = 60.0
_CALL_DELAY = 0.5          # seconds between consecutive API calls (rate-limit guard)
_RETRY_DELAY = 3.0         # seconds before retry on failure
_DRAFT_MAX_TOKENS = 100    # reasoning model needs tokens for thinking
_EVIDENCE_CHARS = 300      # max chars per passage in draft prompt

# Minimal stop words for keyword query construction
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

_DRAFT_PROMPT_TEMPLATE = """\
Evidence: {evidence}
Question: {question}
Answer in 1-5 words:"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _keyword_query(question: str) -> str:
    """Strip stop words and return a focused keyword string."""
    tokens = re.findall(r"\b\w+\b", question.lower())
    keywords = [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]
    return " ".join(keywords) if keywords else question


# ─────────────────────────────────────────────────────────────────────────────
# Draft answer client
# ─────────────────────────────────────────────────────────────────────────────

class _DraftAnswerClient:
    """
    Thin wrapper around OpenRouter for draft answer generation.

    Generates very short (1-5 word) answers to assess answer stability.
    """

    def __init__(self, model: str = _DRAFT_MODEL) -> None:
        self.model = model
        self.client = OpenAI(
            base_url=_API_BASE_URL,
            api_key=_API_KEY,
            timeout=httpx.Timeout(_REQUEST_TIMEOUT),
        )
        self.total_calls: int = 0
        self.total_errors: int = 0

    def generate_draft(self, question: str, workspace_items: list) -> str:
        """
        Generate a short draft answer from current workspace evidence.

        Parameters
        ----------
        question : str
            The natural-language question.
        workspace_items : list[WorkspaceItem]
            Current workspace items to use as evidence.

        Returns
        -------
        str
            Short draft answer string, or empty string on failure.
        """
        if not workspace_items:
            return ""

        # Build evidence text: first _EVIDENCE_CHARS chars of each passage
        evidence_parts = []
        for item in workspace_items:
            content = item.content.strip()
            if content:
                snippet = content[:_EVIDENCE_CHARS]
                evidence_parts.append(snippet)

        if not evidence_parts:
            return ""

        evidence_text = "\n".join(evidence_parts)
        prompt = _DRAFT_PROMPT_TEMPLATE.format(
            evidence=evidence_text,
            question=question,
        )

        # Try primary model, retry on failure
        draft = self._call_api(prompt)
        if draft is None:
            time.sleep(_RETRY_DELAY)
            draft = self._call_api(prompt)

        if draft is None:
            self.total_errors += 1
            return ""

        time.sleep(_CALL_DELAY)
        return draft.strip()

    def _call_api(self, prompt: str) -> Optional[str]:
        """Call the OpenRouter API for a draft answer."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=_DRAFT_MAX_TOKENS,
                temperature=0.0,
            )
            self.total_calls += 1

            choices = response.choices
            if not choices:
                return None

            content = choices[0].message.content
            if content:
                return content

            # Some reasoning models (gpt-oss) put output in reasoning_details
            msg = choices[0].message
            reasoning_details = getattr(msg, "reasoning_details", None)
            if reasoning_details:
                texts = [d.get("text", "") for d in reasoning_details if isinstance(d, dict)]
                combined = " ".join(t.strip() for t in texts if t.strip())
                if combined:
                    return combined

            return ""

        except Exception as exc:
            print(f"    [AnswerStabilityPolicy] API error: {exc}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Policy
# ─────────────────────────────────────────────────────────────────────────────

class AnswerStabilityPolicy(Policy):
    """
    Stops when the draft answer converges (stops changing between steps).

    Step 0: Semantic search → generate draft answer A₀
    Step 1: Lexical search → generate draft answer A₁
            If F1(A₀, A₁) ≥ convergence_threshold → STOP (converged)
            If F1(A₀, A₁) < convergence_threshold → CONTINUE
    Step 2+: Continue until convergence or budget.

    Answer similarity is measured by normalized string matching via the
    HotpotQA F1 metric (lowercase, strip articles/punctuation, token overlap).
    Convergence threshold: F1 ≥ 0.8 between consecutive drafts → STOP.

    Parameters
    ----------
    top_k : int
        Documents per retrieval step.  Default 5.
    max_steps : int
        Hard step cap.  Default 5.
    convergence_threshold : float
        Minimum F1 between consecutive draft answers to declare convergence.
        Default 0.8.
    """

    def __init__(
        self,
        top_k: int = _DEFAULT_TOP_K,
        max_steps: int = _DEFAULT_MAX_STEPS,
        convergence_threshold: float = _DEFAULT_CONVERGENCE_THRESHOLD,
    ) -> None:
        self._top_k = top_k
        self._max_steps = max_steps
        self._convergence_threshold = convergence_threshold
        self._draft_client = _DraftAnswerClient()

        # Per-question state
        self._current_query: str = ""
        self._draft_history: list[str] = []   # draft answers accumulated per question
        self._convergence_step: Optional[int] = None  # step at which convergence was detected

    # ─────────────────────────────────────────────────────────────────────────
    # Policy interface
    # ─────────────────────────────────────────────────────────────────────────

    def name(self) -> str:
        return "pi_answer_stability"

    def select_action(self, state: AgentState) -> Action:
        # Reset per-question state when a new question starts
        if state.step == 0 or state.query != self._current_query:
            self._reset_for_question(state.query)

        # Hard stop conditions
        if state.step >= self._max_steps or state.budget_remaining <= 0.10:
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.STOP,
            )

        # Step 0: always semantic search (no draft yet — workspace is empty)
        if state.step == 0:
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.SEARCH,
                params={"query": state.query, "top_k": self._top_k},
            )

        # Step 1+: generate draft answer from current workspace, then decide
        # NOTE: select_action is called AFTER the previous step's retrieval,
        # so workspace is populated with results from step (state.step - 1).

        draft = self._draft_client.generate_draft(state.query, state.workspace)
        self._draft_history.append(draft)

        # Check convergence: compare to the most recent previous draft
        if len(self._draft_history) >= 2:
            prev_draft = self._draft_history[-2]
            curr_draft = self._draft_history[-1]
            similarity = f1_score(curr_draft, prev_draft)

            if similarity >= self._convergence_threshold:
                self._convergence_step = state.step
                return Action(
                    address_space=AddressSpaceType.SEMANTIC,
                    operation=Operation.STOP,
                )

        # Not yet converged — continue with lexical search (focused keywords)
        kw_query = _keyword_query(state.query)
        return Action(
            address_space=AddressSpaceType.LEXICAL,
            operation=Operation.SEARCH,
            params={
                "query": kw_query,
                "top_k": self._top_k,
            },
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Per-question state management
    # ─────────────────────────────────────────────────────────────────────────

    def _reset_for_question(self, query: str) -> None:
        """Reset all per-question state at the start of a new question."""
        self._current_query = query
        self._draft_history = []
        self._convergence_step = None

    # ─────────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ─────────────────────────────────────────────────────────────────────────

    def get_draft_history(self) -> list[str]:
        """Return the list of draft answers for the current question."""
        return list(self._draft_history)

    def get_convergence_step(self) -> Optional[int]:
        """Return the step at which convergence was detected, or None."""
        return self._convergence_step

    def usage_summary(self) -> dict:
        """Return draft answer API usage statistics."""
        return {
            "total_draft_calls": self._draft_client.total_calls,
            "total_draft_errors": self._draft_client.total_errors,
        }
