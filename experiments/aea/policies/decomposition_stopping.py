"""
Decomposition-based stopping policy for the AEA framework.

Decomposes the question into information requirements using an LLM (once,
at init time per question), then checks if retrieved evidence satisfies
each requirement using simple keyword matching.  Stops when all requirements
are satisfied, or the budget is exhausted.

Design
------
Step 0:  Semantic search to establish anchor documents.
Step 1+: Check which requirements are satisfied by workspace content.
         * If all satisfied → STOP.
         * Otherwise → escalate to lexical (default) or entity search
           depending on which requirements remain unsatisfied.
         * If budget ≤ 10% or max_steps reached → STOP.

The LLM is called ONCE per question at the start of the episode (via
``on_new_question``) and never again during retrieval steps.  This keeps
API costs comparable to using an LLM reader after retrieval, not during it.

Requirements checking uses simple keyword overlap (no LLM needed):
For each requirement phrase, check whether any workspace passage contains
the key non-stop-word tokens of that phrase.

Parameters
----------
top_k : int
    Documents per retrieval step.  Default 5.
max_steps : int
    Hard step cap.  Default 8.
coverage_threshold : float
    Fraction of requirements that must be satisfied before stopping early.
    Default 1.0 (all requirements must be covered).
"""

from __future__ import annotations

import os
import re
import time
from typing import Optional

import httpx
from openai import OpenAI

from ..types import Action, AddressSpaceType, AgentState, Operation
from .base import Policy

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_TOP_K = 5
_DEFAULT_MAX_STEPS = 8
_DEFAULT_COVERAGE_THRESHOLD = 1.0  # Stop only when ALL requirements are satisfied

_DECOMP_MODEL = "openai/gpt-oss-120b"
_API_BASE_URL = "https://openrouter.ai/api/v1"
_API_KEY = os.environ.get(
    "OPENROUTER_API_KEY",
    "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",
)
_REQUEST_TIMEOUT = 60.0
_CALL_DELAY = 0.3   # seconds between consecutive API calls

# Minimal stop words for keyword matching and query construction
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

_DECOMP_PROMPT = """\
Break this question into the specific pieces of information needed to answer it.
List each as a short phrase, one per line.
Do NOT include numbers or bullet points — just bare phrases, one per line.
Be concise: 2-4 phrases maximum.

Question: {question}

Information needed:"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_keywords(phrase: str) -> list[str]:
    """Return non-stop-word tokens from *phrase*, lower-cased."""
    tokens = re.findall(r"\b\w+\b", phrase.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]


def _keyword_query(question: str) -> str:
    """Strip stop words and return a focused keyword string."""
    kws = _extract_keywords(question)
    return " ".join(kws) if kws else question


def _requirement_satisfied_by(requirement: str, workspace_text: str) -> bool:
    """
    Return True if *workspace_text* (concatenation of all workspace passages)
    contains enough keywords from *requirement* to consider it satisfied.

    Criterion: at least 1 keyword from the requirement appears in the text,
    OR all 1-word requirements are found.  For requirements with 2+ keywords,
    we require at least half of them to be present.
    """
    kws = _extract_keywords(requirement)
    if not kws:
        return True  # vacuously satisfied

    ws_lower = workspace_text.lower()
    hits = sum(1 for kw in kws if kw in ws_lower)

    if len(kws) == 1:
        return hits >= 1
    # For multi-keyword requirements, require at least ceil(len/2) hits
    threshold = max(1, (len(kws) + 1) // 2)
    return hits >= threshold


# ─────────────────────────────────────────────────────────────────────────────
# Decomposition client
# ─────────────────────────────────────────────────────────────────────────────

class _DecompositionClient:
    """
    Thin wrapper around the OpenRouter API for decomposition calls only.
    """

    def __init__(self, model: str = _DECOMP_MODEL) -> None:
        self.model = model
        self.client = OpenAI(
            base_url=_API_BASE_URL,
            api_key=_API_KEY,
            timeout=httpx.Timeout(_REQUEST_TIMEOUT),
        )
        self.total_calls: int = 0
        self.total_errors: int = 0

    def decompose(self, question: str) -> list[str]:
        """
        Call the LLM to decompose *question* into requirement phrases.

        Returns a list of requirement strings (may be empty on error).
        """
        prompt = _DECOMP_PROMPT.format(question=question)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.0,
            )
            self.total_calls += 1
            time.sleep(_CALL_DELAY)

            choices = response.choices
            if not choices:
                return []

            content = choices[0].message.content or ""
            if not content:
                # Some reasoning models put output in reasoning_details
                msg = choices[0].message
                reasoning_details = getattr(msg, "reasoning_details", None)
                if reasoning_details:
                    texts = [d.get("text", "") for d in reasoning_details if isinstance(d, dict)]
                    content = " ".join(t.strip() for t in texts if t.strip())

            return self._parse_requirements(content)

        except Exception as exc:
            print(f"    [DecompositionClient] API error: {exc}")
            self.total_errors += 1
            return []

    @staticmethod
    def _parse_requirements(raw: str) -> list[str]:
        """
        Parse a newline-separated list of requirement phrases from raw LLM output.

        Strips leading numbering/bullets and empty lines.
        """
        requirements: list[str] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            # Remove leading numbering like "1.", "1)", "-", "*"
            line = re.sub(r"^[\d]+[.)]\s*", "", line)
            line = re.sub(r"^[-*•]\s*", "", line)
            line = line.strip()
            if line:
                requirements.append(line)
        return requirements


# ─────────────────────────────────────────────────────────────────────────────
# Policy
# ─────────────────────────────────────────────────────────────────────────────

class DecompositionStoppingPolicy(Policy):
    """
    Decomposes the question into information requirements using an LLM (once),
    then checks if retrieved evidence satisfies each requirement using string matching.
    Stops when all requirements are covered.

    Routing sequence
    ----------------
    Step 0  → SemanticAddressSpace SEARCH (anchor documents)
    Step 1+ → Check coverage:
               * All requirements satisfied → STOP
               * Unsatisfied requirements remain → route next search to best
                 address space to fill the gap:
                   - If unsatisfied requirements look entity-like (proper nouns,
                     short bridging phrases) → EntityAddressSpace HOP
                   - Otherwise → LexicalAddressSpace SEARCH with focused keywords
    Any step: budget ≤ 10% or step ≥ max_steps → STOP

    Parameters
    ----------
    top_k : int
        Documents per retrieval step.
    max_steps : int
        Hard step cap.
    coverage_threshold : float
        Fraction of requirements that must be satisfied before stopping early.
        Default 1.0 (all requirements).
    """

    def __init__(
        self,
        top_k: int = _DEFAULT_TOP_K,
        max_steps: int = _DEFAULT_MAX_STEPS,
        coverage_threshold: float = _DEFAULT_COVERAGE_THRESHOLD,
    ) -> None:
        self._top_k = top_k
        self._max_steps = max_steps
        self._coverage_threshold = coverage_threshold
        self._client = _DecompositionClient()

        # Per-question state (reset in on_new_question / first step call)
        self._current_query: str = ""
        self._requirements: list[str] = []
        self._decomposition_done: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    # Policy interface
    # ─────────────────────────────────────────────────────────────────────────

    def name(self) -> str:
        return "pi_decomposition"

    def select_action(self, state: AgentState) -> Action:
        # ── Reset per-question state when a new question starts ───────────────
        if state.step == 0 or state.query != self._current_query:
            self._reset_for_question(state.query)

        # ── Hard stop conditions ──────────────────────────────────────────────
        if state.step >= self._max_steps or state.budget_remaining <= 0.10:
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.STOP,
            )

        # ── Step 0: semantic entry (always) ──────────────────────────────────
        if state.step == 0:
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.SEARCH,
                params={"query": state.query, "top_k": self._top_k},
            )

        # ── Step 1+: coverage check then route ───────────────────────────────
        if not self._requirements:
            # Decomposition returned nothing — fall back to heuristic stopping
            return self._fallback_action(state)

        unsatisfied = self._unsatisfied_requirements(state)

        # Stop if coverage threshold met
        n_total = len(self._requirements)
        n_satisfied = n_total - len(unsatisfied)
        coverage = n_satisfied / n_total if n_total > 0 else 1.0
        if coverage >= self._coverage_threshold:
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.STOP,
            )

        # Route based on what's missing
        return self._route_for_unsatisfied(state, unsatisfied)

    # ─────────────────────────────────────────────────────────────────────────
    # Per-question initialisation
    # ─────────────────────────────────────────────────────────────────────────

    def _reset_for_question(self, query: str) -> None:
        """Decompose the question and cache requirements."""
        self._current_query = query
        self._decomposition_done = False
        self._requirements = []

        reqs = self._client.decompose(query)
        if reqs:
            self._requirements = reqs
            self._decomposition_done = True
            print(
                f"    [DecompositionStopping] Requirements for: {query[:60]}..."
                if len(query) > 60 else
                f"    [DecompositionStopping] Requirements for: {query}"
            )
            for i, r in enumerate(reqs, 1):
                print(f"      {i}. {r}")
        else:
            print(f"    [DecompositionStopping] Decomposition failed — using fallback.")

    # ─────────────────────────────────────────────────────────────────────────
    # Coverage checking
    # ─────────────────────────────────────────────────────────────────────────

    def _workspace_text(self, state: AgentState) -> str:
        """Concatenate all workspace passages into a single string."""
        return " ".join(item.content for item in state.workspace)

    def _unsatisfied_requirements(self, state: AgentState) -> list[str]:
        """Return requirements not yet covered by workspace content."""
        ws_text = self._workspace_text(state)
        return [
            req for req in self._requirements
            if not _requirement_satisfied_by(req, ws_text)
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # Routing logic
    # ─────────────────────────────────────────────────────────────────────────

    def _route_for_unsatisfied(
        self,
        state: AgentState,
        unsatisfied: list[str],
    ) -> Action:
        """
        Choose the best address space to satisfy the remaining requirements.

        Heuristic:
        - If any unsatisfied requirement looks entity-like (mostly proper nouns /
          short 1-2 token phrases that likely name a specific entity), and we have
          not already done an entity hop at this step depth, use entity HOP.
        - Otherwise use lexical search with a keyword query built from the
          unsatisfied requirements.
        """
        # Check if last action was already an entity hop (avoid repeated hops)
        last_space = ""
        if state.history:
            last_action = state.history[-1].get("action", {})
            last_space = last_action.get("address_space", "")

        # Build keyword query from unsatisfied requirements
        combined = " ".join(unsatisfied)
        kw_query = _keyword_query(combined)
        if not kw_query:
            kw_query = _keyword_query(state.query)

        # Decide between entity and lexical
        use_entity = (
            last_space != "entity"
            and state.step <= 3
            and self._looks_entity_bridge(unsatisfied)
        )

        if use_entity:
            return Action(
                address_space=AddressSpaceType.ENTITY,
                operation=Operation.HOP,
                params={
                    "query": kw_query,
                    "depth": 1,
                    "top_k": self._top_k,
                },
            )

        return Action(
            address_space=AddressSpaceType.LEXICAL,
            operation=Operation.SEARCH,
            params={
                "query": kw_query,
                "top_k": self._top_k,
            },
        )

    @staticmethod
    def _looks_entity_bridge(requirements: list[str]) -> bool:
        """
        Return True if any of the unsatisfied requirements looks like an
        entity-bridging phrase (short phrase that names a specific thing).

        Heuristic: short phrase (1-3 tokens) where most tokens are
        capitalised — i.e. proper nouns.
        """
        for req in requirements:
            words = req.split()
            if not words:
                continue
            # A 1-3 word phrase where ≥ 50% words are Title-cased suggests entity
            n_title = sum(1 for w in words if w and w[0].isupper())
            if 1 <= len(words) <= 3 and n_title / len(words) >= 0.5:
                return True
        return False

    def _fallback_action(self, state: AgentState) -> Action:
        """
        Fallback when no requirements are available.

        Uses workspace coverage heuristic identical to AEAHeuristicPolicy.
        """
        high_rel = [i for i in state.workspace if i.relevance_score >= 0.4]
        unique_src = {i.source_id for i in high_rel}
        if len(high_rel) >= 2 and len(unique_src) >= 2:
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.STOP,
            )

        return Action(
            address_space=AddressSpaceType.LEXICAL,
            operation=Operation.SEARCH,
            params={
                "query": _keyword_query(state.query),
                "top_k": self._top_k,
            },
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ─────────────────────────────────────────────────────────────────────────

    def usage_summary(self) -> dict:
        """Return decomposition API usage statistics."""
        return {
            "total_decomposition_calls": self._client.total_calls,
            "total_decomposition_errors": self._client.total_errors,
        }
