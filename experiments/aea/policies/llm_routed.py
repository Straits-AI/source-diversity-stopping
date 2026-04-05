"""
LLM-Routed AEA policy — the LLM itself makes routing decisions.

At each step, the LLM sees:
  - The original question
  - Current workspace contents (retrieved passages, truncated)
  - Available actions (STOP, SEMANTIC_SEARCH, LEXICAL_SEARCH, ENTITY_HOP)
  - How many operations have been used so far

The LLM decides: STOP (evidence sufficient) or which substrate + query to use next.

This is the paper's primary experiment: replacing the hand-designed heuristic
router with an LLM-in-the-loop router that reasons about evidence sufficiency.

API: OpenRouter  (same endpoint as AnswerGenerator)
Model: qwen/qwen3.6-plus:free (spec-specified, intended model)
       Falls back to google/gemma-3-12b-it:free (faster, more reliable free tier)
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


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

# Primary: qwen is the spec-specified model. Fallback: gemma is faster and
# more reliable on the free tier (same as answer_generator.py).
# NOTE: qwen/qwen3.6-plus:free can hang indefinitely on the free tier.
#       We use gemma as primary for reliability; qwen is the intended spec model.
_ROUTER_MODEL = "openai/gpt-oss-120b"   # reasoning model, paid but very cheap (~$0.00002/call)
_ROUTER_SPEC_MODEL = "qwen/qwen3.6-plus:free"  # spec-specified (slow on free tier)
_ROUTER_FALLBACK_MODEL = "meta-llama/llama-3.2-3b-instruct:free"  # second fallback
_API_BASE_URL = "https://openrouter.ai/api/v1"
# Use explicit per-phase timeouts so hung connections are killed after 20s read
_REQUEST_TIMEOUT = httpx.Timeout(connect=10.0, read=20.0, write=10.0, pool=5.0)
_RETRY_DELAY = 2.0        # seconds before fallback retry
_CALL_DELAY = 0.5         # seconds between consecutive calls (paid model, no rate limit concerns)
_DEFAULT_MAX_STEPS = 5    # hard cap per question
_DEFAULT_TOP_K = 5
_CONTENT_PREVIEW_CHARS = 100   # chars per workspace item in the prompt


# ─────────────────────────────────────────────────────────────
# Prompt template
# ─────────────────────────────────────────────────────────────

_ROUTING_PROMPT = """\
You are a retrieval routing agent. Given a question and evidence found so far, decide the next action.

Question: {question}

Evidence found ({n_items} items, {n_ops} operations used):
{workspace_contents}

Available actions:
1. STOP - you have enough evidence to answer the question
2. SEMANTIC_SEARCH - search by meaning (good for paraphrases, concepts)
3. LEXICAL_SEARCH - search by keywords (good for names, IDs, exact terms)
4. ENTITY_HOP - follow entity links from found documents (good for multi-hop: "who directed the movie that...")

Respond with ONLY the action name (e.g., "STOP" or "SEMANTIC_SEARCH" or "ENTITY_HOP"). No explanation."""


# ─────────────────────────────────────────────────────────────
# Action parsing
# ─────────────────────────────────────────────────────────────

# Maps LLM output strings → routing decision strings
_ACTION_KEYWORDS: list[tuple[str, str]] = [
    ("stop", "STOP"),
    ("semantic", "SEMANTIC_SEARCH"),
    ("lexical", "LEXICAL_SEARCH"),
    ("entity", "ENTITY_HOP"),
    ("hop", "ENTITY_HOP"),
]


def _parse_routing_decision(llm_output: str) -> str:
    """
    Parse the LLM response and return one of:
    STOP | SEMANTIC_SEARCH | LEXICAL_SEARCH | ENTITY_HOP

    On failure to parse, returns "STOP" (conservative default).
    """
    text = llm_output.strip().lower()
    # Remove leading numbers/punctuation (e.g. "1. STOP" → "1. stop")
    text = re.sub(r"^\d+[\.\)]\s*", "", text)

    for keyword, decision in _ACTION_KEYWORDS:
        if keyword in text:
            return decision

    # Unparseable → conservative fallback
    return "STOP"


# ─────────────────────────────────────────────────────────────
# Policy
# ─────────────────────────────────────────────────────────────

class LLMRoutedPolicy(Policy):
    """
    Agentic attention: the LLM itself makes routing decisions.

    At each step, the LLM sees:
    - The original question
    - Current workspace contents (retrieved passages, first 100 chars each)
    - Available actions (SEMANTIC_SEARCH, LEXICAL_SEARCH, ENTITY_HOP, STOP)
    - How many operations have been used so far

    The LLM decides: STOP (evidence sufficient) or which substrate to query next.

    Parameters
    ----------
    top_k : int
        Documents to retrieve per step.
    max_steps : int
        Hard cap on steps per question (LLM-independent safety net).
    model : str
        OpenRouter model string.
    """

    def __init__(
        self,
        top_k: int = _DEFAULT_TOP_K,
        max_steps: int = _DEFAULT_MAX_STEPS,
        model: str = _ROUTER_MODEL,
    ) -> None:
        self._top_k = top_k
        self._max_steps = max_steps
        self._model = model

        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        self._client = OpenAI(
            base_url=_API_BASE_URL,
            api_key=api_key,
            timeout=_REQUEST_TIMEOUT,  # already an httpx.Timeout object
        )

        # Usage tracking (accumulated across the whole policy run)
        self.total_routing_calls: int = 0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_api_errors: int = 0

        # Distribution of decisions (for the routing analysis section)
        self.decision_counts: dict[str, int] = {
            "STOP": 0,
            "SEMANTIC_SEARCH": 0,
            "LEXICAL_SEARCH": 0,
            "ENTITY_HOP": 0,
        }

    def name(self) -> str:
        return "pi_llm_routed"

    def select_action(self, state: AgentState) -> Action:
        """
        Call the LLM to decide the next routing action.

        Returns an Action conforming to the Policy base class contract.
        On API error, returns STOP (conservative).
        """
        # Hard caps (bypass LLM call entirely)
        if state.step >= self._max_steps:
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.STOP,
            )
        if state.budget_remaining <= 0.05:
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.STOP,
            )

        # Step 0: always semantic search (need at least one retrieval
        # before the LLM can reason about evidence sufficiency)
        if state.step == 0:
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.SEARCH,
                params={"query": state.query, "top_k": 5},
            )

        # Build prompt
        prompt = self._build_prompt(state)

        # Call LLM
        llm_text = self._call_router(prompt)

        # Parse decision
        decision = _parse_routing_decision(llm_text) if llm_text is not None else "STOP"
        self.decision_counts[decision] = self.decision_counts.get(decision, 0) + 1

        # Rate-limit guard
        time.sleep(_CALL_DELAY)

        # Convert decision to Action
        return self._decision_to_action(decision, state)

    # ─────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────

    def _build_prompt(self, state: AgentState) -> str:
        """Construct the routing prompt from the current agent state."""
        n_ops = len(state.history)

        if state.workspace:
            lines = []
            for item in state.workspace:
                title = item.source_id[:60]
                preview = item.content[:_CONTENT_PREVIEW_CHARS].replace("\n", " ")
                lines.append(f"- [{title}]: {preview}…")
            workspace_contents = "\n".join(lines)
        else:
            workspace_contents = "(none yet)"

        return _ROUTING_PROMPT.format(
            question=state.query,
            n_items=len(state.workspace),
            n_ops=n_ops,
            workspace_contents=workspace_contents,
        )

    def _call_router(self, prompt: str) -> Optional[str]:
        """
        Call the OpenRouter routing model.

        Tries the primary model first (qwen), then falls back to gemma on error.
        Returns the raw text response, or None on persistent error.
        """
        result = self._call_model(prompt, self._model)
        if result is not None:
            return result

        # Primary failed — try fallback after a short pause
        time.sleep(_RETRY_DELAY)
        result = self._call_model(prompt, _ROUTER_FALLBACK_MODEL)
        return result

    def _call_model(self, prompt: str, model: str) -> Optional[str]:
        """Single API call to a specific model. Returns text or None."""
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,     # Reasoning models need tokens for thinking before answering
                temperature=0.0,
            )
            self.total_routing_calls += 1

            usage = getattr(response, "usage", None)
            if usage:
                self.total_prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                self.total_completion_tokens += getattr(usage, "completion_tokens", 0) or 0

            choices = response.choices
            if not choices:
                return None

            content = choices[0].message.content
            if content:
                return content

            # Some reasoning models expose output only in reasoning_details
            msg = choices[0].message
            reasoning_details = getattr(msg, "reasoning_details", None)
            if reasoning_details:
                texts = [d.get("text", "") for d in reasoning_details if isinstance(d, dict)]
                combined = " ".join(t.strip() for t in texts if t.strip())
                return combined if combined else None

            return None

        except Exception as exc:
            print(f"    [LLMRoutedPolicy] Routing API error ({model}): {exc}")
            self.total_api_errors += 1
            return None

    def _decision_to_action(self, decision: str, state: AgentState) -> Action:
        """Convert a parsed routing decision string into an Action."""
        query = state.query

        if decision == "STOP":
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.STOP,
            )
        elif decision == "SEMANTIC_SEARCH":
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.SEARCH,
                params={"query": query, "top_k": self._top_k},
            )
        elif decision == "LEXICAL_SEARCH":
            # Use a keyword-focused form of the query for BM25
            kw_query = _keyword_query(query)
            return Action(
                address_space=AddressSpaceType.LEXICAL,
                operation=Operation.SEARCH,
                params={"query": kw_query, "top_k": self._top_k},
            )
        elif decision == "ENTITY_HOP":
            return Action(
                address_space=AddressSpaceType.ENTITY,
                operation=Operation.HOP,
                params={"query": query, "depth": 1, "top_k": self._top_k},
            )
        else:
            # Unreachable, but be safe
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.STOP,
            )

    def routing_usage_summary(self) -> dict:
        """Return a dict summarising routing API usage and decision distribution."""
        total_decisions = sum(self.decision_counts.values()) or 1
        return {
            "total_routing_calls": self.total_routing_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "total_api_errors": self.total_api_errors,
            "decision_counts": dict(self.decision_counts),
            "decision_pcts": {
                k: round(100.0 * v / total_decisions, 1)
                for k, v in self.decision_counts.items()
            },
        }


# ─────────────────────────────────────────────────────────────
# Helpers (minimal stop-word keyword extraction for LEXICAL)
# ─────────────────────────────────────────────────────────────

_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can",
    "in", "on", "at", "by", "for", "with", "from", "to", "of",
    "and", "or", "but", "so", "yet", "nor", "not",
    "this", "that", "these", "those", "it", "its",
    "what", "which", "who", "whom", "where", "when", "why", "how",
})


def _keyword_query(question: str) -> str:
    """Strip stop words and return a focused keyword string for BM25."""
    tokens = re.findall(r"\b\w+\b", question.lower())
    keywords = [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]
    return " ".join(keywords) if keywords else question
