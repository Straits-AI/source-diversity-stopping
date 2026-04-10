"""
Confidence-gated stopping policy — cheapest possible content-aware stopping.

After the first semantic retrieval step, the policy asks an LLM:
"Can you answer this question from the evidence above?"

  * If YES  → the response IS the draft answer → STOP (1 retrieval op)
  * If NO   → do one lexical retrieval step → STOP regardless (2 retrieval ops)

Total LLM judgment calls: exactly ONE per episode.
Total retrieval operations: 1 or 2.

Design
------
Step 0  → SemanticAddressSpace SEARCH (always)
Step 1  → LLM confidence check (one API call)
          If confident → inject draft answer into history → STOP
          If not confident → LexicalAddressSpace SEARCH
Step 2  → STOP (hard cap)

The confidence prompt is intentionally minimal to keep latency and cost low.
Evidence is truncated to 200 chars per passage.

API: openai/gpt-oss-120b via OpenRouter.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import httpx
from openai import OpenAI

from ..types import Action, AddressSpaceType, AgentState, Operation

# ── Constants ──────────────────────────────────────────────────────────────────

_DEFAULT_TOP_K = 5
_API_BASE_URL = "https://openrouter.ai/api/v1"
_MODEL = "openai/gpt-oss-120b"
_MAX_TOKENS = 150
_CALL_DELAY = 0.5           # seconds between API calls
_REQUEST_TIMEOUT = 60.0     # seconds hard cutoff per call
_EVIDENCE_CHAR_LIMIT = 200  # chars per passage in the confidence prompt

# Signals from the LLM that it cannot answer
_CONTINUE_SIGNALS = frozenset({"need_more", "need more", "insufficient", "cannot"})

_CONFIDENCE_PROMPT = """\
Evidence: {evidence}

Question: {question}

Can you answer this question from the evidence above?
If YES: respond with just the answer (1-5 words).
If NO: respond with exactly "NEED_MORE"."""


class ConfidenceGatedPolicy:
    """
    After first retrieval, asks LLM if it can confidently answer.

    One LLM call total.  Max 2 retrieval steps.

    Parameters
    ----------
    top_k : int
        Documents per retrieval step.
    model : str
        OpenRouter model string for the confidence check.
    api_key : str, optional
        OpenRouter API key.  Falls back to OPENROUTER_API_KEY env var.
    """

    def __init__(
        self,
        top_k: int = _DEFAULT_TOP_K,
        model: str = _MODEL,
        api_key: Optional[str] = None,
    ) -> None:
        self._top_k = top_k
        self._model = model

        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._client = OpenAI(
            base_url=_API_BASE_URL,
            api_key=resolved_key,
            timeout=httpx.Timeout(_REQUEST_TIMEOUT),
        )

        # Per-episode tracking (reset each episode implicitly via history)
        self._confidence_called: dict[int, bool] = {}  # episode_id → called

    def name(self) -> str:
        return "pi_confidence_gated"

    def select_action(self, state: AgentState) -> Action:
        """
        Select the next action.

        Step 0: always semantic search.
        Step 1: LLM confidence check.
                - If confident → inject draft into history, STOP.
                - If not → lexical search.
        Step 2+: STOP (hard cap).
        """
        # ── Step 0: semantic entry ──────────────────────────────────────────
        if state.step == 0:
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.SEARCH,
                params={"query": state.query, "top_k": self._top_k},
            )

        # ── Step 1: LLM confidence check ───────────────────────────────────
        if state.step == 1:
            draft = self._confidence_check(state)
            if draft is not None:
                # Inject draft answer into history so harness._derive_answer
                # picks it up (it reads history entries in reverse for "answer").
                state.history.append({"answer": draft, "step": state.step})
                return Action(
                    address_space=AddressSpaceType.SEMANTIC,
                    operation=Operation.STOP,
                )
            # Not confident → one lexical step
            return Action(
                address_space=AddressSpaceType.LEXICAL,
                operation=Operation.SEARCH,
                params={
                    "query": state.query,
                    "top_k": self._top_k,
                },
            )

        # ── Step 2+: hard stop ─────────────────────────────────────────────
        return Action(
            address_space=AddressSpaceType.SEMANTIC,
            operation=Operation.STOP,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _build_evidence_text(self, state: AgentState) -> str:
        """
        Format workspace passages for the confidence prompt.

        Each passage is truncated to _EVIDENCE_CHAR_LIMIT characters to keep
        the prompt short and cheap.
        """
        snippets = []
        for item in state.workspace:
            snippet = item.content[:_EVIDENCE_CHAR_LIMIT].strip()
            if snippet:
                snippets.append(snippet)
        return "\n---\n".join(snippets) if snippets else "(no evidence)"

    def _confidence_check(self, state: AgentState) -> Optional[str]:
        """
        Ask the LLM whether the current workspace is sufficient to answer.

        Returns
        -------
        str or None
            The draft answer string if the LLM is confident, else None.
        """
        evidence = self._build_evidence_text(state)
        prompt = _CONFIDENCE_PROMPT.format(
            evidence=evidence,
            question=state.query,
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=_MAX_TOKENS,
                temperature=0.0,
            )
            raw = ""
            choices = response.choices
            if choices:
                content = choices[0].message.content
                if content:
                    raw = content.strip()
                else:
                    # gpt-oss-120b may put output in reasoning_details only
                    msg = choices[0].message
                    reasoning_details = getattr(msg, "reasoning_details", None)
                    if reasoning_details:
                        texts = [
                            d.get("text", "")
                            for d in reasoning_details
                            if isinstance(d, dict)
                        ]
                        raw = " ".join(t.strip() for t in texts if t.strip()).strip()

            # Rate-limit guard
            time.sleep(_CALL_DELAY)

            if not raw:
                # Empty response → treat as uncertain → request more retrieval
                return None

            raw_lower = raw.lower()
            for signal in _CONTINUE_SIGNALS:
                if signal in raw_lower:
                    return None

            # Non-NEED_MORE response is the draft answer
            return raw

        except Exception as exc:
            print(f"  [ConfidenceGatedPolicy] API error: {exc}")
            # On error → fall through to lexical step (safe degradation)
            return None
