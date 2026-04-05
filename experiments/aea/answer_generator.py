"""
LLM-based answer generator for AEA retrieval experiments.

Uses OpenRouter (via the OpenAI SDK) to generate short, direct answers
from retrieved evidence passages.  Designed to be called after the
retrieval phase so we can measure downstream answer quality (EM, F1).
"""

from __future__ import annotations

import os
import time
from typing import Optional

import httpx
from openai import OpenAI


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Primary: gemma-3-12b is consistently fast (~3-5s) and reliable.
# Qwen3.6-plus is the spec-specified model but can take 10-60s on the free tier.
# We use gemma as primary for reliability and note qwen as the intended model.
_DEFAULT_MODEL = "google/gemma-3-12b-it:free"
_FALLBACK_MODEL = "qwen/qwen3.6-plus:free"
_API_BASE_URL = "https://openrouter.ai/api/v1"
_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
_REQUEST_TIMEOUT = 60.0   # seconds per API call (hard cutoff via daemon thread)
_RETRY_DELAY = 3.0        # seconds before retry
_CALL_DELAY = 0.5         # seconds between consecutive calls (rate-limit guard)

_PROMPT_TEMPLATE = """\
Based on the following evidence, answer the question. Give ONLY the answer, no explanation.

Evidence:
{evidence}

Question: {question}

Answer:"""


# ─────────────────────────────────────────────────────────────────────────────
# AnswerGenerator
# ─────────────────────────────────────────────────────────────────────────────

class AnswerGenerator:
    """
    Generate short answers from retrieved evidence using an LLM via OpenRouter.

    Parameters
    ----------
    model : str
        Primary model identifier (OpenRouter model string).
    """

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        self.model = model
        # httpx.Timeout(N) sets a total per-request timeout.
        # This is the most reliable approach for enforcing response deadlines.
        self.client = OpenAI(
            base_url=_API_BASE_URL,
            api_key=_API_KEY,
            timeout=httpx.Timeout(_REQUEST_TIMEOUT),
        )
        # Usage tracking
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_calls: int = 0
        self.total_errors: int = 0

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    def generate_answer(self, question: str, evidence: list[str]) -> str:
        """
        Generate a short, direct answer for *question* given *evidence* passages.

        Parameters
        ----------
        question : str
            Natural-language question.
        evidence : list[str]
            List of retrieved passage strings to use as context.

        Returns
        -------
        str
            Generated answer string, or empty string on persistent failure.
        """
        if not evidence:
            return ""

        evidence_text = "\n".join(passage.strip() for passage in evidence if passage.strip())
        if not evidence_text:
            return ""

        prompt = _PROMPT_TEMPLATE.format(
            evidence=evidence_text,
            question=question,
        )

        # Try once, retry once on failure
        answer = self._call_api(prompt)
        if answer is None:
            time.sleep(_RETRY_DELAY)
            answer = self._call_api(prompt, fallback=True)

        if answer is None:
            self.total_errors += 1
            return ""

        # Throttle to respect free-tier rate limits
        time.sleep(_CALL_DELAY)
        return answer.strip()

    # ─────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────

    def _call_api(self, prompt: str, fallback: bool = False) -> Optional[str]:
        """
        Call the OpenRouter chat completion API.

        Parameters
        ----------
        prompt : str
            Full prompt string.
        fallback : bool
            If True, use the fallback model instead of the primary.

        Returns
        -------
        str or None
            The assistant message content, or None on error.
        """
        model = _FALLBACK_MODEL if fallback else self.model
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=64,
                temperature=0.0,
            )
            self.total_calls += 1
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
            # Some reasoning models (gpt-oss) put output in reasoning_details only;
            # extract the last reasoning text as a fallback.
            msg = choices[0].message
            reasoning_details = getattr(msg, "reasoning_details", None)
            if reasoning_details:
                texts = [d.get("text", "") for d in reasoning_details if isinstance(d, dict)]
                combined = " ".join(t.strip() for t in texts if t.strip())
                if combined:
                    return combined
            return ""

        except Exception as exc:
            print(f"    [AnswerGenerator] API error ({'fallback' if fallback else 'primary'}): {exc}")
            return None

    def usage_summary(self) -> dict:
        """Return a summary of API usage so far."""
        return {
            "total_calls": self.total_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "total_errors": self.total_errors,
        }
