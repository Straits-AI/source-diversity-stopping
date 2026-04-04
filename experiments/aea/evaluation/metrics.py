"""
Evaluation metrics for the AEA framework.

CONTRACT: This module is IMMUTABLE after initial publication.  Do not add
optimisation hooks, shortcuts, or behaviour flags.  All metrics must match
the definitions in the paper exactly.

All string metrics apply the same normalisation pipeline:
  1. Lowercase
  2. Strip leading/trailing whitespace
  3. Remove articles (a, an, the)
  4. Remove punctuation
  5. Collapse internal whitespace to single spaces

This matches the SQuAD / HotpotQA official evaluation scripts.
"""

from __future__ import annotations

import re
import string
from collections import Counter


# ─────────────────────────────────────────────────────────────
# String normalisation (shared by all text metrics)
# ─────────────────────────────────────────────────────────────

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """
    Normalise *text* for evaluation.

    Steps
    -----
    1. Lowercase
    2. Strip leading/trailing whitespace
    3. Remove articles (a, an, the)
    4. Remove punctuation
    5. Collapse internal whitespace

    Examples
    --------
    >>> _normalize("The Battle of Waterloo!")
    'battle of waterloo'
    >>> _normalize("  an  Apple  ")
    'apple'
    """
    text = text.lower().strip()
    text = _ARTICLES_RE.sub("", text)
    text = text.translate(_PUNCT_TABLE)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _tokenize(text: str) -> list[str]:
    """Split normalised text into tokens."""
    return _normalize(text).split()


# ─────────────────────────────────────────────────────────────
# Answer quality metrics
# ─────────────────────────────────────────────────────────────

def exact_match(prediction: str, gold: str) -> float:
    """
    Exact match after normalisation.

    Returns 1.0 if the normalised strings are identical, 0.0 otherwise.

    Parameters
    ----------
    prediction : str
        The model's predicted answer string.
    gold : str
        The reference (gold) answer string.

    Returns
    -------
    float
        1.0 on match, 0.0 otherwise.

    Examples
    --------
    >>> exact_match("the Battle of Waterloo", "battle of waterloo")
    1.0
    >>> exact_match("Napoleon", "Wellington")
    0.0
    >>> exact_match("yes", "Yes!")
    1.0
    """
    return 1.0 if _normalize(prediction) == _normalize(gold) else 0.0


def f1_score(prediction: str, gold: str) -> float:
    """
    Token-level F1 between prediction and gold answer.

    Computes precision and recall over the multiset of tokens in each
    string after normalisation, then returns the harmonic mean.

    Parameters
    ----------
    prediction : str
        The model's predicted answer string.
    gold : str
        The reference (gold) answer string.

    Returns
    -------
    float
        Token-level F1 in [0, 1].

    Examples
    --------
    >>> round(f1_score("Scott Derrickson is American", "American"), 4)
    0.4
    >>> f1_score("yes", "yes")
    1.0
    >>> f1_score("", "anything")
    0.0
    """
    pred_tokens = _tokenize(prediction)
    gold_tokens = _tokenize(gold)

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    common = pred_counter & gold_counter
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


# ─────────────────────────────────────────────────────────────
# Evidence retrieval metrics
# ─────────────────────────────────────────────────────────────

def support_recall(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    """
    Fraction of gold supporting facts that were retrieved.

    Parameters
    ----------
    retrieved_ids : list[str]
        IDs of all documents / paragraphs in the agent's workspace or
        evidence bundle.
    gold_ids : list[str]
        IDs of the gold supporting facts for the question.

    Returns
    -------
    float
        |retrieved ∩ gold| / |gold|, or 1.0 if gold_ids is empty.

    Examples
    --------
    >>> support_recall(["a", "b", "c"], ["a", "b"])
    1.0
    >>> support_recall(["a"], ["a", "b"])
    0.5
    >>> support_recall([], [])
    1.0
    """
    if not gold_ids:
        return 1.0
    retrieved_set = set(retrieved_ids)
    gold_set = set(gold_ids)
    return len(retrieved_set & gold_set) / len(gold_set)


def support_precision(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    """
    Fraction of retrieved items that are gold supporting facts.

    Parameters
    ----------
    retrieved_ids : list[str]
        IDs of all documents / paragraphs in the agent's workspace or
        evidence bundle.
    gold_ids : list[str]
        IDs of the gold supporting facts for the question.

    Returns
    -------
    float
        |retrieved ∩ gold| / |retrieved|, or 1.0 if retrieved_ids is empty.

    Examples
    --------
    >>> support_precision(["a", "b"], ["a", "b", "c"])
    1.0
    >>> support_precision(["a", "d"], ["a", "b"])
    0.5
    >>> support_precision([], ["a"])
    1.0
    """
    if not retrieved_ids:
        return 1.0
    retrieved_set = set(retrieved_ids)
    gold_set = set(gold_ids)
    return len(retrieved_set & gold_set) / len(retrieved_set)


# ─────────────────────────────────────────────────────────────
# Evidence bundle metrics
# ─────────────────────────────────────────────────────────────

def bundle_coverage(requirements_satisfied: int, requirements_total: int) -> float:
    """
    Fraction of question requirements covered by the evidence bundle.

    Parameters
    ----------
    requirements_satisfied : int
        Number of distinct information requirements fulfilled.
    requirements_total : int
        Total number of distinct information requirements for the question.

    Returns
    -------
    float
        requirements_satisfied / requirements_total, or 1.0 if total is 0.

    Raises
    ------
    ValueError
        If requirements_satisfied > requirements_total, or if either value
        is negative.

    Examples
    --------
    >>> bundle_coverage(2, 2)
    1.0
    >>> bundle_coverage(1, 2)
    0.5
    >>> bundle_coverage(0, 0)
    1.0
    """
    if requirements_total < 0 or requirements_satisfied < 0:
        raise ValueError(
            "requirements_total and requirements_satisfied must be non-negative."
        )
    if requirements_satisfied > requirements_total:
        raise ValueError(
            "requirements_satisfied cannot exceed requirements_total."
        )
    if requirements_total == 0:
        return 1.0
    return requirements_satisfied / requirements_total


# ─────────────────────────────────────────────────────────────
# Composite utility metric
# ─────────────────────────────────────────────────────────────

def utility_at_budget(
    answer_score: float,
    evidence_score: float,
    cost: float,
    eta: float = 0.5,
    mu: float = 0.3,
) -> float:
    """
    Utility@Budget metric as defined in the AEA paper.

    Formula:
        U = AnswerScore * (1 + eta * EvidenceScore) - mu * Cost

    Penalises high retrieval cost and rewards both answer quality and
    evidence quality simultaneously.

    Parameters
    ----------
    answer_score : float
        Answer quality score in [0, 1] (e.g. F1 or exact match).
    evidence_score : float
        Evidence quality score in [0, 1] (e.g. bundle_coverage or
        support_recall).
    cost : float
        Normalised cost in [0, 1] (output of ``normalize_cost``).
    eta : float
        Weight for the evidence bonus term.  Default 0.5.
    mu : float
        Penalty weight for cost.  Default 0.3.

    Returns
    -------
    float
        Utility value.  Can be negative when cost is high and
        answer quality is low.

    Examples
    --------
    >>> round(utility_at_budget(1.0, 1.0, 0.0), 4)
    1.5
    >>> round(utility_at_budget(0.0, 1.0, 1.0), 4)
    -0.3
    >>> round(utility_at_budget(0.93, 1.0, 0.304, eta=0.5, mu=0.3), 4)
    1.2121
    """
    return answer_score * (1.0 + eta * evidence_score) - mu * cost


# ─────────────────────────────────────────────────────────────
# Cost normalisation
# ─────────────────────────────────────────────────────────────

def normalize_cost(
    tokens: int,
    latency_ms: float,
    operations: int,
    max_tokens: int,
    max_latency: float,
    max_ops: int,
) -> float:
    """
    Normalise a raw cost triplet to a scalar in [0, 1].

    The three cost dimensions are normalised independently and averaged
    with equal weights.  Values are clamped to [0, 1] before averaging
    so that over-budget runs do not produce values outside the range.

    Parameters
    ----------
    tokens : int
        Raw token count consumed.
    latency_ms : float
        Raw wall-clock latency in milliseconds.
    operations : int
        Raw number of index / graph lookup operations.
    max_tokens : int
        Budget cap for tokens (maps tokens → 1.0).
    max_latency : float
        Budget cap for latency in ms (maps latency → 1.0).
    max_ops : int
        Budget cap for operations (maps operations → 1.0).

    Returns
    -------
    float
        Normalised cost in [0, 1].

    Raises
    ------
    ValueError
        If any max value is <= 0.

    Examples
    --------
    >>> normalize_cost(2000, 500.0, 5, 4000, 1000.0, 10)
    0.5
    >>> normalize_cost(4000, 1000.0, 10, 4000, 1000.0, 10)
    1.0
    >>> normalize_cost(0, 0.0, 0, 4000, 1000.0, 10)
    0.0
    """
    if max_tokens <= 0 or max_latency <= 0 or max_ops <= 0:
        raise ValueError("max_tokens, max_latency, and max_ops must be > 0.")

    norm_tokens = min(1.0, tokens / max_tokens)
    norm_latency = min(1.0, latency_ms / max_latency)
    norm_ops = min(1.0, operations / max_ops)

    return (norm_tokens + norm_latency + norm_ops) / 3.0
