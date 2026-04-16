"""
Learned stopping policy for the AEA framework.

Uses a trained classifier to decide when to stop retrieval, replacing the
hand-coded coverage threshold in AEAHeuristicPolicy.

The classifier predicts whether the current workspace state represents the
optimal stopping point, based on features learned from trajectory data.

Design
------
Step 0:  Semantic search (same as all policies — establish anchor documents).
Step 1+: Extract features from workspace state, ask classifier predict stop/continue.
         If continue: use lexical search (keyword-focused).
         If stop or budget low: issue STOP.

The key insight: the classifier learns the stopping threshold from trajectory
data, replacing the hand-coded "2+ items from 2+ sources" rule.
"""

from __future__ import annotations

import os
import pickle
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from ..types import Action, AddressSpaceType, AgentState, Operation
from .base import Policy

# ── Default paths ─────────────────────────────────────────────────────────────
_MODELS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "experiments" / "models"
_DEFAULT_MODEL_PATH = _MODELS_DIR / "stopping_classifier_clean.pkl"

# Minimal stop words for keyword query
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

FEATURE_NAMES = [
    "n_workspace_items",
    "max_relevance",
    "mean_relevance",
    "min_relevance",
    "n_unique_sources",
    "relevance_diversity",
    "step_number",
    "new_items_added",
    "max_relevance_improvement",
]

_DEFAULT_TOP_K = 5
_DEFAULT_MAX_STEPS = 8


def _keyword_query(question: str) -> str:
    """Strip stop words and return a focused keyword string."""
    tokens = re.findall(r"\b\w+\b", question.lower())
    keywords = [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]
    return " ".join(keywords) if keywords else question


class LearnedStoppingPolicy(Policy):
    """
    Uses a trained classifier to decide when to stop.

    Step 0: Semantic search (same as all policies)
    Step 1+: Extract features from workspace -> classifier predicts stop/continue
    If continue: use lexical search (simplest substrate selection)

    The key insight: the classifier learns the STOPPING threshold from data,
    replacing the hand-coded "2+ items from 2+ sources" rule.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to the saved model pickle bundle.  Defaults to
        experiments/models/stopping_classifier_clean.pkl.
    top_k : int
        Documents per retrieval step.  Default 5.
    max_steps : int
        Hard step cap.  Default 8.
    threshold : float, optional
        Probability threshold for stopping.  If None, uses the threshold
        saved in the model bundle (tuned on validation set).
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        top_k: int = _DEFAULT_TOP_K,
        max_steps: int = _DEFAULT_MAX_STEPS,
        threshold: Optional[float] = None,
    ) -> None:
        self._top_k = top_k
        self._max_steps = max_steps
        self._prev_max_relevance: float = 0.0
        self._last_workspace_size: int = 0

        # Load model at init time
        path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        self._model_bundle = self._load_model(path)
        self._classifier = self._model_bundle["model"]
        self._feature_names = self._model_bundle.get("feature_names", FEATURE_NAMES)

        # Use provided threshold, or the one optimised during training
        if threshold is not None:
            self._threshold = threshold
        else:
            self._threshold = self._model_bundle.get("threshold", 0.5)

    # ─────────────────────────────────────────────────────────────────────────
    # Policy interface
    # ─────────────────────────────────────────────────────────────────────────

    def name(self) -> str:
        return "pi_learned_stop"

    def select_action(self, state: AgentState) -> Action:
        # Hard stop conditions
        if state.step >= self._max_steps or state.budget_remaining <= 0.10:
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.STOP,
            )

        # Step 0: always semantic search to establish anchor documents
        if state.step == 0:
            self._prev_max_relevance = 0.0
            self._last_workspace_size = 0
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.SEARCH,
                params={"query": state.query, "top_k": self._top_k},
            )

        # Step 1+: ask the classifier whether to stop
        features = self._extract_features(state)
        should_stop = self._predict_stop(features)

        if should_stop:
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.STOP,
            )

        # Continue: use lexical search with focused keywords
        return Action(
            address_space=AddressSpaceType.LEXICAL,
            operation=Operation.SEARCH,
            params={
                "query": _keyword_query(state.query),
                "top_k": self._top_k,
            },
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Feature extraction
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_features(self, state: AgentState) -> np.ndarray:
        """Extract features from current workspace state."""
        ws = state.workspace
        if not ws:
            features = {
                "n_workspace_items": 0,
                "max_relevance": 0.0,
                "mean_relevance": 0.0,
                "min_relevance": 0.0,
                "n_unique_sources": 0,
                "relevance_diversity": 0.0,
                "step_number": state.step,
                "new_items_added": 0,
                "max_relevance_improvement": 0.0,
            }
        else:
            scores = [item.relevance_score for item in ws]
            max_rel = float(max(scores))
            mean_rel = float(np.mean(scores))
            min_rel = float(min(scores))
            std_rel = float(np.std(scores))
            n_sources = len(set(item.source_id for item in ws))
            new_items = sum(1 for item in ws if item.added_at_step == state.step)

            features = {
                "n_workspace_items": len(ws),
                "max_relevance": max_rel,
                "mean_relevance": mean_rel,
                "min_relevance": min_rel,
                "n_unique_sources": n_sources,
                "relevance_diversity": std_rel,
                "step_number": state.step,
                "new_items_added": new_items,
                "max_relevance_improvement": max_rel - self._prev_max_relevance,
            }
            self._prev_max_relevance = max_rel

        row = [features.get(fname, 0.0) for fname in self._feature_names]
        return np.array(row, dtype=np.float32).reshape(1, -1)

    # ─────────────────────────────────────────────────────────────────────────
    # Classifier prediction
    # ─────────────────────────────────────────────────────────────────────────

    def _predict_stop(self, features: np.ndarray) -> bool:
        """Return True if the classifier recommends stopping."""
        try:
            prob_stop = self._classifier.predict_proba(features)[0, 1]
            return bool(prob_stop >= self._threshold)
        except Exception:
            # Fallback: use default predict
            try:
                pred = self._classifier.predict(features)[0]
                return bool(pred == 1)
            except Exception:
                return False

    # ─────────────────────────────────────────────────────────────────────────
    # Model loading
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_model(path: Path) -> dict:
        """Load the model bundle from disk."""
        if not path.exists():
            raise FileNotFoundError(
                f"Stopping classifier not found at {path}. "
                "Run experiments/train_stopping_model.py first."
            )
        with open(path, "rb") as fh:
            bundle = pickle.load(fh)
        print(f"[LearnedStoppingPolicy] Loaded classifier from {path}")
        print(f"  Model type: {bundle.get('model_name', 'unknown')}")
        print(f"  Threshold:  {bundle.get('threshold', 0.5):.3f}")
        metrics = bundle.get("eval_metrics", {})
        if metrics:
            print(f"  Test F1:    {metrics.get('f1', '?'):.4f}")
        return bundle
