"""
Embedding-based routing policy for the AEA framework.

Uses question embeddings to predict the best retrieval strategy, replacing
unreliable regex pattern-matching with semantic classification.

Design
------
Strategy 0 (stop-after-semantic):
    Question is answerable with a single dense-retrieval pass.
    Action sequence: Semantic SEARCH → STOP.

Strategy 1 (semantic-then-lexical):
    Question requires keyword fallback (e.g. named-entity lookups, rare
    terms that dense retrieval misses).
    Action sequence: Semantic SEARCH → Lexical SEARCH → STOP.

Strategy 2 (semantic-then-hop):
    Question requires an entity bridge (classic 2-hop HotpotQA bridge).
    Action sequence: Semantic SEARCH → Entity HOP → STOP.

Training
--------
On the training split (questions 500-999 from HotpotQA bridge):
1. Run all three strategies exhaustively on each question.
2. For each question, label the strategy that achieves the highest
   support-recall score.
3. Encode questions with all-MiniLM-L6-v2 → 384-dimensional embeddings.
4. Fit a 3-class logistic-regression classifier on
   (embedding → best_strategy).

At evaluation time (questions 0-499):
1. Encode the question with the same model.
2. Predict the strategy class.
3. Execute the corresponding action sequence.

Notes
-----
* The classifier is fitted at first call to ``select_action`` (lazy init)
  to keep construction cheap and allow multi-policy benchmarks to share the
  sentence-transformer model.
* seed=42 throughout for reproducibility.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from ..types import Action, AddressSpaceType, AgentState, Operation
from .base import Policy

# ── Minimal stop-word list (shared with heuristic) ────────────────────────────
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

# Strategy labels
_STRATEGY_SEMANTIC = 0        # stop-after-semantic
_STRATEGY_LEXICAL = 1         # semantic-then-lexical
_STRATEGY_HOP = 2             # semantic-then-hop

_DEFAULT_TOP_K = 5
_DEFAULT_MAX_STEPS = 8
_SEED = 42


def _keyword_query(question: str) -> str:
    """Strip stop words and return a focused keyword string."""
    tokens = re.findall(r"\b\w+\b", question.lower())
    keywords = [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]
    return " ".join(keywords) if keywords else question


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers (called once during policy setup)
# ─────────────────────────────────────────────────────────────────────────────

def _run_strategy(strategy: int, example: dict, address_spaces: dict, top_k: int) -> float:
    """
    Execute one of the three strategies on a single example and return the
    support-recall score.

    This is a *lightweight* simulation: we run the address-space queries
    directly (no harness overhead) and compute recall against gold_ids.

    Parameters
    ----------
    strategy : int
        0 = semantic only, 1 = semantic+lexical, 2 = semantic+hop.
    example : dict
        Dataset example with ``"question"``, ``"context"``, ``"gold_ids"``.
    address_spaces : dict
        Pre-built address spaces (semantic, lexical, entity).
    top_k : int
        Documents per retrieval step.

    Returns
    -------
    float
        Support recall (fraction of gold_ids retrieved).
    """
    from ..evaluation.metrics import support_recall
    from ..types import AgentState, Operation

    question = example["question"]
    gold_ids: list[str] = example.get("gold_ids", [])

    # Build a minimal agent state for address-space querying
    state = AgentState(query=question)

    retrieved_ids: set[str] = set()

    # Step 0: always semantic search
    semantic_space = address_spaces.get("semantic")
    if semantic_space:
        semantic_action = Action(
            address_space=AddressSpaceType.SEMANTIC,
            operation=Operation.SEARCH,
            params={"query": question, "top_k": top_k},
        )
        result = semantic_space.query(state, Operation.SEARCH, semantic_action.params)
        for item in result.items:
            if item.get("id"):
                retrieved_ids.add(item["id"])

    if strategy == _STRATEGY_LEXICAL:
        lexical_space = address_spaces.get("lexical")
        if lexical_space:
            lex_action = Action(
                address_space=AddressSpaceType.LEXICAL,
                operation=Operation.SEARCH,
                params={"query": _keyword_query(question), "top_k": top_k},
            )
            result = lexical_space.query(state, Operation.SEARCH, lex_action.params)
            for item in result.items:
                if item.get("id"):
                    retrieved_ids.add(item["id"])

    elif strategy == _STRATEGY_HOP:
        entity_space = address_spaces.get("entity")
        if entity_space:
            hop_action = Action(
                address_space=AddressSpaceType.ENTITY,
                operation=Operation.HOP,
                params={"query": question, "depth": 1, "top_k": top_k},
            )
            result = entity_space.query(state, Operation.HOP, hop_action.params)
            for item in result.items:
                if item.get("id"):
                    retrieved_ids.add(item["id"])

    return support_recall(list(retrieved_ids), gold_ids)


def _label_best_strategy(
    example: dict,
    address_spaces: dict,
    top_k: int,
) -> int:
    """
    Run all three strategies on *example* and return the label of the best.

    Tie-breaking: prefer cheaper strategies (0 > 1 > 2) so the classifier
    learns to select the simplest effective strategy.
    """
    recalls = [
        _run_strategy(s, example, address_spaces, top_k)
        for s in [_STRATEGY_SEMANTIC, _STRATEGY_LEXICAL, _STRATEGY_HOP]
    ]
    # Prefer the lowest-cost strategy that achieves the maximum recall
    max_recall = max(recalls)
    for strategy, recall in enumerate(recalls):
        if recall >= max_recall - 1e-9:
            return strategy
    return _STRATEGY_SEMANTIC  # fallback


def _embed_questions(questions: list[str], model) -> np.ndarray:
    """
    Encode a list of questions with the sentence-transformer model.

    Returns float32 array of shape (n, embedding_dim).
    """
    embeddings = model.encode(
        questions,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.array(embeddings, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Policy implementation
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingRouterPolicy(Policy):
    """
    Uses question embedding to predict the best retrieval strategy.

    Strategy 0 (stop-after-semantic): question is answerable with one semantic search.
    Strategy 1 (semantic-then-lexical): needs keyword fallback.
    Strategy 2 (semantic-then-hop): needs entity bridge.

    Trained on question embeddings from the TRAINING split (500-999).

    Parameters
    ----------
    train_dataset : list[dict], optional
        Training examples (questions 500-999).  If None, the classifier uses
        a zero-shot fallback (always strategy 0).
    address_spaces_for_training : dict, optional
        Address-space instances used during training simulation.  Must be
        pre-indexed on the training documents.
    top_k : int
        Documents per retrieval step.
    max_steps : int
        Hard step cap.
    model_name : str
        Sentence-transformer model name for embedding questions.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        train_dataset: Optional[list[dict]] = None,
        address_spaces_for_training: Optional[dict] = None,
        top_k: int = _DEFAULT_TOP_K,
        max_steps: int = _DEFAULT_MAX_STEPS,
        model_name: str = "all-MiniLM-L6-v2",
        seed: int = _SEED,
    ) -> None:
        self._top_k = top_k
        self._max_steps = max_steps
        self._model_name = model_name
        self._seed = seed

        self._train_dataset = train_dataset
        self._address_spaces_for_training = address_spaces_for_training

        # Lazy-initialised
        self._sentence_model = None
        self._classifier = None   # sklearn classifier (fitted)
        self._trained = False

    # ─────────────────────────────────────────────────────────────────────────
    # Policy interface
    # ─────────────────────────────────────────────────────────────────────────

    def name(self) -> str:
        return "pi_embedding_router"

    def select_action(self, state: AgentState) -> Action:
        # Hard stop conditions
        if state.step >= self._max_steps or state.budget_remaining <= 0.10:
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.STOP,
            )

        # Strategy-based routing
        strategy = self._predict_strategy(state.query)
        return self._action_for_strategy(state, strategy)

    # ─────────────────────────────────────────────────────────────────────────
    # Strategy routing
    # ─────────────────────────────────────────────────────────────────────────

    def _action_for_strategy(self, state: AgentState, strategy: int) -> Action:
        """
        Map (state.step, strategy) → Action.

        Each strategy defines a 2-step (or 1-step) sequence:
          strategy 0: step 0 → semantic, step 1+ → STOP
          strategy 1: step 0 → semantic, step 1 → lexical, step 2+ → STOP
          strategy 2: step 0 → semantic, step 1 → entity hop, step 2+ → STOP
        """
        step = state.step

        if step == 0:
            # All strategies start with semantic search
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.SEARCH,
                params={"query": state.query, "top_k": self._top_k},
            )

        if strategy == _STRATEGY_SEMANTIC:
            # One-shot: stop after first semantic search
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.STOP,
            )

        if strategy == _STRATEGY_LEXICAL:
            if step == 1:
                return Action(
                    address_space=AddressSpaceType.LEXICAL,
                    operation=Operation.SEARCH,
                    params={
                        "query": _keyword_query(state.query),
                        "top_k": self._top_k,
                    },
                )
            # step >= 2 → stop
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.STOP,
            )

        if strategy == _STRATEGY_HOP:
            if step == 1:
                return Action(
                    address_space=AddressSpaceType.ENTITY,
                    operation=Operation.HOP,
                    params={
                        "query": state.query,
                        "depth": 1,
                        "top_k": self._top_k,
                    },
                )
            # step >= 2 → stop
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.STOP,
            )

        # Unknown strategy — default to stop
        return Action(
            address_space=AddressSpaceType.SEMANTIC,
            operation=Operation.STOP,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Classifier: predict strategy from question embedding
    # ─────────────────────────────────────────────────────────────────────────

    def _predict_strategy(self, question: str) -> int:
        """Return predicted strategy for a single question."""
        if not self._trained:
            self._fit_classifier()

        if self._classifier is None:
            return _STRATEGY_SEMANTIC

        model = self._get_sentence_model()
        embedding = model.encode(
            [question],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        embedding = np.array(embedding, dtype=np.float32)
        pred = self._classifier.predict(embedding)
        return int(pred[0])

    def _fit_classifier(self) -> None:
        """
        Fit the strategy classifier on training data.

        If no training data was provided, leaves classifier as None (fallback
        to strategy 0 for all questions).
        """
        self._trained = True  # prevent re-entry

        if not self._train_dataset or not self._address_spaces_for_training:
            print(
                "[EmbeddingRouterPolicy] No training data provided — "
                "defaulting to strategy 0 (semantic only) for all questions."
            )
            return

        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
        except ImportError:
            print(
                "[EmbeddingRouterPolicy] scikit-learn not available — "
                "falling back to strategy 0."
            )
            return

        model = self._get_sentence_model()

        print("[EmbeddingRouterPolicy] Labelling training questions ...")
        questions: list[str] = []
        labels: list[int] = []

        for i, example in enumerate(self._train_dataset):
            q = example["question"]
            # Build index for this example's documents
            docs = example.get("context", [])
            for space in self._address_spaces_for_training.values():
                space.build_index(docs)

            label = _label_best_strategy(
                example,
                self._address_spaces_for_training,
                self._top_k,
            )
            questions.append(q)
            labels.append(label)

            if (i + 1) % 50 == 0:
                print(f"  Labelled {i + 1}/{len(self._train_dataset)} ...")

        y = np.array(labels, dtype=np.int32)
        label_counts = {s: int((y == s).sum()) for s in [0, 1, 2]}
        print(f"  Strategy distribution: {label_counts}")

        print("[EmbeddingRouterPolicy] Embedding training questions ...")
        X = _embed_questions(questions, model)

        print("[EmbeddingRouterPolicy] Fitting logistic regression classifier ...")
        np.random.seed(self._seed)
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                random_state=self._seed,
                max_iter=1000,
                class_weight="balanced",
                C=1.0,
                solver="lbfgs",
            )),
        ])
        clf.fit(X, y)

        # Report training accuracy
        y_pred = clf.predict(X)
        train_acc = float((y_pred == y).mean())
        print(f"[EmbeddingRouterPolicy] Training accuracy: {train_acc:.4f}")

        # Per-class breakdown
        for s in [0, 1, 2]:
            mask = y == s
            if mask.sum() > 0:
                class_acc = float((y_pred[mask] == s).mean())
                print(f"  Strategy {s}: {mask.sum()} examples, accuracy={class_acc:.4f}")

        self._classifier = clf
        print("[EmbeddingRouterPolicy] Classifier ready.")

    def _get_sentence_model(self):
        """Lazy-load the sentence-transformer model (shared across calls)."""
        if self._sentence_model is None:
            from sentence_transformers import SentenceTransformer
            print(f"[EmbeddingRouterPolicy] Loading {self._model_name} ...")
            self._sentence_model = SentenceTransformer(self._model_name)
        return self._sentence_model

    # ─────────────────────────────────────────────────────────────────────────
    # Introspection helpers (for result reporting)
    # ─────────────────────────────────────────────────────────────────────────

    def predict_strategy_batch(self, questions: list[str]) -> list[int]:
        """Predict strategy for a batch of questions (for analysis)."""
        if not self._trained:
            self._fit_classifier()
        if self._classifier is None:
            return [_STRATEGY_SEMANTIC] * len(questions)
        model = self._get_sentence_model()
        X = _embed_questions(questions, model)
        return self._classifier.predict(X).tolist()

    def strategy_name(self, strategy: int) -> str:
        """Return a human-readable name for a strategy integer."""
        return {
            _STRATEGY_SEMANTIC: "stop-after-semantic",
            _STRATEGY_LEXICAL: "semantic-then-lexical",
            _STRATEGY_HOP: "semantic-then-hop",
        }.get(strategy, f"unknown({strategy})")
