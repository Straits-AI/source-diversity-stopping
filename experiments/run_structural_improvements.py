"""
Structural Improvements to the AEA Heuristic Policy.

Three approaches tested on N=500 HotpotQA bridge questions:
  - Training split: questions 500-999 (for threshold grid search)
  - Test split:     questions 0-499   (held-out evaluation)

Approach A: Threshold Optimization
    Collect per-step workspace snapshots from a SINGLE harness run (with all
    stopping disabled), then simulate different (min_items, min_sources,
    min_relevance) thresholds on those snapshots to find the best config.
    Evaluate the best config with a real harness run on the test split.

Approach B: Novelty-Based Stopping
    Stop when new retrievals are redundant (cosine > novelty_threshold with
    existing workspace items) OR when the original coverage condition is met.
    Uses the SemanticAddressSpace's pre-computed embeddings to avoid
    re-encoding on the fly.

Approach C: Dual Structural Signal
    Adds relevance-gap signal: if top-2 relevance gap < gap_threshold
    AND from different sources, stop early.

All policies are purely structural — no LLM calls, no learned models.
Metrics reported with paired t-tests against the original heuristic.

Usage
-----
    python experiments/run_structural_improvements.py

Results saved to: experiments/results/structural_improvements.json
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from scipy import stats  # type: ignore
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# ── Prevent HuggingFace tokenizer deadlocks on macOS ─────────────────────────
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── Make sure project root is on sys.path when run directly ──────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── AEA imports ──────────────────────────────────────────────────────────────
from experiments.aea.address_spaces.semantic import SemanticAddressSpace
from experiments.aea.address_spaces.lexical import LexicalAddressSpace
from experiments.aea.address_spaces.entity_graph import EntityGraphAddressSpace
from experiments.aea.evaluation.harness import EvaluationHarness
from experiments.aea.policies.heuristic import (
    AEAHeuristicPolicy,
    _keyword_query,
    _looks_multi_hop,
)
from experiments.aea.policies.base import Policy
from experiments.aea.types import Action, AddressSpaceType, AgentState, Operation

# Reuse data-loading helpers
from experiments.run_hotpotqa_baselines import (
    load_hotpotqa,
    filter_bridge,
    convert_example,
)

# ── Constants ─────────────────────────────────────────────────────────────────
N_TRAIN = 500       # indices 500-999 of bridge questions (for threshold search)
N_TEST = 500        # indices 0-499 of bridge questions (held-out evaluation)
SEED = 42
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_FILE = RESULTS_DIR / "structural_improvements.json"

# Grid search space for Approach A
A_MIN_ITEMS = [1, 2, 3]
A_MIN_SOURCES = [1, 2]
A_MIN_RELEVANCE = [0.2, 0.3, 0.4, 0.5, 0.6]

# Novelty threshold for Approach B
B_NOVELTY_THRESHOLD = 0.8  # cosine similarity above which a new item is "redundant"

# Relevance gap threshold for Approach C
C_GAP_THRESHOLD = 0.1      # top-1 minus top-2 < this → converged


# ── Shared encoder ────────────────────────────────────────────────────────────

_SHARED_ENCODER = None


def _get_shared_encoder():
    global _SHARED_ENCODER
    if _SHARED_ENCODER is None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        print("  [setup] Loading SentenceTransformer model (once)…")
        _SHARED_ENCODER = SentenceTransformer("all-MiniLM-L6-v2")
        print("  [setup] Model loaded.")
    return _SHARED_ENCODER


def make_address_spaces() -> dict:
    """Instantiate fresh address spaces with shared encoder."""
    sem_space = SemanticAddressSpace(model_name="all-MiniLM-L6-v2")
    sem_space._encoder = _get_shared_encoder()
    return {
        "semantic": sem_space,
        "lexical": LexicalAddressSpace(),
        "entity": EntityGraphAddressSpace(),
    }


# ── Trajectory-collecting policy (for Approach A grid search) ─────────────────

class TrajectoryCollectorPolicy(AEAHeuristicPolicy):
    """
    A modified heuristic that records workspace snapshots at every step but
    never stops early on coverage (coverage stopping is disabled so we collect
    full trajectory data for post-hoc threshold simulation).

    Budget and max_steps limits remain active.
    """

    def __init__(self, top_k: int = 5, max_steps: int = 8) -> None:
        # Pass a very high coverage_threshold so the coverage check never fires
        super().__init__(
            top_k=top_k,
            coverage_threshold=9999.0,  # effectively disabled
            max_steps=max_steps,
        )
        # Per-episode workspace snapshots: list of list of (source_id, relevance_score, step)
        self.episode_snapshots: list[list[tuple[str, float, int]]] = []
        self._current_snapshots: list[tuple[str, float, int]] = []

    def name(self) -> str:
        return "pi_trajectory_collector"

    def select_action(self, state: AgentState) -> Action:
        # Record workspace state BEFORE the action
        snapshot = [
            (item.source_id, item.relevance_score, item.added_at_step)
            for item in state.workspace
        ]
        self._current_snapshots = snapshot
        return super().select_action(state)

    def on_episode_end(self) -> None:
        """Call after each episode to save the trajectory."""
        self.episode_snapshots.append(list(self._current_snapshots))
        self._current_snapshots = []


# ── Post-hoc threshold simulation (Approach A) ───────────────────────────────

def simulate_threshold(
    per_example_traces: list[dict],
    min_items: int,
    min_sources: int,
    min_relevance: float,
    gold_ids_list: list[list[str]],
) -> float:
    """
    Given per-example traces (from a single harness run), simulate stopping
    at the FIRST step where the threshold (min_items/min_sources/min_relevance)
    would have been met.

    Returns mean utility_at_budget across examples (using support_recall as
    a proxy for quality and steps_taken as cost).
    """
    from experiments.aea.evaluation.metrics import (
        support_recall,
        support_precision,
        utility_at_budget,
        normalize_cost,
    )

    ub_vals = []
    for trace_result, gold_ids in zip(per_example_traces, gold_ids_list):
        trace = trace_result.get("trace", [])
        total_steps = trace_result.get("steps_taken", len(trace))
        tokens_used = trace_result.get("tokens_used", 0)
        operations_used = trace_result.get("operations_used", 0)

        # Find the step at which the threshold would fire
        # We need to rebuild workspace state per step from the trace
        # Trace format: list of {step, action, result_items, workspace_size, tokens_used}
        # We use the per_example workspace snapshot at end (from retrieved_ids)
        # For threshold simulation, we approximate by checking workspace items per step

        # Since we don't have per-step full workspace data in the trace,
        # use the final retrieved_ids and the step at which each was added
        # from the trace's result_items.
        workspace: dict[str, float] = {}  # source_id -> best relevance
        stop_step = total_steps
        stop_ops = operations_used
        stop_tokens = tokens_used

        for step_entry in trace:
            step_idx = step_entry["step"]
            for item in step_entry.get("result_items", []):
                src_id = item["id"]
                score = float(item.get("score", 0.0))
                if src_id not in workspace or workspace[src_id] < score:
                    workspace[src_id] = score

            # Check threshold after this step
            high_rel = {sid for sid, sc in workspace.items() if sc >= min_relevance}
            if len(high_rel) >= min_items and len(high_rel) >= min_sources:
                # Would stop here — estimate cost proportionally
                frac = (step_idx + 1) / max(total_steps, 1)
                stop_ops = operations_used * frac
                stop_tokens = tokens_used * frac
                stop_step = step_idx + 1
                break

        retrieved_ids = list(workspace.keys())
        sp_recall = support_recall(retrieved_ids, gold_ids)

        # Normalise cost
        from experiments.aea.evaluation.harness import (
            _DEFAULT_TOKEN_BUDGET,
            _DEFAULT_MAX_STEPS,
        )
        norm_cost = normalize_cost(
            tokens=int(stop_tokens),
            latency_ms=0.0,
            operations=stop_ops,
            max_tokens=_DEFAULT_TOKEN_BUDGET,
            max_latency=30_000.0,
            max_ops=_DEFAULT_MAX_STEPS * 2,
        )
        # Use f1=support_recall as proxy (no LLM answer)
        ub = utility_at_budget(
            answer_score=sp_recall,
            evidence_score=sp_recall,
            cost=norm_cost,
        )
        ub_vals.append(ub)

    return float(np.mean(ub_vals)) if ub_vals else 0.0


# ── Approach A policy (for test evaluation) ───────────────────────────────────

class ThresholdOptimizedPolicy(AEAHeuristicPolicy):
    """AEA heuristic with configurable (min_items, min_sources, min_relevance)."""

    def __init__(
        self,
        min_items: int = 2,
        min_sources: int = 2,
        min_relevance: float = 0.4,
        top_k: int = 5,
        max_steps: int = 6,
        coverage_threshold: float = 0.5,
    ) -> None:
        super().__init__(
            top_k=top_k,
            coverage_threshold=coverage_threshold,
            max_steps=max_steps,
        )
        self._min_items = min_items
        self._min_sources = min_sources
        self._min_relevance = min_relevance

    def name(self) -> str:
        return (
            f"pi_threshold_{self._min_items}_{self._min_sources}_{self._min_relevance}"
        )

    def select_action(self, state: AgentState) -> Action:
        if state.step > 0:
            self._manage_workspace(state)
        if self._should_stop(state):
            return Action(address_space=AddressSpaceType.SEMANTIC, operation=Operation.STOP)
        if state.step == 0:
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.SEARCH,
                params={"query": state.query, "top_k": self._top_k},
            )

        high_rel_items = [i for i in state.workspace if i.relevance_score >= self._min_relevance]
        unique_sources = set(i.source_id for i in high_rel_items)

        if len(high_rel_items) >= self._min_items and len(unique_sources) >= self._min_sources:
            return Action(address_space=AddressSpaceType.SEMANTIC, operation=Operation.STOP)

        if len(high_rel_items) >= 1 and len(unique_sources) <= 1 and _looks_multi_hop(state.query):
            return Action(
                address_space=AddressSpaceType.ENTITY,
                operation=Operation.HOP,
                params={"query": state.query, "depth": 1, "top_k": self._top_k},
            )

        last_action = state.history[-1].get("action", {}) if state.history else {}
        last_space = last_action.get("address_space", "")
        last_n_items = state.history[-1].get("n_items", 0) if state.history else 0

        if last_space == "entity" and last_n_items > 0 and state.step <= 3:
            return Action(
                address_space=AddressSpaceType.ENTITY,
                operation=Operation.HOP,
                params={"query": state.query, "depth": 1, "top_k": self._top_k},
            )

        return Action(
            address_space=AddressSpaceType.LEXICAL,
            operation=Operation.SEARCH,
            params={"query": _keyword_query(state.query), "top_k": self._top_k},
        )


# ── Approach B: Novelty-Based Stopping ───────────────────────────────────────

class NoveltyStoppingPolicy(AEAHeuristicPolicy):
    """
    Stops when new retrievals are redundant OR when the original coverage
    condition is met. Uses the pre-built embeddings from SemanticAddressSpace
    injected at episode start (avoids re-encoding from scratch every step).

    Embeddings are indexed by source_id using the address space's internal
    document index. We access sem_space._embeddings and sem_space._documents
    once per episode to look up pre-computed vectors.
    """

    def __init__(
        self,
        novelty_threshold: float = B_NOVELTY_THRESHOLD,
        min_items: int = 2,
        min_sources: int = 2,
        min_relevance: float = 0.4,
        top_k: int = 5,
        max_steps: int = 6,
        coverage_threshold: float = 0.5,
    ) -> None:
        super().__init__(
            top_k=top_k,
            coverage_threshold=coverage_threshold,
            max_steps=max_steps,
        )
        self._novelty_threshold = novelty_threshold
        self._min_items = min_items
        self._min_sources = min_sources
        self._min_relevance = min_relevance
        # Injected per-episode: source_id -> L2-normed embedding vector
        self._doc_embeddings: dict[str, np.ndarray] = {}

    def name(self) -> str:
        return f"pi_novelty_{self._novelty_threshold}"

    def inject_embeddings(self, sem_space: SemanticAddressSpace) -> None:
        """
        Extract pre-computed L2-normalised embeddings from the SemanticAddressSpace.
        Call this once after the harness has built the index for each episode.
        """
        self._doc_embeddings = {}
        if sem_space._embeddings is None or not sem_space._documents:
            return
        for doc, emb in zip(sem_space._documents, sem_space._embeddings):
            self._doc_embeddings[doc["id"]] = emb

    def _novelty_exhausted(self, state: AgentState) -> bool:
        """
        True if all workspace items added at the last step are redundant
        (cosine > threshold) relative to items added at earlier steps.
        """
        if state.step < 2:
            return False
        last_step = state.step - 1
        last_items = [i for i in state.workspace if i.added_at_step == last_step]
        prev_items = [i for i in state.workspace if i.added_at_step < last_step]

        if not last_items or not prev_items:
            return False

        prev_embs = np.stack([
            self._doc_embeddings[i.source_id]
            for i in prev_items
            if i.source_id in self._doc_embeddings
        ], axis=0)
        if prev_embs.ndim < 2 or prev_embs.shape[0] == 0:
            return False

        for last_item in last_items:
            if last_item.source_id not in self._doc_embeddings:
                continue
            emb = self._doc_embeddings[last_item.source_id]
            sims = prev_embs @ emb
            if float(np.max(sims)) <= self._novelty_threshold:
                return False  # This item is novel → not exhausted

        return True  # All last-step items are redundant

    def select_action(self, state: AgentState) -> Action:
        if state.step == 0:
            self._doc_embeddings = {}  # reset (will be re-injected per episode)

        if state.step > 0:
            self._manage_workspace(state)
        if self._should_stop(state):
            return Action(address_space=AddressSpaceType.SEMANTIC, operation=Operation.STOP)
        if state.step == 0:
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.SEARCH,
                params={"query": state.query, "top_k": self._top_k},
            )

        high_rel_items = [i for i in state.workspace if i.relevance_score >= self._min_relevance]
        unique_sources = set(i.source_id for i in high_rel_items)
        source_diverse = (
            len(high_rel_items) >= self._min_items and len(unique_sources) >= self._min_sources
        )
        novelty_exhausted = self._novelty_exhausted(state)

        if source_diverse or novelty_exhausted:
            return Action(address_space=AddressSpaceType.SEMANTIC, operation=Operation.STOP)

        if len(high_rel_items) >= 1 and len(unique_sources) <= 1 and _looks_multi_hop(state.query):
            return Action(
                address_space=AddressSpaceType.ENTITY,
                operation=Operation.HOP,
                params={"query": state.query, "depth": 1, "top_k": self._top_k},
            )

        last_action = state.history[-1].get("action", {}) if state.history else {}
        last_space = last_action.get("address_space", "")
        last_n_items = state.history[-1].get("n_items", 0) if state.history else 0

        if last_space == "entity" and last_n_items > 0 and state.step <= 3:
            return Action(
                address_space=AddressSpaceType.ENTITY,
                operation=Operation.HOP,
                params={"query": state.query, "depth": 1, "top_k": self._top_k},
            )

        return Action(
            address_space=AddressSpaceType.LEXICAL,
            operation=Operation.SEARCH,
            params={"query": _keyword_query(state.query), "top_k": self._top_k},
        )


class NoveltyHarness(EvaluationHarness):
    """
    Thin wrapper around EvaluationHarness that injects per-episode embeddings
    into a NoveltyStoppingPolicy before the policy runs.
    """

    def __init__(self, novelty_policy: NoveltyStoppingPolicy, **kwargs) -> None:
        super().__init__(**kwargs)
        self._novelty_policy = novelty_policy

    def evaluate_single(self, policy, example: dict) -> dict:
        # Build indices (done inside parent evaluate_single for each example)
        result = super().evaluate_single(policy, example)
        return result

    def _post_build_hook(self, policy, example: dict) -> None:
        """Inject embeddings right after index is built."""
        sem_space = self.address_spaces.get("semantic")
        if sem_space is not None and isinstance(policy, NoveltyStoppingPolicy):
            policy.inject_embeddings(sem_space)

    def evaluate(self, policy, dataset: list[dict]) -> dict:
        """
        Override to inject embeddings per episode.
        Replicate minimal harness logic to inject between build_index and policy.
        """
        per_example_results = []
        n_errors = 0

        for example in dataset:
            try:
                # Build index for this example first
                question = example["question"]
                documents = example.get("context", [])
                for i, doc in enumerate(documents):
                    if "id" not in doc:
                        doc["id"] = doc.get("title", f"doc_{i}")
                for space in self.address_spaces.values():
                    space.build_index(documents)

                # Inject embeddings into novelty policy
                sem_space = self.address_spaces.get("semantic")
                if sem_space is not None and isinstance(policy, NoveltyStoppingPolicy):
                    policy.inject_embeddings(sem_space)

                result = self.evaluate_single(policy, example)
                per_example_results.append(result)
            except Exception as exc:
                n_errors += 1
                per_example_results.append({
                    "id": example.get("id", "unknown"),
                    "error": str(exc),
                    "exact_match": 0.0, "f1": 0.0,
                    "support_recall": 0.0, "support_precision": 0.0,
                    "bundle_coverage": 0.0, "utility_at_budget": 0.0,
                    "steps_taken": 0, "tokens_used": 0, "operations_used": 0,
                })

        aggregated = self._aggregate(per_example_results)
        return {
            "per_example": per_example_results,
            "aggregated": aggregated,
            "policy_name": policy.name(),
            "n_examples": len(dataset),
            "n_errors": n_errors,
        }


# ── Approach C: Dual Structural Signal ───────────────────────────────────────

class DualSignalPolicy(AEAHeuristicPolicy):
    """
    Combines source diversity with relevance-gap convergence:
    STOP if EITHER (diversity condition) OR (top-2 gap < gap_threshold and
    top-2 are from different sources).
    """

    def __init__(
        self,
        gap_threshold: float = C_GAP_THRESHOLD,
        min_items: int = 2,
        min_sources: int = 2,
        min_relevance: float = 0.4,
        top_k: int = 5,
        max_steps: int = 6,
        coverage_threshold: float = 0.5,
    ) -> None:
        super().__init__(
            top_k=top_k,
            coverage_threshold=coverage_threshold,
            max_steps=max_steps,
        )
        self._gap_threshold = gap_threshold
        self._min_items = min_items
        self._min_sources = min_sources
        self._min_relevance = min_relevance

    def name(self) -> str:
        return f"pi_dual_signal_{self._gap_threshold}"

    def _relevance_converged(self, state: AgentState) -> bool:
        eligible = sorted(
            [i for i in state.workspace if i.relevance_score >= self._min_relevance],
            key=lambda x: x.relevance_score,
            reverse=True,
        )
        if len(eligible) < 2:
            return False
        gap = eligible[0].relevance_score - eligible[1].relevance_score
        diff_sources = eligible[0].source_id != eligible[1].source_id
        return gap < self._gap_threshold and diff_sources

    def select_action(self, state: AgentState) -> Action:
        if state.step > 0:
            self._manage_workspace(state)
        if self._should_stop(state):
            return Action(address_space=AddressSpaceType.SEMANTIC, operation=Operation.STOP)
        if state.step == 0:
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.SEARCH,
                params={"query": state.query, "top_k": self._top_k},
            )

        high_rel_items = [i for i in state.workspace if i.relevance_score >= self._min_relevance]
        unique_sources = set(i.source_id for i in high_rel_items)
        source_diverse = (
            len(high_rel_items) >= self._min_items and len(unique_sources) >= self._min_sources
        )
        converged = self._relevance_converged(state)

        if source_diverse or converged:
            return Action(address_space=AddressSpaceType.SEMANTIC, operation=Operation.STOP)

        if len(high_rel_items) >= 1 and len(unique_sources) <= 1 and _looks_multi_hop(state.query):
            return Action(
                address_space=AddressSpaceType.ENTITY,
                operation=Operation.HOP,
                params={"query": state.query, "depth": 1, "top_k": self._top_k},
            )

        last_action = state.history[-1].get("action", {}) if state.history else {}
        last_space = last_action.get("address_space", "")
        last_n_items = state.history[-1].get("n_items", 0) if state.history else 0

        if last_space == "entity" and last_n_items > 0 and state.step <= 3:
            return Action(
                address_space=AddressSpaceType.ENTITY,
                operation=Operation.HOP,
                params={"query": state.query, "depth": 1, "top_k": self._top_k},
            )

        return Action(
            address_space=AddressSpaceType.LEXICAL,
            operation=Operation.SEARCH,
            params={"query": _keyword_query(state.query), "top_k": self._top_k},
        )


# ── Evaluation helpers ────────────────────────────────────────────────────────

def run_policy(policy, dataset: list[dict], seed: int = SEED, harness_cls=None) -> dict:
    """Run a single policy on dataset; return full harness result."""
    address_spaces = make_address_spaces()
    if harness_cls is None:
        harness_cls = EvaluationHarness
    harness_kwargs = dict(
        address_spaces=address_spaces,
        max_steps=10,
        token_budget=4000,
        seed=seed,
    )
    if harness_cls is NoveltyHarness:
        harness = harness_cls(novelty_policy=policy, **harness_kwargs)
    else:
        harness = harness_cls(**harness_kwargs)
    t0 = time.perf_counter()
    result = harness.evaluate(policy, dataset)
    result["runtime_seconds"] = round(time.perf_counter() - t0, 2)
    return result


def extract_ub_values(result: dict) -> list[float]:
    return [ex["utility_at_budget"] for ex in result["per_example"]]


def extract_recall_values(result: dict) -> list[float]:
    return [ex["support_recall"] for ex in result["per_example"]]


def paired_ttest(a: list[float], b: list[float]) -> tuple[float, float]:
    """Paired t-test. Returns (t_stat, p_value)."""
    if _HAS_SCIPY:
        t_stat, p_value = stats.ttest_rel(a, b)
        return float(t_stat), float(p_value)
    else:
        # Manual paired t-test
        diffs = np.array(a) - np.array(b)
        n = len(diffs)
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs, ddof=1)
        if std_diff == 0:
            return 0.0, 1.0
        t_stat = mean_diff / (std_diff / np.sqrt(n))
        # Two-tailed p-value using normal approximation for large n
        from math import erfc, sqrt
        p_value = float(erfc(abs(t_stat) / sqrt(2)))
        return float(t_stat), p_value


# ── Approach A: Post-hoc threshold simulation ─────────────────────────────────

def grid_search_thresholds_posthoc(
    train_result: dict,
    train_gold_ids: list[list[str]],
) -> tuple[dict, list[dict]]:
    """
    Simulate different stopping thresholds on per-example traces from a
    single harness run. O(30 * N) instead of O(30 * N * harness_calls).
    Returns (best_config, all_grid_results).
    """
    from experiments.aea.evaluation.metrics import (
        support_recall,
        utility_at_budget,
        normalize_cost,
    )
    from experiments.aea.evaluation.harness import _DEFAULT_TOKEN_BUDGET

    configs = list(product(A_MIN_ITEMS, A_MIN_SOURCES, A_MIN_RELEVANCE))
    print(f"  Post-hoc grid search: {len(configs)} configurations…")

    per_example = train_result["per_example"]

    grid_results = []
    best_ub = -float("inf")
    best_config = None

    for min_items, min_sources, min_relevance in configs:
        ub_vals = []
        recall_vals = []

        for trace_result, gold_ids in zip(per_example, train_gold_ids):
            trace = trace_result.get("trace", [])
            total_steps = trace_result.get("steps_taken", len(trace))
            tokens_used = trace_result.get("tokens_used", 0)
            ops_used = trace_result.get("operations_used", 0)

            # Replay workspace from trace, find first step meeting threshold
            workspace_scores: dict[str, float] = {}  # source_id -> best score
            stop_frac = 1.0

            for step_entry in trace:
                step_idx = step_entry["step"]
                for item in step_entry.get("result_items", []):
                    sid = item["id"]
                    score = float(item.get("score", 0.0))
                    if sid not in workspace_scores or workspace_scores[sid] < score:
                        workspace_scores[sid] = score

                high_rel = {sid for sid, sc in workspace_scores.items() if sc >= min_relevance}
                if len(high_rel) >= min_items and len(high_rel) >= min_sources:
                    stop_frac = (step_idx + 1) / max(total_steps, 1)
                    break

            retrieved_ids = list(workspace_scores.keys())
            sp_recall = support_recall(retrieved_ids, gold_ids)

            norm_cost = normalize_cost(
                tokens=int(tokens_used * stop_frac),
                latency_ms=0.0,
                operations=ops_used * stop_frac,
                max_tokens=_DEFAULT_TOKEN_BUDGET,
                max_latency=30_000.0,
                max_ops=10 * 2,  # harness default max_steps * 2
            )
            ub = utility_at_budget(
                answer_score=sp_recall,
                evidence_score=sp_recall,
                cost=norm_cost,
            )
            ub_vals.append(ub)
            recall_vals.append(sp_recall)

        mean_ub = float(np.mean(ub_vals))
        mean_recall = float(np.mean(recall_vals))

        grid_results.append({
            "min_items": min_items,
            "min_sources": min_sources,
            "min_relevance": min_relevance,
            "utility_at_budget": mean_ub,
            "support_recall": mean_recall,
        })

        if mean_ub > best_ub:
            best_ub = mean_ub
            best_config = {
                "min_items": min_items,
                "min_sources": min_sources,
                "min_relevance": min_relevance,
            }

        print(
            f"    items={min_items} src={min_sources} rel={min_relevance:.1f} → "
            f"U@B={mean_ub:.4f}  recall={mean_recall:.4f}"
        )

    print(f"  Best config: {best_config}  (train U@B={best_ub:.4f})")
    return best_config, grid_results


# ── Main ──────────────────────────────────────────────────────────────────────

def run_structural_improvements(
    seed: int = SEED,
    results_path: Optional[Path] = RESULTS_FILE,
) -> dict:
    """
    Full structural improvements experiment.

    1. Load 1000 bridge questions (0-499 = test, 500-999 = train).
    2. Run original heuristic on BOTH splits (max_steps=8, no early stop for
       train trajectory collection; normal for test evaluation).
    3. Approach A: post-hoc grid search on train traces; evaluate best on test.
    4. Approach B: novelty stopping on test.
    5. Approach C: dual signal on test.
    6. Paired t-tests vs original.
    7. Save and print.
    """
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print(f"Structural Improvements Experiment  (seed={seed})")
    print("=" * 70)

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    print("\n[Step 1] Loading data…")
    raw_data = load_hotpotqa()
    bridge_1000 = filter_bridge(raw_data, n=1000)
    actual = len(bridge_1000)
    print(f"  Available bridge questions: {actual}")
    n_test = min(N_TEST, actual // 2)
    n_train = min(N_TRAIN, actual - n_test)

    test_raw = bridge_1000[:n_test]
    train_raw = bridge_1000[n_test: n_test + n_train]
    test_dataset = [convert_example(ex) for ex in test_raw]
    train_dataset = [convert_example(ex) for ex in train_raw]
    print(f"  Test  split: {len(test_dataset)} examples (idx 0–{n_test-1})")
    print(f"  Train split: {len(train_dataset)} examples (idx {n_test}–{n_test+n_train-1})")

    train_gold_ids = [ex["gold_ids"] for ex in train_dataset]
    test_gold_ids = [ex["gold_ids"] for ex in test_dataset]

    # ── Step 2: Original heuristic on TEST ────────────────────────────────────
    print("\n[Step 2] Original heuristic (2/2/0.4) on test split…")
    t0 = time.perf_counter()
    original_policy = AEAHeuristicPolicy(top_k=5, coverage_threshold=0.5, max_steps=6)
    original_result = run_policy(original_policy, test_dataset, seed=seed)
    print(
        f"  Original  — U@B={original_result['aggregated']['utility_at_budget']:.4f}  "
        f"recall={original_result['aggregated']['support_recall']:.4f}  "
        f"ops={original_result['aggregated']['operations_used']:.2f}  "
        f"runtime={original_result['runtime_seconds']:.1f}s"
    )

    # ── Step 3: Collect training trajectories (for Approach A grid search) ────
    print("\n[Step 3] Collecting training trajectories for grid search…")
    # Run a version of the heuristic with coverage stopping disabled on train split
    train_collector = AEAHeuristicPolicy(
        top_k=5,
        coverage_threshold=9999.0,  # effectively disables coverage stop
        max_steps=8,
    )
    train_trajectory_result = run_policy(train_collector, train_dataset, seed=seed)
    print(f"  Trajectories collected: {len(train_trajectory_result['per_example'])} examples  "
          f"runtime={train_trajectory_result['runtime_seconds']:.1f}s")

    # ── Step 4: Approach A — Post-hoc grid search ─────────────────────────────
    print("\n[Step 4] Approach A: Post-hoc threshold grid search…")
    best_config_a, grid_results_a = grid_search_thresholds_posthoc(
        train_trajectory_result, train_gold_ids
    )

    print(f"\n  Evaluating best config {best_config_a} on TEST split…")
    policy_a = ThresholdOptimizedPolicy(
        min_items=best_config_a["min_items"],
        min_sources=best_config_a["min_sources"],
        min_relevance=best_config_a["min_relevance"],
        top_k=5,
        max_steps=6,
    )
    result_a = run_policy(policy_a, test_dataset, seed=seed)
    print(
        f"  Approach A — U@B={result_a['aggregated']['utility_at_budget']:.4f}  "
        f"recall={result_a['aggregated']['support_recall']:.4f}  "
        f"ops={result_a['aggregated']['operations_used']:.2f}  "
        f"runtime={result_a['runtime_seconds']:.1f}s"
    )

    # ── Step 5: Approach B — Novelty-Based Stopping ───────────────────────────
    print("\n[Step 5] Approach B: Novelty-based stopping on test split…")
    policy_b = NoveltyStoppingPolicy(
        novelty_threshold=B_NOVELTY_THRESHOLD,
        min_items=2,
        min_sources=2,
        min_relevance=0.4,
        top_k=5,
        max_steps=6,
    )
    result_b = run_policy(policy_b, test_dataset, seed=seed, harness_cls=NoveltyHarness)
    print(
        f"  Approach B — U@B={result_b['aggregated']['utility_at_budget']:.4f}  "
        f"recall={result_b['aggregated']['support_recall']:.4f}  "
        f"ops={result_b['aggregated']['operations_used']:.2f}  "
        f"runtime={result_b['runtime_seconds']:.1f}s"
    )

    # ── Step 6: Approach C — Dual Structural Signal ───────────────────────────
    print("\n[Step 6] Approach C: Dual structural signal on test split…")
    policy_c = DualSignalPolicy(
        gap_threshold=C_GAP_THRESHOLD,
        min_items=2,
        min_sources=2,
        min_relevance=0.4,
        top_k=5,
        max_steps=6,
    )
    result_c = run_policy(policy_c, test_dataset, seed=seed)
    print(
        f"  Approach C — U@B={result_c['aggregated']['utility_at_budget']:.4f}  "
        f"recall={result_c['aggregated']['support_recall']:.4f}  "
        f"ops={result_c['aggregated']['operations_used']:.2f}  "
        f"runtime={result_c['runtime_seconds']:.1f}s"
    )

    # ── Step 7: Paired t-tests ────────────────────────────────────────────────
    print("\n[Step 7] Paired t-tests vs original heuristic (U@B)…")
    orig_ub_vals = extract_ub_values(original_result)
    comparisons = [
        ("Approach A (optimized thresholds)", extract_ub_values(result_a)),
        ("Approach B (novelty stopping)",     extract_ub_values(result_b)),
        ("Approach C (dual signal)",          extract_ub_values(result_c)),
    ]

    stat_tests = []
    for label, vals in comparisons:
        t_stat, p_val = paired_ttest(vals, orig_ub_vals)
        delta = float(np.mean(vals)) - float(np.mean(orig_ub_vals))
        sig = "YES" if p_val < 0.05 else "NO"
        stat_tests.append({
            "label": label,
            "delta_ub": round(delta, 6),
            "t_stat": round(t_stat, 4),
            "p_value": round(p_val, 6),
            "significant": p_val < 0.05,
        })
        print(
            f"  {label}: Δ={delta:+.4f}  t={t_stat:.3f}  "
            f"p={p_val:.4f}  Significant? {sig}"
        )

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"=== Structural Improvements Summary (N={n_test} test examples) ===")
    print("=" * 72)
    orig_ub = original_result["aggregated"]["utility_at_budget"]
    orig_recall = original_result["aggregated"]["support_recall"]
    ub_a = result_a["aggregated"]["utility_at_budget"]
    ub_b = result_b["aggregated"]["utility_at_budget"]
    ub_c = result_c["aggregated"]["utility_at_budget"]

    header = (
        f"| {'Policy':<34} "
        f"| {'U@B':>8} "
        f"| {'Recall':>8} "
        f"| {'Ops':>6} "
        f"| {'vs Orig':>9} |"
    )
    sep = f"| {'-'*34} | {'-'*8} | {'-'*8} | {'-'*6} | {'-'*9} |"
    print(header)
    print(sep)

    rows = [
        ("Original (2/2/0.4)",
         orig_ub, orig_recall,
         original_result["aggregated"]["operations_used"], "—"),
        (f"Approach A ({best_config_a['min_items']}/{best_config_a['min_sources']}/{best_config_a['min_relevance']})",
         ub_a, result_a["aggregated"]["support_recall"],
         result_a["aggregated"]["operations_used"],
         f"{ub_a - orig_ub:+.4f}"),
        (f"Approach B (novelty>{B_NOVELTY_THRESHOLD})",
         ub_b, result_b["aggregated"]["support_recall"],
         result_b["aggregated"]["operations_used"],
         f"{ub_b - orig_ub:+.4f}"),
        (f"Approach C (gap<{C_GAP_THRESHOLD})",
         ub_c, result_c["aggregated"]["support_recall"],
         result_c["aggregated"]["operations_used"],
         f"{ub_c - orig_ub:+.4f}"),
    ]

    for plabel, ub, rec, ops, vs_orig in rows:
        print(
            f"| {plabel:<34} "
            f"| {ub:>8.4f} "
            f"| {rec:>8.4f} "
            f"| {ops:>6.2f} "
            f"| {vs_orig:>9} |"
        )
    print()

    # ── Build and save results ────────────────────────────────────────────────
    full_results = {
        "experiment": "structural_improvements",
        "seed": seed,
        "n_test": n_test,
        "n_train": n_train,
        "approach_a": {
            "description": "Threshold optimization (post-hoc grid search on train trajectories)",
            "grid_search_space": {
                "min_items": A_MIN_ITEMS,
                "min_sources": A_MIN_SOURCES,
                "min_relevance": A_MIN_RELEVANCE,
            },
            "best_config": best_config_a,
            "grid_results": grid_results_a,
            "test_aggregated": result_a["aggregated"],
        },
        "approach_b": {
            "description": "Novelty-based stopping (cosine > threshold → redundant)",
            "novelty_threshold": B_NOVELTY_THRESHOLD,
            "test_aggregated": result_b["aggregated"],
        },
        "approach_c": {
            "description": "Dual structural signal (source diversity OR relevance convergence)",
            "gap_threshold": C_GAP_THRESHOLD,
            "test_aggregated": result_c["aggregated"],
        },
        "original_heuristic": {
            "description": "Original AEA heuristic (min_items=2, min_sources=2, min_relevance=0.4)",
            "test_aggregated": original_result["aggregated"],
        },
        "statistical_tests": stat_tests,
        "summary": {
            "original_ub": orig_ub,
            "approach_a_ub": ub_a,
            "approach_b_ub": ub_b,
            "approach_c_ub": ub_c,
            "best_approach": max(
                [("approach_a", ub_a), ("approach_b", ub_b), ("approach_c", ub_c)],
                key=lambda x: x[1],
            )[0],
            "any_significant_improvement": any(
                t["significant"] and t["delta_ub"] > 0 for t in stat_tests
            ),
        },
    }

    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump(full_results, fh, indent=2)
        print(f"Results saved to: {results_path}")

    return full_results


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_structural_improvements()
