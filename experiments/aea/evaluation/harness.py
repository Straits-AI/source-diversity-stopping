"""
Evaluation harness for the AEA framework.

The harness:
  1. Initialises an AgentState per example
  2. Builds address-space indices from the example's documents
  3. Runs a policy until it issues STOP or the token budget is exhausted
  4. Updates the workspace after each action
  5. Scores the final answer and evidence against gold

CONTRACT: This module is IMMUTABLE — do not add optimisation hooks,
learned-scoring callbacks, or any behaviour that is not pure evaluation.
"""

from __future__ import annotations

import random
import time
from copy import deepcopy
from typing import Any, Optional

import numpy as np

from ..address_spaces.base import AddressSpace
from ..types import (
    Action,
    ActionResult,
    AgentState,
    DiscoveryEntry,
    Operation,
    WorkspaceItem,
)
from .metrics import (
    bundle_coverage,
    exact_match,
    f1_score,
    normalize_cost,
    support_precision,
    support_recall,
    utility_at_budget,
)


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

_DEFAULT_MAX_STEPS = 10
_DEFAULT_TOKEN_BUDGET = 4000
_DEFAULT_SEED = 42
_MAX_WORKSPACE_ITEMS = 20       # Hard cap; harness evicts lowest-relevance items
_RELEVANCE_EVICT_THRESHOLD = 0.2


class EvaluationHarness:
    """
    Runs a policy on a dataset and collects metrics.

    CONTRACT: This class is immutable.  All evaluation behaviour is fixed
    so that results are comparable across experiments and paper revisions.

    Parameters
    ----------
    address_spaces : dict[str, AddressSpace]
        Mapping from address-space name to its implementation.  The keys
        must match the ``AddressSpaceType.value`` strings used by policies
        (e.g. ``"semantic"``, ``"lexical"``, ``"entity"``).
    max_steps : int
        Maximum number of policy steps per example.
    token_budget : int
        Total token budget per example.  When consumed, the episode ends.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        address_spaces: dict[str, AddressSpace],
        max_steps: int = _DEFAULT_MAX_STEPS,
        token_budget: int = _DEFAULT_TOKEN_BUDGET,
        seed: int = _DEFAULT_SEED,
    ) -> None:
        self.address_spaces = address_spaces
        self.max_steps = max_steps
        self.token_budget = token_budget
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    def evaluate(self, policy, dataset: list[dict]) -> dict:
        """
        Run *policy* on every example in *dataset* and aggregate metrics.

        Parameters
        ----------
        policy : Policy
            An instance of a class that inherits from
            ``experiments.aea.policies.base.Policy``.
        dataset : list[dict]
            Each example must contain:

            * ``"id"``          — unique identifier string
            * ``"question"``    — natural-language question string
            * ``"answer"``      — gold answer string
            * ``"context"``     — list of ``{"id": str, "content": str, …}``
            * ``"gold_ids"``    — list of supporting document id strings

        Returns
        -------
        dict
            Keys:

            * ``"per_example"``  — list of per-example result dicts
            * ``"aggregated"``   — dict of averaged metric values
            * ``"policy_name"``  — policy.name()
            * ``"n_examples"``   — number of examples evaluated
            * ``"n_errors"``     — number of examples that raised exceptions
        """
        per_example_results: list[dict] = []
        n_errors = 0

        for example in dataset:
            try:
                result = self.evaluate_single(policy, example)
                per_example_results.append(result)
            except Exception as exc:
                n_errors += 1
                per_example_results.append(
                    {
                        "id": example.get("id", "unknown"),
                        "error": str(exc),
                        "exact_match": 0.0,
                        "f1": 0.0,
                        "support_recall": 0.0,
                        "support_precision": 0.0,
                        "bundle_coverage": 0.0,
                        "utility_at_budget": 0.0,
                        "steps_taken": 0,
                        "tokens_used": 0,
                        "operations_used": 0,
                    }
                )

        aggregated = self._aggregate(per_example_results)

        return {
            "per_example": per_example_results,
            "aggregated": aggregated,
            "policy_name": policy.name(),
            "n_examples": len(dataset),
            "n_errors": n_errors,
        }

    def evaluate_single(self, policy, example: dict) -> dict:
        """
        Run *policy* on a single *example* and return a detailed trace.

        Parameters
        ----------
        policy : Policy
            Policy instance.
        example : dict
            Single dataset example (see ``evaluate`` for schema).

        Returns
        -------
        dict
            Per-example metrics and full action trace.  Keys:

            * ``"id"``
            * ``"question"``
            * ``"predicted_answer"``
            * ``"gold_answer"``
            * ``"exact_match"``
            * ``"f1"``
            * ``"support_recall"``
            * ``"support_precision"``
            * ``"bundle_coverage"``
            * ``"utility_at_budget"``
            * ``"steps_taken"``
            * ``"tokens_used"``
            * ``"latency_ms_total"``
            * ``"operations_used"``
            * ``"stopped_reason"``   — "stop_action" | "budget" | "max_steps"
            * ``"trace"``           — list of step dicts
        """
        question = example["question"]
        gold_answer = example.get("answer", "")
        gold_ids: list[str] = example.get("gold_ids", [])
        documents: list[dict] = example.get("context", [])

        # Ensure every document has a stable "id" field
        for i, doc in enumerate(documents):
            if "id" not in doc:
                title = doc.get("title", f"doc_{i}")
                doc["id"] = title

        # Build indices
        for space in self.address_spaces.values():
            space.build_index(documents)

        # Initialise agent state
        state = AgentState(
            query=question,
            budget_remaining=1.0,
        )

        # Counters
        tokens_used = 0
        latency_ms_total = 0.0
        operations_used = 0
        trace: list[dict] = []
        stopped_reason = "max_steps"

        for step_idx in range(self.max_steps):
            state.step = step_idx

            # Budget check
            if tokens_used >= self.token_budget:
                stopped_reason = "budget"
                break

            # Policy selects action
            action: Action = policy.select_action(state)

            if action.operation == Operation.STOP:
                stopped_reason = "stop_action"
                break

            # Execute action in the appropriate address space
            space_name = action.address_space.value
            if space_name not in self.address_spaces:
                # Unknown address space — log and skip
                trace.append(
                    {
                        "step": step_idx,
                        "action": _action_to_dict(action),
                        "result_items": [],
                        "error": f"Unknown address space: {space_name}",
                    }
                )
                continue

            space = self.address_spaces[space_name]
            t0 = time.perf_counter()
            result: ActionResult = space.query(state, action.operation, action.params)
            step_latency = (time.perf_counter() - t0) * 1000

            # Accumulate costs
            tokens_used += result.cost_tokens
            latency_ms_total += step_latency
            operations_used += result.cost_operations

            # Update budget
            state.budget_remaining = max(
                0.0, 1.0 - tokens_used / self.token_budget
            )

            # Update workspace from result
            self._update_workspace(state, result, action, step_idx)

            # Append to history
            history_entry = {
                "step": step_idx,
                "action": _action_to_dict(action),
                "n_items": len(result.items),
                "cost_tokens": result.cost_tokens,
            }
            state.history.append(history_entry)

            # Trace
            trace.append(
                {
                    "step": step_idx,
                    "action": _action_to_dict(action),
                    "result_items": [
                        {"id": it.get("id", ""), "score": it.get("score", 0.0)}
                        for it in result.items
                    ],
                    "workspace_size": len(state.workspace),
                    "tokens_used": tokens_used,
                }
            )

        # Derive answer from workspace
        predicted_answer = self._derive_answer(state, example)

        # Collect retrieved IDs
        retrieved_ids = [item.source_id for item in state.workspace]

        # Score
        em = exact_match(predicted_answer, gold_answer)
        f1 = f1_score(predicted_answer, gold_answer)
        sp_recall = support_recall(retrieved_ids, gold_ids)
        sp_precision = support_precision(retrieved_ids, gold_ids)

        # Bundle coverage: treat gold_ids as requirements
        reqs_total = len(gold_ids)
        reqs_satisfied = len(set(retrieved_ids) & set(gold_ids))
        bc = bundle_coverage(reqs_satisfied, reqs_total) if reqs_total > 0 else 1.0

        # Normalised cost (use token dimension as primary)
        norm_cost = normalize_cost(
            tokens=tokens_used,
            latency_ms=latency_ms_total,
            operations=operations_used,
            max_tokens=self.token_budget,
            max_latency=30_000.0,    # 30 s cap
            max_ops=self.max_steps * 2,
        )

        u_budget = utility_at_budget(
            answer_score=f1,
            evidence_score=sp_recall,
            cost=norm_cost,
        )

        return {
            "id": example.get("id", "unknown"),
            "question": question,
            "predicted_answer": predicted_answer,
            "gold_answer": gold_answer,
            "exact_match": em,
            "f1": f1,
            "support_recall": sp_recall,
            "support_precision": sp_precision,
            "bundle_coverage": bc,
            "utility_at_budget": u_budget,
            "steps_taken": len(trace),
            "tokens_used": tokens_used,
            "latency_ms_total": latency_ms_total,
            "operations_used": operations_used,
            "stopped_reason": stopped_reason,
            "trace": trace,
        }

    # ─────────────────────────────────────────────────────────
    # Workspace management (immutable rules)
    # ─────────────────────────────────────────────────────────

    def _update_workspace(
        self,
        state: AgentState,
        result: ActionResult,
        action: Action,
        step: int,
    ) -> None:
        """
        Add retrieved items to workspace and evict low-relevance items.

        Rules:
        1. Add each result item as a WorkspaceItem if not already present.
        2. If EVICT operation, remove matching items by source_id.
        3. Enforce _MAX_WORKSPACE_ITEMS cap by evicting non-pinned items
           with lowest relevance_score.
        """
        if action.operation == Operation.EVICT:
            evict_ids = set(action.params.get("ids", []))
            state.workspace = [
                item for item in state.workspace
                if item.source_id not in evict_ids or item.pinned
            ]
            return

        # Build set of existing source_ids for dedup
        existing_ids = {item.source_id for item in state.workspace}

        for result_item in result.items:
            doc_id = result_item.get("id", "")
            content = result_item.get("content", "")
            score = float(result_item.get("score", 0.0))

            if doc_id in existing_ids:
                # Update score if the new retrieval is more relevant
                for ws_item in state.workspace:
                    if ws_item.source_id == doc_id and ws_item.relevance_score < score:
                        ws_item.relevance_score = score
                continue

            ws_item = WorkspaceItem(
                content=content,
                source_id=doc_id,
                relevance_score=score,
                pinned=False,
                compressed=False,
                added_at_step=step,
            )
            state.workspace.append(ws_item)
            existing_ids.add(doc_id)

            # Also add to discovery if not yet fully loaded
            already_discovered = any(
                d.source_id == doc_id for d in state.discovery
            )
            if not already_discovered:
                title = result_item.get("title", doc_id)
                snippet = content[:200] if content else ""
                state.discovery.append(
                    DiscoveryEntry(
                        source_id=doc_id,
                        source_type=result_item.get("source_type", "document"),
                        description=snippet,
                        confidence=score,
                        discovered_at_step=step,
                    )
                )

        # Enforce workspace cap: evict non-pinned, lowest relevance first
        if len(state.workspace) > _MAX_WORKSPACE_ITEMS:
            unpinned = [item for item in state.workspace if not item.pinned]
            pinned = [item for item in state.workspace if item.pinned]
            unpinned.sort(key=lambda x: x.relevance_score, reverse=True)
            keep_unpinned = _MAX_WORKSPACE_ITEMS - len(pinned)
            state.workspace = pinned + unpinned[:max(0, keep_unpinned)]

    # ─────────────────────────────────────────────────────────
    # Answer derivation (heuristic — no LLM)
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _derive_answer(state: AgentState, example: dict) -> str:
        """
        Derive a predicted answer string from the agent state.

        Current strategy (no LLM):
          1. Look for an explicit "answer" key set by the policy in the
             last history entry.
          2. Otherwise, concatenate the content of the top-scored workspace
             item.  This is a placeholder — real experiments will use an
             LLM reader over the workspace.

        Parameters
        ----------
        state : AgentState
        example : dict

        Returns
        -------
        str
            Predicted answer string (may be empty if workspace is empty).
        """
        # Check if policy injected an answer via history
        for entry in reversed(state.history):
            if "answer" in entry:
                return str(entry["answer"])

        if not state.workspace:
            return ""

        # Return content of highest-relevance workspace item as a stand-in
        top = max(state.workspace, key=lambda x: x.relevance_score)
        return top.content

    # ─────────────────────────────────────────────────────────
    # Aggregation
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _aggregate(results: list[dict]) -> dict:
        """Compute mean of all float metrics across examples."""
        numeric_keys = [
            "exact_match", "f1", "support_recall", "support_precision",
            "bundle_coverage", "utility_at_budget",
            "steps_taken", "tokens_used", "latency_ms_total", "operations_used",
        ]
        aggregated: dict[str, float] = {}
        for key in numeric_keys:
            values = [r[key] for r in results if key in r and isinstance(r[key], (int, float))]
            aggregated[key] = float(np.mean(values)) if values else 0.0
        return aggregated


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _action_to_dict(action: Action) -> dict:
    """Serialise an Action to a JSON-safe dict for trace logging."""
    return {
        "address_space": action.address_space.value,
        "operation": action.operation.value,
        "params": {
            k: v for k, v in action.params.items()
            if isinstance(v, (str, int, float, bool, list, type(None)))
        },
    }
