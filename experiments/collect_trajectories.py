"""
Trajectory collector for the learned stopping classifier.

Runs a FULL retrieval trace (all substrates, no early stopping) on
HotpotQA bridge questions 200-700 (training split — no overlap with the
eval split that uses questions 0-100).

At each step t records:
  - Features: observable workspace state at step t
  - Labels: oracle utility computed post-hoc

Output: experiments/data/trajectories.json
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np

# ── Project root on sys.path ─────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.run_hotpotqa_baselines import (
    load_hotpotqa,
    filter_bridge,
    convert_example,
)
from experiments.aea.address_spaces.semantic import SemanticAddressSpace
from experiments.aea.address_spaces.lexical import LexicalAddressSpace
from experiments.aea.address_spaces.entity_graph import EntityGraphAddressSpace
from experiments.aea.types import (
    Action,
    AddressSpaceType,
    AgentState,
    Operation,
    WorkspaceItem,
    DiscoveryEntry,
)
from experiments.aea.evaluation.metrics import support_recall, utility_at_budget, normalize_cost

# ── Constants ────────────────────────────────────────────────────────────────
SEED = 42
# Training split: questions 200-700 (no overlap with eval 0-100)
TRAIN_START = 200
TRAIN_END = 700
N_TRAIN = 500

OUTPUT_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_FILE = OUTPUT_DIR / "trajectories.json"

MAX_STEPS = 8   # Full trace, no early stopping
TOP_K = 5
TOKEN_BUDGET = 4000
MAX_WORKSPACE_ITEMS = 20
EVICT_THRESHOLD = 0.2

# Substrate sequence for full trace (same pattern as ensemble, repeated)
_SPACE_SEQUENCE = [
    (AddressSpaceType.SEMANTIC, Operation.SEARCH, {}),
    (AddressSpaceType.LEXICAL, Operation.SEARCH, {}),
    (AddressSpaceType.ENTITY, Operation.SEARCH, {}),
]


def _make_address_spaces() -> dict:
    return {
        "semantic": SemanticAddressSpace(model_name="all-MiniLM-L6-v2"),
        "lexical": LexicalAddressSpace(),
        "entity": EntityGraphAddressSpace(),
    }


def _extract_features(state: AgentState, prev_max_relevance: float) -> dict:
    """Extract observable workspace features at the current step."""
    ws = state.workspace
    if not ws:
        return {
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

    scores = [item.relevance_score for item in ws]
    max_rel = float(max(scores))
    mean_rel = float(np.mean(scores))
    min_rel = float(min(scores))
    std_rel = float(np.std(scores))
    n_sources = len(set(item.source_id for item in ws))

    # New items added this step
    new_items = sum(1 for item in ws if item.added_at_step == state.step)

    return {
        "n_workspace_items": len(ws),
        "max_relevance": max_rel,
        "mean_relevance": mean_rel,
        "min_relevance": min_rel,
        "n_unique_sources": n_sources,
        "relevance_diversity": std_rel,
        "step_number": state.step,
        "new_items_added": new_items,
        "max_relevance_improvement": max_rel - prev_max_relevance,
    }


def _update_workspace(state: AgentState, result, action, step: int) -> None:
    """Mirror of harness workspace update logic."""
    if action.operation == Operation.EVICT:
        evict_ids = set(action.params.get("ids", []))
        state.workspace = [
            item for item in state.workspace
            if item.source_id not in evict_ids or item.pinned
        ]
        return

    existing_ids = {item.source_id for item in state.workspace}

    for result_item in result.items:
        doc_id = result_item.get("id", "")
        content = result_item.get("content", "")
        score = float(result_item.get("score", 0.0))

        if doc_id in existing_ids:
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

        already_discovered = any(d.source_id == doc_id for d in state.discovery)
        if not already_discovered:
            state.discovery.append(
                DiscoveryEntry(
                    source_id=doc_id,
                    source_type=result_item.get("source_type", "document"),
                    description=(content[:200] if content else ""),
                    confidence=score,
                    discovered_at_step=step,
                )
            )

    # Enforce workspace cap
    if len(state.workspace) > MAX_WORKSPACE_ITEMS:
        unpinned = [item for item in state.workspace if not item.pinned]
        pinned = [item for item in state.workspace if item.pinned]
        unpinned.sort(key=lambda x: x.relevance_score, reverse=True)
        keep_unpinned = MAX_WORKSPACE_ITEMS - len(pinned)
        state.workspace = pinned + unpinned[:max(0, keep_unpinned)]


def _run_full_trace(example: dict, address_spaces: dict) -> list[dict]:
    """
    Run a full retrieval trace (no stopping) and return per-step snapshots.

    Returns list of dicts, each with keys: step_state_snapshot, n_items.
    """
    question = example["question"]
    documents = example["context"]

    # Ensure every doc has an id
    for i, doc in enumerate(documents):
        if "id" not in doc:
            doc["id"] = doc.get("title", f"doc_{i}")

    # Build indices
    for space in address_spaces.values():
        space.build_index(documents)

    state = AgentState(query=question, budget_remaining=1.0)
    tokens_used = 0
    step_snapshots: list[dict] = []

    for step_idx in range(MAX_STEPS):
        state.step = step_idx

        # Record pre-step state snapshot (workspace state BEFORE this action)
        step_snapshots.append({
            "step": step_idx,
            "workspace_snapshot": deepcopy(state.workspace),
            "n_workspace_items": len(state.workspace),
        })

        if tokens_used >= TOKEN_BUDGET:
            break

        # Pick next substrate (round-robin across all substrates)
        space_type, operation, extra = _SPACE_SEQUENCE[step_idx % len(_SPACE_SEQUENCE)]
        params = {"query": question, "top_k": TOP_K}
        params.update(extra)

        action = Action(
            address_space=space_type,
            operation=operation,
            params=params,
        )

        space_name = space_type.value
        if space_name not in address_spaces:
            continue

        space = address_spaces[space_name]
        result = space.query(state, operation, params)

        tokens_used += result.cost_tokens
        state.budget_remaining = max(0.0, 1.0 - tokens_used / TOKEN_BUDGET)

        _update_workspace(state, result, action, step_idx)

        state.history.append({
            "step": step_idx,
            "action": {"address_space": space_name, "operation": operation.value},
            "n_items": len(result.items),
            "cost_tokens": result.cost_tokens,
        })

    # Final snapshot (after last step)
    step_snapshots.append({
        "step": MAX_STEPS,
        "workspace_snapshot": deepcopy(state.workspace),
        "n_workspace_items": len(state.workspace),
    })

    return step_snapshots


def _compute_step_labels(
    step_snapshots: list[dict],
    gold_ids: list[str],
) -> list[dict]:
    """
    Compute oracle labels for each step.

    For each step t, compute the support_recall and utility we'd get
    if we stopped at that step. Also find the optimal stop step.
    """
    step_utilities = []

    for snap in step_snapshots:
        ws = snap["workspace_snapshot"]
        step = snap["step"]

        retrieved_ids = [item.source_id for item in ws]
        recall_t = support_recall(retrieved_ids, gold_ids)

        # Simulate cost: normalise step number as proxy for ops cost
        ops_cost = normalize_cost(
            tokens=step * (TOKEN_BUDGET // MAX_STEPS),  # approximate tokens
            latency_ms=step * 100.0,
            operations=step,
            max_tokens=TOKEN_BUDGET,
            max_latency=MAX_STEPS * 100.0,
            max_ops=MAX_STEPS,
        )

        # Use recall as both answer score proxy and evidence score
        # (no LLM in trajectory collection)
        utility_t = utility_at_budget(
            answer_score=recall_t,
            evidence_score=recall_t,
            cost=ops_cost,
            eta=0.5,
            mu=0.3,
        )

        step_utilities.append({
            "step": step,
            "support_recall_at_t": recall_t,
            "utility_at_t": utility_t,
        })

    # Find optimal stop step (highest utility)
    if step_utilities:
        best_utility = max(su["utility_at_t"] for su in step_utilities)
        for su in step_utilities:
            su["is_optimal_stop"] = 1 if abs(su["utility_at_t"] - best_utility) < 1e-9 else 0
    else:
        for su in step_utilities:
            su["is_optimal_stop"] = 0

    return step_utilities


def collect_trajectories(
    n_examples: int = N_TRAIN,
    output_file: Path = OUTPUT_FILE,
    seed: int = SEED,
) -> list[dict]:
    """
    Collect trajectories for the training split.

    Returns list of trajectory dicts.
    """
    random.seed(seed)
    np.random.seed(seed)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data — use training split (questions 200-700)
    print("Loading HotpotQA...")
    raw_data = load_hotpotqa()
    all_bridge = [ex for ex in raw_data if ex.get("type") == "bridge"]
    print(f"Total bridge questions: {len(all_bridge)}")

    # Training split: questions 200-700
    train_raw = all_bridge[TRAIN_START:TRAIN_START + n_examples]
    print(f"Training split: questions {TRAIN_START} to {TRAIN_START + len(train_raw)} ({len(train_raw)} examples)")

    dataset = [convert_example(ex) for ex in train_raw]

    # Create shared address spaces once — build_index() will update per example
    print("Initialising address spaces (loading encoder once) ...")
    address_spaces = _make_address_spaces()
    print("  Done.")

    trajectories = []
    n_errors = 0

    for i, example in enumerate(dataset):
        if i % 50 == 0:
            print(f"  Processing example {i}/{len(dataset)} ...")

        try:
            # Run full trace (reuse shared address spaces — build_index resets state)
            step_snapshots = _run_full_trace(example, address_spaces)

            # Compute labels
            step_labels = _compute_step_labels(step_snapshots, example["gold_ids"])

            # Assemble per-step records
            prev_max_relevance = 0.0
            steps = []

            for snap, labels in zip(step_snapshots, step_labels):
                # Build a temporary state to extract features
                tmp_state = AgentState(
                    query=example["question"],
                    workspace=snap["workspace_snapshot"],
                    step=snap["step"],
                )

                features = _extract_features(tmp_state, prev_max_relevance)
                prev_max_relevance = features["max_relevance"]

                steps.append({
                    "features": features,
                    "labels": labels,
                })

            trajectories.append({
                "question_id": example["id"],
                "question": example["question"],
                "gold_ids": example["gold_ids"],
                "steps": steps,
            })

        except Exception as exc:
            n_errors += 1
            print(f"  ERROR on example {i} ({example.get('id', '?')}): {exc}")
            if n_errors > 20:
                print("  Too many errors, stopping early.")
                break

    print(f"\nCollected {len(trajectories)} trajectories ({n_errors} errors)")

    # Save
    print(f"Saving to {output_file} ...")
    with open(output_file, "w", encoding="utf-8") as fh:
        json.dump(trajectories, fh, indent=2)
    print(f"Saved {len(trajectories)} trajectories.")

    return trajectories


if __name__ == "__main__":
    collect_trajectories()
