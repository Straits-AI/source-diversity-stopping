"""
Heterogeneous Benchmark Runner — AEA framework evaluation.

Runs all five policies (SemanticOnly, LexicalOnly, EntityOnly, Ensemble,
AEAHeuristic) on the 100-question heterogeneous benchmark and reports:

  * Overall aggregated metrics
  * Per-task-type breakdown
  * Substrate switching analysis for the AEA heuristic policy

Usage
-----
    python experiments/run_heterogeneous_benchmark.py

or as a module::

    from experiments.run_heterogeneous_benchmark import run_heterogeneous
    results = run_heterogeneous()
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ── Ensure project root is on sys.path ───────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── AEA imports ──────────────────────────────────────────────────────────────
from experiments.aea.address_spaces.entity_graph import EntityGraphAddressSpace
from experiments.aea.address_spaces.lexical import LexicalAddressSpace
from experiments.aea.address_spaces.semantic import SemanticAddressSpace
from experiments.aea.evaluation.harness import EvaluationHarness
from experiments.aea.policies.ensemble import EnsemblePolicy
from experiments.aea.policies.heuristic import AEAHeuristicPolicy
from experiments.aea.policies.single_substrate import (
    EntityOnlyPolicy,
    LexicalOnlyPolicy,
    SemanticOnlyPolicy,
)
from experiments.benchmarks.heterogeneous_benchmark import HeterogeneousBenchmark

# ── Constants ────────────────────────────────────────────────────────────────
SEED = 42
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_FILE = RESULTS_DIR / "heterogeneous_benchmark.json"

_TASK_TYPES = [
    "entity_bridge",
    "implicit_bridge",
    "semantic_computation",
    "low_lexical_overlap",
    "multi_hop_chain",
    "discovery_extraction",
]

_TASK_TYPE_LABELS = {
    "entity_bridge":        "Entity Bridge",
    "implicit_bridge":      "Implicit Bridge",
    "semantic_computation": "Semantic + Computation",
    "low_lexical_overlap":  "Low Lexical Overlap",
    "multi_hop_chain":      "Multi-Hop Chain",
    "discovery_extraction": "Discovery + Extraction",
}


# ─────────────────────────────────────────────────────────────────────────────
# Factory helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_policies() -> list:
    """Instantiate all five baseline policies."""
    return [
        SemanticOnlyPolicy(top_k=5, max_steps=2),
        LexicalOnlyPolicy(top_k=5, max_steps=2),
        EntityOnlyPolicy(top_k=5, max_steps=3),
        EnsemblePolicy(top_k=5, max_steps=3),
        AEAHeuristicPolicy(top_k=5, coverage_threshold=0.5, max_steps=6),
    ]


def make_address_spaces() -> dict:
    """Return fresh address space instances."""
    return {
        "semantic": SemanticAddressSpace(model_name="all-MiniLM-L6-v2"),
        "lexical": LexicalAddressSpace(),
        "entity": EntityGraphAddressSpace(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_dataset(examples: list[dict]) -> list[dict]:
    """
    Convert heterogeneous benchmark examples to the EvaluationHarness schema.

    Harness expects each example to have:
      * ``"id"``       — unique string
      * ``"question"`` — NL question
      * ``"answer"``   — gold answer
      * ``"context"``  — list of ``{"id": str, "content": str, ...}``
      * ``"gold_ids"`` — list of supporting doc id strings
    """
    dataset = []
    for ex in examples:
        # The context docs already have "id" and "content"; harness is happy.
        dataset.append({
            "id":       ex["id"],
            "question": ex["question"],
            "answer":   ex["answer"],
            "context":  ex["context"],
            "gold_ids": ex["gold_ids"],
            # Extra metadata carried through for per-type analysis
            "task_type": ex["task_type"],
            "num_hops":  ex.get("num_hops", 1),
        })
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# Substrate switching analysis helpers
# ─────────────────────────────────────────────────────────────────────────────

def _count_unique_substrates_used(trace: list[dict]) -> int:
    """Count distinct address spaces visited in a policy trace."""
    seen: set[str] = set()
    for step in trace:
        space = step.get("action", {}).get("address_space", "")
        if space:
            seen.add(space)
    return len(seen)


def _aea_substrate_analysis(
    aea_results: list[dict],
    best_single_results_by_id: dict[str, float],
) -> dict:
    """
    Summarise substrate switching behaviour for the AEA heuristic.

    Parameters
    ----------
    aea_results : list[dict]
        Per-example result dicts for the AEA heuristic policy.
    best_single_results_by_id : dict[str, float]
        Maps example id → best Utility@Budget across single-substrate policies.

    Returns
    -------
    dict
        Keys: questions_multi_substrate, questions_aea_wins, avg_substrates
    """
    multi_count = 0
    aea_wins = 0
    total_substrates: list[float] = []

    for ex_result in aea_results:
        if "error" in ex_result:
            continue
        trace = ex_result.get("trace", [])
        n_substrates = _count_unique_substrates_used(trace)
        total_substrates.append(n_substrates)

        if n_substrates > 1:
            multi_count += 1

        ex_id = ex_result["id"]
        best_single = best_single_results_by_id.get(ex_id, 0.0)
        aea_score   = ex_result.get("utility_at_budget", 0.0)
        if aea_score > best_single:
            aea_wins += 1

    return {
        "questions_multi_substrate": multi_count,
        "questions_aea_wins":        aea_wins,
        "avg_substrates":            float(np.mean(total_substrates)) if total_substrates else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Printing helpers
# ─────────────────────────────────────────────────────────────────────────────

_COL_POLICY  = 22
_COL_RECALL  = 14
_COL_PREC    = 14
_COL_OPS     = 9
_COL_UTILITY = 15


def _make_header(include_precision: bool = True) -> tuple[str, str]:
    if include_precision:
        h = (
            f"| {'Policy':<{_COL_POLICY}} "
            f"| {'SupportRecall':>{_COL_RECALL}} "
            f"| {'SupportPrec':>{_COL_PREC}} "
            f"| {'AvgOps':>{_COL_OPS}} "
            f"| {'Utility@Budget':>{_COL_UTILITY}} |"
        )
        sep = (
            f"| {'-'*_COL_POLICY} "
            f"| {'-'*_COL_RECALL} "
            f"| {'-'*_COL_PREC} "
            f"| {'-'*_COL_OPS} "
            f"| {'-'*_COL_UTILITY} |"
        )
    else:
        h = (
            f"| {'Policy':<{_COL_POLICY}} "
            f"| {'SupportRecall':>{_COL_RECALL}} "
            f"| {'AvgOps':>{_COL_OPS}} "
            f"| {'Utility@Budget':>{_COL_UTILITY}} |"
        )
        sep = (
            f"| {'-'*_COL_POLICY} "
            f"| {'-'*_COL_RECALL} "
            f"| {'-'*_COL_OPS} "
            f"| {'-'*_COL_UTILITY} |"
        )
    return h, sep


def _make_row(pname: str, agg: dict, include_precision: bool = True) -> str:
    if include_precision:
        return (
            f"| {pname:<{_COL_POLICY}} "
            f"| {agg['support_recall']:>{_COL_RECALL}.4f} "
            f"| {agg['support_precision']:>{_COL_PREC}.4f} "
            f"| {agg['operations_used']:>{_COL_OPS}.2f} "
            f"| {agg['utility_at_budget']:>{_COL_UTILITY}.4f} |"
        )
    else:
        return (
            f"| {pname:<{_COL_POLICY}} "
            f"| {agg['support_recall']:>{_COL_RECALL}.4f} "
            f"| {agg['operations_used']:>{_COL_OPS}.2f} "
            f"| {agg['utility_at_budget']:>{_COL_UTILITY}.4f} |"
        )


def _print_overall_table(all_results: dict, n: int) -> None:
    print(f"\n{'='*70}")
    print(f"=== Heterogeneous Benchmark Results (N={n}) ===")
    print(f"{'='*70}")
    print("\nOverall:")
    h, sep = _make_header(include_precision=True)
    print(h)
    print(sep)
    for pname, res in all_results.items():
        print(_make_row(pname, res["aggregated"], include_precision=True))
    print()


def _print_per_type_tables(
    all_results: dict,
    dataset: list[dict],
) -> None:
    print("By Task Type:\n")

    for task_type in _TASK_TYPES:
        label = _TASK_TYPE_LABELS[task_type]
        type_ids = {ex["id"] for ex in dataset if ex["task_type"] == task_type}
        n_type = len(type_ids)
        if n_type == 0:
            continue

        print(f"[{label}] (N={n_type})")
        h, sep = _make_header(include_precision=False)
        print(h)
        print(sep)

        for pname, res in all_results.items():
            # Filter per_example to this task type
            type_examples = [
                ex for ex in res["per_example"]
                if ex["id"] in type_ids and "error" not in ex
            ]
            if not type_examples:
                print(f"| {pname:<{_COL_POLICY}} | (no results) |")
                continue

            avg_recall  = float(np.mean([e["support_recall"]    for e in type_examples]))
            avg_ops     = float(np.mean([e["operations_used"]   for e in type_examples]))
            avg_utility = float(np.mean([e["utility_at_budget"] for e in type_examples]))

            agg_mini = {
                "support_recall":    avg_recall,
                "operations_used":   avg_ops,
                "utility_at_budget": avg_utility,
            }
            print(_make_row(pname, agg_mini, include_precision=False))
        print()


def _print_substrate_switching(
    aea_name: str,
    all_results: dict,
    n_total: int,
) -> None:
    print("Substrate Switching Analysis:")

    aea_res = all_results.get(aea_name)
    if aea_res is None:
        print(f"  AEA heuristic ({aea_name}) not found in results.")
        return

    # Single-substrate policy names (all except ensemble and AEA)
    single_names = [k for k in all_results if k not in {aea_name, "pi_ensemble"}]

    # Build best-single utility per example id
    best_single_by_id: dict[str, float] = {}
    for pname in single_names:
        for ex in all_results[pname]["per_example"]:
            if "error" in ex:
                continue
            eid = ex["id"]
            score = ex.get("utility_at_budget", 0.0)
            if eid not in best_single_by_id or score > best_single_by_id[eid]:
                best_single_by_id[eid] = score

    analysis = _aea_substrate_analysis(
        aea_results=aea_res["per_example"],
        best_single_results_by_id=best_single_by_id,
    )

    print(
        f"- Questions where AEA used multiple substrates: "
        f"{analysis['questions_multi_substrate']}/{n_total}"
    )
    print(
        f"- Questions where AEA outperformed best single-substrate: "
        f"{analysis['questions_aea_wins']}/{n_total}"
    )
    print(
        f"- Average substrates used per question by AEA: "
        f"{analysis['avg_substrates']:.2f}"
    )
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation function
# ─────────────────────────────────────────────────────────────────────────────

def run_heterogeneous(
    seed: int = SEED,
    results_path: Optional[Path] = RESULTS_FILE,
) -> dict:
    """
    Run all five policies on the heterogeneous benchmark and report results.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    results_path : Path, optional
        Path to write the detailed JSON results.  Pass None to skip.

    Returns
    -------
    dict
        ``{policy_name: evaluation_result_dict}``
    """
    # ── Generate benchmark ───────────────────────────────────────────────────
    print("Generating heterogeneous benchmark ...")
    bench = HeterogeneousBenchmark(seed=seed)
    examples = bench.generate()
    dataset  = prepare_dataset(examples)
    print(f"  Generated {len(dataset)} examples across {len(_TASK_TYPES)} task types.\n")

    # Print breakdown
    for task_type in _TASK_TYPES:
        n_t = sum(1 for ex in dataset if ex["task_type"] == task_type)
        print(f"  {_TASK_TYPE_LABELS[task_type]:<30}: {n_t:3d}")
    print()

    policies  = make_policies()
    all_results: dict = {}

    # ── Run each policy ──────────────────────────────────────────────────────
    for policy in policies:
        pname = policy.name()
        print(f"{'─' * 65}")
        print(f"Running policy: {pname}")
        print(f"{'─' * 65}")

        address_spaces = make_address_spaces()
        harness = EvaluationHarness(
            address_spaces=address_spaces,
            max_steps=10,
            token_budget=4000,
            seed=seed,
        )

        t_start = time.perf_counter()
        result  = harness.evaluate(policy, dataset)
        elapsed = time.perf_counter() - t_start

        result["runtime_seconds"] = round(elapsed, 2)
        all_results[pname] = result

        agg = result["aggregated"]
        print(f"  support_recall:    {agg['support_recall']:.4f}")
        print(f"  support_precision: {agg['support_precision']:.4f}")
        print(f"  avg_operations:    {agg['operations_used']:.2f}")
        print(f"  utility@budget:    {agg['utility_at_budget']:.4f}")
        print(f"  n_errors:          {result['n_errors']}")
        print(f"  runtime:           {elapsed:.1f}s\n")

    # ── Print summary tables ─────────────────────────────────────────────────
    _print_overall_table(all_results, n=len(dataset))
    _print_per_type_tables(all_results, dataset)

    aea_policy_name = AEAHeuristicPolicy().name()
    _print_substrate_switching(aea_policy_name, all_results, n_total=len(dataset))

    # ── Save results ─────────────────────────────────────────────────────────
    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        # Strip non-serialisable AddressSpaceType values from the benchmark
        # examples before embedding them in the JSON output.
        serialisable_examples = []
        for ex in examples:
            ex_copy = dict(ex)
            ex_copy["required_substrates"] = [
                s.value for s in ex.get("required_substrates", [])
            ]
            serialisable_examples.append(ex_copy)

        payload = {
            "benchmark_examples": serialisable_examples,
            "policy_results":     all_results,
        }
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Detailed results saved to: {results_path}")

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_heterogeneous()
