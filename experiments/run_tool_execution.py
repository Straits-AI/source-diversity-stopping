"""
Tool Execution Experiment — AEA Framework.

Research question
-----------------
Does adding an EXECUTABLE address space (A_tool) change the stopping picture?
Is source-diversity STILL the right stopping signal when one substrate can COMPUTE?

Design
------
Four configurations run on the computational benchmark (100 questions:
50 comparison + 50 arithmetic):

  pi_semantic    — dense retrieval only
  pi_lexical     — BM25 only
  pi_executable  — executable address space only (SEARCH + TOOL_CALL)
  pi_ensemble_tool — all three substrates round-robin (semantic + lexical + executable)

The heuristic stopping rule (AEAHeuristicPolicy pattern) is extended via a
new ExecutablePolicy that:
  Step 0  → executable SEARCH (find numeric passages)
  Step 1  → executable TOOL_CALL (compute the answer from workspace)
  Step 2+ → stop (computation is complete after one TOOL_CALL)

Metrics reported
----------------
  support_recall   — fraction of gold passages retrieved
  utility@budget   — F1 × recall / norm_cost
  steps_taken      — mean steps before stopping
  stopped_reason   — distribution of stop reasons

Interpretation guide
---------------------
If pi_executable beats pi_semantic / pi_lexical on utility@budget:
  → The executable substrate adds unique signal for computation questions.
If pi_ensemble_tool is substantially better than pi_executable alone:
  → Source diversity still matters even when one substrate can compute.
If pi_ensemble_tool is NOT better:
  → For computation tasks, a single executable substrate is sufficient;
    source diversity provides diminishing returns.

Usage
-----
    python experiments/run_tool_execution.py

Results saved to: experiments/results/tool_execution.json
"""

from __future__ import annotations

import json
import random
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
from experiments.aea.address_spaces.executable import ExecutableAddressSpace
from experiments.aea.address_spaces.lexical import LexicalAddressSpace
from experiments.aea.address_spaces.semantic import SemanticAddressSpace
from experiments.aea.evaluation.harness import EvaluationHarness
from experiments.aea.policies.base import Policy
from experiments.aea.policies.single_substrate import LexicalOnlyPolicy, SemanticOnlyPolicy
from experiments.aea.types import Action, AddressSpaceType, AgentState, Operation
from experiments.benchmarks.computational_benchmark import ComputationalBenchmark

# ── Constants ────────────────────────────────────────────────────────────────
SEED = 42
N_EXAMPLES = 100
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_FILE = RESULTS_DIR / "tool_execution.json"


# ─────────────────────────────────────────────────────────────────────────────
# Policies
# ─────────────────────────────────────────────────────────────────────────────

class ExecutableOnlyPolicy(Policy):
    """
    pi_executable: executable address space only.

    Step 0: SEARCH to find passages with numeric quantities.
    Step 1: TOOL_CALL to compute the answer from the workspace.
    Step 2: STOP (computation is done).

    This is the canonical single-substrate policy for A_tool.

    Parameters
    ----------
    top_k : int
        Number of passages to retrieve in the SEARCH step.
    """

    def __init__(self, top_k: int = 5) -> None:
        self._top_k = top_k

    def name(self) -> str:
        return "pi_executable"

    def select_action(self, state: AgentState) -> Action:
        if state.step == 0:
            return Action(
                address_space=AddressSpaceType.EXECUTABLE,
                operation=Operation.SEARCH,
                params={"query": state.query, "top_k": self._top_k},
            )
        if state.step == 1:
            return Action(
                address_space=AddressSpaceType.EXECUTABLE,
                operation=Operation.TOOL_CALL,
                params={"query": state.query},
            )
        return Action(
            address_space=AddressSpaceType.EXECUTABLE,
            operation=Operation.STOP,
        )


class EnsembleToolPolicy(Policy):
    """
    pi_ensemble_tool: all three substrates (semantic + lexical + executable).

    Round-robin:
      step 0 → semantic SEARCH
      step 1 → lexical  SEARCH
      step 2 → executable SEARCH
      step 3 → executable TOOL_CALL (compute from enriched workspace)
      step 4 → STOP

    This gives each substrate one retrieval pass, then lets the executable
    substrate compute over the combined workspace.

    Parameters
    ----------
    top_k : int
        Documents to retrieve per SEARCH step.
    """

    def __init__(self, top_k: int = 5) -> None:
        self._top_k = top_k

    def name(self) -> str:
        return "pi_ensemble_tool"

    def select_action(self, state: AgentState) -> Action:
        if state.budget_remaining <= 0.10:
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.STOP,
            )

        step = state.step

        if step == 0:
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.SEARCH,
                params={"query": state.query, "top_k": self._top_k},
            )
        if step == 1:
            return Action(
                address_space=AddressSpaceType.LEXICAL,
                operation=Operation.SEARCH,
                params={"query": state.query, "top_k": self._top_k},
            )
        if step == 2:
            return Action(
                address_space=AddressSpaceType.EXECUTABLE,
                operation=Operation.SEARCH,
                params={"query": state.query, "top_k": self._top_k},
            )
        if step == 3:
            return Action(
                address_space=AddressSpaceType.EXECUTABLE,
                operation=Operation.TOOL_CALL,
                params={"query": state.query},
            )
        return Action(
            address_space=AddressSpaceType.SEMANTIC,
            operation=Operation.STOP,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Address-space factories
# ─────────────────────────────────────────────────────────────────────────────

def make_address_spaces_for(policy_name: str) -> dict:
    """Return the minimal set of address spaces required for *policy_name*."""
    spaces = {}
    if "semantic" in policy_name or "ensemble" in policy_name:
        spaces["semantic"] = SemanticAddressSpace(model_name="all-MiniLM-L6-v2")
    if "lexical" in policy_name or "ensemble" in policy_name:
        spaces["lexical"] = LexicalAddressSpace()
    if "executable" in policy_name or "ensemble" in policy_name:
        spaces["executable"] = ExecutableAddressSpace()
    return spaces


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_computational_dataset(seed: int = SEED, n: int = N_EXAMPLES) -> list[dict]:
    bench = ComputationalBenchmark(seed=seed)
    examples = bench.generate()
    return examples[:n]


# ─────────────────────────────────────────────────────────────────────────────
# Core runner
# ─────────────────────────────────────────────────────────────────────────────

def run_policy(
    policy: Policy,
    dataset: list[dict],
    seed: int = SEED,
    max_steps: int = 10,
    token_budget: int = 4000,
) -> dict:
    """Run *policy* on *dataset* and return the harness result dict."""
    address_spaces = make_address_spaces_for(policy.name())
    harness = EvaluationHarness(
        address_spaces=address_spaces,
        max_steps=max_steps,
        token_budget=token_budget,
        seed=seed,
    )
    return harness.evaluate(policy, dataset)


# ─────────────────────────────────────────────────────────────────────────────
# Stopping-picture analysis
# ─────────────────────────────────────────────────────────────────────────────

def _stopped_reason_dist(result: dict) -> dict[str, int]:
    """Count stopped_reason values across all per-example results."""
    counts: dict[str, int] = {}
    for ex in result.get("per_example", []):
        reason = ex.get("stopped_reason", "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _task_split_metrics(result: dict, dataset: list[dict]) -> dict:
    """
    Break down support_recall and utility@budget by task_type
    (comparison vs. arithmetic).
    """
    by_id = {ex["id"]: ex["task_type"] for ex in dataset}
    buckets: dict[str, list] = {"comparison": [], "arithmetic": []}

    for ex_result in result.get("per_example", []):
        task = by_id.get(ex_result.get("id", ""), "unknown")
        if task in buckets:
            buckets[task].append(ex_result)

    out = {}
    for task, items in buckets.items():
        if not items:
            continue
        out[task] = {
            "n": len(items),
            "support_recall": float(np.mean([r["support_recall"] for r in items])),
            "utility_at_budget": float(np.mean([r["utility_at_budget"] for r in items])),
            "f1": float(np.mean([r["f1"] for r in items])),
        }
    return out


def analyze_stopping(all_results: dict, dataset: list[dict]) -> dict:
    """
    Compute the stopping-picture analysis.

    Returns a dict with:
      per_policy_stopping : dict[policy_name → stopped_reason_counts]
      task_split          : dict[policy_name → {comparison: metrics, arithmetic: metrics}]
      source_diversity_verdict : str — summary verdict on the stopping question
    """
    stopping: dict = {}
    splits: dict = {}

    for pname, result in all_results.items():
        stopping[pname] = _stopped_reason_dist(result)
        splits[pname] = _task_split_metrics(result, dataset)

    # Verdict: compare ensemble_tool vs executable_only on utility@budget
    u_exec = all_results.get("pi_executable", {}).get("aggregated", {}).get("utility_at_budget", 0.0)
    u_ens  = all_results.get("pi_ensemble_tool", {}).get("aggregated", {}).get("utility_at_budget", 0.0)
    delta = u_ens - u_exec
    threshold = 0.01  # 1% relative gain is considered meaningful

    if delta > threshold:
        verdict = (
            f"Source diversity STILL matters: pi_ensemble_tool ({u_ens:.4f}) beats "
            f"pi_executable ({u_exec:.4f}) by {delta:.4f} U@B. "
            f"Even when one substrate can compute, combining sources improves stopping."
        )
    elif delta < -threshold:
        verdict = (
            f"Executable substrate DOMINATES: pi_executable ({u_exec:.4f}) beats "
            f"pi_ensemble_tool ({u_ens:.4f}) by {-delta:.4f} U@B. "
            f"For computation tasks, source diversity adds overhead without benefit."
        )
    else:
        verdict = (
            f"Parity: pi_executable ({u_exec:.4f}) ≈ pi_ensemble_tool ({u_ens:.4f}), "
            f"delta = {delta:.4f}. "
            f"Source diversity neither helps nor hurts for pure computation tasks."
        )

    return {
        "per_policy_stopping": stopping,
        "task_split": splits,
        "source_diversity_verdict": verdict,
        "u_executable": u_exec,
        "u_ensemble_tool": u_ens,
        "delta_ensemble_minus_executable": delta,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Printing helpers
# ─────────────────────────────────────────────────────────────────────────────

_W_POLICY  = 22
_W_RECALL  = 14
_W_F1      = 8
_W_OPS     = 8
_W_UTILITY = 15
_W_STEPS   = 9


def _header() -> str:
    h = (
        f"| {'Policy':<{_W_POLICY}} "
        f"| {'SupportRecall':>{_W_RECALL}} "
        f"| {'F1':>{_W_F1}} "
        f"| {'AvgOps':>{_W_OPS}} "
        f"| {'Utility@Budget':>{_W_UTILITY}} "
        f"| {'AvgSteps':>{_W_STEPS}} |"
    )
    sep = (
        f"| {'-'*_W_POLICY} "
        f"| {'-'*_W_RECALL} "
        f"| {'-'*_W_F1} "
        f"| {'-'*_W_OPS} "
        f"| {'-'*_W_UTILITY} "
        f"| {'-'*_W_STEPS} |"
    )
    return h + "\n" + sep


def _row(label: str, agg: dict) -> str:
    return (
        f"| {label:<{_W_POLICY}} "
        f"| {agg['support_recall']:>{_W_RECALL}.4f} "
        f"| {agg['f1']:>{_W_F1}.4f} "
        f"| {agg['operations_used']:>{_W_OPS}.2f} "
        f"| {agg['utility_at_budget']:>{_W_UTILITY}.4f} "
        f"| {agg['steps_taken']:>{_W_STEPS}.2f} |"
    )


def print_results_table(all_results: dict) -> None:
    print("\n[Computational Benchmark — N=100  (50 comparison + 50 arithmetic)]")
    print(_header())
    _ORDER = [
        ("pi_semantic",      "pi_semantic"),
        ("pi_lexical",       "pi_lexical"),
        ("pi_executable",    "pi_executable"),
        ("pi_ensemble_tool", "pi_ensemble_tool"),
    ]
    for pname, plabel in _ORDER:
        if pname not in all_results:
            continue
        print(_row(plabel, all_results[pname]["aggregated"]))


def print_stopping_analysis(analysis: dict) -> None:
    print("\n[Stopping reason distribution]")
    for pname, reasons in analysis["per_policy_stopping"].items():
        total = sum(reasons.values())
        parts = "  ".join(f"{r}: {n}/{total}" for r, n in sorted(reasons.items()))
        print(f"  {pname:<22}  {parts}")

    print("\n[Task-type breakdown]")
    for pname, splits in analysis["task_split"].items():
        print(f"  {pname}:")
        for task, m in splits.items():
            print(
                f"    {task:<14}  recall={m['support_recall']:.4f}  "
                f"f1={m['f1']:.4f}  U@B={m['utility_at_budget']:.4f}  (n={m['n']})"
            )

    print(f"\n[Source-diversity verdict]")
    print(f"  {analysis['source_diversity_verdict']}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_tool_execution(
    seed: int = SEED,
    results_path: Optional[Path] = RESULTS_FILE,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print("TOOL EXECUTION EXPERIMENT — AEA Framework")
    print("=" * 70)
    print(
        "\nResearch question: Does A_tool (executable addressing) change the stopping\n"
        "picture?  Is source diversity still the right signal when one substrate\n"
        "can COMPUTE?\n"
    )

    # ── Load dataset ─────────────────────────────────────────────────────────
    print("Loading computational benchmark …")
    dataset = load_computational_dataset(seed=seed, n=N_EXAMPLES)
    comp_count = sum(1 for ex in dataset if ex["task_type"] == "comparison")
    arith_count = sum(1 for ex in dataset if ex["task_type"] == "arithmetic")
    print(f"  {len(dataset)} examples  ({comp_count} comparison, {arith_count} arithmetic)\n")

    # ── Policies ─────────────────────────────────────────────────────────────
    policies = [
        SemanticOnlyPolicy(top_k=5, max_steps=2),
        LexicalOnlyPolicy(top_k=5, max_steps=2),
        ExecutableOnlyPolicy(top_k=5),
        EnsembleToolPolicy(top_k=5),
    ]

    # ── Run ───────────────────────────────────────────────────────────────────
    all_results: dict = {}
    print("Running policies …")
    for policy in policies:
        pname = policy.name()
        print(f"  [{pname}]")
        t0 = time.perf_counter()
        result = run_policy(policy, dataset, seed=seed)
        elapsed = time.perf_counter() - t0
        result["runtime_seconds"] = round(elapsed, 2)
        all_results[pname] = result
        agg = result["aggregated"]
        print(
            f"    support_recall={agg['support_recall']:.4f}  "
            f"f1={agg['f1']:.4f}  "
            f"utility@budget={agg['utility_at_budget']:.4f}  "
            f"avg_steps={agg['steps_taken']:.2f}  "
            f"({elapsed:.1f}s)"
        )

    # ── Analysis ─────────────────────────────────────────────────────────────
    analysis = analyze_stopping(all_results, dataset)

    # ── Print tables ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("=== RESULTS ===")
    print("=" * 70)
    print_results_table(all_results)
    print_stopping_analysis(analysis)

    # ── Save ─────────────────────────────────────────────────────────────────
    payload = {
        "experiment": "tool_execution",
        "seed": seed,
        "n_examples": len(dataset),
        "task_breakdown": {"comparison": comp_count, "arithmetic": arith_count},
        "results": all_results,
        "stopping_analysis": analysis,
    }

    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nDetailed results saved to: {results_path}")

    return payload


if __name__ == "__main__":
    run_tool_execution()
