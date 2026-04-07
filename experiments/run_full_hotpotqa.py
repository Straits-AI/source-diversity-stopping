"""
Full HotpotQA Evaluation — ALL question types (bridge + comparison).

Addresses reviewer concern: "Results may be specific to bridge questions."

Strategy
--------
- Load the full HotpotQA distractor validation set (7405 questions total).
- Take the first 1000 questions (mixture of bridge and comparison).
- Run retrieval-only for: pi_semantic, pi_lexical, pi_ensemble, pi_aea_heuristic.
- Report metrics broken down by question type (bridge vs comparison).
- Run paired t-test (heuristic vs ensemble) on Utility@Budget (retrieval).

Key hypothesis: does pi_aea_heuristic beat pi_ensemble on COMPARISON questions
                as well as bridge questions?

Usage
-----
    python experiments/run_full_hotpotqa.py

Results saved to experiments/results/full_hotpotqa.json
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats

# ── Project root on sys.path ─────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── AEA imports ──────────────────────────────────────────────────────────────
from experiments.aea.address_spaces.semantic import SemanticAddressSpace
from experiments.aea.address_spaces.lexical import LexicalAddressSpace
from experiments.aea.address_spaces.entity_graph import EntityGraphAddressSpace
from experiments.aea.evaluation.harness import EvaluationHarness
from experiments.aea.policies.single_substrate import (
    SemanticOnlyPolicy,
    LexicalOnlyPolicy,
)
from experiments.aea.policies.ensemble import EnsemblePolicy
from experiments.aea.policies.heuristic import AEAHeuristicPolicy

# Re-use data loading utilities from baselines
from experiments.run_hotpotqa_baselines import load_hotpotqa, convert_example

# ── Constants ─────────────────────────────────────────────────────────────────
N_EXAMPLES = 1000
SEED = 42
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_FILE = RESULTS_DIR / "full_hotpotqa.json"


# ── Policy factory ────────────────────────────────────────────────────────────

def make_policies():
    """Instantiate the four retrieval-only policies."""
    return [
        SemanticOnlyPolicy(top_k=5, max_steps=2),
        LexicalOnlyPolicy(top_k=5, max_steps=2),
        EnsemblePolicy(top_k=5, max_steps=3),
        AEAHeuristicPolicy(top_k=5, coverage_threshold=0.5, max_steps=6),
    ]


def make_address_spaces():
    """Instantiate fresh address spaces."""
    return {
        "semantic": SemanticAddressSpace(model_name="all-MiniLM-L6-v2"),
        "lexical": LexicalAddressSpace(),
        "entity": EntityGraphAddressSpace(),
    }


# ── Statistical utilities ─────────────────────────────────────────────────────

def paired_ttest(scores_a: list[float], scores_b: list[float]) -> dict:
    """
    Paired two-sided t-test: H0 = mean(A) - mean(B) = 0.

    Returns dict with delta, p_value, ci_95_lower, ci_95_upper, n.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    diffs = a - b
    n = len(diffs)
    delta = float(diffs.mean())
    t_stat, p_value = stats.ttest_rel(a, b)

    # 95% CI for the mean difference
    se = float(diffs.std(ddof=1) / np.sqrt(n))
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    ci_lower = delta - t_crit * se
    ci_upper = delta + t_crit * se

    # Cohen's d (paired)
    std_diff = float(diffs.std(ddof=1))
    cohens_d = delta / std_diff if std_diff > 0 else 0.0

    return {
        "n": n,
        "delta": round(delta, 6),
        "t_stat": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "ci_95_lower": round(ci_lower, 6),
        "ci_95_upper": round(ci_upper, 6),
        "cohens_d": round(cohens_d, 4),
    }


# ── Main evaluation ───────────────────────────────────────────────────────────

def run_full_hotpotqa(
    n_examples: int = N_EXAMPLES,
    seed: int = SEED,
    results_path: Optional[Path] = RESULTS_FILE,
) -> dict:
    """
    Run all four retrieval policies on the first n_examples questions
    (bridge + comparison) from HotpotQA distractor validation set.
    """
    random.seed(seed)
    np.random.seed(seed)

    # ── Load data ────────────────────────────────────────────────────────────
    raw_data = load_hotpotqa()
    print(f"Total questions in HotpotQA distractor validation: {len(raw_data)}")

    # Count types before slicing
    type_counts = {}
    for ex in raw_data:
        t = ex.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"Question type distribution (full set): {type_counts}")

    # Take first N (mixture of bridge + comparison)
    subset = raw_data[:n_examples]
    type_counts_subset = {}
    for ex in subset:
        t = ex.get("type", "unknown")
        type_counts_subset[t] = type_counts_subset.get(t, 0) + 1
    print(f"Question type distribution (first {n_examples}): {type_counts_subset}")

    dataset = [convert_example(ex) for ex in subset]

    # Tag each converted example with its original type
    for ex_orig, ex_conv in zip(subset, dataset):
        ex_conv["question_type"] = ex_orig.get("type", "unknown")

    print(f"Converted {len(dataset)} examples for evaluation.\n")

    policies = make_policies()
    all_results: dict = {}

    # ── Run each policy ──────────────────────────────────────────────────────
    for policy in policies:
        pname = policy.name()
        print(f"{'=' * 70}")
        print(f"Policy: {pname}")
        print(f"{'=' * 70}")

        address_spaces = make_address_spaces()
        harness = EvaluationHarness(
            address_spaces=address_spaces,
            max_steps=10,
            token_budget=4000,
            seed=seed,
        )

        t_start = time.perf_counter()
        result = harness.evaluate(policy, dataset)
        elapsed = time.perf_counter() - t_start
        result["runtime_seconds"] = round(elapsed, 2)

        # Attach question_type to per_example results
        for i, ex_result in enumerate(result["per_example"]):
            if i < len(dataset):
                ex_result["question_type"] = dataset[i].get("question_type", "unknown")

        all_results[pname] = result

        agg = result["aggregated"]
        print(f"  OVERALL  support_recall={agg['support_recall']:.4f}  "
              f"utility@budget={agg['utility_at_budget']:.4f}  "
              f"ops={agg['operations_used']:.2f}  "
              f"errors={result['n_errors']}  runtime={elapsed:.1f}s")

        # Break down by question type
        for qtype in sorted(type_counts_subset.keys()):
            type_examples = [
                r for r in result["per_example"]
                if r.get("question_type") == qtype
            ]
            if type_examples:
                sr_vals = [r.get("support_recall", 0.0) for r in type_examples]
                ub_vals = [r.get("utility_at_budget", 0.0) for r in type_examples]
                print(f"  [{qtype:12s}] N={len(type_examples):4d}  "
                      f"support_recall={np.mean(sr_vals):.4f}  "
                      f"utility@budget={np.mean(ub_vals):.4f}")
        print()

    # ── Paired t-tests: heuristic vs ensemble ────────────────────────────────
    print("=" * 70)
    print("Paired t-test: pi_aea_heuristic vs pi_ensemble")
    print("=" * 70)

    heuristic_results = all_results.get("pi_aea_heuristic", {})
    ensemble_results = all_results.get("pi_ensemble", {})

    stat_tests: dict = {}

    if heuristic_results and ensemble_results:
        heur_per = heuristic_results["per_example"]
        ens_per = ensemble_results["per_example"]

        # Overall
        heur_ub = [r.get("utility_at_budget", 0.0) for r in heur_per]
        ens_ub = [r.get("utility_at_budget", 0.0) for r in ens_per]
        overall_test = paired_ttest(heur_ub, ens_ub)
        stat_tests["overall"] = overall_test
        print(f"\nOverall (N={overall_test['n']}):")
        print(f"  delta = {overall_test['delta']:+.6f}")
        print(f"  t = {overall_test['t_stat']:.4f}, p = {overall_test['p_value']:.6f}")
        print(f"  95% CI: [{overall_test['ci_95_lower']:+.6f}, {overall_test['ci_95_upper']:+.6f}]")
        print(f"  Cohen's d = {overall_test['cohens_d']:.4f}")

        # Per question type
        for qtype in sorted(type_counts_subset.keys()):
            heur_type = [
                r.get("utility_at_budget", 0.0) for r in heur_per
                if r.get("question_type") == qtype
            ]
            ens_type = [
                r.get("utility_at_budget", 0.0) for r in ens_per
                if r.get("question_type") == qtype
            ]
            if len(heur_type) > 1 and len(ens_type) > 1:
                type_test = paired_ttest(heur_type, ens_type)
                stat_tests[qtype] = type_test
                sig = "**" if type_test["p_value"] < 0.05 else ""
                print(f"\n{qtype} (N={type_test['n']}):{sig}")
                print(f"  delta = {type_test['delta']:+.6f}")
                print(f"  t = {type_test['t_stat']:.4f}, p = {type_test['p_value']:.6f}")
                print(f"  95% CI: [{type_test['ci_95_lower']:+.6f}, {type_test['ci_95_upper']:+.6f}]")
                print(f"  Cohen's d = {type_test['cohens_d']:.4f}")

    # ── Summary table ────────────────────────────────────────────────────────
    _print_summary_table(all_results, n_examples, type_counts_subset)

    # ── Assemble output ───────────────────────────────────────────────────────
    output = {
        "experiment": "full_hotpotqa",
        "n_examples": n_examples,
        "seed": seed,
        "type_distribution": type_counts_subset,
        "policy_results": all_results,
        "stat_tests_heuristic_vs_ensemble": stat_tests,
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2)
        print(f"\nDetailed results saved to: {results_path}")

    return output


def _print_summary_table(
    all_results: dict,
    n_examples: int,
    type_counts: dict,
) -> None:
    """Print a Markdown-style summary table."""
    print(f"\n=== Full HotpotQA Evaluation (N={n_examples}, types={type_counts}) ===\n")

    col = {"Policy": 22, "SR": 12, "SP": 12, "Ops": 8, "U@B": 12}
    header = (
        f"| {'Policy':<{col['Policy']}} "
        f"| {'SuppRecall':>{col['SR']}} "
        f"| {'SuppPrec':>{col['SP']}} "
        f"| {'AvgOps':>{col['Ops']}} "
        f"| {'Utility@B':>{col['U@B']}} |"
    )
    sep = (
        f"| {'-'*col['Policy']} "
        f"| {'-'*col['SR']} "
        f"| {'-'*col['SP']} "
        f"| {'-'*col['Ops']} "
        f"| {'-'*col['U@B']} |"
    )

    print(header)
    print(sep)
    for pname, result in all_results.items():
        agg = result["aggregated"]
        print(
            f"| {pname:<{col['Policy']}} "
            f"| {agg['support_recall']:>{col['SR']}.4f} "
            f"| {agg['support_precision']:>{col['SP']}.4f} "
            f"| {agg['operations_used']:>{col['Ops']}.2f} "
            f"| {agg['utility_at_budget']:>{col['U@B']}.4f} |"
        )
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_full_hotpotqa()
