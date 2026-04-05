"""
Multi-Seed HotpotQA Evaluation with Statistical Inference.

Addresses the reviewer concern: "N=100 with single seed provides no error bars,
confidence intervals, or significance tests."

Strategy:
- Use ALL bridge questions (up to 500) from HotpotQA distractor validation.
- Since retrieval is deterministic, we use bootstrap resampling for CIs.
- We run with 5 seeds to shuffle question order; variance comes from subsampling
  if N < total, or bootstrap if all data is used.
- Paired permutation test: AEA vs best baseline for statistical significance.
- Effect size: Cohen's d.

Usage
-----
    python experiments/run_multiseed.py

Results saved to experiments/results/multiseed_hotpotqa.json
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ── Make sure project root is on sys.path when run directly ──────────────────
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
    EntityOnlyPolicy,
)
from experiments.aea.policies.ensemble import EnsemblePolicy
from experiments.aea.policies.heuristic import AEAHeuristicPolicy

# Re-use data loading utilities from the baseline script
from experiments.run_hotpotqa_baselines import (
    load_hotpotqa,
    convert_example,
)

# ── Constants ────────────────────────────────────────────────────────────────
N_MAX = 500           # Use up to this many bridge questions
SEEDS = [42, 123, 456, 789, 1024]
N_BOOTSTRAP = 1000    # Bootstrap resamples for CIs
N_PERMUTATIONS = 10_000  # Permutation test iterations
ALPHA = 0.05          # Significance level
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_FILE = RESULTS_DIR / "multiseed_hotpotqa.json"


# ── Policy factory ────────────────────────────────────────────────────────────

def make_policies():
    """Instantiate all baseline policies."""
    return [
        SemanticOnlyPolicy(top_k=5, max_steps=2),
        LexicalOnlyPolicy(top_k=5, max_steps=2),
        EntityOnlyPolicy(top_k=5, max_steps=3),
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

def bootstrap_ci(
    values: list[float],
    n_resamples: int = N_BOOTSTRAP,
    alpha: float = ALPHA,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float]:
    """
    Compute a bootstrap confidence interval for the mean of values.

    Parameters
    ----------
    values : list[float]
    n_resamples : int
    alpha : float  Significance level (e.g. 0.05 for 95% CI).
    rng : numpy Generator

    Returns
    -------
    (lower, upper) tuple of floats
    """
    if rng is None:
        rng = np.random.default_rng(42)
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    boot_means = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        sample = arr[rng.integers(0, n, size=n)]
        boot_means[i] = sample.mean()
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lower, upper


def permutation_test(
    scores_a: list[float],
    scores_b: list[float],
    n_permutations: int = N_PERMUTATIONS,
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float]:
    """
    Paired permutation test: H0 = mean(A) - mean(B) = 0.

    Returns (observed_delta, p_value).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    diffs = a - b
    observed_delta = float(diffs.mean())

    # Permute signs of paired differences
    count_extreme = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(diffs))
        permuted_mean = float((signs * diffs).mean())
        if abs(permuted_mean) >= abs(observed_delta):
            count_extreme += 1

    p_value = count_extreme / n_permutations
    return observed_delta, p_value


def cohens_d(scores_a: list[float], scores_b: list[float]) -> float:
    """
    Cohen's d effect size (paired).

    d = mean(A - B) / std(A - B)
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    diffs = a - b
    std_diff = float(diffs.std(ddof=1))
    if std_diff == 0.0:
        return 0.0
    return float(diffs.mean() / std_diff)


# ── Main evaluation ───────────────────────────────────────────────────────────

def run_multiseed(
    n_max: int = N_MAX,
    seeds: list[int] = SEEDS,
    results_path: Optional[Path] = RESULTS_FILE,
) -> dict:
    """
    Run all policies on the full bridge dataset (up to n_max examples).

    Since retrieval is deterministic (embeddings + BM25 + entity graph), we
    run once to get per-example metrics, then use bootstrap resampling for
    confidence intervals and permutation tests for significance.

    The seeds parameter controls which subsets we use if n_max < total bridge
    questions. If n_max >= total, all seeds evaluate the same data and we
    rely on bootstrap for statistical inference.
    """
    # ── Load data ────────────────────────────────────────────────────────────
    raw_data = load_hotpotqa()
    all_bridge = [ex for ex in raw_data if ex.get("type") == "bridge"]
    print(f"Total bridge questions available: {len(all_bridge)}")

    n_examples = min(n_max, len(all_bridge))
    print(f"Using N={n_examples} bridge questions\n")

    # For each seed, shuffle and take the first n_examples.
    # If n_examples == len(all_bridge), all seeds give the same data;
    # variance comes entirely from bootstrap resampling.
    seed_datasets: dict[int, list[dict]] = {}
    for seed in seeds:
        rng = random.Random(seed)
        shuffled = all_bridge.copy()
        rng.shuffle(shuffled)
        subset = shuffled[:n_examples]
        seed_datasets[seed] = [convert_example(ex) for ex in subset]

    # Check if all seeds yield identical data (n_examples == total bridge)
    all_same_data = (n_examples == len(all_bridge))
    if all_same_data:
        print("N >= total bridge questions: all seeds will evaluate same data.")
        print("Statistical inference via bootstrap resampling and permutation test.\n")

    policies = make_policies()
    policy_names = [p.name() for p in policies]

    # Structure: per_example_by_policy[policy_name][seed] = list of metric dicts
    per_example_by_policy: dict[str, dict[int, list[dict]]] = {
        pname: {} for pname in policy_names
    }

    # ── Run each policy on each seed ─────────────────────────────────────────
    total_runs = len(policies) * len(seeds)
    run_idx = 0

    for policy in policies:
        pname = policy.name()
        print(f"{'=' * 70}")
        print(f"Policy: {pname}")
        print(f"{'=' * 70}")

        # If all seeds give same data, run only once and replicate results
        if all_same_data:
            seed = seeds[0]
            dataset = seed_datasets[seed]
            print(f"  Running once (deterministic) on N={len(dataset)} examples ...")
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
            agg = result["aggregated"]
            print(f"    SupportRecall={agg['support_recall']:.4f}  "
                  f"AvgOps={agg['operations_used']:.2f}  "
                  f"U@B={agg['utility_at_budget']:.4f}  "
                  f"({elapsed:.1f}s)")

            # All seeds share the same per_example results
            for s in seeds:
                per_example_by_policy[pname][s] = result["per_example"]

        else:
            for seed in seeds:
                run_idx += 1
                dataset = seed_datasets[seed]
                print(f"  Seed {seed} ({run_idx}/{total_runs}, N={len(dataset)}) ...")
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
                per_example_by_policy[pname][seed] = result["per_example"]
                agg = result["aggregated"]
                print(f"    SupportRecall={agg['support_recall']:.4f}  "
                      f"AvgOps={agg['operations_used']:.2f}  "
                      f"U@B={agg['utility_at_budget']:.4f}  "
                      f"({elapsed:.1f}s)")
        print()

    # ── Compute statistics ────────────────────────────────────────────────────
    print("\nComputing bootstrap CIs and permutation tests ...\n")
    rng = np.random.default_rng(42)

    metrics_of_interest = [
        "support_recall",
        "support_precision",
        "operations_used",
        "utility_at_budget",
    ]

    # Gather pooled per-example metrics across seeds for each policy
    # If all seeds gave same data, pooling is identical to using one seed.
    # We still compute per-seed means to report std.
    stats: dict[str, dict] = {}

    for pname in policy_names:
        seed_means: dict[str, list[float]] = {m: [] for m in metrics_of_interest}
        all_values: dict[str, list[float]] = {m: [] for m in metrics_of_interest}

        for seed in seeds:
            per_ex = per_example_by_policy[pname][seed]
            for metric in metrics_of_interest:
                vals = [
                    ex[metric] for ex in per_ex
                    if metric in ex and isinstance(ex[metric], (int, float))
                ]
                seed_means[metric].append(float(np.mean(vals)) if vals else 0.0)
                # Only add to all_values once per seed if same data
                # (no double-counting when all seeds are identical)
            if not all_same_data or seed == seeds[0]:
                for metric in metrics_of_interest:
                    vals = [
                        ex[metric] for ex in per_ex
                        if metric in ex and isinstance(ex[metric], (int, float))
                    ]
                    all_values[metric].extend(vals)

        # If same data for all seeds, all_values was populated only once
        # Bootstrap CIs on the pooled per-example values
        ci: dict[str, tuple[float, float]] = {}
        for metric in metrics_of_interest:
            vals = all_values[metric]
            if vals:
                ci[metric] = bootstrap_ci(vals, rng=rng)
            else:
                ci[metric] = (0.0, 0.0)

        # Mean ± std across seed-level means
        mean_std: dict[str, tuple[float, float]] = {}
        for metric in metrics_of_interest:
            sm = seed_means[metric]
            mean_std[metric] = (float(np.mean(sm)), float(np.std(sm, ddof=1) if len(sm) > 1 else 0.0))

        stats[pname] = {
            "mean_std": mean_std,
            "ci_95": ci,
            "all_values": {m: all_values[m] for m in metrics_of_interest},
        }

    # ── Significance tests: AEA vs baselines ─────────────────────────────────
    aea_name = "pi_aea_heuristic"
    baselines_to_test = ["pi_lexical", "pi_semantic", "pi_ensemble"]

    # Collect per-example U@B for AEA (from first seed since deterministic)
    aea_ub = stats[aea_name]["all_values"]["utility_at_budget"]

    sig_tests: dict[str, dict] = {}
    for baseline in baselines_to_test:
        if baseline not in stats:
            continue
        baseline_ub = stats[baseline]["all_values"]["utility_at_budget"]
        # Truncate to same length (should already be equal)
        n = min(len(aea_ub), len(baseline_ub))
        delta, pval = permutation_test(aea_ub[:n], baseline_ub[:n], rng=rng)
        d = cohens_d(aea_ub[:n], baseline_ub[:n])
        sig_tests[baseline] = {
            "delta": delta,
            "p_value": pval,
            "cohens_d": d,
            "significant": pval < ALPHA,
        }

    # ── Print summary table ───────────────────────────────────────────────────
    _print_summary(stats, sig_tests, n_examples, seeds, all_same_data)

    # ── Assemble results dict ─────────────────────────────────────────────────
    output = {
        "meta": {
            "n_examples": n_examples,
            "n_seeds": len(seeds),
            "seeds": seeds,
            "all_same_data": all_same_data,
            "n_bootstrap": N_BOOTSTRAP,
            "n_permutations": N_PERMUTATIONS,
            "alpha": ALPHA,
        },
        "stats": {
            pname: {
                "mean_std": stats[pname]["mean_std"],
                "ci_95": {m: list(v) for m, v in stats[pname]["ci_95"].items()},
            }
            for pname in policy_names
        },
        "significance_tests": sig_tests,
        # Include raw per-example values for reproducibility
        "per_example_values": {
            pname: {
                m: stats[pname]["all_values"][m]
                for m in metrics_of_interest
            }
            for pname in policy_names
        },
    }

    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2)
        print(f"\nDetailed results saved to: {results_path}")

    return output


def _print_summary(
    stats: dict,
    sig_tests: dict,
    n_examples: int,
    seeds: list[int],
    all_same_data: bool,
) -> None:
    """Print the multi-seed results table."""
    policy_display = {
        "pi_semantic": "pi_semantic",
        "pi_lexical": "pi_lexical",
        "pi_entity": "pi_entity",
        "pi_ensemble": "pi_ensemble",
        "pi_aea_heuristic": "pi_aea",
    }

    note = "(bootstrap CIs, same data all seeds)" if all_same_data else "(across seeds)"

    print(f"\n{'=' * 80}")
    print(f"=== Multi-Seed Results (N={n_examples}, Seeds={len(seeds)}) ===")
    print(f"Statistical inference {note}")
    print(f"{'=' * 80}\n")

    header = (
        f"| {'Policy':<15} | {'SupportRecall':>22} "
        f"| {'AvgOps':>10} | {'Utility@Budget':>35} |"
    )
    sep = f"| {'-' * 15} | {'-' * 22} | {'-' * 10} | {'-' * 35} |"
    print(header)
    print(sep)

    policy_order = [
        "pi_semantic", "pi_lexical", "pi_entity", "pi_ensemble", "pi_aea_heuristic"
    ]

    for pname in policy_order:
        if pname not in stats:
            continue
        s = stats[pname]
        disp = policy_display.get(pname, pname)

        sr_mean, sr_std = s["mean_std"]["support_recall"]
        ops_mean, _ = s["mean_std"]["operations_used"]
        ub_mean, ub_std = s["mean_std"]["utility_at_budget"]
        ub_lo, ub_hi = s["ci_95"]["utility_at_budget"]
        sr_lo, sr_hi = s["ci_95"]["support_recall"]

        sr_col = f"{sr_mean:.4f} +/- {sr_std:.4f} [{sr_lo:.4f}, {sr_hi:.4f}]"
        ub_col = f"{ub_mean:.4f} +/- {ub_std:.4f} [{ub_lo:.4f}, {ub_hi:.4f}]"
        ops_col = f"{ops_mean:.2f}"

        print(
            f"| {disp:<15} | {sr_col:>22} | {ops_col:>10} | {ub_col:>35} |"
        )

    print()
    print("Statistical Tests (AEA vs baselines, metric: Utility@Budget):")
    print(f"  {'Comparison':<30} {'Delta':>10} {'p-value':>10} {'Cohen d':>10} {'Sig?':>8}")
    print(f"  {'-' * 30} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 8}")
    for baseline, t in sig_tests.items():
        sig = "YES" if t["significant"] else "NO"
        print(
            f"  AEA vs {baseline:<23} "
            f"{t['delta']:>10.4f} "
            f"{t['p_value']:>10.4f} "
            f"{t['cohens_d']:>10.4f} "
            f"{sig:>8}"
        )
    print()

    # Determine best baseline by U@B mean
    best_baseline = max(
        ["pi_semantic", "pi_lexical", "pi_entity", "pi_ensemble"],
        key=lambda p: stats[p]["mean_std"]["utility_at_budget"][0] if p in stats else -1,
    )
    print(f"Best baseline: {best_baseline}")
    if best_baseline in sig_tests:
        t = sig_tests[best_baseline]
        sig = "YES" if t["significant"] else "NO"
        print(
            f"AEA vs best baseline ({best_baseline}): "
            f"Delta={t['delta']:.4f}, p={t['p_value']:.4f}, "
            f"Cohen's d={t['cohens_d']:.4f}, Significant at p<{ALPHA}? {sig}"
        )
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.perf_counter()
    run_multiseed()
    elapsed = time.perf_counter() - t0
    print(f"Total runtime: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
