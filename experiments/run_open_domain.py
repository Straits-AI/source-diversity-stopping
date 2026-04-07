"""
Open-Domain Retrieval Evaluation — expanded candidate set.

Addresses reviewer concern: "Results may be specific to 10-paragraph closed sets."

Strategy
--------
- Take 200 HotpotQA bridge questions.
- For each question, EXPAND the candidate pool:
    Original:  10 paragraphs (2 gold + 8 distractors)
    Expanded:  50 paragraphs (2 gold + 8 original distractors
                              + 40 distractors sampled from OTHER questions)
- This simulates open-domain retrieval where the gold signal is diluted (5x).
- Run: pi_semantic, pi_lexical, pi_ensemble, pi_aea_heuristic.
- Compare to the 10-paragraph baseline.
- Key question: does pi_aea_heuristic still beat pi_ensemble when the
  retrieval space is 5x larger?

Usage
-----
    python experiments/run_open_domain.py

Results saved to experiments/results/open_domain.json
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
N_BRIDGE = 200          # Number of bridge questions to evaluate
N_EXTRA_DISTRACTORS = 40  # Distractor paragraphs sampled from OTHER questions
SEED = 42
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_FILE = RESULTS_DIR / "open_domain.json"


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


# ── Data expansion ────────────────────────────────────────────────────────────

def expand_context(
    example: dict,
    distractor_pool: list[dict],
    n_extra: int,
    rng: random.Random,
) -> dict:
    """
    Expand the candidate context of an example by adding n_extra distractor
    paragraphs sampled from distractor_pool (paragraphs from OTHER questions).

    The gold paragraphs are preserved; only the distractor count increases.

    Parameters
    ----------
    example : dict
        Converted example (output of convert_example).
    distractor_pool : list[dict]
        List of {"id": str, "title": str, "content": str} dicts from OTHER
        questions. May contain duplicates by title — we deduplicate by id.
    n_extra : int
        Number of extra distractors to add.
    rng : random.Random
        Seeded RNG for reproducibility.

    Returns
    -------
    dict
        A copy of the example with an expanded ``context`` list.
    """
    import copy
    expanded = copy.deepcopy(example)

    # IDs already in the example
    existing_ids = {doc["id"] for doc in expanded["context"]}

    # Candidates: distractors not already present
    candidates = [
        doc for doc in distractor_pool
        if doc["id"] not in existing_ids
    ]

    # Sample without replacement (or with, if pool is too small)
    if len(candidates) >= n_extra:
        sampled = rng.sample(candidates, n_extra)
    else:
        # Smaller pool than requested — sample with replacement from candidates
        sampled = [rng.choice(candidates) for _ in range(n_extra)] if candidates else []

    expanded["context"] = expanded["context"] + sampled
    return expanded


def build_distractor_pool(raw_data: list[dict], exclude_ids: set[str]) -> list[dict]:
    """
    Build a pool of paragraph dicts from questions NOT in exclude_ids.

    Each paragraph becomes {"id": title, "title": title, "content": text}.
    We use all paragraphs (gold and non-gold) from non-selected questions as
    distractors — they are all distractors for our selected questions.
    """
    pool: list[dict] = []
    seen_titles: set[str] = set()
    for ex in raw_data:
        if ex.get("_id", "") in exclude_ids:
            continue
        for entry in ex.get("context", []):
            title: str = entry[0]
            sentences: list[str] = entry[1]
            if title in seen_titles:
                continue
            seen_titles.add(title)
            content = " ".join(s.strip() for s in sentences)
            pool.append({"id": title, "title": title, "content": content})
    return pool


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

    se = float(diffs.std(ddof=1) / np.sqrt(n))
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    ci_lower = delta - t_crit * se
    ci_upper = delta + t_crit * se

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


# ── Single-setting evaluation ─────────────────────────────────────────────────

def run_setting(
    policies: list,
    dataset: list[dict],
    seed: int,
    label: str,
) -> dict:
    """
    Run all policies on a dataset and return {policy_name: result}.
    """
    setting_results = {}
    for policy in policies:
        pname = policy.name()
        print(f"  [{label}] {pname} ...", end=" ", flush=True)
        address_spaces = make_address_spaces()
        harness = EvaluationHarness(
            address_spaces=address_spaces,
            max_steps=10,
            token_budget=4000,
            seed=seed,
        )
        t0 = time.perf_counter()
        result = harness.evaluate(policy, dataset)
        elapsed = time.perf_counter() - t0
        result["runtime_seconds"] = round(elapsed, 2)
        setting_results[pname] = result

        agg = result["aggregated"]
        print(
            f"SR={agg['support_recall']:.4f}  "
            f"U@B={agg['utility_at_budget']:.4f}  "
            f"ops={agg['operations_used']:.2f}  "
            f"({elapsed:.1f}s)"
        )
    return setting_results


# ── Main evaluation ───────────────────────────────────────────────────────────

def run_open_domain(
    n_bridge: int = N_BRIDGE,
    n_extra_distractors: int = N_EXTRA_DISTRACTORS,
    seed: int = SEED,
    results_path: Optional[Path] = RESULTS_FILE,
) -> dict:
    """
    Run retrieval evaluation on two settings for the same bridge questions:
      (A) 10-paragraph closed set (original HotpotQA setting)
      (B) 50-paragraph open-domain expanded setting

    Compare heuristic vs ensemble in both settings to test robustness.
    """
    rng_py = random.Random(seed)
    np.random.seed(seed)

    # ── Load data ────────────────────────────────────────────────────────────
    raw_data = load_hotpotqa()
    bridge_raw = [ex for ex in raw_data if ex.get("type") == "bridge"]
    print(f"Total bridge questions: {len(bridge_raw)}")
    print(f"Using first {n_bridge} bridge questions")

    selected_raw = bridge_raw[:n_bridge]
    selected_ids = {ex.get("_id", "") for ex in selected_raw}

    # ── Build distractor pool from the REST of the dataset ───────────────────
    print(f"\nBuilding distractor pool from non-selected questions ...")
    distractor_pool = build_distractor_pool(raw_data, exclude_ids=selected_ids)
    print(f"Distractor pool size: {len(distractor_pool)} unique paragraphs")

    # ── Convert to framework format ───────────────────────────────────────────
    dataset_10 = [convert_example(ex) for ex in selected_raw]

    # Expanded: add n_extra_distractors to each example
    print(f"\nExpanding each example from 10 → {10 + n_extra_distractors} paragraphs ...")
    dataset_50 = [
        expand_context(ex, distractor_pool, n_extra_distractors, rng_py)
        for ex in dataset_10
    ]

    # Verify expansion
    ctx_sizes_10 = [len(ex["context"]) for ex in dataset_10]
    ctx_sizes_50 = [len(ex["context"]) for ex in dataset_50]
    print(f"  10-para: min={min(ctx_sizes_10)} max={max(ctx_sizes_10)} mean={np.mean(ctx_sizes_10):.1f}")
    print(f"  50-para: min={min(ctx_sizes_50)} max={max(ctx_sizes_50)} mean={np.mean(ctx_sizes_50):.1f}")

    policies = make_policies()

    # ── Setting A: 10-paragraph closed set ───────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Setting A: 10-paragraph closed set (N={n_bridge})")
    print(f"{'='*70}")
    results_10 = run_setting(policies, dataset_10, seed=seed, label="10-para")

    # ── Setting B: 50-paragraph open-domain ──────────────────────────────────
    # Re-instantiate policies so internal state is fresh
    policies = make_policies()
    print(f"\n{'='*70}")
    print(f"Setting B: 50-paragraph open-domain (N={n_bridge})")
    print(f"{'='*70}")
    results_50 = run_setting(policies, dataset_50, seed=seed, label="50-para")

    # ── Paired t-tests: heuristic vs ensemble, per setting ────────────────────
    print(f"\n{'='*70}")
    print("Statistical tests: pi_aea_heuristic vs pi_ensemble")
    print(f"{'='*70}")

    stat_tests: dict = {}

    for label, results in [("10-para", results_10), ("50-para", results_50)]:
        heur = results.get("pi_aea_heuristic", {}).get("per_example", [])
        ens = results.get("pi_ensemble", {}).get("per_example", [])
        if heur and ens:
            heur_ub = [r.get("utility_at_budget", 0.0) for r in heur]
            ens_ub = [r.get("utility_at_budget", 0.0) for r in ens]
            test = paired_ttest(heur_ub, ens_ub)
            stat_tests[label] = test
            sig = "**" if test["p_value"] < 0.05 else "  "
            print(f"\n{label} (N={test['n']}){sig}:")
            print(f"  delta (heuristic - ensemble) = {test['delta']:+.6f}")
            print(f"  t = {test['t_stat']:.4f},  p = {test['p_value']:.6f}")
            print(f"  95% CI: [{test['ci_95_lower']:+.6f}, {test['ci_95_upper']:+.6f}]")
            print(f"  Cohen's d = {test['cohens_d']:.4f}")

    # ── Comparison: heuristic in 10-para vs 50-para ───────────────────────────
    heur_10 = results_10.get("pi_aea_heuristic", {}).get("per_example", [])
    heur_50 = results_50.get("pi_aea_heuristic", {}).get("per_example", [])
    if heur_10 and heur_50:
        ub_10 = [r.get("utility_at_budget", 0.0) for r in heur_10]
        ub_50 = [r.get("utility_at_budget", 0.0) for r in heur_50]
        degradation_test = paired_ttest(ub_10, ub_50)
        stat_tests["heuristic_10_vs_50"] = degradation_test
        print(f"\nHeuristic degradation 10-para → 50-para (N={degradation_test['n']}):")
        print(f"  delta (10-para - 50-para) = {degradation_test['delta']:+.6f}")
        print(f"  t = {degradation_test['t_stat']:.4f},  p = {degradation_test['p_value']:.6f}")
        print(f"  95% CI: [{degradation_test['ci_95_lower']:+.6f}, {degradation_test['ci_95_upper']:+.6f}]")

    # ── Summary tables ────────────────────────────────────────────────────────
    _print_summary_table(results_10, label="10-para")
    _print_summary_table(results_50, label="50-para")

    # ── Print head-to-head comparison ─────────────────────────────────────────
    _print_comparison(results_10, results_50)

    # ── Assemble output ───────────────────────────────────────────────────────
    output = {
        "experiment": "open_domain",
        "n_bridge": n_bridge,
        "n_extra_distractors": n_extra_distractors,
        "total_context_size_10para": int(np.mean(ctx_sizes_10)),
        "total_context_size_50para": int(np.mean(ctx_sizes_50)),
        "seed": seed,
        "distractor_pool_size": len(distractor_pool),
        "results_10para": results_10,
        "results_50para": results_50,
        "stat_tests": stat_tests,
    }

    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2)
        print(f"\nDetailed results saved to: {results_path}")

    return output


def _print_summary_table(results: dict, label: str) -> None:
    print(f"\n=== Setting: {label} ===")
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
    for pname, result in results.items():
        agg = result["aggregated"]
        print(
            f"| {pname:<{col['Policy']}} "
            f"| {agg['support_recall']:>{col['SR']}.4f} "
            f"| {agg['support_precision']:>{col['SP']}.4f} "
            f"| {agg['operations_used']:>{col['Ops']}.2f} "
            f"| {agg['utility_at_budget']:>{col['U@B']}.4f} |"
        )


def _print_comparison(results_10: dict, results_50: dict) -> None:
    """Print side-by-side comparison of 10-para vs 50-para U@B."""
    print("\n=== Head-to-head: 10-para vs 50-para (Utility@Budget) ===")
    col = {"Policy": 22, "10": 10, "50": 10, "Delta": 10}
    header = (
        f"| {'Policy':<{col['Policy']}} "
        f"| {'10-para':>{col['10']}} "
        f"| {'50-para':>{col['50']}} "
        f"| {'Delta':>{col['Delta']}} |"
    )
    sep = (
        f"| {'-'*col['Policy']} "
        f"| {'-'*col['10']} "
        f"| {'-'*col['50']} "
        f"| {'-'*col['Delta']} |"
    )
    print(header)
    print(sep)
    for pname in results_10:
        if pname not in results_50:
            continue
        ub_10 = results_10[pname]["aggregated"]["utility_at_budget"]
        ub_50 = results_50[pname]["aggregated"]["utility_at_budget"]
        delta = ub_50 - ub_10
        print(
            f"| {pname:<{col['Policy']}} "
            f"| {ub_10:>{col['10']}.4f} "
            f"| {ub_50:>{col['50']}.4f} "
            f"| {delta:>+{col['Delta']}.4f} |"
        )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_open_domain()
