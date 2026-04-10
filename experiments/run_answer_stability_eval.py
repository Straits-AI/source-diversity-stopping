"""
Answer-Stability Stopping — AEA framework evaluation script.

Compares three policies on N=200 HotpotQA bridge questions:
  - pi_ensemble
  - pi_aea_heuristic
  - pi_answer_stability  (new: stops when draft answer converges)

For all three policies:
  1. Run retrieval-only phase (fast, no external API)
  2. Generate LLM answers using gpt-oss-120b via OpenRouter
  3. Compute retrieval + E2E metrics + stability-specific diagnostics
  4. Run paired t-tests: stability vs heuristic, stability vs ensemble

Usage
-----
    export OPENROUTER_API_KEY="sk-or-..."
    python experiments/run_answer_stability_eval.py
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Prevent HuggingFace tokenizer deadlocks on macOS
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Set API key BEFORE any AEA module imports — answer_generator.py reads the
# key at import time via os.environ.get(), so it must be set first.
_OPENROUTER_KEY = "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"
os.environ.setdefault("OPENROUTER_API_KEY", _OPENROUTER_KEY)

# Make sure project root is on sys.path when run directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.aea.address_spaces.semantic import SemanticAddressSpace
from experiments.aea.address_spaces.lexical import LexicalAddressSpace
from experiments.aea.address_spaces.entity_graph import EntityGraphAddressSpace
from experiments.aea.evaluation.harness import EvaluationHarness
from experiments.aea.policies.ensemble import EnsemblePolicy
from experiments.aea.policies.heuristic import AEAHeuristicPolicy
from experiments.aea.policies.answer_stability import AnswerStabilityPolicy
from experiments.aea.answer_generator import AnswerGenerator
from experiments.aea.evaluation.metrics import exact_match, f1_score

from experiments.run_hotpotqa_baselines import (
    load_hotpotqa,
    filter_bridge,
    convert_example,
)

# Constants
N_EXAMPLES = 200
SEED = 42
MU = 0.3
MAX_OPS = 3.0
N_BOOTSTRAP = 1000
API_CALL_DELAY = 0.1
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_FILE = RESULTS_DIR / "answer_stability.json"
CHECKPOINT_FILE = RESULTS_DIR / "answer_stability_checkpoint.json"

# Shared encoder (avoids macOS deadlock from multiple SentenceTransformer inits)
_SHARED_ENCODER = None


def _get_shared_encoder():
    global _SHARED_ENCODER
    if _SHARED_ENCODER is None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        print("  [setup] Loading SentenceTransformer model (once)...")
        _SHARED_ENCODER = SentenceTransformer("all-MiniLM-L6-v2")
        print("  [setup] Model loaded.")
    return _SHARED_ENCODER


def make_address_spaces() -> dict:
    sem_space = SemanticAddressSpace(model_name="all-MiniLM-L6-v2")
    sem_space._encoder = _get_shared_encoder()
    return {
        "semantic": sem_space,
        "lexical": LexicalAddressSpace(),
        "entity": EntityGraphAddressSpace(),
    }


def make_policies() -> list:
    return [
        EnsemblePolicy(top_k=5, max_steps=3),
        AEAHeuristicPolicy(top_k=5, coverage_threshold=0.5, max_steps=6),
        AnswerStabilityPolicy(top_k=5, max_steps=5, convergence_threshold=0.8),
    ]


def compute_e2e_ub(f1: float, recall: float, ops: float) -> float:
    return f1 * (1.0 + 0.5 * recall) - MU * (ops / MAX_OPS)


def bootstrap_ci(
    values: list[float],
    n_resamples: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=np.float64)
    means = np.array([
        np.mean(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_resamples)
    ])
    ci_low = float(np.percentile(means, 2.5))
    ci_high = float(np.percentile(means, 97.5))
    return float(np.mean(arr)), ci_low, ci_high


def paired_ttest(a: list[float], b: list[float]) -> tuple[float, float]:
    """Two-sided paired t-test. Returns (t_statistic, p_value)."""
    try:
        from scipy import stats  # type: ignore
        result = stats.ttest_rel(a, b)
        return float(result.statistic), float(result.pvalue)
    except ImportError:
        pass
    # Manual fallback
    a_arr = np.array(a, dtype=np.float64)
    b_arr = np.array(b, dtype=np.float64)
    diff = a_arr - b_arr
    n = len(diff)
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    if std_diff == 0.0 or n <= 1:
        return 0.0, 1.0
    t_stat = mean_diff / (std_diff / np.sqrt(n))
    # Two-tailed p-value via normal approximation (valid for n >= 30)
    z = abs(t_stat)
    p_approx = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
    return float(t_stat), float(p_approx)


def extract_evidence(
    ex_result: dict,
    dataset_by_id: dict[str, dict],
) -> list[str]:
    retrieved_ids: list[str] = []
    for trace_step in ex_result.get("trace", []):
        for item in trace_step.get("result_items", []):
            rid = item.get("id", "")
            if rid and rid not in retrieved_ids:
                retrieved_ids.append(rid)
    ex_id = ex_result.get("id", "unknown")
    example = dataset_by_id.get(ex_id, {})
    id_to_content: dict[str, str] = {
        doc["id"]: doc.get("content", "")
        for doc in example.get("context", [])
    }
    return [
        id_to_content[rid]
        for rid in retrieved_ids
        if rid in id_to_content and id_to_content[rid]
    ]


def generate_answers_for_policy(
    harness_result: dict,
    dataset_by_id: dict[str, dict],
    answer_gen: AnswerGenerator,
    policy_name: str,
) -> list[dict]:
    per_example = harness_result["per_example"]
    total = len(per_example)
    e2e_results = []
    for idx, ex_result in enumerate(per_example):
        ex_id = ex_result.get("id", "unknown")
        question = ex_result.get("question", "")
        gold_answer = ex_result.get("gold_answer", "")
        support_recall_val = float(ex_result.get("support_recall", 0.0))
        operations_used = float(ex_result.get("operations_used", 0))
        evidence_passages = extract_evidence(ex_result, dataset_by_id)
        if evidence_passages:
            time.sleep(API_CALL_DELAY)
            llm_answer = answer_gen.generate_answer(question, evidence_passages)
        else:
            llm_answer = ""
        em = exact_match(llm_answer, gold_answer)
        f1 = f1_score(llm_answer, gold_answer)
        e2e_ub = compute_e2e_ub(f1, support_recall_val, operations_used)
        e2e_results.append({
            "id": ex_id,
            "question": question,
            "gold_answer": gold_answer,
            "llm_answer": llm_answer,
            "em": em,
            "f1": f1,
            "support_recall": support_recall_val,
            "operations_used": operations_used,
            "e2e_ub": e2e_ub,
        })
        if (idx + 1) % 25 == 0 or idx == 0:
            pct = (idx + 1) / total * 100
            usage = answer_gen.usage_summary()
            print(
                f"    [{policy_name}] {idx+1}/{total} ({pct:.0f}%) -- "
                f"API calls: {usage['total_calls']}, errors: {usage['total_errors']}"
            )
    return e2e_results


def aggregate_e2e(e2e_results: list[dict]) -> dict:
    out: dict = {}
    for key in ["em", "f1", "support_recall", "operations_used", "e2e_ub"]:
        vals = [r[key] for r in e2e_results if key in r]
        out[f"{key}_mean"] = float(np.mean(vals)) if vals else 0.0
        out[f"{key}_std"] = float(np.std(vals)) if vals else 0.0
    return out


def compute_stability_diagnostics(harness_result: dict) -> dict:
    per_example = harness_result["per_example"]
    steps_list = [r.get("steps_taken", 0) for r in per_example if "error" not in r]
    stopped_by_stop = [
        r for r in per_example
        if r.get("stopped_reason") == "stop_action" and "error" not in r
    ]
    n_total = len(steps_list)
    if n_total == 0:
        return {
            "avg_steps_taken": 0.0,
            "pct_converged_step1": 0.0,
            "pct_converged_step2": 0.0,
            "pct_stopped_by_stop_action": 0.0,
        }
    n_step1 = sum(
        1 for r in per_example
        if r.get("steps_taken", 0) <= 1 and r.get("stopped_reason") == "stop_action"
    )
    n_step2 = sum(
        1 for r in per_example
        if r.get("steps_taken", 0) == 2 and r.get("stopped_reason") == "stop_action"
    )
    return {
        "avg_steps_taken": float(np.mean(steps_list)),
        "pct_converged_step1": n_step1 / n_total * 100.0,
        "pct_converged_step2": n_step2 / n_total * 100.0,
        "pct_stopped_by_stop_action": len(stopped_by_stop) / n_total * 100.0,
    }


def print_retrieval_table(policy_names: list[str], all_retrieval: dict) -> None:
    print(f"\n=== Retrieval Metrics (N={N_EXAMPLES}) ===\n")
    header = (
        f"| {'Policy':<24} "
        f"| {'SupportRecall':>14} "
        f"| {'AvgOps':>8} "
        f"| {'U@Budget':>10} |"
    )
    sep = f"| {'-'*24} | {'-'*14} | {'-'*8} | {'-'*10} |"
    print(header)
    print(sep)
    for pname in policy_names:
        if pname not in all_retrieval:
            continue
        agg = all_retrieval[pname]["aggregated"]
        print(
            f"| {pname:<24} "
            f"| {agg['support_recall']:>14.4f} "
            f"| {agg['operations_used']:>8.2f} "
            f"| {agg['utility_at_budget']:>10.4f} |"
        )
    print()


def print_e2e_table(
    policy_names: list[str],
    agg_stats: dict,
    ci_stats: dict,
) -> None:
    print(f"=== E2E Metrics (N={N_EXAMPLES}, gpt-oss-120b) ===\n")
    header = (
        f"| {'Policy':<24} "
        f"| {'EM':>8} "
        f"| {'F1':>8} "
        f"| {'Recall':>8} "
        f"| {'Ops':>6} "
        f"| {'E2E U@B [95% CI]':>30} |"
    )
    sep = f"| {'-'*24} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*6} | {'-'*30} |"
    print(header)
    print(sep)
    for pname in policy_names:
        if pname not in agg_stats:
            continue
        agg = agg_stats[pname]
        ci = ci_stats.get(pname, (0.0, 0.0, 0.0))
        print(
            f"| {pname:<24} "
            f"| {agg['em_mean']:>8.4f} "
            f"| {agg['f1_mean']:>8.4f} "
            f"| {agg['support_recall_mean']:>8.4f} "
            f"| {agg['operations_used_mean']:>6.2f} "
            f"| {ci[0]:.4f} [{ci[1]:.4f}, {ci[2]:.4f}]   |"
        )
    print()


def print_stability_diag(diagnostics: dict) -> None:
    print("=== Stability Diagnostics (pi_answer_stability) ===\n")
    pname = "pi_answer_stability"
    if pname not in diagnostics:
        print("  No stability diagnostics available.\n")
        return
    d = diagnostics[pname]
    print(f"  Avg steps taken:            {d['avg_steps_taken']:.2f}")
    print(f"  % stopped by STOP action:   {d['pct_stopped_by_stop_action']:.1f}%")
    print(f"  % converged at step <= 1:   {d['pct_converged_step1']:.1f}%")
    print(f"  % converged at step == 2:   {d['pct_converged_step2']:.1f}%")
    print()


def run_answer_stability_eval(
    n_examples: int = N_EXAMPLES,
    seed: int = SEED,
    results_path: Optional[Path] = RESULTS_FILE,
) -> dict:
    """Full evaluation pipeline for answer-stability stopping."""
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print(f"Answer-Stability Evaluation -- N={n_examples}, seed={seed}")
    print("=" * 70)

    # Phase 1: Load data
    print("\n[Phase 1] Loading HotpotQA data...")
    raw_data = load_hotpotqa()
    bridge = filter_bridge(raw_data, n=n_examples)
    dataset = [convert_example(ex) for ex in bridge]
    dataset_by_id = {ex["id"]: ex for ex in dataset}
    print(f"  Loaded {len(dataset)} bridge examples.\n")

    # Phase 2: Retrieval-only phase (with checkpoint support)
    print("[Phase 2] Running retrieval-only phase for 3 policies...")
    policies = make_policies()
    policy_names_all = [p.name() for p in policies]
    all_retrieval: dict = {}

    # Load checkpoint if it exists (allows resuming Phase 3 after Phase 2 crash)
    checkpoint_path = CHECKPOINT_FILE
    if checkpoint_path.exists():
        print(f"  Found checkpoint at {checkpoint_path} — loading Phase 2 results...")
        with open(checkpoint_path, "r", encoding="utf-8") as fh:
            checkpoint_data = json.load(fh)
        if checkpoint_data.get("n_examples") == n_examples and checkpoint_data.get("seed") == seed:
            all_retrieval = checkpoint_data.get("retrieval_results", {})
            print(f"  Loaded {len(all_retrieval)} policy retrieval results from checkpoint.")
            for pname, result in all_retrieval.items():
                agg = result["aggregated"]
                print(f"    [checkpoint] {pname}: recall={agg['support_recall']:.4f}, ops={agg['operations_used']:.2f}")
        else:
            print("  Checkpoint mismatch (n_examples or seed differ) — re-running Phase 2.")
            all_retrieval = {}

    for policy in policies:
        pname = policy.name()
        if pname in all_retrieval:
            print(f"\n  Policy {pname}: skipping (loaded from checkpoint)")
            continue
        print(f"\n  Policy: {pname}")
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
        all_retrieval[pname] = result

        agg = result["aggregated"]
        print(f"    support_recall:    {agg['support_recall']:.4f}")
        print(f"    support_precision: {agg['support_precision']:.4f}")
        print(f"    avg_operations:    {agg['operations_used']:.2f}")
        print(f"    utility@budget:    {agg['utility_at_budget']:.4f}")
        print(f"    n_errors:          {result['n_errors']}")
        print(f"    runtime:           {elapsed:.1f}s")

        # Save checkpoint after each policy completes
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, "w", encoding="utf-8") as fh:
            json.dump({"n_examples": n_examples, "seed": seed, "retrieval_results": all_retrieval}, fh, indent=2)
        print(f"    [checkpoint saved to {checkpoint_path}]")

    # Phase 3: LLM answer generation
    policy_names = policy_names_all
    print(f"\n[Phase 3] Generating LLM answers for {policy_names}...")
    answer_gen = AnswerGenerator()
    all_e2e_per_example: dict[str, list[dict]] = {}

    for pname in policy_names:
        if pname not in all_retrieval:
            print(f"  WARNING: {pname} not found in retrieval results, skipping.")
            continue
        print(f"\n  Generating answers for: {pname}")
        e2e_results = generate_answers_for_policy(
            all_retrieval[pname], dataset_by_id, answer_gen, pname
        )
        all_e2e_per_example[pname] = e2e_results
        agg = aggregate_e2e(e2e_results)
        print(
            f"    EM={agg['em_mean']:.4f}, F1={agg['f1_mean']:.4f}, "
            f"E2E U@B={agg['e2e_ub_mean']:.4f}"
        )

    usage = answer_gen.usage_summary()
    print(f"\n  Total API calls: {usage['total_calls']}")
    print(f"  Total tokens:    {usage['total_tokens']}")
    print(f"  Errors:          {usage['total_errors']}")
    est_cost = usage["total_calls"] * 0.00002
    print(f"  Estimated cost:  ~${est_cost:.4f}")

    # Phase 4: Aggregate and bootstrap CIs
    print("\n[Phase 4] Computing bootstrap CIs...")
    agg_stats: dict[str, dict] = {}
    ci_stats: dict[str, tuple] = {}

    for pname in policy_names:
        if pname not in all_e2e_per_example:
            continue
        results = all_e2e_per_example[pname]
        agg_stats[pname] = aggregate_e2e(results)
        e2e_ub_vals = [r["e2e_ub"] for r in results]
        mean_ub, ci_low, ci_high = bootstrap_ci(e2e_ub_vals, n_resamples=N_BOOTSTRAP, seed=seed)
        ci_stats[pname] = (mean_ub, ci_low, ci_high)
        print(f"  {pname}: E2E U@B mean={mean_ub:.4f} [{ci_low:.4f}, {ci_high:.4f}]")

    # Phase 5: Paired t-tests
    print("\n[Phase 5] Paired t-tests on E2E U@B...")
    test_pairs = [
        ("pi_answer_stability", "pi_aea_heuristic", "AnswerStability vs Heuristic"),
        ("pi_answer_stability", "pi_ensemble",     "AnswerStability vs Ensemble"),
    ]

    stat_tests: list[dict] = []
    for name_a, name_b, label in test_pairs:
        if name_a not in all_e2e_per_example or name_b not in all_e2e_per_example:
            print(f"  Skipping {label}: one or both policies missing.")
            continue
        vals_a = [r["e2e_ub"] for r in all_e2e_per_example[name_a]]
        vals_b = [r["e2e_ub"] for r in all_e2e_per_example[name_b]]
        delta = float(np.mean(vals_a)) - float(np.mean(vals_b))
        t_stat, p_val = paired_ttest(vals_a, vals_b)
        diff = np.array(vals_a) - np.array(vals_b)
        pooled_std = float(np.std(diff, ddof=1))
        cohens_d = delta / pooled_std if pooled_std > 0 else 0.0
        stat_tests.append({
            "label": label,
            "policy_a": name_a,
            "policy_b": name_b,
            "delta": delta,
            "t_statistic": t_stat,
            "p_value": p_val,
            "cohens_d": cohens_d,
        })
        sig = "YES" if p_val < 0.05 else "NO"
        print(
            f"  {label}: delta={delta:.4f}, t={t_stat:.3f}, "
            f"p={p_val:.4f}, d={cohens_d:.3f}, Significant? {sig}"
        )

    # Phase 6: Stability diagnostics
    stability_diagnostics: dict[str, dict] = {}
    for pname in policy_names:
        if pname in all_retrieval:
            stability_diagnostics[pname] = compute_stability_diagnostics(all_retrieval[pname])

    # Phase 7: Print tables
    print_retrieval_table(policy_names, all_retrieval)
    print_e2e_table(policy_names, agg_stats, ci_stats)
    print_stability_diag(stability_diagnostics)

    print("Statistical Tests (E2E U@B, paired t-test):")
    for t in stat_tests:
        sig = "YES" if t["p_value"] < 0.05 else "NO"
        print(
            f"  {t['label']}: "
            f"delta={t['delta']:.4f}, t={t['t_statistic']:.3f}, "
            f"p={t['p_value']:.4f}, d={t['cohens_d']:.3f}, "
            f"Significant? {sig}"
        )
    print()

    # Phase 8: Save results
    full_results = {
        "n_examples": n_examples,
        "seed": seed,
        "mu": MU,
        "max_ops": MAX_OPS,
        "n_bootstrap": N_BOOTSTRAP,
        "policies_evaluated": policy_names,
        "retrieval_aggregated": {
            pname: result["aggregated"]
            for pname, result in all_retrieval.items()
        },
        "e2e_per_example": all_e2e_per_example,
        "e2e_aggregated": agg_stats,
        "bootstrap_ci": {
            pname: {"mean": ci[0], "ci_low": ci[1], "ci_high": ci[2]}
            for pname, ci in ci_stats.items()
        },
        "statistical_tests": stat_tests,
        "stability_diagnostics": stability_diagnostics,
        "answer_generator_usage": usage,
        "estimated_api_cost_usd": est_cost,
    }

    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump(full_results, fh, indent=2)
        print(f"Full results saved to: {results_path}")

    return full_results


if __name__ == "__main__":
    run_answer_stability_eval()
