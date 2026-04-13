"""
Confidence-gated stopping benchmark — N=200 HotpotQA bridge questions.

Compares three policies on the eval split (indices 0-199):
  1. pi_ensemble           — always 3 ops, strong evidence recall
  2. pi_aea_heuristic      — heuristic routing, ~1.16 ops
  3. pi_confidence_gated   — 1 LLM call after first retrieval; 1 or 2 ops

For each policy:
  - Retrieval phase: harness-level evaluation (no LLM for retrieval decisions)
  - E2E phase: LLM (gpt-oss-120b) generates final answers from workspace

Key metrics reported:
  - EM, F1           (E2E answer quality)
  - Ops              (mean retrieval operations)
  - E2E U@B          (utility-at-budget)
  - % stopped at 1 op / 2 ops (confidence-gated only)
  - Paired t-tests vs heuristic and ensemble

Results saved to: experiments/results/confidence_gated.json

Usage
-----
    export OPENROUTER_API_KEY="sk-or-..."
    python experiments/run_confidence_gated_eval.py
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

# Prevent HuggingFace tokenizer deadlocks on macOS
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Project root on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# AEA imports
from experiments.aea.address_spaces.semantic import SemanticAddressSpace
from experiments.aea.address_spaces.lexical import LexicalAddressSpace
from experiments.aea.address_spaces.entity_graph import EntityGraphAddressSpace
from experiments.aea.evaluation.harness import EvaluationHarness
from experiments.aea.evaluation.metrics import exact_match, f1_score
from experiments.aea.policies.ensemble import EnsemblePolicy
from experiments.aea.policies.heuristic import AEAHeuristicPolicy
from experiments.aea.policies.confidence_gated import ConfidenceGatedPolicy
from experiments.aea.answer_generator import AnswerGenerator

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
API_CALL_DELAY = 0.1
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_FILE = RESULTS_DIR / "confidence_gated.json"

os.environ.setdefault(
    "OPENROUTER_API_KEY",
    "REPLACE_WITH_YOUR_OPENROUTER_API_KEY",
)

# Shared SentenceTransformer encoder
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
        ConfidenceGatedPolicy(top_k=5),
    ]


def compute_e2e_ub(f1: float, recall: float, ops: float) -> float:
    """E2E U@B = f1 * (1 + 0.5 * recall) - mu * (ops / max_ops)."""
    return f1 * (1.0 + 0.5 * recall) - MU * (ops / MAX_OPS)


def extract_evidence(ex_result: dict, dataset_by_id: dict) -> list[str]:
    """Extract retrieved passage texts from a per-example harness result."""
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


def generate_llm_answers(
    harness_result: dict,
    dataset_by_id: dict,
    answer_gen: AnswerGenerator,
    policy_name: str,
) -> list[dict]:
    """Generate LLM E2E answers for all examples in a harness result."""
    per_example = harness_result["per_example"]
    total = len(per_example)
    results = []

    for idx, ex_result in enumerate(per_example):
        ex_id = ex_result.get("id", "unknown")
        question = ex_result.get("question", "")
        gold_answer = ex_result.get("gold_answer", "")
        support_recall_val = ex_result.get("support_recall", 0.0)
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

        results.append({
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

        if (idx + 1) % 50 == 0 or idx == 0:
            usage = answer_gen.usage_summary()
            print(
                f"    [{policy_name}] {idx+1}/{total} "
                f"({(idx+1)/total*100:.0f}%) — "
                f"API calls: {usage['total_calls']}, "
                f"errors: {usage['total_errors']}"
            )

    return results


def aggregate_e2e(e2e_results: list[dict]) -> dict:
    out = {}
    for key in ["em", "f1", "support_recall", "operations_used", "e2e_ub"]:
        vals = [r[key] for r in e2e_results if key in r]
        out[f"{key}_mean"] = float(np.mean(vals)) if vals else 0.0
        out[f"{key}_std"] = float(np.std(vals)) if vals else 0.0
    return out


def compute_stopping_breakdown(harness_result: dict) -> dict:
    """Count episodes stopped at 1 vs 2 retrieval ops."""
    per_example = harness_result["per_example"]
    stopped_at_1 = sum(1 for ex in per_example if ex.get("operations_used", 0) <= 1)
    stopped_at_2 = len(per_example) - stopped_at_1
    total = len(per_example)
    return {
        "stopped_at_1_op": stopped_at_1,
        "stopped_at_2_ops": stopped_at_2,
        "total": total,
        "pct_stopped_at_1": stopped_at_1 / total if total > 0 else 0.0,
        "pct_stopped_at_2": stopped_at_2 / total if total > 0 else 0.0,
    }


def paired_t_test(a: list[float], b: list[float]) -> tuple[float, float, float]:
    """Two-sided paired t-test. Returns (delta, p_value, cohens_d)."""
    from scipy import stats  # type: ignore
    a_arr = np.array(a, dtype=np.float64)
    b_arr = np.array(b, dtype=np.float64)
    diff = a_arr - b_arr
    _t, p_val = stats.ttest_rel(a_arr, b_arr)
    delta = float(np.mean(diff))
    pooled_std = float(np.std(diff, ddof=1))
    cohens_d = delta / pooled_std if pooled_std > 0 else 0.0
    return delta, float(p_val), cohens_d


def run_benchmark(
    n_examples: int = N_EXAMPLES,
    seed: int = SEED,
    results_path: Optional[Path] = RESULTS_FILE,
) -> dict:
    """Full benchmark pipeline: retrieval + E2E + stats."""
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print(f"Confidence-Gated Benchmark — N={n_examples}, seed={seed}")
    print("=" * 70)

    # Phase 1: Load data
    print("\n[Phase 1] Loading HotpotQA bridge questions...")
    raw_data = load_hotpotqa()
    bridge = filter_bridge(raw_data, n=n_examples)
    dataset = [convert_example(ex) for ex in bridge]
    dataset_by_id = {ex["id"]: ex for ex in dataset}
    print(f"  Loaded {len(dataset)} examples.\n")

    # Phase 2: Retrieval phase for all 3 policies
    print("[Phase 2] Retrieval phase (3 policies)...")
    policies = make_policies()
    all_retrieval: dict = {}

    for policy in policies:
        pname = policy.name()
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

    # Stopping breakdown for confidence-gated
    cg_result = all_retrieval.get("pi_confidence_gated", {})
    stopping_breakdown = compute_stopping_breakdown(cg_result)
    print(f"\n  [pi_confidence_gated] Stopping breakdown:")
    print(f"    Stopped at 1 op  (LLM confident): "
          f"{stopping_breakdown['stopped_at_1_op']} "
          f"({stopping_breakdown['pct_stopped_at_1']:.1%})")
    print(f"    Stopped at 2 ops (needed more):   "
          f"{stopping_breakdown['stopped_at_2_ops']} "
          f"({stopping_breakdown['pct_stopped_at_2']:.1%})")

    # Phase 3: LLM answer generation
    print(f"\n[Phase 3] Generating LLM answers (gpt-oss-120b)...")
    answer_gen = AnswerGenerator()
    all_e2e: dict[str, list[dict]] = {}

    policy_names = ["pi_ensemble", "pi_aea_heuristic", "pi_confidence_gated"]
    for pname in policy_names:
        if pname not in all_retrieval:
            print(f"  WARNING: {pname} not found, skipping.")
            continue
        print(f"\n  Generating answers: {pname}")
        e2e = generate_llm_answers(
            all_retrieval[pname], dataset_by_id, answer_gen, pname
        )
        all_e2e[pname] = e2e
        agg = aggregate_e2e(e2e)
        print(
            f"    EM={agg['em_mean']:.4f}, "
            f"F1={agg['f1_mean']:.4f}, "
            f"Ops={agg['operations_used_mean']:.2f}, "
            f"E2E U@B={agg['e2e_ub_mean']:.4f}"
        )

    usage = answer_gen.usage_summary()
    print(f"\n  Total API calls: {usage['total_calls']}")
    print(f"  Total tokens: {usage['total_tokens']}")
    print(f"  Errors: {usage['total_errors']}")

    # Phase 4: Aggregate + stats
    print("\n[Phase 4] Aggregating metrics and running paired t-tests...")
    agg_stats: dict[str, dict] = {}
    for pname in policy_names:
        if pname in all_e2e:
            agg_stats[pname] = aggregate_e2e(all_e2e[pname])

    test_pairs = [
        ("pi_confidence_gated", "pi_aea_heuristic", "ConfGated vs Heuristic"),
        ("pi_confidence_gated", "pi_ensemble",      "ConfGated vs Ensemble"),
        ("pi_aea_heuristic",    "pi_ensemble",      "Heuristic vs Ensemble"),
    ]
    stat_tests: list[dict] = []
    try:
        for name_a, name_b, label in test_pairs:
            if name_a not in all_e2e or name_b not in all_e2e:
                continue
            vals_a = [r["e2e_ub"] for r in all_e2e[name_a]]
            vals_b = [r["e2e_ub"] for r in all_e2e[name_b]]
            delta, p_val, d = paired_t_test(vals_a, vals_b)
            sig = "YES" if p_val < 0.05 else "NO"
            stat_tests.append({
                "label": label,
                "policy_a": name_a,
                "policy_b": name_b,
                "delta": delta,
                "p_value": p_val,
                "cohens_d": d,
                "significant": sig,
            })
            print(
                f"  {label}: delta={delta:.4f}, p={p_val:.4f}, "
                f"d={d:.3f}, Significant? {sig}"
            )
    except ImportError:
        print("  scipy not available; skipping t-tests.")

    # Phase 5: Print summary table
    print("\n=== Results (N=200, gpt-oss-120b) ===\n")
    header = (
        f"| {'Policy':<24} "
        f"| {'EM':>7} "
        f"| {'F1':>7} "
        f"| {'Recall':>7} "
        f"| {'Ops':>5} "
        f"| {'E2E U@B':>9} |"
    )
    sep = f"| {'-'*24} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*5} | {'-'*9} |"
    print(header)
    print(sep)
    for pname in policy_names:
        if pname not in agg_stats:
            continue
        agg = agg_stats[pname]
        row = (
            f"| {pname:<24} "
            f"| {agg['em_mean']:>7.4f} "
            f"| {agg['f1_mean']:>7.4f} "
            f"| {agg['support_recall_mean']:>7.4f} "
            f"| {agg['operations_used_mean']:>5.2f} "
            f"| {agg['e2e_ub_mean']:>9.4f} |"
        )
        print(row)
    print()

    print("Confidence-Gated Stopping Breakdown:")
    print(f"  Stopped at 1 op  (LLM confident): "
          f"{stopping_breakdown['stopped_at_1_op']} "
          f"({stopping_breakdown['pct_stopped_at_1']:.1%})")
    print(f"  Stopped at 2 ops (needed more):   "
          f"{stopping_breakdown['stopped_at_2_ops']} "
          f"({stopping_breakdown['pct_stopped_at_2']:.1%})")
    print()

    if stat_tests:
        print("Statistical Tests (E2E U@B, paired t-test):")
        for t in stat_tests:
            print(
                f"  {t['label']}: "
                f"delta={t['delta']:.4f}, p={t['p_value']:.4f}, "
                f"d={t['cohens_d']:.3f}, Significant? {t['significant']}"
            )
        print()

    # Phase 6: Save results
    full_results = {
        "n_examples": n_examples,
        "seed": seed,
        "mu": MU,
        "max_ops": MAX_OPS,
        "retrieval_aggregated": {
            pname: result["aggregated"]
            for pname, result in all_retrieval.items()
        },
        "e2e_per_example": all_e2e,
        "e2e_aggregated": agg_stats,
        "confidence_gated_stopping_breakdown": stopping_breakdown,
        "statistical_tests": stat_tests,
        "answer_generator_usage": usage,
    }

    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump(full_results, fh, indent=2)
        print(f"Full results saved to: {results_path}")

    return full_results


if __name__ == "__main__":
    run_benchmark()
