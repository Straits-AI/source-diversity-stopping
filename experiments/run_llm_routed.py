"""
LLM-Routed AEA vs Baselines — HotpotQA Bridge evaluation.

Runs five policies and generates LLM answers for each, reporting both
retrieval-only and end-to-end metrics.

Policies evaluated
------------------
  pi_semantic       — SemanticOnlyPolicy (baseline)
  pi_lexical        — LexicalOnlyPolicy  (baseline)
  pi_ensemble       — EnsemblePolicy     (upper bound)
  pi_aea_heuristic  — AEAHeuristicPolicy (current method)
  pi_llm_routed     — LLMRoutedPolicy    (new — LLM makes routing decisions)

Usage
-----
    export OPENROUTER_API_KEY="sk-or-..."
    python experiments/run_llm_routed.py

Or with a smaller N for a smoke-test:
    python experiments/run_llm_routed.py --n 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ── Make sure project root is on sys.path when run directly ──────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── AEA imports ───────────────────────────────────────────────────────────────
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
from experiments.aea.policies.llm_routed import LLMRoutedPolicy
from experiments.aea.answer_generator import AnswerGenerator
from experiments.aea.evaluation.metrics import exact_match, f1_score, utility_at_budget, normalize_cost

# Reuse data-loading helpers from the baselines script
from experiments.run_hotpotqa_baselines import (
    load_hotpotqa,
    filter_bridge,
    convert_example,
    make_address_spaces,
)

# ── Constants ─────────────────────────────────────────────────────────────────
N_EXAMPLES = 100
SEED = 42
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_FILE = RESULTS_DIR / "llm_routed.json"


# ── Policy factory ─────────────────────────────────────────────────────────────

def make_policies(llm_routed: LLMRoutedPolicy) -> list:
    """Return ordered list of policies to evaluate."""
    return [
        SemanticOnlyPolicy(top_k=5, max_steps=2),
        LexicalOnlyPolicy(top_k=5, max_steps=2),
        EnsemblePolicy(top_k=5, max_steps=3),
        AEAHeuristicPolicy(top_k=5, coverage_threshold=0.5, max_steps=6),
        llm_routed,
    ]


# ── End-to-end answer generation ─────────────────────────────────────────────

def generate_e2e_answers(
    harness_result: dict,
    dataset: dict[str, dict],
    answer_gen: AnswerGenerator,
) -> list[dict]:
    """
    For each per-example result, generate an LLM answer from workspace evidence
    and compute EM / F1 against the gold answer.

    Parameters
    ----------
    harness_result : dict
        Output from ``EvaluationHarness.evaluate``.
    dataset : dict[str, dict]
        Mapping of example id → example dict (for gold answer lookup).
    answer_gen : AnswerGenerator
        Shared answer generator.

    Returns
    -------
    list[dict]
        Per-example dicts with keys: id, question, gold_answer,
        llm_answer, em, f1, support_recall, operations_used, utility_at_budget.
    """
    e2e_results = []

    per_example = harness_result["per_example"]
    total = len(per_example)

    for idx, ex_result in enumerate(per_example):
        ex_id = ex_result.get("id", "unknown")
        question = ex_result.get("question", "")
        gold_answer = ex_result.get("gold_answer", "")
        support_recall = ex_result.get("support_recall", 0.0)
        operations_used = ex_result.get("operations_used", 0)
        tokens_used = ex_result.get("tokens_used", 0)

        # Gather workspace evidence from trace
        # The harness doesn't serialise workspace contents in the result dict,
        # so we use the retrieval IDs from result_items and the original dataset
        # to reconstruct passage text.
        retrieved_ids: list[str] = []
        for trace_step in ex_result.get("trace", []):
            for item in trace_step.get("result_items", []):
                rid = item.get("id", "")
                if rid and rid not in retrieved_ids:
                    retrieved_ids.append(rid)

        # Look up passage text from the original dataset example
        example = dataset.get(ex_id, {})
        id_to_content: dict[str, str] = {
            doc["id"]: doc.get("content", "")
            for doc in example.get("context", [])
        }

        evidence_passages = [
            id_to_content[rid]
            for rid in retrieved_ids
            if rid in id_to_content and id_to_content[rid]
        ]

        # Generate answer
        if evidence_passages:
            llm_answer = answer_gen.generate_answer(question, evidence_passages)
        else:
            llm_answer = ""

        em = exact_match(llm_answer, gold_answer)
        f1 = f1_score(llm_answer, gold_answer)

        # Recompute utility@budget with LLM answer quality
        norm_cost = normalize_cost(
            tokens=tokens_used,
            latency_ms=0.0,
            operations=int(operations_used),
            max_tokens=4000,
            max_latency=30_000.0,
            max_ops=20,
        )
        u_at_b = utility_at_budget(
            answer_score=f1,
            evidence_score=support_recall,
            cost=norm_cost,
        )

        e2e_results.append({
            "id": ex_id,
            "question": question,
            "gold_answer": gold_answer,
            "llm_answer": llm_answer,
            "em": em,
            "f1": f1,
            "support_recall": support_recall,
            "operations_used": operations_used,
            "utility_at_budget": u_at_b,
        })

        if (idx + 1) % 10 == 0:
            print(f"    [{idx + 1}/{total}] answers generated…")

    return e2e_results


def _aggregate_e2e(e2e_results: list[dict]) -> dict:
    """Compute mean of numeric fields across e2e results."""
    keys = ["em", "f1", "support_recall", "operations_used", "utility_at_budget"]
    out = {}
    for k in keys:
        vals = [r[k] for r in e2e_results if k in r]
        out[k] = float(np.mean(vals)) if vals else 0.0
    return out


# ── Routing analysis (LLM-routed only) ────────────────────────────────────────

def compute_routing_analysis(
    llm_policy: LLMRoutedPolicy,
    harness_result: dict,
) -> dict:
    """
    Compute routing-specific statistics from the harness trace.

    Returns
    -------
    dict
        avg_steps_before_stop, action_distribution (pcts),
        pct_stopped_after_1_step, pct_used_3plus_steps,
        and the raw routing usage summary from the policy.
    """
    per_example = harness_result["per_example"]
    steps_list = [ex.get("steps_taken", 0) for ex in per_example]

    avg_steps = float(np.mean(steps_list)) if steps_list else 0.0
    pct_1_step = 100.0 * sum(1 for s in steps_list if s <= 1) / len(steps_list) if steps_list else 0.0
    pct_3plus = 100.0 * sum(1 for s in steps_list if s >= 3) / len(steps_list) if steps_list else 0.0

    routing_usage = llm_policy.routing_usage_summary()

    return {
        "avg_steps_before_stop": round(avg_steps, 2),
        "action_distribution_pcts": routing_usage["decision_pcts"],
        "action_distribution_counts": routing_usage["decision_counts"],
        "pct_stopped_after_1_step": round(pct_1_step, 1),
        "pct_used_3plus_steps": round(pct_3plus, 1),
        "api_usage": {
            "total_routing_calls": routing_usage["total_routing_calls"],
            "total_tokens": routing_usage["total_tokens"],
            "total_errors": routing_usage["total_api_errors"],
        },
    }


# ── Pretty-print tables ────────────────────────────────────────────────────────

def _print_retrieval_table(all_retrieval: dict, n: int) -> None:
    """Print retrieval-only metrics table."""
    print(f"\n=== LLM-Routed AEA vs Baselines (N={n}) ===\n")
    print("Retrieval Metrics:")
    header = (
        f"| {'Policy':<22} "
        f"| {'SupportRecall':>15} "
        f"| {'SupportPrec':>13} "
        f"| {'AvgOps':>8} "
        f"| {'Retrieval U@B':>14} |"
    )
    sep = f"| {'-'*22} | {'-'*15} | {'-'*13} | {'-'*8} | {'-'*14} |"
    print(header)
    print(sep)

    for pname, result in all_retrieval.items():
        agg = result["aggregated"]
        row = (
            f"| {pname:<22} "
            f"| {agg['support_recall']:>15.4f} "
            f"| {agg['support_precision']:>13.4f} "
            f"| {agg['operations_used']:>8.2f} "
            f"| {agg['utility_at_budget']:>14.4f} |"
        )
        print(row)
    print()


def _print_e2e_table(all_e2e: dict, n: int) -> None:
    """Print end-to-end metrics table."""
    print("End-to-End Metrics (with LLM answers):")
    header = (
        f"| {'Policy':<22} "
        f"| {'EM':>6} "
        f"| {'F1':>8} "
        f"| {'SupportRecall':>14} "
        f"| {'AvgOps':>8} "
        f"| {'End-to-End U@B':>16} |"
    )
    sep = f"| {'-'*22} | {'-'*6} | {'-'*8} | {'-'*14} | {'-'*8} | {'-'*16} |"
    print(header)
    print(sep)

    for pname, agg in all_e2e.items():
        row = (
            f"| {pname:<22} "
            f"| {agg['em']:>6.4f} "
            f"| {agg['f1']:>8.4f} "
            f"| {agg['support_recall']:>14.4f} "
            f"| {agg['operations_used']:>8.2f} "
            f"| {agg['utility_at_budget']:>16.4f} |"
        )
        print(row)
    print()


def _print_routing_analysis(analysis: dict) -> None:
    """Print routing-specific statistics."""
    print("Routing Analysis (LLM-routed only):")
    dist = analysis["action_distribution_pcts"]
    print(f"  Average steps before STOP: {analysis['avg_steps_before_stop']:.2f}")
    print(
        f"  Action distribution: "
        f"STOP={dist.get('STOP', 0)}%, "
        f"SEMANTIC={dist.get('SEMANTIC_SEARCH', 0)}%, "
        f"LEXICAL={dist.get('LEXICAL_SEARCH', 0)}%, "
        f"ENTITY_HOP={dist.get('ENTITY_HOP', 0)}%"
    )
    print(f"  Questions where LLM stopped after 1 step: {analysis['pct_stopped_after_1_step']}%")
    print(f"  Questions where LLM used 3+ steps: {analysis['pct_used_3plus_steps']}%")
    api = analysis["api_usage"]
    print(
        f"  Routing API: {api['total_routing_calls']} calls, "
        f"{api['total_tokens']} tokens, "
        f"{api['total_errors']} errors"
    )
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def run_llm_routed(
    n_examples: int = N_EXAMPLES,
    seed: int = SEED,
    results_path: Optional[Path] = RESULTS_FILE,
) -> dict:
    """
    Run all five policies on the first *n_examples* HotpotQA bridge questions,
    generate LLM answers for each, and report retrieval + end-to-end metrics.

    Returns
    -------
    dict
        Full results dict (saved to *results_path* if provided).
    """
    import random
    random.seed(seed)
    np.random.seed(seed)

    # ── Load data ────────────────────────────────────────────────────────────
    raw_data = load_hotpotqa()
    bridge = filter_bridge(raw_data, n=n_examples)
    dataset = [convert_example(ex) for ex in bridge]
    # Build id → example dict for answer generation evidence lookup
    dataset_by_id = {ex["id"]: ex for ex in dataset}
    print(f"Converted {len(dataset)} examples for evaluation.\n")

    # ── Shared answer generator ───────────────────────────────────────────────
    answer_gen = AnswerGenerator()  # uses default model (gemma-3-12b)

    # ── LLM-routed policy (shared so we accumulate routing stats) ────────────
    llm_policy = LLMRoutedPolicy(top_k=5, max_steps=5)
    policies = make_policies(llm_policy)

    all_retrieval: dict = {}
    all_e2e_agg: dict = {}
    all_e2e_per_example: dict = {}
    routing_analysis: Optional[dict] = None

    # ── Run each policy ───────────────────────────────────────────────────────
    for policy in policies:
        pname = policy.name()
        print(f"{'─' * 60}")
        print(f"Running policy: {pname}")
        print(f"{'─' * 60}")

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
        print(f"  retrieval support_recall:    {agg['support_recall']:.4f}")
        print(f"  retrieval support_precision: {agg['support_precision']:.4f}")
        print(f"  retrieval avg_operations:    {agg['operations_used']:.2f}")
        print(f"  retrieval utility@budget:    {agg['utility_at_budget']:.4f}")
        print(f"  n_errors:                    {result['n_errors']}")
        print(f"  runtime:                     {elapsed:.1f}s")

        # ── Generate LLM answers ──────────────────────────────────────────────
        print(f"  Generating LLM answers for {pname}…")
        e2e_results = generate_e2e_answers(result, dataset_by_id, answer_gen)
        e2e_agg = _aggregate_e2e(e2e_results)
        all_e2e_agg[pname] = e2e_agg
        all_e2e_per_example[pname] = e2e_results
        print(f"  e2e EM: {e2e_agg['em']:.4f}  F1: {e2e_agg['f1']:.4f}  U@B: {e2e_agg['utility_at_budget']:.4f}\n")

        # Collect routing stats for LLM-routed policy
        if pname == "pi_llm_routed":
            routing_analysis = compute_routing_analysis(llm_policy, result)

    # ── Print summary tables ──────────────────────────────────────────────────
    _print_retrieval_table(all_retrieval, n_examples)
    _print_e2e_table(all_e2e_agg, n_examples)
    if routing_analysis:
        _print_routing_analysis(routing_analysis)

    # ── Answer generator usage ────────────────────────────────────────────────
    ans_usage = answer_gen.usage_summary()
    print("Answer Generator Usage:")
    print(f"  Total calls:  {ans_usage['total_calls']}")
    print(f"  Total tokens: {ans_usage['total_tokens']}")
    print(f"  Errors:       {ans_usage['total_errors']}")
    print()

    # ── Assemble full output ───────────────────────────────────────────────────
    full_results = {
        "n_examples": n_examples,
        "seed": seed,
        "retrieval": all_retrieval,
        "e2e_aggregated": all_e2e_agg,
        "e2e_per_example": all_e2e_per_example,
        "routing_analysis": routing_analysis,
        "answer_generator_usage": ans_usage,
    }

    # ── Save results ──────────────────────────────────────────────────────────
    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump(full_results, fh, indent=2)
        print(f"Detailed results saved to: {results_path}")

    return full_results


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM-routed AEA experiment")
    parser.add_argument(
        "--n", type=int, default=N_EXAMPLES,
        help=f"Number of HotpotQA bridge questions to evaluate (default: {N_EXAMPLES})"
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help=f"Random seed (default: {SEED})"
    )
    args = parser.parse_args()

    run_llm_routed(n_examples=args.n, seed=args.seed)
