"""
Decomposition Stopping vs Baselines — End-to-End Evaluation.

Evaluates three policies on 100 HotpotQA bridge questions:
  1. pi_ensemble       (upper bound — exhaustive)
  2. pi_aea_heuristic  (current method)
  3. pi_decomposition  (NEW — LLM decomposes question once; content-aware stopping)

Also generates LLM answers (gpt-oss-120b via OpenRouter) for E2E EM/F1.

The LLM decomposition call happens ONCE per question (100 calls total).
LLM answer generation covers all three policies (~300 calls).

Usage
-----
    python experiments/run_decomposition_eval.py

or with API key override::

    OPENROUTER_API_KEY=sk-... python experiments/run_decomposition_eval.py
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

# -- Set API key BEFORE any imports that read os.environ at module level -------
_HARDCODED_KEY = "REPLACE_WITH_YOUR_OPENROUTER_API_KEY"
if not os.environ.get("OPENROUTER_API_KEY"):
    os.environ["OPENROUTER_API_KEY"] = _HARDCODED_KEY

# -- Project root on sys.path -------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# -- AEA imports --------------------------------------------------------------
from experiments.aea.address_spaces.semantic import SemanticAddressSpace
from experiments.aea.address_spaces.lexical import LexicalAddressSpace
from experiments.aea.address_spaces.entity_graph import EntityGraphAddressSpace
from experiments.aea.evaluation.harness import EvaluationHarness
from experiments.aea.evaluation.metrics import (
    exact_match, f1_score as em_f1, support_recall, utility_at_budget, normalize_cost
)
from experiments.aea.policies.ensemble import EnsemblePolicy
from experiments.aea.policies.heuristic import AEAHeuristicPolicy
from experiments.aea.policies.decomposition_stopping import DecompositionStoppingPolicy
from experiments.aea.answer_generator import AnswerGenerator
from experiments.run_hotpotqa_baselines import (
    load_hotpotqa,
    filter_bridge,
    convert_example,
)

# -- Constants ----------------------------------------------------------------
SEED = 42
N_EVAL = 100
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_FILE = RESULTS_DIR / "decomposition_eval_results.json"

OPENROUTER_API_KEY = os.environ.get(
    "OPENROUTER_API_KEY",
    "REPLACE_WITH_YOUR_OPENROUTER_API_KEY",
)

# Sensitivity analysis mu values (cost-vs-quality trade-off parameter)
MU_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5]


# -- Helpers ------------------------------------------------------------------

def make_address_spaces() -> dict:
    """Fresh set of address spaces (rebuilt per policy)."""
    return {
        "semantic": SemanticAddressSpace(model_name="all-MiniLM-L6-v2"),
        "lexical": LexicalAddressSpace(),
        "entity": EntityGraphAddressSpace(),
    }


def make_policies() -> list:
    """Instantiate the three comparison policies."""
    return [
        EnsemblePolicy(top_k=5, max_steps=3),
        AEAHeuristicPolicy(top_k=5, coverage_threshold=0.5, max_steps=6),
        DecompositionStoppingPolicy(top_k=5, max_steps=8, coverage_threshold=1.0),
    ]


def load_eval_dataset(n: int = N_EVAL, seed: int = SEED) -> list[dict]:
    """Load *n* HotpotQA bridge questions."""
    random.seed(seed)
    np.random.seed(seed)
    raw_data = load_hotpotqa()
    bridge = filter_bridge(raw_data, n=n)
    dataset = [convert_example(ex) for ex in bridge]
    print(f"Loaded {len(dataset)} evaluation examples.")
    return dataset


# -- Retrieval evaluation -----------------------------------------------------

def run_retrieval_eval(dataset: list[dict], seed: int = SEED) -> dict:
    """Run all policies on the dataset; return per-policy result dicts."""
    policies = make_policies()
    all_results: dict = {}

    for policy in policies:
        pname = policy.name()
        print(f"\n{'--' * 30}")
        print(f"Running policy: {pname}")
        print(f"{'--' * 30}")

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
        all_results[pname] = result

        agg = result["aggregated"]
        print(f"  support_recall:  {agg['support_recall']:.4f}")
        print(f"  operations_used: {agg['operations_used']:.2f}")
        print(f"  utility@budget:  {agg['utility_at_budget']:.4f}")
        print(f"  n_errors:        {result['n_errors']}")
        print(f"  runtime:         {elapsed:.1f}s")

        if hasattr(policy, "usage_summary"):
            print(f"  decomposition:   {policy.usage_summary()}")

    return all_results


# -- LLM answer generation ----------------------------------------------------

def generate_llm_answers(
    dataset: list[dict],
    all_results: dict,
    policies_to_evaluate: Optional[list[str]] = None,
) -> dict:
    """
    Generate LLM answers using retrieved evidence for each policy.

    For each example, collects evidence passages from the trace (all steps),
    then calls gpt-oss-120b to generate a short answer.
    """
    if not OPENROUTER_API_KEY:
        print("\nNo OPENROUTER_API_KEY -- skipping LLM answer generation.")
        return all_results

    os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
    generator = AnswerGenerator()

    ex_lookup = {ex["id"]: ex for ex in dataset}
    target_policies = policies_to_evaluate or list(all_results.keys())

    for pname, result in all_results.items():
        if pname not in target_policies:
            for ex_result in result["per_example"]:
                ex_result["llm_answer"] = ""
                ex_result["llm_em"] = 0.0
                ex_result["llm_f1"] = 0.0
            result["aggregated"]["llm_em"] = 0.0
            result["aggregated"]["llm_f1"] = 0.0
            continue

        print(f"\nGenerating LLM answers for policy: {pname}")
        per_example = result["per_example"]
        n_total = len(per_example)

        for i, ex_result in enumerate(per_example):
            if i % 10 == 0:
                print(f"  {i}/{n_total} ...")

            if "error" in ex_result:
                ex_result["llm_answer"] = ""
                ex_result["llm_em"] = 0.0
                ex_result["llm_f1"] = 0.0
                continue

            question = ex_result.get("question", "")
            gold_answer = ex_result.get("gold_answer", "")

            # Collect evidence from all trace steps
            trace = ex_result.get("trace", [])
            ex_id = ex_result.get("id", "")
            ex_data = ex_lookup.get(ex_id, {})
            context_docs = {doc["id"]: doc["content"] for doc in ex_data.get("context", [])}

            seen_ids: set = set()
            evidence: list = []
            for step_entry in trace:
                for it in step_entry.get("result_items", []):
                    doc_id = it.get("id", "")
                    if doc_id and doc_id not in seen_ids and doc_id in context_docs:
                        evidence.append(context_docs[doc_id])
                        seen_ids.add(doc_id)

            if not evidence:
                pred = ex_result.get("predicted_answer", "")
                if pred:
                    evidence = [pred]

            # Limit evidence to top 5 passages to keep prompts short
            evidence = evidence[:5]

            llm_ans = generator.generate_answer(question, evidence)
            ex_result["llm_answer"] = llm_ans
            ex_result["llm_em"] = exact_match(llm_ans, gold_answer)
            ex_result["llm_f1"] = em_f1(llm_ans, gold_answer)

        all_llm_em = [r.get("llm_em", 0.0) for r in per_example]
        all_llm_f1 = [r.get("llm_f1", 0.0) for r in per_example]
        result["aggregated"]["llm_em"] = float(np.mean(all_llm_em))
        result["aggregated"]["llm_f1"] = float(np.mean(all_llm_f1))

        print(f"  LLM EM={result['aggregated']['llm_em']:.4f}  F1={result['aggregated']['llm_f1']:.4f}")
        print(f"  API usage: {generator.usage_summary()}")

    return all_results


# -- Sensitivity analysis -----------------------------------------------------

def compute_sensitivity(all_results: dict, mu_values: list) -> dict:
    """Compute E2E U@B across different mu values (cost-quality trade-off)."""
    sensitivity: dict = {}

    for pname, result in all_results.items():
        per_example = result["per_example"]
        sensitivity[pname] = {}

        for mu in mu_values:
            utils = []
            for ex_r in per_example:
                if "error" in ex_r:
                    utils.append(0.0)
                    continue

                ans_score = ex_r.get("llm_f1", ex_r.get("f1", 0.0))
                ev_score = ex_r.get("support_recall", 0.0)
                ops = ex_r.get("operations_used", 0)
                tokens = ex_r.get("tokens_used", 0)
                latency = ex_r.get("latency_ms_total", 0.0)

                cost = normalize_cost(
                    tokens=tokens,
                    latency_ms=latency,
                    operations=ops,
                    max_tokens=4000,
                    max_latency=30_000.0,
                    max_ops=20,
                )
                u = utility_at_budget(
                    answer_score=ans_score,
                    evidence_score=ev_score,
                    cost=cost,
                    eta=0.5,
                    mu=mu,
                )
                utils.append(u)
            sensitivity[pname][mu] = float(np.mean(utils))

    return sensitivity


# -- Reporting ----------------------------------------------------------------

def print_results(all_results: dict, sensitivity: dict, n_examples: int) -> None:
    """Print full results table to stdout."""
    print("\n" + "=" * 70)
    print(f"=== Decomposition Stopping vs Baselines (N={n_examples}) ===")
    print("=" * 70)

    print("\nRetrieval:")
    print(f"| {'Policy':<25} | {'Recall':>8} | {'Ops':>6} | {'Retrieval U@B':>14} |")
    print(f"| {'-'*25} | {'-'*8} | {'-'*6} | {'-'*14} |")
    for pname, result in all_results.items():
        agg = result["aggregated"]
        print(
            f"| {pname:<25} "
            f"| {agg['support_recall']:>8.4f} "
            f"| {agg['operations_used']:>6.2f} "
            f"| {agg['utility_at_budget']:>14.4f} |"
        )

    print("\nEnd-to-End:")
    has_llm = any("llm_em" in r["aggregated"] for r in all_results.values())
    if has_llm:
        print(
            f"| {'Policy':<25} | {'EM':>6} | {'F1':>6} | {'Recall':>8} "
            f"| {'Ops':>6} | {'E2E U@B (mu=0.3)':>17} |"
        )
        print(f"| {'-'*25} | {'-'*6} | {'-'*6} | {'-'*8} | {'-'*6} | {'-'*17} |")
        for pname, result in all_results.items():
            agg = result["aggregated"]
            mu03 = sensitivity.get(pname, {}).get(0.3, 0.0)
            print(
                f"| {pname:<25} "
                f"| {agg.get('llm_em', 0.0):>6.4f} "
                f"| {agg.get('llm_f1', 0.0):>6.4f} "
                f"| {agg['support_recall']:>8.4f} "
                f"| {agg['operations_used']:>6.2f} "
                f"| {mu03:>17.4f} |"
            )
    else:
        print("  (No LLM answers -- set OPENROUTER_API_KEY to enable.)")

    print("\nSensitivity (E2E U@B across mu values):")
    header_parts = [f"| {'Policy':<25}"]
    for mu in MU_VALUES:
        header_parts.append(f" | {'mu='+str(mu):>8}")
    header_parts.append(" | Winner |")
    print("".join(header_parts))

    sep_parts = [f"| {'-'*25}"]
    for mu in MU_VALUES:
        sep_parts.append(f" | {'-'*8}")
    sep_parts.append(" | ------ |")
    print("".join(sep_parts))

    for mu in MU_VALUES:
        best_policy = max(sensitivity.keys(), key=lambda p: sensitivity[p].get(mu, 0.0))
        row_parts = [f"| {'mu='+str(mu):<25}"]
        for pname in all_results.keys():
            val = sensitivity.get(pname, {}).get(mu, 0.0)
            row_parts.append(f" | {val:>8.4f}")
        row_parts.append(f" | {best_policy:<6} |")
        print("".join(row_parts))

    print()


# -- Main pipeline ------------------------------------------------------------

def run_evaluation(
    n_examples: int = N_EVAL,
    seed: int = SEED,
    results_path: Optional[Path] = RESULTS_FILE,
    run_llm: bool = True,
) -> dict:
    """Full evaluation pipeline."""
    random.seed(seed)
    np.random.seed(seed)

    print("Loading evaluation dataset ...")
    dataset = load_eval_dataset(n=n_examples, seed=seed)

    print("\nRunning retrieval evaluation ...")
    all_results = run_retrieval_eval(dataset, seed=seed)

    if run_llm and OPENROUTER_API_KEY:
        target_policies = list(all_results.keys())
        print(f"\nGenerating LLM answers for: {target_policies}")
        all_results = generate_llm_answers(dataset, all_results, policies_to_evaluate=target_policies)
    else:
        if not OPENROUTER_API_KEY:
            print("\nOPENROUTER_API_KEY not set -- skipping LLM answer generation.")
        for result in all_results.values():
            result["aggregated"]["llm_em"] = 0.0
            result["aggregated"]["llm_f1"] = 0.0

    sensitivity = compute_sensitivity(all_results, MU_VALUES)
    print_results(all_results, sensitivity, n_examples)

    if results_path:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        serialisable = {}
        for pname, result in all_results.items():
            serialisable[pname] = {
                k: v for k, v in result.items()
                if k != "per_example"
            }
            serialisable[pname]["per_example"] = [
                {k2: v2 for k2, v2 in ex.items() if not isinstance(v2, list) or k2 != "trace"}
                for ex in result.get("per_example", [])
            ]
        serialisable["sensitivity"] = {
            p: {str(k): v for k, v in s.items()} for p, s in sensitivity.items()
        }
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump(serialisable, fh, indent=2, default=str)
        print(f"Results saved to: {results_path}")

    return all_results


if __name__ == "__main__":
    run_evaluation()
