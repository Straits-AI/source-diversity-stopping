"""
Learned Stopping vs Baselines — End-to-End Evaluation.

Evaluates five policies on 100 HotpotQA bridge questions (eval split 0-100):
  1. pi_semantic     (baseline)
  2. pi_lexical      (baseline)
  3. pi_ensemble     (upper bound)
  4. pi_aea_heuristic (current method)
  5. pi_learned_stop (NEW — learned classifier)

Also generates LLM answers (gpt-oss-120b via OpenRouter) for E2E EM/F1.

Usage
-----
    OPENROUTER_API_KEY=sk-... python experiments/run_learned_stopping.py
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

# ── Project root on sys.path ─────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── AEA imports ──────────────────────────────────────────────────────────────
from experiments.aea.address_spaces.semantic import SemanticAddressSpace
from experiments.aea.address_spaces.lexical import LexicalAddressSpace
from experiments.aea.address_spaces.entity_graph import EntityGraphAddressSpace
from experiments.aea.evaluation.harness import EvaluationHarness
from experiments.aea.evaluation.metrics import (
    exact_match, f1_score as em_f1, support_recall, utility_at_budget, normalize_cost
)
from experiments.aea.policies.single_substrate import (
    SemanticOnlyPolicy,
    LexicalOnlyPolicy,
)
from experiments.aea.policies.ensemble import EnsemblePolicy
from experiments.aea.policies.heuristic import AEAHeuristicPolicy
from experiments.aea.policies.learned_stopping import LearnedStoppingPolicy
from experiments.aea.answer_generator import AnswerGenerator
from experiments.run_hotpotqa_baselines import (
    load_hotpotqa,
    filter_bridge,
    convert_example,
)

# ── Constants ────────────────────────────────────────────────────────────────
SEED = 42
N_EVAL = 100   # Evaluation split: questions 0-100 (no overlap with training 200-700)
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_FILE = RESULTS_DIR / "learned_stopping_results.json"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "stopping_classifier.pkl"

OPENROUTER_API_KEY = os.environ.get(
    "OPENROUTER_API_KEY",
    "REPLACE_WITH_YOUR_OPENROUTER_API_KEY",
)

# Sensitivity analysis mu values
MU_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5]


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_address_spaces() -> dict:
    return {
        "semantic": SemanticAddressSpace(model_name="all-MiniLM-L6-v2"),
        "lexical": LexicalAddressSpace(),
        "entity": EntityGraphAddressSpace(),
    }


def make_policies() -> list:
    return [
        SemanticOnlyPolicy(top_k=5, max_steps=2),
        LexicalOnlyPolicy(top_k=5, max_steps=2),
        EnsemblePolicy(top_k=5, max_steps=3),
        AEAHeuristicPolicy(top_k=5, coverage_threshold=0.5, max_steps=6),
        LearnedStoppingPolicy(model_path=MODEL_PATH, top_k=5, max_steps=8),
    ]


def load_eval_dataset(n: int = N_EVAL, seed: int = SEED) -> list[dict]:
    """Load the evaluation split (questions 0-100)."""
    raw_data = load_hotpotqa()
    bridge = filter_bridge(raw_data, n=n)
    dataset = [convert_example(ex) for ex in bridge]
    print(f"Loaded {len(dataset)} evaluation examples (questions 0-{n}).")
    return dataset


def run_retrieval_eval(
    dataset: list[dict],
    seed: int = SEED,
) -> dict:
    """Run all policies on the dataset and collect retrieval metrics."""
    policies = make_policies()
    all_results: dict = {}

    for policy in policies:
        pname = policy.name()
        print(f"\n{'─' * 60}")
        print(f"Running policy: {pname}")
        print(f"{'─' * 60}")

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

    return all_results


def generate_llm_answers(
    dataset: list[dict],
    all_results: dict,
    policies_to_evaluate: Optional[list[str]] = None,
) -> dict:
    """
    Generate LLM answers for each policy's workspace evidence.

    Adds 'llm_answers' to each per-example result.

    Parameters
    ----------
    policies_to_evaluate : list of policy names to run LLM for.
        Defaults to all policies. Pass a subset to save API calls.
    """
    if not OPENROUTER_API_KEY:
        print("\nNo OPENROUTER_API_KEY set — skipping LLM answer generation.")
        return all_results

    os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
    generator = AnswerGenerator()

    # Build a lookup from question_id -> context docs
    ex_lookup = {ex["id"]: ex for ex in dataset}

    # Determine which policies to run LLM for
    target_policies = policies_to_evaluate or list(all_results.keys())

    for pname, result in all_results.items():
        if pname not in target_policies:
            # Still set metrics to 0 for consistency
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

            # Collect evidence from ALL trace steps (not just the last step)
            # The trace stores result_items per step; union gives full workspace
            trace = ex_result.get("trace", [])
            ex_id = ex_result.get("id", "")
            ex_data = ex_lookup.get(ex_id, {})
            context_docs = {doc["id"]: doc["content"] for doc in ex_data.get("context", [])}

            seen_ids: set[str] = set()
            evidence: list[str] = []
            for step_entry in trace:
                for it in step_entry.get("result_items", []):
                    doc_id = it.get("id", "")
                    if doc_id and doc_id not in seen_ids and doc_id in context_docs:
                        evidence.append(context_docs[doc_id])
                        seen_ids.add(doc_id)

            # Fallback: use predicted_answer as evidence if empty
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

        # Recompute aggregated llm metrics
        all_llm_em = [r.get("llm_em", 0.0) for r in per_example]
        all_llm_f1 = [r.get("llm_f1", 0.0) for r in per_example]
        result["aggregated"]["llm_em"] = float(np.mean(all_llm_em))
        result["aggregated"]["llm_f1"] = float(np.mean(all_llm_f1))

        print(f"  LLM EM={result['aggregated']['llm_em']:.4f}  F1={result['aggregated']['llm_f1']:.4f}")
        print(f"  API usage: {generator.usage_summary()}")

    return all_results


def compute_sensitivity(all_results: dict, mu_values: list[float]) -> dict:
    """
    Compute E2E U@B across different mu values for each policy.

    Uses support_recall as the evidence score and LLM F1 (or heuristic F1)
    as the answer score.
    """
    sensitivity: dict[str, dict[float, float]] = {}

    for pname, result in all_results.items():
        per_example = result["per_example"]
        sensitivity[pname] = {}

        for mu in mu_values:
            utils = []
            for ex_r in per_example:
                if "error" in ex_r:
                    utils.append(0.0)
                    continue

                # Prefer LLM F1 for answer score; fall back to heuristic F1
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


def print_results(all_results: dict, sensitivity: dict, n_examples: int) -> None:
    """Print full results table."""
    print("\n" + "=" * 70)
    print(f"=== Learned Stopping vs Baselines (N={n_examples}) ===")
    print("=" * 70)

    # ── Retrieval table ──────────────────────────────────────────────────────
    print("\nRetrieval:")
    print(f"| {'Policy':<22} | {'Recall':>8} | {'Ops':>6} | {'Retrieval U@B':>14} |")
    print(f"| {'-'*22} | {'-'*8} | {'-'*6} | {'-'*14} |")
    for pname, result in all_results.items():
        agg = result["aggregated"]
        print(
            f"| {pname:<22} "
            f"| {agg['support_recall']:>8.4f} "
            f"| {agg['operations_used']:>6.2f} "
            f"| {agg['utility_at_budget']:>14.4f} |"
        )

    # ── End-to-End table ─────────────────────────────────────────────────────
    print("\nEnd-to-End:")
    has_llm = any("llm_em" in r["aggregated"] for r in all_results.values())
    if has_llm:
        print(
            f"| {'Policy':<22} | {'EM':>6} | {'F1':>6} | {'Recall':>8} "
            f"| {'Ops':>6} | {'E2E U@B (mu=0.3)':>17} |"
        )
        print(f"| {'-'*22} | {'-'*6} | {'-'*6} | {'-'*8} | {'-'*6} | {'-'*17} |")
        for pname, result in all_results.items():
            agg = result["aggregated"]
            mu03 = sensitivity.get(pname, {}).get(0.3, 0.0)
            print(
                f"| {pname:<22} "
                f"| {agg.get('llm_em', 0.0):>6.4f} "
                f"| {agg.get('llm_f1', 0.0):>6.4f} "
                f"| {agg['support_recall']:>8.4f} "
                f"| {agg['operations_used']:>6.2f} "
                f"| {mu03:>17.4f} |"
            )
    else:
        print("  (No LLM answers generated — set OPENROUTER_API_KEY to enable.)")

    # ── Sensitivity table ────────────────────────────────────────────────────
    print("\nSensitivity (E2E U@B across mu values):")
    mu_cols = [0.1, 0.2, 0.3, 0.4, 0.5]
    header_parts = [f"| {'Policy':<22}"]
    for mu in mu_cols:
        header_parts.append(f" | {'mu='+str(mu):>8}")
    header_parts.append(" | Winner |")
    print("".join(header_parts))

    sep_parts = [f"| {'-'*22}"]
    for mu in mu_cols:
        sep_parts.append(f" | {'-'*8}")
    sep_parts.append(" | ------ |")
    print("".join(sep_parts))

    for mu in mu_cols:
        best_policy = max(sensitivity.keys(), key=lambda p: sensitivity[p].get(mu, 0.0))
        row_parts = [f"| {'mu='+str(mu):<22}"]
        for pname in all_results.keys():
            val = sensitivity.get(pname, {}).get(mu, 0.0)
            row_parts.append(f" | {val:>8.4f}")
        row_parts.append(f" | {best_policy:<6} |")
        print("".join(row_parts))

    # ── Classifier stats ─────────────────────────────────────────────────────
    learned_result = all_results.get("pi_learned_stop")
    if learned_result:
        try:
            import pickle
            model_bundle_path = Path(__file__).resolve().parent / "models" / "stopping_classifier.pkl"
            if model_bundle_path.exists():
                with open(model_bundle_path, "rb") as fh:
                    bundle = pickle.load(fh)
                metrics = bundle.get("eval_metrics", {})
                importances = bundle.get("feature_importances", [])
                threshold = bundle.get("threshold", 0.5)
                print(f"\nClassifier Stats:")
                print(f"  Test accuracy:      {metrics.get('accuracy', 0):.4f}")
                print(f"  Test precision:     {metrics.get('precision', 0):.4f}")
                print(f"  Test recall:        {metrics.get('recall', 0):.4f}")
                print(f"  Test F1:            {metrics.get('f1', 0):.4f}")
                print(f"  Optimal threshold:  {threshold:.2f}")
                print(f"  Feature importances:")
                sorted_imp = sorted(importances, key=lambda x: abs(x[1]), reverse=True)
                for fname, imp in sorted_imp[:5]:
                    print(f"    {fname:<30}: {imp:+.4f}")
        except Exception as exc:
            print(f"  (Could not load classifier stats: {exc})")

    print()


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
        # Only generate LLM answers for the 3 key policies (saves ~300 API calls)
        # pi_semantic (low recall baseline), pi_aea_heuristic (current method), pi_learned_stop (new)
        key_policies = ["pi_semantic", "pi_aea_heuristic", "pi_learned_stop"]
        print(f"\nGenerating LLM answers for key policies: {key_policies} ...")
        all_results = generate_llm_answers(dataset, all_results, policies_to_evaluate=key_policies)
    else:
        if not OPENROUTER_API_KEY:
            print("\nOPENROUTER_API_KEY not set — skipping LLM answer generation.")
        for result in all_results.values():
            result["aggregated"]["llm_em"] = 0.0
            result["aggregated"]["llm_f1"] = 0.0

    sensitivity = compute_sensitivity(all_results, MU_VALUES)

    print_results(all_results, sensitivity, n_examples)

    if results_path:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        # Serialise safely (skip arrays)
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
        serialisable["sensitivity"] = {p: {str(k): v for k, v in s.items()} for p, s in sensitivity.items()}
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump(serialisable, fh, indent=2, default=str)
        print(f"Results saved to: {results_path}")

    return all_results


if __name__ == "__main__":
    run_evaluation()
