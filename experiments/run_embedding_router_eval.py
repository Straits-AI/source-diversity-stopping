"""
Embedding Router vs Baselines — End-to-End Evaluation.

Tests whether question-level routing based on question EMBEDDINGS outperforms
the hand-coded heuristic (pi_aea_heuristic).

Experimental design
-------------------
* Training split  : questions 500-999 (bridge questions from HotpotQA dev)
* Evaluation split: questions   0-499 (clean, no overlap with training)

Policies evaluated
------------------
1. pi_ensemble          — upper bound (all substrates, 3 steps)
2. pi_aea_heuristic     — current method (regex-based routing)
3. pi_embedding_router  — NEW: embedding-based routing classifier

Key question
------------
Does semantic (embedding-based) question routing beat the regex heuristic?

Usage
-----
    python experiments/run_embedding_router_eval.py

Seed: 42
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

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
    support_recall,
    utility_at_budget,
    normalize_cost,
)
from experiments.aea.policies.ensemble import EnsemblePolicy
from experiments.aea.policies.heuristic import AEAHeuristicPolicy
from experiments.aea.policies.embedding_router import EmbeddingRouterPolicy
from experiments.run_hotpotqa_baselines import (
    load_hotpotqa,
    filter_bridge,
    convert_example,
)

# -- Constants ----------------------------------------------------------------
SEED = 42
N_EVAL = 500          # Evaluation split: questions 0-499
N_TRAIN = 500         # Training split  : next 500 questions (500-999)
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_FILE = RESULTS_DIR / "embedding_router_results.json"


# -- Data loading -------------------------------------------------------------

def load_splits(seed: int = SEED) -> tuple[list[dict], list[dict]]:
    """
    Load HotpotQA and split into eval (0-499) and train (500-999).

    Returns
    -------
    eval_dataset : list[dict]
    train_dataset : list[dict]
    """
    raw_data = load_hotpotqa()
    bridge = filter_bridge(raw_data, n=1000)

    if len(bridge) < N_EVAL + N_TRAIN:
        print(
            f"Warning: only {len(bridge)} bridge questions available "
            f"(need {N_EVAL + N_TRAIN}).  Adjusting splits."
        )
        n_eval = min(N_EVAL, len(bridge) // 2)
        n_train = len(bridge) - n_eval
    else:
        n_eval = N_EVAL
        n_train = N_TRAIN

    eval_examples  = [convert_example(ex) for ex in bridge[:n_eval]]
    train_examples = [convert_example(ex) for ex in bridge[n_eval:n_eval + n_train]]

    print(f"Eval split  : {len(eval_examples)} questions (idx 0-{n_eval-1})")
    print(f"Train split : {len(train_examples)} questions (idx {n_eval}-{n_eval+n_train-1})")
    return eval_examples, train_examples


# -- Address-space factory -----------------------------------------------------

def make_address_spaces() -> dict:
    """Create fresh address-space instances (indices rebuilt per example in harness)."""
    return {
        "semantic": SemanticAddressSpace(model_name="all-MiniLM-L6-v2"),
        "lexical": LexicalAddressSpace(),
        "entity": EntityGraphAddressSpace(),
    }


# -- Policy construction -------------------------------------------------------

def build_embedding_router(
    train_dataset: list[dict],
    seed: int = SEED,
) -> EmbeddingRouterPolicy:
    """
    Build and train the EmbeddingRouterPolicy on the training split.

    We pass dedicated address-space instances for training so the evaluation
    harness address spaces are not polluted with training indices.
    """
    train_address_spaces = make_address_spaces()
    policy = EmbeddingRouterPolicy(
        train_dataset=train_dataset,
        address_spaces_for_training=train_address_spaces,
        top_k=5,
        max_steps=8,
        model_name="all-MiniLM-L6-v2",
        seed=seed,
    )
    # Trigger training immediately (also validates the pipeline)
    policy._fit_classifier()
    return policy


# -- Evaluation ----------------------------------------------------------------

def run_retrieval_eval(
    eval_dataset: list[dict],
    train_dataset: list[dict],
    seed: int = SEED,
) -> dict:
    """Run all three policies on the eval split and collect retrieval metrics."""

    print("\nBuilding embedding router (train on questions 500-999) ...")
    embedding_router = build_embedding_router(train_dataset, seed=seed)

    policies = [
        EnsemblePolicy(top_k=5, max_steps=3),
        AEAHeuristicPolicy(top_k=5, coverage_threshold=0.5, max_steps=6),
        embedding_router,
    ]

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
        result = harness.evaluate(policy, eval_dataset)
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


def compute_sensitivity(all_results: dict, mu_values: list[float]) -> dict:
    """Compute U@B across different mu values for each policy."""
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

                ans_score = ex_r.get("f1", 0.0)
                ev_score  = ex_r.get("support_recall", 0.0)
                ops       = ex_r.get("operations_used", 0)
                tokens    = ex_r.get("tokens_used", 0)
                latency   = ex_r.get("latency_ms_total", 0.0)

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


def analyse_routing_decisions(
    policy: EmbeddingRouterPolicy,
    eval_dataset: list[dict],
) -> dict:
    """
    Analyse which strategy the embedding router assigned to each question.

    Returns a summary dict with per-strategy counts and example questions.
    """
    questions = [ex["question"] for ex in eval_dataset]
    strategies = policy.predict_strategy_batch(questions)

    strategy_names = {
        0: "stop-after-semantic",
        1: "semantic-then-lexical",
        2: "semantic-then-hop",
    }

    analysis: dict = {"counts": {}, "examples": {}}
    for s in [0, 1, 2]:
        name = strategy_names[s]
        idxs = [i for i, st in enumerate(strategies) if st == s]
        analysis["counts"][name] = len(idxs)
        # Show up to 3 example questions per strategy
        analysis["examples"][name] = [
            questions[i] for i in idxs[:3]
        ]

    return analysis


def print_results(
    all_results: dict,
    sensitivity: dict,
    routing_analysis: Optional[dict],
    n_eval: int,
    n_train: int,
) -> None:
    """Print full results table."""
    mu_values = [0.1, 0.2, 0.3, 0.4, 0.5]

    print("\n" + "=" * 70)
    print(f"=== Embedding Router vs Baselines (eval N={n_eval}, train N={n_train}) ===")
    print("=" * 70)

    # -- Retrieval table ------------------------------------------------------
    print("\nRetrieval Metrics:")
    print(f"| {'Policy':<25} | {'Recall':>8} | {'Ops':>6} | {'U@B':>8} |")
    print(f"| {'-'*25} | {'-'*8} | {'-'*6} | {'-'*8} |")

    for pname, result in all_results.items():
        agg = result["aggregated"]
        print(
            f"| {pname:<25} "
            f"| {agg['support_recall']:>8.4f} "
            f"| {agg['operations_used']:>6.2f} "
            f"| {agg['utility_at_budget']:>8.4f} |"
        )

    # -- Comparison: embedding_router vs heuristic ----------------------------
    heuristic = all_results.get("pi_aea_heuristic", {}).get("aggregated", {})
    router    = all_results.get("pi_embedding_router", {}).get("aggregated", {})
    if heuristic and router:
        delta_recall = router.get("support_recall", 0) - heuristic.get("support_recall", 0)
        delta_ops    = router.get("operations_used", 0) - heuristic.get("operations_used", 0)
        delta_uab    = router.get("utility_at_budget", 0) - heuristic.get("utility_at_budget", 0)
        print(f"\nEmbedding Router vs Heuristic (delta):")
        print(f"  support_recall:  {delta_recall:+.4f}")
        print(f"  operations_used: {delta_ops:+.2f}")
        print(f"  utility@budget:  {delta_uab:+.4f}")
        verdict = "BETTER" if delta_uab > 0 else "WORSE"
        print(f"  Verdict: Embedding router is {verdict} than heuristic on U@B.")

    # -- Sensitivity table ----------------------------------------------------
    print("\nSensitivity (U@B across mu values):")
    header_parts = [f"| {'Policy':<25}"]
    for mu in mu_values:
        header_parts.append(f" | {'mu='+str(mu):>8}")
    header_parts.append(" |")
    print("".join(header_parts))

    sep_parts = [f"| {'-'*25}"]
    for mu in mu_values:
        sep_parts.append(f" | {'-'*8}")
    sep_parts.append(" |")
    print("".join(sep_parts))

    for pname in all_results.keys():
        row_parts = [f"| {pname:<25}"]
        for mu in mu_values:
            val = sensitivity.get(pname, {}).get(mu, 0.0)
            row_parts.append(f" | {val:>8.4f}")
        row_parts.append(" |")
        print("".join(row_parts))

    # -- Winner per mu --------------------------------------------------------
    print("\nWinner per mu:")
    for mu in mu_values:
        best = max(all_results.keys(), key=lambda p: sensitivity.get(p, {}).get(mu, 0.0))
        print(f"  mu={mu}: {best}")

    # -- Routing analysis -----------------------------------------------------
    if routing_analysis:
        print("\nEmbedding Router -- Strategy Distribution (eval set):")
        for strategy_name, count in routing_analysis["counts"].items():
            pct = 100.0 * count / n_eval if n_eval > 0 else 0.0
            print(f"  {strategy_name:<25}: {count:>4} ({pct:.1f}%)")
        print("\nExample questions per strategy:")
        for strategy_name, examples in routing_analysis["examples"].items():
            print(f"  [{strategy_name}]")
            for q in examples:
                print(f"    * {q[:90]}")

    print()


def run_evaluation(
    n_eval: int = N_EVAL,
    n_train: int = N_TRAIN,
    seed: int = SEED,
    results_path: Optional[Path] = RESULTS_FILE,
) -> dict:
    """Full evaluation pipeline."""
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print("Embedding Router Evaluation")
    print(f"  Eval split  : questions 0-{n_eval-1}")
    print(f"  Train split : questions {n_eval}-{n_eval+n_train-1}")
    print(f"  Seed        : {seed}")
    print("=" * 70)

    # Load data
    eval_dataset, train_dataset = load_splits(seed=seed)

    # Run retrieval assessment
    print("\nRunning retrieval assessment ...")
    all_results = run_retrieval_eval(eval_dataset, train_dataset, seed=seed)

    # Sensitivity analysis
    mu_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    sensitivity = compute_sensitivity(all_results, mu_values)

    # Routing analysis (re-train to get the policy object back)
    routing_analysis: Optional[dict] = None
    try:
        router_policy = build_embedding_router(train_dataset, seed=seed)
        routing_analysis = analyse_routing_decisions(router_policy, eval_dataset)
    except Exception as exc:
        print(f"[Warning] Could not compute routing analysis: {exc}")

    print_results(all_results, sensitivity, routing_analysis, n_eval, n_train)

    # Save results
    if results_path:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        serialisable: dict = {}
        for pname, result in all_results.items():
            serialisable[pname] = {
                k: v for k, v in result.items()
                if k != "per_example"
            }
            serialisable[pname]["per_example"] = [
                {k2: v2 for k2, v2 in ex.items() if k2 != "trace"}
                for ex in result.get("per_example", [])
            ]
        serialisable["sensitivity"] = {
            p: {str(k): v for k, v in s.items()}
            for p, s in sensitivity.items()
        }
        if routing_analysis:
            serialisable["routing_analysis"] = routing_analysis
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump(serialisable, fh, indent=2, default=str)
        print(f"Results saved to: {results_path}")

    return all_results


if __name__ == "__main__":
    run_evaluation()
