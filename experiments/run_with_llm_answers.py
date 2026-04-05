"""
HotpotQA Bridge — AEA evaluation with LLM-generated answers.

Runs all five policies through the standard retrieval harness (same as
run_hotpotqa_baselines.py), then calls an LLM (via OpenRouter) to generate
a short answer for each question from the retrieved workspace evidence.
Reports EM, F1, SupportRecall, AvgOps, and Utility@Budget (F1-based).

Usage
-----
    python experiments/run_with_llm_answers.py

Results are saved to experiments/results/llm_answers.json.
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
    exact_match,
    f1_score,
    support_precision,
    utility_at_budget,
)
from experiments.aea.policies.single_substrate import (
    SemanticOnlyPolicy,
    LexicalOnlyPolicy,
    EntityOnlyPolicy,
)
from experiments.aea.policies.ensemble import EnsemblePolicy
from experiments.aea.policies.heuristic import AEAHeuristicPolicy
from experiments.aea.answer_generator import AnswerGenerator

# Re-use data-loading helpers from the baseline runner
from experiments.run_hotpotqa_baselines import (
    load_hotpotqa,
    filter_bridge,
    convert_example,
)

# ── Constants ────────────────────────────────────────────────────────────────
N_EXAMPLES = 50    # Reduced for API cost/latency on free tier (baseline used 100)
SEED = 42
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_FILE = RESULTS_DIR / "llm_answers.json"


# ── Factory helpers ───────────────────────────────────────────────────────────

def make_policies() -> list:
    """Instantiate all five policies."""
    return [
        SemanticOnlyPolicy(top_k=5, max_steps=2),
        LexicalOnlyPolicy(top_k=5, max_steps=2),
        EntityOnlyPolicy(top_k=5, max_steps=3),
        EnsemblePolicy(top_k=5, max_steps=3),
        AEAHeuristicPolicy(top_k=5, coverage_threshold=0.5, max_steps=6),
    ]


def make_address_spaces() -> dict:
    """Create fresh address-space instances (rebuilt per policy run)."""
    return {
        "semantic": SemanticAddressSpace(model_name="all-MiniLM-L6-v2"),
        "lexical": LexicalAddressSpace(),
        "entity": EntityGraphAddressSpace(),
    }


# ── Workspace content reconstruction ─────────────────────────────────────────

def extract_workspace_passages(per_ex: dict, example: dict) -> list[str]:
    """
    Reconstruct workspace passage texts from the trace result_items.

    The harness trace records which document IDs were retrieved in each step.
    We look those IDs up in the example's context documents.

    Parameters
    ----------
    per_ex : dict
        Per-example result from EvaluationHarness.evaluate().
    example : dict
        Original converted example with "context" list.

    Returns
    -------
    list[str]
        Passage strings for the top retrieved docs (best-score first).
    """
    id_to_content = {doc["id"]: doc["content"] for doc in example.get("context", [])}

    # Collect retrieved IDs from trace, keeping best score per ID
    best_score: dict[str, float] = {}
    for step in per_ex.get("trace", []):
        for item in step.get("result_items", []):
            doc_id = item.get("id", "")
            score = float(item.get("score", 0.0))
            if doc_id and (doc_id not in best_score or best_score[doc_id] < score):
                best_score[doc_id] = score

    # Sort by score descending
    sorted_ids = sorted(best_score, key=lambda k: -best_score[k])
    return [id_to_content[rid] for rid in sorted_ids if rid in id_to_content]


# ── LLM answer generation pass ────────────────────────────────────────────────

def generate_policy_answers(
    per_example: list[dict],
    dataset: list[dict],
    generator: AnswerGenerator,
    policy_name: str,
) -> list[dict]:
    """
    For each example, call the AnswerGenerator with retrieved workspace passages.
    Returns enriched per-example dicts with llm_answer, llm_em, llm_f1 added.
    """
    id_to_example = {ex["id"]: ex for ex in dataset}
    enriched: list[dict] = []
    total = len(per_example)

    for idx, ex in enumerate(per_example):
        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"    [{policy_name}] LLM answer {idx + 1}/{total} …", flush=True)

        example = id_to_example.get(ex.get("id", ""), {})
        passages = extract_workspace_passages(ex, example)
        question = ex.get("question", "")
        gold = ex.get("gold_answer", "")

        llm_answer = generator.generate_answer(question, passages)
        em = exact_match(llm_answer, gold)
        f1 = f1_score(llm_answer, gold)

        enriched_ex = dict(ex)
        enriched_ex["llm_answer"] = llm_answer
        enriched_ex["llm_em"] = em
        enriched_ex["llm_f1"] = f1
        enriched.append(enriched_ex)

    return enriched


# ── Utility@Budget with LLM F1 ───────────────────────────────────────────────

_MAX_OPS = 18   # Max expected ops (6 steps × 3 ops/step for AEA)


def compute_llm_utility(ex: dict) -> float:
    """
    U@B = F1 * (1 + 0.5 * SupportPrecision) - 0.3 * NormalisedOps

    NormalisedOps = min(1, operations_used / MAX_OPS)
    """
    f1 = ex.get("llm_f1", 0.0)
    sp_prec = ex.get("support_precision", 0.0)
    ops = ex.get("operations_used", 0)
    norm_ops = min(1.0, ops / _MAX_OPS)
    return utility_at_budget(
        answer_score=f1,
        evidence_score=sp_prec,
        cost=norm_ops,
    )


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_llm(per_example: list[dict]) -> dict:
    """Average LLM metrics across examples."""
    def mean(key: str) -> float:
        vals = [ex[key] for ex in per_example if key in ex and isinstance(ex[key], (int, float))]
        return float(np.mean(vals)) if vals else 0.0

    return {
        "em": mean("llm_em"),
        "f1": mean("llm_f1"),
        "support_recall": mean("support_recall"),
        "support_precision": mean("support_precision"),
        "avg_ops": mean("operations_used"),
        "utility_at_budget": mean("llm_utility_at_budget"),
    }


# ── Main evaluation loop ──────────────────────────────────────────────────────

def run_with_llm_answers(
    n_examples: int = N_EXAMPLES,
    seed: int = SEED,
    results_path: Optional[Path] = RESULTS_FILE,
) -> dict:
    """
    Full pipeline: retrieval + LLM answer generation for all policies.

    Returns
    -------
    dict
        Keyed by policy name; each value has "aggregated" and "per_example".
    """
    random.seed(seed)
    np.random.seed(seed)

    # ── Load data ────────────────────────────────────────────────────────────
    raw_data = load_hotpotqa()
    bridge = filter_bridge(raw_data, n=n_examples)
    dataset = [convert_example(ex) for ex in bridge]
    actual_n = len(dataset)
    print(f"Converted {actual_n} examples for evaluation.\n", flush=True)

    policies = make_policies()
    generator = AnswerGenerator()
    all_results: dict = {}

    # ── Run each policy ──────────────────────────────────────────────────────
    for policy in policies:
        pname = policy.name()
        print(f"{'─' * 60}", flush=True)
        print(f"Policy: {pname}", flush=True)
        print(f"{'─' * 60}", flush=True)

        # Step 1 — retrieval pass via standard harness
        print(f"  Retrieval phase …", flush=True)
        address_spaces = make_address_spaces()
        harness = EvaluationHarness(
            address_spaces=address_spaces,
            max_steps=10,
            token_budget=4000,
            seed=seed,
        )
        t0 = time.perf_counter()
        harness_result = harness.evaluate(policy, dataset)
        retrieval_time = time.perf_counter() - t0
        per_example = harness_result["per_example"]
        print(f"  Retrieval done in {retrieval_time:.1f}s  (n_errors={harness_result['n_errors']})", flush=True)

        # Step 2 — LLM answer generation
        print(f"  LLM answer generation …", flush=True)
        t1 = time.perf_counter()
        per_example = generate_policy_answers(per_example, dataset, generator, pname)
        llm_time = time.perf_counter() - t1
        print(f"  LLM generation done in {llm_time:.1f}s", flush=True)

        # Step 3 — compute LLM Utility@Budget per example
        for ex in per_example:
            ex["llm_utility_at_budget"] = compute_llm_utility(ex)

        # Step 4 — aggregate
        aggregated = aggregate_llm(per_example)
        n_errors = harness_result["n_errors"]

        all_results[pname] = {
            "per_example": per_example,
            "aggregated": aggregated,
            "policy_name": pname,
            "n_examples": actual_n,
            "n_errors": n_errors,
            "retrieval_time_s": round(retrieval_time, 2),
            "llm_time_s": round(llm_time, 2),
        }

        agg = aggregated
        print(f"  EM:               {agg['em']:.4f}", flush=True)
        print(f"  F1:               {agg['f1']:.4f}", flush=True)
        print(f"  SupportRecall:    {agg['support_recall']:.4f}", flush=True)
        print(f"  SupportPrecision: {agg['support_precision']:.4f}", flush=True)
        print(f"  AvgOps:           {agg['avg_ops']:.2f}", flush=True)
        print(f"  Utility@Budget:   {agg['utility_at_budget']:.4f}", flush=True)
        print(f"  n_errors:         {n_errors}\n", flush=True)

    # ── Print summary table ──────────────────────────────────────────────────
    _print_summary_table(all_results, actual_n)

    # ── API usage ────────────────────────────────────────────────────────────
    usage = generator.usage_summary()
    print(f"API usage: {usage['total_calls']} calls, "
          f"{usage['total_tokens']} tokens total, "
          f"{usage['total_errors']} errors", flush=True)

    # ── Save ─────────────────────────────────────────────────────────────────
    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        serialisable = {}
        for pname, res in all_results.items():
            # Omit heavy trace fields to keep file size manageable
            slim_per_example = []
            for ex in res["per_example"]:
                ex_copy = {k: v for k, v in ex.items() if k != "trace"}
                slim_per_example.append(ex_copy)
            serialisable[pname] = {**res, "per_example": slim_per_example}

        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump({"results": serialisable, "api_usage": usage}, fh, indent=2)
        print(f"\nResults saved to: {results_path}", flush=True)

    return all_results


# ── Summary table ─────────────────────────────────────────────────────────────

def _print_summary_table(all_results: dict, n_examples: int) -> None:
    col = {"Policy": 22, "EM": 8, "F1": 8, "SR": 15, "AvgOps": 9, "U@B": 14}

    def hline() -> str:
        return (
            f"| {'-' * col['Policy']} "
            f"| {'-' * col['EM']} "
            f"| {'-' * col['F1']} "
            f"| {'-' * col['SR']} "
            f"| {'-' * col['AvgOps']} "
            f"| {'-' * col['U@B']} |"
        )

    print(f"\n=== HotpotQA with LLM Answer Generation (N={n_examples}) ===\n", flush=True)
    header = (
        f"| {'Policy':<{col['Policy']}} "
        f"| {'EM':>{col['EM']}} "
        f"| {'F1':>{col['F1']}} "
        f"| {'SupportRecall':>{col['SR']}} "
        f"| {'AvgOps':>{col['AvgOps']}} "
        f"| {'Utility@Budget':>{col['U@B']}} |"
    )
    print(header, flush=True)
    print(hline(), flush=True)

    for pname, result in all_results.items():
        agg = result["aggregated"]
        row = (
            f"| {pname:<{col['Policy']}} "
            f"| {agg['em']:>{col['EM']}.4f} "
            f"| {agg['f1']:>{col['F1']}.4f} "
            f"| {agg['support_recall']:>{col['SR']}.4f} "
            f"| {agg['avg_ops']:>{col['AvgOps']}.2f} "
            f"| {agg['utility_at_budget']:>{col['U@B']}.4f} |"
        )
        print(row, flush=True)

    print(flush=True)

    # ── Key finding ───────────────────────────────────────────────────────────
    aea_key = next((k for k in all_results if "aea" in k.lower()), None)
    if aea_key:
        aea_agg = all_results[aea_key]["aggregated"]
        baselines = {k: v for k, v in all_results.items() if k != aea_key}
        if baselines:
            best_em = max(v["aggregated"]["em"] for v in baselines.values())
            best_f1 = max(v["aggregated"]["f1"] for v in baselines.values())
            aea_em = aea_agg["em"]
            aea_f1 = aea_agg["f1"]
            print("Key finding: Does better retrieval (AEA) translate to better answers?", flush=True)
            print(f"  AEA EM vs best baseline EM:  {aea_em:.4f} vs {best_em:.4f}  "
                  f"(delta = {aea_em - best_em:+.4f})", flush=True)
            print(f"  AEA F1 vs best baseline F1:  {aea_f1:.4f} vs {best_f1:.4f}  "
                  f"(delta = {aea_f1 - best_f1:+.4f})", flush=True)
            print(flush=True)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_with_llm_answers()
