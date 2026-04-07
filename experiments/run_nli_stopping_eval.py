"""
NLI-based bundle sufficiency stopping policy evaluation — N=500 HotpotQA bridge questions.

Compares three policies:
  1. pi_ensemble          — brute-force baseline (all substrates each step)
  2. pi_aea_heuristic     — workspace-statistics stopping (current best)
  3. pi_nli_stopping      — NLI bundle sufficiency check (principled content-aware baseline)

Key research question: does NLI-based bundle assessment (set function) beat the
workspace-statistics heuristic?

  - If NLI ALSO fails to beat the heuristic → strong negative result supporting
    the structural signal thesis (workspace statistics encode something content
    signals miss).
  - If NLI beats the heuristic → demonstrates a working content-aware method,
    completing the story.

Usage
-----
    python experiments/run_nli_stopping_eval.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

# Ensure stdout is unbuffered so output appears in background task logs
os.environ.setdefault("PYTHONUNBUFFERED", "1")
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass

import numpy as np
from scipy import stats  # type: ignore

# Prevent HuggingFace tokenizer deadlocks on macOS
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Set API key before importing answer_generator (reads env var at import time)
_OPENROUTER_KEY = "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"
os.environ.setdefault("OPENROUTER_API_KEY", _OPENROUTER_KEY)

# Project root on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.aea.address_spaces.semantic import SemanticAddressSpace
from experiments.aea.address_spaces.lexical import LexicalAddressSpace
from experiments.aea.address_spaces.entity_graph import EntityGraphAddressSpace
from experiments.aea.evaluation.harness import EvaluationHarness
from experiments.aea.policies.ensemble import EnsemblePolicy
from experiments.aea.policies.heuristic import AEAHeuristicPolicy
from experiments.aea.policies.nli_stopping import NLIStoppingPolicy
from experiments.aea.answer_generator import AnswerGenerator
from experiments.aea.evaluation.metrics import exact_match, f1_score
from experiments.run_hotpotqa_baselines import (
    load_hotpotqa,
    filter_bridge,
    convert_example,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

N_EXAMPLES: int = 500
SEED: int = 42
MU: float = 0.3
MAX_OPS: float = 3.0
N_BOOTSTRAP: int = 1000
API_CALL_DELAY: float = 0.1

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_FILE = RESULTS_DIR / "nli_stopping.json"

LLM_POLICIES = ["pi_ensemble", "pi_aea_heuristic", "pi_nli_stopping"]

# Shared SentenceTransformer encoder (avoid macOS deadlocks)
_SHARED_ENCODER = None


def _get_shared_encoder():
    global _SHARED_ENCODER
    if _SHARED_ENCODER is None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        print("  [setup] Loading SentenceTransformer model (once)...")
        _SHARED_ENCODER = SentenceTransformer("all-MiniLM-L6-v2")
        print("  [setup] SentenceTransformer loaded.")
    return _SHARED_ENCODER


def make_address_spaces() -> dict:
    sem = SemanticAddressSpace(model_name="all-MiniLM-L6-v2")
    sem._encoder = _get_shared_encoder()
    return {
        "semantic": sem,
        "lexical": LexicalAddressSpace(),
        "entity": EntityGraphAddressSpace(),
    }


def make_policies() -> list:
    return [
        EnsemblePolicy(top_k=5, max_steps=3),
        AEAHeuristicPolicy(top_k=5, coverage_threshold=0.5, max_steps=6),
        NLIStoppingPolicy(top_k=5, max_steps=8),
    ]


def compute_e2e_ub(f1: float, recall: float, ops: float) -> float:
    """E2E U@B = f1 * (1 + 0.5 * recall) - mu * (ops / max_ops)"""
    return f1 * (1.0 + 0.5 * recall) - MU * (ops / MAX_OPS)


def extract_evidence(ex_result: dict, dataset_by_id: dict) -> list:
    retrieved_ids: list = []
    for trace_step in ex_result.get("trace", []):
        for item in trace_step.get("result_items", []):
            rid = item.get("id", "")
            if rid and rid not in retrieved_ids:
                retrieved_ids.append(rid)
    example = dataset_by_id.get(ex_result.get("id", ""), {})
    id_to_content = {doc["id"]: doc.get("content", "") for doc in example.get("context", [])}
    return [id_to_content[rid] for rid in retrieved_ids if rid in id_to_content and id_to_content[rid]]


def bootstrap_ci(values: list, n_resamples: int = N_BOOTSTRAP, seed: int = SEED):
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=np.float64)
    boot_means = np.array([
        np.mean(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_resamples)
    ])
    return float(np.mean(arr)), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def paired_ttest(a: list, b: list):
    """Two-sided paired t-test. Returns (delta, p_value, cohens_d)."""
    a_arr = np.array(a, dtype=np.float64)
    b_arr = np.array(b, dtype=np.float64)
    diff = a_arr - b_arr
    delta = float(np.mean(diff))
    _, p_val = stats.ttest_rel(a_arr, b_arr)
    pooled_std = float(np.std(diff, ddof=1))
    cohens_d = delta / pooled_std if pooled_std > 0 else 0.0
    return delta, float(p_val), cohens_d


def generate_answers_for_policy(harness_result, dataset_by_id, answer_gen, policy_name):
    per_example = harness_result["per_example"]
    total = len(per_example)
    e2e_results = []

    for idx, ex_result in enumerate(per_example):
        ex_id = ex_result.get("id", "unknown")
        question = ex_result.get("question", "")
        gold_answer = ex_result.get("gold_answer", "")
        support_recall_val = float(ex_result.get("support_recall", 0.0))
        ops = float(ex_result.get("operations_used", 0.0))

        passages = extract_evidence(ex_result, dataset_by_id)

        if passages:
            time.sleep(API_CALL_DELAY)
            llm_answer = answer_gen.generate_answer(question, passages)
        else:
            llm_answer = ""

        em = exact_match(llm_answer, gold_answer)
        f1 = f1_score(llm_answer, gold_answer)
        e2e_ub = compute_e2e_ub(f1, support_recall_val, ops)

        e2e_results.append({
            "id": ex_id,
            "question": question,
            "gold_answer": gold_answer,
            "llm_answer": llm_answer,
            "em": em,
            "f1": f1,
            "support_recall": support_recall_val,
            "operations_used": ops,
            "e2e_ub": e2e_ub,
        })

        if (idx + 1) % 50 == 0 or idx == 0:
            usage = answer_gen.usage_summary()
            print(
                f"    [{policy_name}] {idx+1}/{total} "
                f"({(idx+1)/total*100:.0f}%) - "
                f"calls: {usage['total_calls']}, errors: {usage['total_errors']}"
            )

    return e2e_results


def aggregate_e2e(results: list) -> dict:
    out = {}
    for key in ["em", "f1", "support_recall", "operations_used", "e2e_ub"]:
        vals = [r[key] for r in results if key in r]
        out[f"{key}_mean"] = float(np.mean(vals)) if vals else 0.0
        out[f"{key}_std"] = float(np.std(vals)) if vals else 0.0
    return out


def print_retrieval_table(all_retrieval):
    print("\n=== Retrieval-only Metrics ===\n")
    header = (
        f"| {'Policy':<24} "
        f"| {'SupportRecall':>14} "
        f"| {'SupportPrec':>12} "
        f"| {'AvgOps':>8} "
        f"| {'U@Budget':>10} |"
    )
    sep = f"| {'-'*24} | {'-'*14} | {'-'*12} | {'-'*8} | {'-'*10} |"
    print(header)
    print(sep)
    for pname, result in all_retrieval.items():
        agg = result["aggregated"]
        print(
            f"| {pname:<24} "
            f"| {agg['support_recall']:>14.4f} "
            f"| {agg['support_precision']:>12.4f} "
            f"| {agg['operations_used']:>8.2f} "
            f"| {agg['utility_at_budget']:>10.4f} |"
        )
    print()


def print_e2e_table(policy_names, agg_stats, ci_stats):
    print("=== E2E Evaluation (N=500) ===\n")
    header = (
        f"| {'Policy':<24} "
        f"| {'EM':>10} "
        f"| {'F1':>10} "
        f"| {'Recall':>8} "
        f"| {'Ops':>6} "
        f"| {'E2E U@B [95% CI]':>30} |"
    )
    sep = f"| {'-'*24} | {'-'*10} | {'-'*10} | {'-'*8} | {'-'*6} | {'-'*30} |"
    print(header)
    print(sep)
    for pname in policy_names:
        if pname not in agg_stats:
            continue
        agg = agg_stats[pname]
        ci = ci_stats.get(pname, (0.0, 0.0, 0.0))
        print(
            f"| {pname:<24} "
            f"| {agg['em_mean']:>10.4f} "
            f"| {agg['f1_mean']:>10.4f} "
            f"| {agg['support_recall_mean']:>8.4f} "
            f"| {agg['operations_used_mean']:>6.2f} "
            f"| {ci[0]:.4f} [{ci[1]:.4f}, {ci[2]:.4f}]   |"
        )
    print()


def print_stat_tests(test_results):
    print("Statistical Tests (paired t-test on E2E U@B):")
    for t in test_results:
        sig = "YES" if t["p_value"] < 0.05 else "NO"
        print(
            f"  {t['label']}: "
            f"delta={t['delta']:+.4f}, "
            f"p={t['p_value']:.4f}, "
            f"d={t['cohens_d']:.3f}, "
            f"significant? {sig}"
        )
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_nli_stopping_assessment(
    n_examples: int = N_EXAMPLES,
    seed: int = SEED,
    results_path=RESULTS_FILE,
) -> dict:
    """Full NLI stopping evaluation pipeline."""
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print(f"NLI Bundle Sufficiency Stopping — N={n_examples}, seed={seed}")
    print("=" * 70)
    print()
    print("Research question: does NLI bundle assessment (set function)")
    print("beat the workspace-statistics heuristic?")
    print()

    # Phase 1: Load data
    print("[Phase 1] Loading HotpotQA data...")
    _LOCAL_PATHS = [
        "/tmp/hotpot_dev_distractor_repaired.json",
        "/tmp/hotpot_dev_distractor_v1_new.json",
        "/tmp/hotpot_dev_distractor_v1.json",
    ]
    raw_data = load_hotpotqa(local_paths=_LOCAL_PATHS)
    bridge = filter_bridge(raw_data, n=n_examples)
    dataset = [convert_example(ex) for ex in bridge]
    dataset_by_id = {ex["id"]: ex for ex in dataset}
    print(f"  Loaded {len(dataset)} bridge examples.\n")

    # Phase 2: Retrieval-only evaluation
    print("[Phase 2] Running retrieval-only evaluation...")
    policies = make_policies()
    all_retrieval = {}

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

    print_retrieval_table(all_retrieval)

    # Phase 3: LLM answer generation
    print(f"[Phase 3] Generating LLM answers for: {LLM_POLICIES}...")
    answer_gen = AnswerGenerator()
    all_e2e = {}

    for pname in LLM_POLICIES:
        if pname not in all_retrieval:
            print(f"  WARNING: {pname} not in retrieval results - skipping.")
            continue
        print(f"\n  Generating answers for: {pname}")
        e2e_results = generate_answers_for_policy(
            all_retrieval[pname], dataset_by_id, answer_gen, pname
        )
        all_e2e[pname] = e2e_results
        agg = aggregate_e2e(e2e_results)
        print(
            f"    EM={agg['em_mean']:.4f}, F1={agg['f1_mean']:.4f}, "
            f"E2E U@B={agg['e2e_ub_mean']:.4f}"
        )

    usage = answer_gen.usage_summary()
    est_cost = usage["total_calls"] * 0.00002
    print(f"\n  Total API calls: {usage['total_calls']}, tokens: {usage['total_tokens']}")
    print(f"  Errors: {usage['total_errors']}, estimated cost: ~${est_cost:.4f}")

    # Phase 4: Statistics
    print("\n[Phase 4] Bootstrap CIs and paired t-tests...")
    agg_stats = {}
    ci_stats = {}

    for pname in LLM_POLICIES:
        if pname not in all_e2e:
            continue
        agg_stats[pname] = aggregate_e2e(all_e2e[pname])
        e2e_vals = [r["e2e_ub"] for r in all_e2e[pname]]
        mean_ub, ci_lo, ci_hi = bootstrap_ci(e2e_vals, n_resamples=N_BOOTSTRAP, seed=seed)
        ci_stats[pname] = (mean_ub, ci_lo, ci_hi)
        print(f"  {pname}: E2E U@B = {mean_ub:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")

    test_pairs = [
        ("pi_nli_stopping",  "pi_ensemble",      "NLI vs Ensemble"),
        ("pi_nli_stopping",  "pi_aea_heuristic",  "NLI vs AEA Heuristic"),
        ("pi_aea_heuristic", "pi_ensemble",       "AEA Heuristic vs Ensemble"),
    ]

    stat_tests = []
    for name_a, name_b, label in test_pairs:
        if name_a not in all_e2e or name_b not in all_e2e:
            print(f"  Skipping {label}: one or both policies missing.")
            continue
        vals_a = [r["e2e_ub"] for r in all_e2e[name_a]]
        vals_b = [r["e2e_ub"] for r in all_e2e[name_b]]
        delta, p_val, d = paired_ttest(vals_a, vals_b)
        stat_tests.append({
            "label": label,
            "policy_a": name_a,
            "policy_b": name_b,
            "delta": delta,
            "p_value": p_val,
            "cohens_d": d,
        })
        sig = "YES" if p_val < 0.05 else "NO"
        print(f"  {label}: delta={delta:+.4f}, p={p_val:.4f}, d={d:.3f}, sig? {sig}")

    # Phase 5: Print tables
    print_e2e_table(LLM_POLICIES, agg_stats, ci_stats)
    print_stat_tests(stat_tests)

    # Interpret key result
    nli_vs_heuristic = next(
        (t for t in stat_tests if t["label"] == "NLI vs AEA Heuristic"), None
    )
    if nli_vs_heuristic:
        print("=== KEY FINDING ===")
        d = nli_vs_heuristic["delta"]
        p = nli_vs_heuristic["p_value"]
        if d > 0 and p < 0.05:
            print(
                f"NLI BEATS heuristic (delta={d:+.4f}, p={p:.4f}).\n"
                "Content-aware bundle assessment works. Story: NLI solves the set\n"
                "function problem that per-passage cross-encoder fails at."
            )
        elif d <= 0 and p < 0.05:
            print(
                f"Heuristic BEATS NLI (delta={d:+.4f}, p={p:.4f}).\n"
                "STRONG negative result: even principled NLI bundle assessment\n"
                "cannot beat workspace-statistics stopping. The structural signal\n"
                "thesis is strongly supported — query-content interaction is not\n"
                "what drives stopping quality; it's workspace coverage statistics."
            )
        else:
            print(
                f"No significant difference (delta={d:+.4f}, p={p:.4f}).\n"
                "NLI bundle assessment is equivalent to the heuristic, providing\n"
                "indirect support that content-based signals do not add beyond\n"
                "workspace statistics for this task."
            )
        print()

    # Save results
    full_results = {
        "experiment": "nli_stopping",
        "n_examples": n_examples,
        "seed": seed,
        "mu": MU,
        "max_ops": MAX_OPS,
        "n_bootstrap": N_BOOTSTRAP,
        "nli_model": "cross-encoder/nli-deberta-v3-small",
        "entailment_threshold": 0.7,
        "retrieval_aggregated": {
            pname: result["aggregated"]
            for pname, result in all_retrieval.items()
        },
        "e2e_per_example": all_e2e,
        "e2e_aggregated": agg_stats,
        "bootstrap_ci": {
            pname: {"mean": ci[0], "ci_low": ci[1], "ci_high": ci[2]}
            for pname, ci in ci_stats.items()
        },
        "statistical_tests": stat_tests,
        "answer_generator_usage": usage,
        "estimated_api_cost_usd": est_cost,
    }

    if results_path is not None:
        results_path = Path(results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump(full_results, fh, indent=2)
        print(f"Full results saved to: {results_path}")

    return full_results


if __name__ == "__main__":
    run_nli_stopping_assessment()
