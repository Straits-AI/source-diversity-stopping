"""
End-to-end evaluation at N=500 with statistical significance tests.

Evaluates 5 retrieval policies on HotpotQA bridge questions and generates
LLM answers for 3 key policies, then runs bootstrap CIs and paired
permutation tests to determine statistical significance.

Usage
-----
    export OPENROUTER_API_KEY="sk-or-..."
    python experiments/run_e2e_n500.py
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

# ── Prevent HuggingFace tokenizer deadlocks on macOS ─────────────────────────
# HuggingFace fast tokenizers spawn worker threads that can deadlock when
# multiple SentenceTransformer model instances are created back-to-back.
# Setting this env var before any HuggingFace import prevents the issue.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


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
from experiments.aea.policies.learned_stopping import LearnedStoppingPolicy
from experiments.aea.answer_generator import AnswerGenerator
from experiments.aea.evaluation.metrics import exact_match, f1_score

# Reuse data-loading helpers from the baselines script
from experiments.run_hotpotqa_baselines import (
    load_hotpotqa,
    filter_bridge,
    convert_example,
)

# ── Constants ─────────────────────────────────────────────────────────────────
N_EXAMPLES = 500
SEED = 42
MU = 0.3
MAX_OPS = 3.0
N_BOOTSTRAP = 1000
N_PERMUTATIONS = 10_000
API_CALL_DELAY = 0.1          # seconds between API calls (AnswerGenerator adds 0.5s internally)
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_FILE = RESULTS_DIR / "e2e_n500.json"
MODELS_DIR = Path(__file__).resolve().parent / "models"
STOPPING_MODEL_PATH = MODELS_DIR / "stopping_classifier_clean.pkl"

# Set API key
os.environ.setdefault(
    "OPENROUTER_API_KEY",
    "",
)

# ── Policies for retrieval-only phase ────────────────────────────────────────
RETRIEVAL_POLICIES_NAMES = [
    "pi_semantic",
    "pi_lexical",
    "pi_ensemble",
    "pi_aea_heuristic",
    "pi_learned_stop",
]

# ── Policies for LLM answer generation ───────────────────────────────────────
LLM_POLICIES_NAMES = [
    "pi_ensemble",
    "pi_aea_heuristic",
    "pi_learned_stop",
]


# ── Shared semantic encoder (avoid reloading SentenceTransformer 5× on macOS) ─
# Creating 5 SentenceTransformer instances back-to-back can deadlock on macOS
# due to HuggingFace tokenizer thread-pool contention.  We create ONE encoder
# and inject it into each SemanticAddressSpace via the _encoder attribute.
_SHARED_ENCODER = None


def _get_shared_encoder():
    """
    Load the SentenceTransformer model once and reuse it.

    On macOS with Apple Silicon, multiple SentenceTransformer instantiations
    back-to-back deadlock because each creates a HuggingFace tokenizer
    thread pool.  Loading the model once and sharing the encoder object
    avoids repeated thread-pool creation.
    """
    global _SHARED_ENCODER
    if _SHARED_ENCODER is None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        print("  [setup] Loading SentenceTransformer model (once)…")
        _SHARED_ENCODER = SentenceTransformer("all-MiniLM-L6-v2")
        print("  [setup] Model loaded.")
    return _SHARED_ENCODER


# ── Address-space factory ─────────────────────────────────────────────────────

def make_address_spaces() -> dict:
    """
    Instantiate fresh address spaces.

    The SemanticAddressSpace normally loads the SentenceTransformer model
    lazily.  We pre-inject the shared encoder to prevent the model from being
    loaded multiple times (which causes deadlocks on macOS due to tokenizer
    thread-pool contention).
    """
    sem_space = SemanticAddressSpace(model_name="all-MiniLM-L6-v2")
    sem_space._encoder = _get_shared_encoder()  # inject shared encoder
    return {
        "semantic": sem_space,
        "lexical": LexicalAddressSpace(),
        "entity": EntityGraphAddressSpace(),
    }


# ── Policy factory ────────────────────────────────────────────────────────────

def make_all_policies() -> list:
    """Return all 5 policies in evaluation order."""
    return [
        SemanticOnlyPolicy(top_k=5, max_steps=2),
        LexicalOnlyPolicy(top_k=5, max_steps=2),
        EnsemblePolicy(top_k=5, max_steps=3),
        AEAHeuristicPolicy(top_k=5, coverage_threshold=0.5, max_steps=6),
        LearnedStoppingPolicy(model_path=STOPPING_MODEL_PATH, top_k=5, max_steps=8),
    ]


# ── Per-example E2E utility ───────────────────────────────────────────────────

def compute_e2e_ub(f1: float, recall: float, ops: float) -> float:
    """
    Per-example E2E U@B:
        e2e_ub_i = f1_i * (1 + 0.5 * recall_i) - mu * (ops_i / max_ops)
    """
    return f1 * (1.0 + 0.5 * recall) - MU * (ops / MAX_OPS)


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_ci(values: list[float], n_resamples: int = N_BOOTSTRAP, seed: int = SEED) -> tuple[float, float, float]:
    """
    Compute mean and 95% bootstrap CI for a list of values.

    Returns
    -------
    (mean, ci_low, ci_high)
    """
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=np.float64)
    means = np.array([
        np.mean(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_resamples)
    ])
    ci_low = float(np.percentile(means, 2.5))
    ci_high = float(np.percentile(means, 97.5))
    return float(np.mean(arr)), ci_low, ci_high


# ── Paired permutation test ───────────────────────────────────────────────────

def paired_permutation_test(
    a: list[float],
    b: list[float],
    n_permutations: int = N_PERMUTATIONS,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """
    Two-sided paired permutation test for H0: mean(a) == mean(b).

    Returns
    -------
    (delta, p_value, cohens_d)
        delta  = mean(a) - mean(b)
        p_value = fraction of permutations with |delta| >= |observed delta|
        cohens_d = delta / pooled std
    """
    rng = np.random.default_rng(seed)
    a_arr = np.array(a, dtype=np.float64)
    b_arr = np.array(b, dtype=np.float64)

    diff = a_arr - b_arr
    observed_delta = float(np.mean(diff))

    # Permutation: randomly flip signs of paired differences
    extreme = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1.0, 1.0], size=len(diff))
        perm_delta = float(np.mean(diff * signs))
        if abs(perm_delta) >= abs(observed_delta):
            extreme += 1

    p_value = (extreme + 1) / (n_permutations + 1)

    # Cohen's d = mean difference / pooled SD
    pooled_std = float(np.std(diff, ddof=1))
    cohens_d = observed_delta / pooled_std if pooled_std > 0 else 0.0

    return observed_delta, p_value, cohens_d


# ── Crossover analysis ────────────────────────────────────────────────────────

def crossover_mu(
    f1_a: list[float],
    recall_a: list[float],
    ops_a: list[float],
    f1_b: list[float],
    recall_b: list[float],
    ops_b: list[float],
) -> float:
    """
    Find mu at which mean(E2E U@B for a) == mean(E2E U@B for b).

    E2E U@B_i(mu) = f1_i * (1 + 0.5 * recall_i) - mu * (ops_i / max_ops)

    At crossover:
        mean_a_quality - mu * mean_a_cost = mean_b_quality - mu * mean_b_cost
        mu = (mean_a_quality - mean_b_quality) / (mean_a_cost - mean_b_cost)
    """
    a_quality = float(np.mean([f * (1 + 0.5 * r) for f, r in zip(f1_a, recall_a)]))
    b_quality = float(np.mean([f * (1 + 0.5 * r) for f, r in zip(f1_b, recall_b)]))
    a_cost = float(np.mean([o / MAX_OPS for o in ops_a]))
    b_cost = float(np.mean([o / MAX_OPS for o in ops_b]))

    cost_diff = a_cost - b_cost
    if abs(cost_diff) < 1e-10:
        return float("inf")
    return (a_quality - b_quality) / cost_diff


# ── Evidence extraction from trace ───────────────────────────────────────────

def extract_evidence(
    ex_result: dict,
    dataset_by_id: dict[str, dict],
) -> list[str]:
    """
    Extract retrieved passage texts from a per-example harness result.
    Uses the trace's result_items and looks up content from original dataset.
    """
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


# ── LLM answer generation for a policy's results ─────────────────────────────

def generate_answers_for_policy(
    harness_result: dict,
    dataset_by_id: dict[str, dict],
    answer_gen: AnswerGenerator,
    policy_name: str,
) -> list[dict]:
    """
    Generate LLM answers for all examples in a policy's harness result.
    Returns per-example dicts with em, f1, e2e_ub.
    """
    per_example = harness_result["per_example"]
    total = len(per_example)
    e2e_results = []

    for idx, ex_result in enumerate(per_example):
        ex_id = ex_result.get("id", "unknown")
        question = ex_result.get("question", "")
        gold_answer = ex_result.get("gold_answer", "")
        support_recall_val = ex_result.get("support_recall", 0.0)
        operations_used = float(ex_result.get("operations_used", 0))

        evidence_passages = extract_evidence(ex_result, dataset_by_id)

        if evidence_passages:
            # Rate-limit delay before each API call
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

        if (idx + 1) % 50 == 0 or idx == 0:
            elapsed_pct = (idx + 1) / total * 100
            usage = answer_gen.usage_summary()
            print(
                f"    [{policy_name}] {idx+1}/{total} ({elapsed_pct:.0f}%) — "
                f"API calls: {usage['total_calls']}, errors: {usage['total_errors']}"
            )

    return e2e_results


# ── Aggregate helpers ─────────────────────────────────────────────────────────

def aggregate_e2e(e2e_results: list[dict]) -> dict:
    """Compute mean ± std for key metrics."""
    out = {}
    for key in ["em", "f1", "support_recall", "operations_used", "e2e_ub"]:
        vals = [r[key] for r in e2e_results if key in r]
        out[f"{key}_mean"] = float(np.mean(vals)) if vals else 0.0
        out[f"{key}_std"] = float(np.std(vals)) if vals else 0.0
    return out


# ── Printing helpers ──────────────────────────────────────────────────────────

def print_results_table(
    policy_names: list[str],
    agg_stats: dict[str, dict],
    ci_stats: dict[str, tuple],
) -> None:
    """Print the main results table."""
    print("\n=== E2E Evaluation (N=500, gpt-oss-120b) ===\n")
    header = (
        f"| {'Policy':<22} "
        f"| {'EM (mean±std)':>16} "
        f"| {'F1 (mean±std)':>16} "
        f"| {'Recall':>8} "
        f"| {'Ops':>6} "
        f"| {'E2E U@B [95% CI]':>30} |"
    )
    sep = f"| {'-'*22} | {'-'*16} | {'-'*16} | {'-'*8} | {'-'*6} | {'-'*30} |"
    print(header)
    print(sep)

    for pname in policy_names:
        if pname not in agg_stats:
            continue
        agg = agg_stats[pname]
        ci = ci_stats.get(pname, (0.0, 0.0, 0.0))
        row = (
            f"| {pname:<22} "
            f"| {agg['em_mean']:.4f}±{agg['em_std']:.4f}  "
            f"| {agg['f1_mean']:.4f}±{agg['f1_std']:.4f}  "
            f"| {agg['support_recall_mean']:>8.4f} "
            f"| {agg['operations_used_mean']:>6.2f} "
            f"| {ci[0]:.4f} [{ci[1]:.4f}, {ci[2]:.4f}]   |"
        )
        print(row)
    print()


def print_statistical_tests(
    test_results: list[dict],
) -> None:
    """Print statistical test results."""
    print("Statistical Tests (E2E U@B):")
    for t in test_results:
        sig = "YES" if t["p_value"] < 0.05 else "NO"
        print(
            f"  {t['label']}: "
            f"Δ={t['delta']:.4f}, "
            f"p={t['p_value']:.4f}, "
            f"d={t['cohens_d']:.3f}, "
            f"Significant? {sig}"
        )
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def run_e2e_n500(
    n_examples: int = N_EXAMPLES,
    seed: int = SEED,
    results_path: Optional[Path] = RESULTS_FILE,
) -> dict:
    """
    Full E2E evaluation pipeline.

    1. Load N=500 HotpotQA bridge questions
    2. Run 5 retrieval policies (fast, no API)
    3. Generate LLM answers for 3 key policies
    4. Compute bootstrap CIs and paired permutation tests
    5. Print formatted output table
    6. Save full results to JSON
    """
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print(f"E2E Evaluation — N={n_examples}, seed={seed}")
    print("=" * 70)

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    print("\n[Phase 1] Loading data…")
    raw_data = load_hotpotqa()
    bridge = filter_bridge(raw_data, n=n_examples)
    dataset = [convert_example(ex) for ex in bridge]
    dataset_by_id = {ex["id"]: ex for ex in dataset}
    print(f"  Loaded {len(dataset)} examples.\n")

    # ── Step 2: Run retrieval-only phase for all 5 policies ───────────────────
    print("[Phase 2] Running retrieval-only phase for 5 policies…")
    policies = make_all_policies()
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

    # ── Step 3: Generate LLM answers for 3 key policies ──────────────────────
    print(f"\n[Phase 3] Generating LLM answers for {LLM_POLICIES_NAMES}…")
    answer_gen = AnswerGenerator()  # uses gpt-oss-120b by default
    all_e2e_per_example: dict[str, list[dict]] = {}

    for pname in LLM_POLICIES_NAMES:
        if pname not in all_retrieval:
            print(f"  WARNING: {pname} not found in retrieval results, skipping.")
            continue
        print(f"\n  Generating answers for: {pname}")
        harness_result = all_retrieval[pname]
        e2e_results = generate_answers_for_policy(
            harness_result, dataset_by_id, answer_gen, pname
        )
        all_e2e_per_example[pname] = e2e_results

        agg = aggregate_e2e(e2e_results)
        print(
            f"    EM={agg['em_mean']:.4f}, F1={agg['f1_mean']:.4f}, "
            f"E2E U@B={agg['e2e_ub_mean']:.4f}"
        )

    usage = answer_gen.usage_summary()
    print(f"\n  Total API calls: {usage['total_calls']}")
    print(f"  Total tokens: {usage['total_tokens']}")
    print(f"  Errors: {usage['total_errors']}")
    # Cost estimate: gpt-oss-120b ~$0.00002/call rough estimate
    est_cost = usage["total_calls"] * 0.00002
    print(f"  Estimated cost: ~${est_cost:.4f}")

    # ── Step 4: Compute statistics ────────────────────────────────────────────
    print("\n[Phase 4] Computing bootstrap CIs and statistical tests…")

    agg_stats: dict[str, dict] = {}
    ci_stats: dict[str, tuple] = {}

    for pname in LLM_POLICIES_NAMES:
        if pname not in all_e2e_per_example:
            continue
        results = all_e2e_per_example[pname]
        agg_stats[pname] = aggregate_e2e(results)
        e2e_ub_vals = [r["e2e_ub"] for r in results]
        mean_ub, ci_low, ci_high = bootstrap_ci(e2e_ub_vals, n_resamples=N_BOOTSTRAP, seed=seed)
        ci_stats[pname] = (mean_ub, ci_low, ci_high)
        print(f"  {pname}: E2E U@B mean={mean_ub:.4f} [{ci_low:.4f}, {ci_high:.4f}]")

    # Paired permutation tests
    test_pairs = [
        ("pi_aea_heuristic", "pi_ensemble",  "AEA vs Ensemble"),
        ("pi_aea_heuristic", "pi_learned_stop", "AEA vs Learned"),
        ("pi_learned_stop",  "pi_ensemble",  "Learned vs Ensemble"),
    ]

    stat_tests: list[dict] = []
    for name_a, name_b, label in test_pairs:
        if name_a not in all_e2e_per_example or name_b not in all_e2e_per_example:
            print(f"  Skipping {label}: one or both policies missing.")
            continue
        vals_a = [r["e2e_ub"] for r in all_e2e_per_example[name_a]]
        vals_b = [r["e2e_ub"] for r in all_e2e_per_example[name_b]]
        delta, p_val, d = paired_permutation_test(vals_a, vals_b, n_permutations=N_PERMUTATIONS, seed=seed)
        stat_tests.append({
            "label": label,
            "policy_a": name_a,
            "policy_b": name_b,
            "delta": delta,
            "p_value": p_val,
            "cohens_d": d,
        })
        sig = "YES" if p_val < 0.05 else "NO"
        print(f"  {label}: Δ={delta:.4f}, p={p_val:.4f}, d={d:.3f}, Significant? {sig}")

    # ── Crossover analysis ────────────────────────────────────────────────────
    crossover = None
    if "pi_aea_heuristic" in all_e2e_per_example and "pi_ensemble" in all_e2e_per_example:
        aea_results = all_e2e_per_example["pi_aea_heuristic"]
        ens_results = all_e2e_per_example["pi_ensemble"]
        crossover = crossover_mu(
            f1_a=[r["f1"] for r in aea_results],
            recall_a=[r["support_recall"] for r in aea_results],
            ops_a=[r["operations_used"] for r in aea_results],
            f1_b=[r["f1"] for r in ens_results],
            recall_b=[r["support_recall"] for r in ens_results],
            ops_b=[r["operations_used"] for r in ens_results],
        )
        print(f"\n  Crossover: AEA overtakes Ensemble at mu = {crossover:.4f}")

    # ── Step 5: Print formatted output ───────────────────────────────────────
    print_results_table(LLM_POLICIES_NAMES, agg_stats, ci_stats)
    print_statistical_tests(stat_tests)

    if crossover is not None:
        print("Sensitivity (crossover analysis):")
        print(f"  AEA overtakes Ensemble at μ = {crossover:.4f}")
        print()

    # Answer generator usage summary
    print("Answer Generator Usage Summary:")
    print(f"  Total calls:             {usage['total_calls']}")
    print(f"  Total prompt tokens:     {usage['total_prompt_tokens']}")
    print(f"  Total completion tokens: {usage['total_completion_tokens']}")
    print(f"  Total tokens:            {usage['total_tokens']}")
    print(f"  Total errors:            {usage['total_errors']}")
    print(f"  Estimated cost:          ~${est_cost:.4f}")
    print()

    # ── Step 6: Save results ──────────────────────────────────────────────────
    full_results = {
        "n_examples": n_examples,
        "seed": seed,
        "mu": MU,
        "max_ops": MAX_OPS,
        "n_bootstrap": N_BOOTSTRAP,
        "n_permutations": N_PERMUTATIONS,
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
        "crossover_mu_aea_vs_ensemble": crossover,
        "answer_generator_usage": usage,
        "estimated_api_cost_usd": est_cost,
    }

    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump(full_results, fh, indent=2)
        print(f"Full results saved to: {results_path}")

    return full_results


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_e2e_n500()
