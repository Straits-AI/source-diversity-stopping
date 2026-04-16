"""
Confidence-Gated Stopping — BRIGHT Benchmark (Reviewer generalization concern).

Key question: Does confidence-gated stopping work on reasoning-intensive retrieval?

Runs pi_aea_heuristic and pi_confidence_gated on 200 BRIGHT questions.
Retrieval-only — no E2E answer generation needed.

The confidence call still happens during retrieval as part of pi_confidence_gated:
  Step 0: Semantic SEARCH (always)
  Step 1: LLM confidence check — if confident, STOP with draft; else lexical SEARCH
  Step 2: Hard STOP

BRIGHT is challenging because queries and gold docs have low lexical AND
semantic overlap — correct retrieval requires reasoning to connect query
intent to document content.

Usage
-----
    export OPENROUTER_API_KEY="sk-or-..."
    python experiments/run_confidence_gated_bright.py

Results saved to: experiments/results/confidence_gated_bright.json
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
from scipy import stats

# ── Prevent HuggingFace tokenizer deadlocks on macOS ─────────────────────────
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── Project root ──────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── AEA imports ───────────────────────────────────────────────────────────────
from experiments.aea.address_spaces.semantic import SemanticAddressSpace
from experiments.aea.address_spaces.lexical import LexicalAddressSpace
from experiments.aea.address_spaces.entity_graph import EntityGraphAddressSpace
from experiments.aea.evaluation.harness import EvaluationHarness
from experiments.aea.policies.heuristic import AEAHeuristicPolicy
from experiments.aea.policies.confidence_gated import ConfidenceGatedPolicy

# Reuse BRIGHT data loading helpers from run_bright.py
from experiments.run_bright import (
    build_bright_dataset,
    build_synthetic_bright,
    DOMAINS,
    N_PER_DOMAIN,
)

# ── Constants ─────────────────────────────────────────────────────────────────
N_EXAMPLES = 200
SEED = 42

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_FILE = RESULTS_DIR / "confidence_gated_bright.json"

# Set API key
os.environ.setdefault(
    "OPENROUTER_API_KEY",
    "",
)

POLICIES_TO_RUN = ["pi_aea_heuristic", "pi_confidence_gated"]


# ── Shared encoder ────────────────────────────────────────────────────────────

_SHARED_ENCODER = None


def _get_shared_encoder():
    global _SHARED_ENCODER
    if _SHARED_ENCODER is None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        print("  [setup] Loading SentenceTransformer model (once)…")
        _SHARED_ENCODER = SentenceTransformer("all-MiniLM-L6-v2")
        print("  [setup] Model loaded.")
    return _SHARED_ENCODER


# ── Address-space factory ─────────────────────────────────────────────────────

def make_address_spaces() -> dict:
    sem_space = SemanticAddressSpace(model_name="all-MiniLM-L6-v2")
    sem_space._encoder = _get_shared_encoder()
    return {
        "semantic": sem_space,
        "lexical": LexicalAddressSpace(),
        "entity": EntityGraphAddressSpace(),
    }


# ── Statistics helpers ────────────────────────────────────────────────────────

def paired_ttest(a: list[float], b: list[float]) -> tuple[float, float]:
    """Paired t-test; returns (t_stat, p_value)."""
    n = min(len(a), len(b))
    if n < 2:
        return float("nan"), float("nan")
    result = stats.ttest_rel(np.array(a[:n]), np.array(b[:n]))
    return float(result.statistic), float(result.pvalue)


def cohens_d(a: list[float], b: list[float]) -> float:
    """Cohen's d effect size for paired samples."""
    n = min(len(a), len(b))
    if n < 2:
        return float("nan")
    diffs = np.array(a[:n]) - np.array(b[:n])
    return float(np.mean(diffs) / (np.std(diffs, ddof=1) + 1e-12))


def extract_metric(per_example: list[dict], key: str) -> list[float]:
    return [ex[key] for ex in per_example if key in ex and isinstance(ex[key], (int, float))]


# ── Per-domain breakdown ──────────────────────────────────────────────────────

def compute_domain_breakdown(per_example: list[dict], dataset: list[dict]) -> dict:
    """Compute per-domain support_recall and utility_at_budget."""
    id_to_domain = {ex["id"]: ex.get("domain", "unknown") for ex in dataset}
    domain_metrics: dict[str, dict[str, list[float]]] = {}

    for ex_result in per_example:
        domain = id_to_domain.get(ex_result.get("id", ""), "unknown")
        if domain not in domain_metrics:
            domain_metrics[domain] = {"support_recall": [], "utility_at_budget": []}
        domain_metrics[domain]["support_recall"].append(
            ex_result.get("support_recall", 0.0)
        )
        domain_metrics[domain]["utility_at_budget"].append(
            ex_result.get("utility_at_budget", 0.0)
        )

    return {
        domain: {
            "support_recall_mean": float(np.mean(vals["support_recall"])),
            "utility_at_budget_mean": float(np.mean(vals["utility_at_budget"])),
            "n": len(vals["support_recall"]),
        }
        for domain, vals in domain_metrics.items()
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run_confidence_gated_bright(
    n_examples: int = N_EXAMPLES,
    seed: int = SEED,
    results_path: Optional[Path] = RESULTS_FILE,
) -> dict:
    """
    BRIGHT retrieval evaluation of pi_aea_heuristic vs pi_confidence_gated.

    Key question: Does the confidence-gated stopping signal work on
    reasoning-intensive retrieval where standard methods struggle?
    """
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print("Confidence-Gated BRIGHT Evaluation — Generalization Test")
    print(f"N={n_examples}, seed={seed}")
    print("=" * 70)

    # ── Phase 1: Load BRIGHT data ─────────────────────────────────────────────
    print("\n[Phase 1] Loading BRIGHT dataset…")
    dataset: list[dict] = []
    is_synthetic = False

    try:
        from datasets import load_dataset  # type: ignore  # noqa: F401
        print("  HuggingFace datasets available. Attempting real BRIGHT data…")
        dataset = build_bright_dataset(
            n_per_domain=n_examples // len(DOMAINS), seed=seed
        )
        if len(dataset) < 50:
            print(f"  Only {len(dataset)} real examples (< 50). Falling back to synthetic.")
            raise RuntimeError("Too few real examples")
    except Exception as exc:
        print(f"  Real BRIGHT loading failed: {exc}")
        print("  Using synthetic BRIGHT-style dataset…")
        dataset = build_synthetic_bright(n=n_examples, seed=seed)
        is_synthetic = True

    n_actual = len(dataset)
    print(f"\nDataset ready: {n_actual} examples  (synthetic={is_synthetic})")

    if not is_synthetic:
        domain_counts: dict[str, int] = {}
        for ex in dataset:
            d = ex.get("domain", "unknown")
            domain_counts[d] = domain_counts.get(d, 0) + 1
        for d, c in sorted(domain_counts.items()):
            print(f"  {d}: {c}")

    # ── Phase 2: Retrieval evaluation ─────────────────────────────────────────
    print(f"\n[Phase 2] Retrieval evaluation (policies: {POLICIES_TO_RUN})…")
    policies = [
        AEAHeuristicPolicy(top_k=5, coverage_threshold=0.5, max_steps=6),
        ConfidenceGatedPolicy(top_k=5),
    ]
    all_results: dict = {}

    for policy in policies:
        pname = policy.name()
        print(f"\n{'─' * 60}")
        print(f"Policy: {pname}")
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
        result["is_synthetic"] = is_synthetic

        # Inject domain into per_example
        id_to_domain = {ex["id"]: ex.get("domain", "unknown") for ex in dataset}
        id_to_category = {ex["id"]: ex.get("category", "unknown") for ex in dataset}
        for ex_result in result["per_example"]:
            ex_result["domain"] = id_to_domain.get(ex_result.get("id", ""), "unknown")
            ex_result["category"] = id_to_category.get(ex_result.get("id", ""), "unknown")

        agg = result["aggregated"]
        print(f"  support_recall:    {agg['support_recall']:.4f}")
        print(f"  support_precision: {agg['support_precision']:.4f}")
        print(f"  utility@budget:    {agg['utility_at_budget']:.4f}")
        print(f"  avg_operations:    {agg['operations_used']:.2f}")
        print(f"  n_errors:          {result['n_errors']}")
        print(f"  runtime:           {elapsed:.1f}s")

        # Per-domain breakdown
        breakdown = compute_domain_breakdown(result["per_example"], dataset)
        result["domain_breakdown"] = breakdown
        if breakdown:
            print("  Per-domain support_recall:")
            for domain, dm in sorted(breakdown.items()):
                print(f"    {domain}: {dm['support_recall_mean']:.4f} (n={dm['n']})")

        all_results[pname] = result

    # ── Phase 3: Statistical analysis ─────────────────────────────────────────
    print(f"\n[Phase 3] Statistical Tests (paired t-test on U@B)…")
    stat_tests: list[dict] = []

    if "pi_confidence_gated" in all_results and "pi_aea_heuristic" in all_results:
        cg_ub = extract_metric(
            all_results["pi_confidence_gated"]["per_example"], "utility_at_budget"
        )
        aea_ub = extract_metric(
            all_results["pi_aea_heuristic"]["per_example"], "utility_at_budget"
        )
        cg_sr = extract_metric(
            all_results["pi_confidence_gated"]["per_example"], "support_recall"
        )
        aea_sr = extract_metric(
            all_results["pi_aea_heuristic"]["per_example"], "support_recall"
        )

        # Test on U@B
        t_ub, p_ub = paired_ttest(cg_ub, aea_ub)
        d_ub = cohens_d(cg_ub, aea_ub)
        delta_ub = float(np.mean(cg_ub)) - float(np.mean(aea_ub))
        sig_ub = p_ub < 0.05

        stat_tests.append({
            "label": "ConfGated vs AEAHeuristic (U@B)",
            "policy_a": "pi_confidence_gated",
            "policy_b": "pi_aea_heuristic",
            "metric": "utility_at_budget",
            "mean_a": float(np.mean(cg_ub)),
            "mean_b": float(np.mean(aea_ub)),
            "delta": delta_ub,
            "t_statistic": float(t_ub),
            "p_value": float(p_ub),
            "cohens_d": d_ub,
            "significant": sig_ub,
            "n": min(len(cg_ub), len(aea_ub)),
        })
        sig_str = "YES" if sig_ub else "NO"
        print(
            f"  ConfGated vs AEAHeuristic (U@B): "
            f"Δ={delta_ub:.4f}, t={t_ub:.3f}, p={p_ub:.4f}, d={d_ub:.3f}, Sig? {sig_str}"
        )

        # Test on support_recall
        t_sr, p_sr = paired_ttest(cg_sr, aea_sr)
        d_sr = cohens_d(cg_sr, aea_sr)
        delta_sr = float(np.mean(cg_sr)) - float(np.mean(aea_sr))
        sig_sr = p_sr < 0.05

        stat_tests.append({
            "label": "ConfGated vs AEAHeuristic (support_recall)",
            "policy_a": "pi_confidence_gated",
            "policy_b": "pi_aea_heuristic",
            "metric": "support_recall",
            "mean_a": float(np.mean(cg_sr)),
            "mean_b": float(np.mean(aea_sr)),
            "delta": delta_sr,
            "t_statistic": float(t_sr),
            "p_value": float(p_sr),
            "cohens_d": d_sr,
            "significant": sig_sr,
            "n": min(len(cg_sr), len(aea_sr)),
        })
        sig_str = "YES" if sig_sr else "NO"
        print(
            f"  ConfGated vs AEAHeuristic (recall): "
            f"Δ={delta_sr:.4f}, t={t_sr:.3f}, p={p_sr:.4f}, d={d_sr:.3f}, Sig? {sig_str}"
        )

    # ── Phase 4: Summary table ────────────────────────────────────────────────
    print(f"\n=== BRIGHT — Confidence-Gated vs AEA Heuristic (N={n_actual}) ===\n")
    print(
        f"  {'Policy':<25} {'SR':>8} {'SP':>10} {'Ops':>6} "
        f"{'U@B':>8} {'Errors':>8}"
    )
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*6} {'-'*8} {'-'*8}")
    for pname in POLICIES_TO_RUN:
        if pname not in all_results:
            continue
        agg = all_results[pname]["aggregated"]
        n_err = all_results[pname]["n_errors"]
        print(
            f"  {pname:<25} {agg['support_recall']:>8.4f} "
            f"{agg['support_precision']:>10.4f} {agg['operations_used']:>6.2f} "
            f"{agg['utility_at_budget']:>8.4f} {n_err:>8}"
        )

    print(f"\n  Data source: {'Real BRIGHT (xlangai/BRIGHT)' if not is_synthetic else 'Synthetic BRIGHT-style'}")
    print(f"  Key question: Does confidence-gated stopping generalize to reasoning-intensive retrieval?")

    print("\nPaired t-test results:")
    for t in stat_tests:
        sig = "YES" if t["p_value"] < 0.05 else "NO"
        print(
            f"  {t['label']}: "
            f"Δ={t['delta']:.4f}, t={t['t_statistic']:.3f}, "
            f"p={t['p_value']:.4f}, d={t['cohens_d']:.3f}, Sig? {sig}"
        )

    # Answer the key question
    print("\nKey finding:")
    if "pi_confidence_gated" in all_results and "pi_aea_heuristic" in all_results:
        cg_agg = all_results["pi_confidence_gated"]["aggregated"]
        aea_agg = all_results["pi_aea_heuristic"]["aggregated"]
        ub_diff = cg_agg["utility_at_budget"] - aea_agg["utility_at_budget"]
        sr_diff = cg_agg["support_recall"] - aea_agg["support_recall"]
        ops_diff = cg_agg["operations_used"] - aea_agg["operations_used"]

        direction_ub = "BETTER" if ub_diff > 0 else "WORSE"
        direction_sr = "higher" if sr_diff > 0 else "lower"
        direction_ops = "more" if ops_diff > 0 else "fewer"

        print(
            f"  Confidence-gated is {direction_ub} on U@B (Δ={ub_diff:+.4f}), "
            f"{direction_sr} recall (Δ={sr_diff:+.4f}), "
            f"{direction_ops} ops (Δ={ops_diff:+.2f})."
        )
        ub_sig = any(t["metric"] == "utility_at_budget" and t["significant"] for t in stat_tests)
        if ub_sig:
            print("  Result is STATISTICALLY SIGNIFICANT — confidence-gated stopping generalizes to BRIGHT.")
        else:
            print("  Difference is NOT statistically significant at α=0.05.")

    # ── Save results ──────────────────────────────────────────────────────────
    slim_results: dict = {}
    for pname, res in all_results.items():
        slim_per = [
            {k: v for k, v in ex.items() if k != "trace"}
            for ex in res.get("per_example", [])
        ]
        slim_results[pname] = {**res, "per_example": slim_per}

    output = {
        "experiment": "confidence_gated_bright",
        "benchmark": "BRIGHT",
        "benchmark_description": (
            "Reasoning-Intensive Retrieval Benchmark — "
            "low lexical AND semantic overlap between queries and gold docs"
        ),
        "n_examples": n_actual,
        "is_synthetic": is_synthetic,
        "domains": DOMAINS,
        "seed": seed,
        "policies_evaluated": POLICIES_TO_RUN,
        "retrieval_aggregated": {
            pname: res["aggregated"]
            for pname, res in all_results.items()
        },
        "domain_breakdowns": {
            pname: res.get("domain_breakdown", {})
            for pname, res in all_results.items()
        },
        "results": slim_results,
        "statistical_tests": stat_tests,
        "key_question": "Does confidence-gated stopping generalize to reasoning-intensive retrieval?",
    }

    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2)
        print(f"\nResults saved to {results_path}")

    return all_results


if __name__ == "__main__":
    run_confidence_gated_bright()
