"""
Confidence-Gated Stopping — N=500 HotpotQA (Reviewer underpowered concern).

Runs pi_aea_heuristic and pi_confidence_gated on 500 HotpotQA bridge
questions (eval split 0-499, same clean split as e2e_n500_clean.json).

Skips pi_ensemble because those results already exist in e2e_n500_clean.json
(pi_ensemble: F1=0.6814, E2E U@B=0.7068, ops=3.00).

For pi_confidence_gated, the adjusted U@B uses the +0.23 unified LLM cost:
    adj_ub_i = f1_i * (1 + 0.5 * recall_i) - 0.3 * ((ops_i + 0.23) / 3.0)

Statistical tests: paired t-test (scipy.stats.ttest_rel).

Usage
-----
    export OPENROUTER_API_KEY="sk-or-..."
    python experiments/run_confidence_gated_n500.py

Results saved to: experiments/results/confidence_gated_n500.json
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

# ── Set API key BEFORE any AEA imports (answer_generator reads key at import) ─
_API_KEY_VALUE = "REPLACE_WITH_YOUR_OPENROUTER_API_KEY"
os.environ["OPENROUTER_API_KEY"] = _API_KEY_VALUE

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
from experiments.aea.answer_generator import AnswerGenerator
from experiments.aea.evaluation.metrics import exact_match, f1_score

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
LLM_COST_OFFSET = 0.23    # unified LLM cost for confidence-gated stopping call
API_CALL_DELAY = 0.5       # seconds between LLM answer-gen calls

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_FILE = RESULTS_DIR / "confidence_gated_n500.json"

# Set API key
os.environ.setdefault(
    "OPENROUTER_API_KEY",
    "REPLACE_WITH_YOUR_OPENROUTER_API_KEY",
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


# ── U@B helpers ───────────────────────────────────────────────────────────────

def compute_ub(f1: float, recall: float, ops: float) -> float:
    """Standard U@B (no LLM cost offset)."""
    return f1 * (1.0 + 0.5 * recall) - MU * (ops / MAX_OPS)


def compute_adj_ub(f1: float, recall: float, ops: float) -> float:
    """Adjusted U@B with +0.23 LLM cost offset for confidence-gated policy."""
    return f1 * (1.0 + 0.5 * recall) - MU * ((ops + LLM_COST_OFFSET) / MAX_OPS)


# ── Evidence extraction ───────────────────────────────────────────────────────

def extract_evidence(ex_result: dict, dataset_by_id: dict[str, dict]) -> list[str]:
    """
    Extract retrieved passage texts.

    Tries (in order):
    1. trace → result_items (full trace present)
    2. retrieved_ids field (compact checkpoint)
    3. Falls back to empty list if neither available
    """
    retrieved_ids: list[str] = []

    # Method 1: full trace
    if ex_result.get("trace"):
        for trace_step in ex_result["trace"]:
            for item in trace_step.get("result_items", []):
                rid = item.get("id", "")
                if rid and rid not in retrieved_ids:
                    retrieved_ids.append(rid)
    # Method 2: compact retrieved_ids field (stored in checkpoint)
    elif ex_result.get("retrieved_ids"):
        retrieved_ids = ex_result["retrieved_ids"]

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


# ── LLM answer generation ─────────────────────────────────────────────────────

def generate_answers_for_policy(
    harness_result: dict,
    dataset_by_id: dict[str, dict],
    answer_gen: AnswerGenerator,
    policy_name: str,
    is_confidence_gated: bool = False,
) -> list[dict]:
    """
    Generate LLM answers and compute per-example metrics.

    For confidence_gated, also compute the adjusted U@B with LLM cost offset.
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
            time.sleep(API_CALL_DELAY)
            llm_answer = answer_gen.generate_answer(question, evidence_passages)
        else:
            llm_answer = ""

        em = exact_match(llm_answer, gold_answer)
        f1 = f1_score(llm_answer, gold_answer)

        ub = compute_ub(f1, support_recall_val, operations_used)
        adj_ub = compute_adj_ub(f1, support_recall_val, operations_used) if is_confidence_gated else ub

        entry = {
            "id": ex_id,
            "question": question,
            "gold_answer": gold_answer,
            "llm_answer": llm_answer,
            "em": em,
            "f1": f1,
            "support_recall": support_recall_val,
            "operations_used": operations_used,
            "e2e_ub": ub,
        }
        if is_confidence_gated:
            entry["adj_e2e_ub"] = adj_ub

        e2e_results.append(entry)

        if (idx + 1) % 50 == 0 or idx == 0:
            pct = (idx + 1) / total * 100
            usage = answer_gen.usage_summary()
            print(
                f"    [{policy_name}] {idx+1}/{total} ({pct:.0f}%) — "
                f"API calls: {usage['total_calls']}, errors: {usage['total_errors']}"
            )

    return e2e_results


# ── Aggregate helpers ─────────────────────────────────────────────────────────

def aggregate_e2e(e2e_results: list[dict], is_confidence_gated: bool = False) -> dict:
    out = {}
    keys = ["em", "f1", "support_recall", "operations_used", "e2e_ub"]
    if is_confidence_gated:
        keys.append("adj_e2e_ub")
    for key in keys:
        vals = [r[key] for r in e2e_results if key in r]
        out[f"{key}_mean"] = float(np.mean(vals)) if vals else 0.0
        out[f"{key}_std"] = float(np.std(vals)) if vals else 0.0
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def run_confidence_gated_n500(
    n_examples: int = N_EXAMPLES,
    seed: int = SEED,
    results_path: Optional[Path] = RESULTS_FILE,
) -> dict:
    """
    N=500 HotpotQA evaluation of pi_aea_heuristic vs pi_confidence_gated.

    Returns
    -------
    dict with all retrieval metrics, E2E metrics, and statistical test results.
    """
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print(f"Confidence-Gated N=500 Evaluation — seed={seed}")
    print("=" * 70)

    # ── Phase 1: Load data ────────────────────────────────────────────────────
    print("\n[Phase 1] Loading HotpotQA data (eval split 0-499)…")
    raw_data = load_hotpotqa()
    bridge = filter_bridge(raw_data, n=n_examples)
    dataset = [convert_example(ex) for ex in bridge]
    dataset_by_id = {ex["id"]: ex for ex in dataset}
    print(f"  Loaded {len(dataset)} bridge examples.\n")

    # ── Phase 2: Retrieval-only phase ─────────────────────────────────────────
    # Checkpoint path: save retrieval results to avoid re-running the expensive
    # confidence-gated retrieval (500 LLM calls) if phase 3 fails.
    checkpoint_path = RESULTS_DIR / "confidence_gated_n500_retrieval_checkpoint.json"
    all_retrieval: dict = {}

    if checkpoint_path.exists():
        print("[Phase 2] Loading retrieval checkpoint (skipping re-run)…")
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        # Restore per_example results but not the full trace (saves memory)
        all_retrieval = checkpoint
        for pname, res in all_retrieval.items():
            agg = res["aggregated"]
            print(f"  [checkpoint] {pname}: SR={agg['support_recall']:.4f}, ops={agg['operations_used']:.2f}")
    else:
        print("[Phase 2] Running retrieval-only phase…")
        policies = [
            AEAHeuristicPolicy(top_k=5, coverage_threshold=0.5, max_steps=6),
            ConfidenceGatedPolicy(top_k=5),
        ]

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

        # Save checkpoint so Phase 3 can be retried without re-running retrieval.
        # We store a compact version: per_example has retrieved_ids extracted from trace,
        # dropping the full trace to keep the file small.
        checkpoint = {}
        for pname, res in all_retrieval.items():
            compact_per = []
            for ex in res.get("per_example", []):
                # Extract retrieved_ids from trace
                rids: list[str] = []
                for ts in ex.get("trace", []):
                    for item in ts.get("result_items", []):
                        rid = item.get("id", "")
                        if rid and rid not in rids:
                            rids.append(rid)
                compact_ex = {k: v for k, v in ex.items() if k != "trace"}
                compact_ex["retrieved_ids"] = rids
                compact_per.append(compact_ex)
            checkpoint[pname] = {**res, "per_example": compact_per}
            checkpoint[pname].pop("per_example", None)
            checkpoint[pname]["per_example"] = compact_per

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)
        print(f"\n  [checkpoint saved to {checkpoint_path}]")

    # ── Phase 3: LLM answer generation ───────────────────────────────────────
    print(f"\n[Phase 3] Generating LLM answers for {POLICIES_TO_RUN} via gpt-oss-120b…")
    answer_gen = AnswerGenerator()
    all_e2e: dict[str, list[dict]] = {}

    for pname in POLICIES_TO_RUN:
        if pname not in all_retrieval:
            print(f"  WARNING: {pname} missing, skipping.")
            continue
        is_cg = pname == "pi_confidence_gated"
        print(f"\n  Generating answers for: {pname}")
        e2e_results = generate_answers_for_policy(
            all_retrieval[pname], dataset_by_id, answer_gen, pname,
            is_confidence_gated=is_cg,
        )
        all_e2e[pname] = e2e_results

        agg = aggregate_e2e(e2e_results, is_confidence_gated=is_cg)
        print(f"    EM={agg['em_mean']:.4f}, F1={agg['f1_mean']:.4f}, E2E U@B={agg['e2e_ub_mean']:.4f}", end="")
        if is_cg:
            print(f", Adj U@B={agg['adj_e2e_ub_mean']:.4f}")
        else:
            print()

    usage = answer_gen.usage_summary()
    print(f"\n  Total API calls: {usage['total_calls']}")
    print(f"  Total tokens: {usage['total_tokens']}")
    print(f"  Errors: {usage['total_errors']}")

    # ── Phase 4: Statistics ───────────────────────────────────────────────────
    print("\n[Phase 4] Statistical tests (paired t-test on E2E U@B)…")

    agg_stats: dict[str, dict] = {}
    for pname, results in all_e2e.items():
        is_cg = pname == "pi_confidence_gated"
        agg_stats[pname] = aggregate_e2e(results, is_confidence_gated=is_cg)

    stat_tests: list[dict] = []

    # Test 1: confidence_gated (standard U@B) vs aea_heuristic
    if "pi_confidence_gated" in all_e2e and "pi_aea_heuristic" in all_e2e:
        cg_ub = [r["e2e_ub"] for r in all_e2e["pi_confidence_gated"]]
        aea_ub = [r["e2e_ub"] for r in all_e2e["pi_aea_heuristic"]]
        tstat, pval = stats.ttest_rel(cg_ub, aea_ub)
        delta = float(np.mean(cg_ub)) - float(np.mean(aea_ub))
        d = delta / float(np.std(np.array(cg_ub) - np.array(aea_ub), ddof=1)) if np.std(np.array(cg_ub) - np.array(aea_ub), ddof=1) > 0 else 0.0
        test_entry = {
            "label": "ConfGated vs AEAHeuristic (standard U@B)",
            "policy_a": "pi_confidence_gated",
            "policy_b": "pi_aea_heuristic",
            "delta": delta,
            "t_statistic": float(tstat),
            "p_value": float(pval),
            "cohens_d": d,
            "significant": bool(pval < 0.05),
        }
        stat_tests.append(test_entry)
        sig = "YES" if pval < 0.05 else "NO"
        print(f"  ConfGated vs AEAHeuristic (U@B): Δ={delta:.4f}, t={tstat:.3f}, p={pval:.4f}, d={d:.3f}, Sig? {sig}")

    # Test 2: confidence_gated (adjusted U@B) vs aea_heuristic (standard U@B)
    if "pi_confidence_gated" in all_e2e and "pi_aea_heuristic" in all_e2e:
        cg_adj = [r["adj_e2e_ub"] for r in all_e2e["pi_confidence_gated"]]
        aea_ub = [r["e2e_ub"] for r in all_e2e["pi_aea_heuristic"]]
        tstat, pval = stats.ttest_rel(cg_adj, aea_ub)
        delta = float(np.mean(cg_adj)) - float(np.mean(aea_ub))
        d = delta / float(np.std(np.array(cg_adj) - np.array(aea_ub), ddof=1)) if np.std(np.array(cg_adj) - np.array(aea_ub), ddof=1) > 0 else 0.0
        test_entry = {
            "label": "ConfGated (adj +0.23 LLM cost) vs AEAHeuristic",
            "policy_a": "pi_confidence_gated_adj",
            "policy_b": "pi_aea_heuristic",
            "delta": delta,
            "t_statistic": float(tstat),
            "p_value": float(pval),
            "cohens_d": d,
            "significant": bool(pval < 0.05),
        }
        stat_tests.append(test_entry)
        sig = "YES" if pval < 0.05 else "NO"
        print(f"  ConfGated-adj vs AEAHeuristic (Adj U@B): Δ={delta:.4f}, t={tstat:.3f}, p={pval:.4f}, d={d:.3f}, Sig? {sig}")

    # ── Phase 5: Print summary table ──────────────────────────────────────────
    print("\n=== N=500 HotpotQA — Confidence-Gated vs AEA Heuristic ===\n")

    # Retrieval-only summary
    print("Retrieval-only metrics:")
    print(f"  {'Policy':<25} {'Recall':>8} {'Precision':>10} {'Ops':>6} {'U@B':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*6} {'-'*8}")
    for pname in POLICIES_TO_RUN:
        if pname not in all_retrieval:
            continue
        agg = all_retrieval[pname]["aggregated"]
        print(
            f"  {pname:<25} {agg['support_recall']:>8.4f} "
            f"{agg['support_precision']:>10.4f} {agg['operations_used']:>6.2f} "
            f"{agg['utility_at_budget']:>8.4f}"
        )

    # E2E summary
    print("\nE2E metrics (with LLM answer generation):")
    print(f"  {'Policy':<25} {'EM':>7} {'F1':>7} {'Recall':>8} {'Ops':>6} {'U@B':>8} {'Adj U@B':>9}")
    print(f"  {'-'*25} {'-'*7} {'-'*7} {'-'*8} {'-'*6} {'-'*8} {'-'*9}")
    for pname in POLICIES_TO_RUN:
        if pname not in agg_stats:
            continue
        agg = agg_stats[pname]
        is_cg = pname == "pi_confidence_gated"
        adj_str = f"{agg['adj_e2e_ub_mean']:>9.4f}" if is_cg and "adj_e2e_ub_mean" in agg else "       N/A"
        print(
            f"  {pname:<25} {agg['em_mean']:>7.4f} {agg['f1_mean']:>7.4f} "
            f"{agg['support_recall_mean']:>8.4f} {agg['operations_used_mean']:>6.2f} "
            f"{agg['e2e_ub_mean']:>8.4f} {adj_str}"
        )

    # Ensemble reference (from e2e_n500_clean.json)
    print("\n  [Reference from e2e_n500_clean.json — pi_ensemble N=500]")
    print("  pi_ensemble               F1=0.6814, E2E U@B=0.7068, ops=3.00")

    print("\nPaired t-test results:")
    for t in stat_tests:
        sig = "YES" if t["p_value"] < 0.05 else "NO"
        print(
            f"  {t['label']}: "
            f"Δ={t['delta']:.4f}, t={t['t_statistic']:.3f}, "
            f"p={t['p_value']:.4f}, d={t['cohens_d']:.3f}, Significant? {sig}"
        )

    # ── Save results ──────────────────────────────────────────────────────────
    full_results = {
        "experiment": "confidence_gated_n500",
        "n_examples": n_examples,
        "seed": seed,
        "mu": MU,
        "max_ops": MAX_OPS,
        "llm_cost_offset_confidence_gated": LLM_COST_OFFSET,
        "retrieval_aggregated": {
            pname: result["aggregated"]
            for pname, result in all_retrieval.items()
        },
        "e2e_per_example": all_e2e,
        "e2e_aggregated": agg_stats,
        "statistical_tests": stat_tests,
        "api_usage": usage,
        "note": (
            "Ensemble reference (from e2e_n500_clean.json): "
            "F1=0.6814, E2E U@B=0.7068, ops=3.00"
        ),
    }

    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)

        def _to_serializable(obj):
            """Recursively convert non-JSON-serializable types."""
            if isinstance(obj, dict):
                return {k: _to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_serializable(v) for v in obj]
            # Must check numpy types BEFORE Python built-in types because
            # numpy.bool_ is NOT a subclass of Python bool in modern numpy,
            # but numpy.bool_.__class__.__name__ == 'bool' confuses json.
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            return obj

        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump(_to_serializable(full_results), fh, indent=2)
        print(f"\nResults saved to {results_path}")

    return full_results


if __name__ == "__main__":
    run_confidence_gated_n500()
