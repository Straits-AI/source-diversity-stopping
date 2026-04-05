"""
Ablation Study Runner — AEA framework.

Tests hypothesis H4: "Router ablation accounts for >50% of total improvement."

Runs the following policies on BOTH benchmarks:
  pi_aea_heuristic     — full system (baseline for ablation delta)
  pi_lexical           — best HotpotQA single-substrate baseline
  pi_semantic          — best heterogeneous v2 single-substrate baseline
  abl_no_early_stop    — disable coverage-driven early stopping
  abl_semantic_only_smart_stop — semantic only + keep smart stop
  abl_no_entity_hop    — no entity graph hops
  abl_always_hop       — always hop after semantic
  abl_no_workspace_mgmt — no pin/evict management

Usage
-----
    python experiments/run_ablations.py

Results are saved to experiments/results/ablation_study.json
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

# ── Ensure project root is on sys.path ───────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── AEA imports ──────────────────────────────────────────────────────────────
from experiments.aea.address_spaces.entity_graph import EntityGraphAddressSpace
from experiments.aea.address_spaces.lexical import LexicalAddressSpace
from experiments.aea.address_spaces.semantic import SemanticAddressSpace
from experiments.aea.evaluation.harness import EvaluationHarness
from experiments.aea.policies.heuristic import AEAHeuristicPolicy
from experiments.aea.policies.single_substrate import (
    LexicalOnlyPolicy,
    SemanticOnlyPolicy,
)
from experiments.aea.policies.ablations import (
    AblAlwaysHop,
    AblNoEarlyStop,
    AblNoEntityHop,
    AblNoWorkspaceMgmt,
    AblSemanticOnlySmartStop,
)
from experiments.benchmarks.heterogeneous_benchmark_v2 import HeterogeneousBenchmarkV2

# ── Constants ────────────────────────────────────────────────────────────────
SEED = 42
N_EXAMPLES = 100
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_FILE = RESULTS_DIR / "ablation_study.json"

HOTPOTQA_LOCAL_PATHS = [
    "/tmp/hotpot_dev_distractor_v1_new.json",
    "/tmp/hotpot_dev_distractor_v1.json",
    "/tmp/hotpot_dev_distractor_v1_downloaded.json",
]
HOTPOTQA_URL = (
    "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _download_hotpotqa(dest_path: str) -> None:
    print(f"Downloading HotpotQA from {HOTPOTQA_URL} …")
    urllib.request.urlretrieve(HOTPOTQA_URL, dest_path)
    print(f"Saved to {dest_path}")


def load_hotpotqa(local_paths: list[str] = HOTPOTQA_LOCAL_PATHS) -> list[dict]:
    for path in local_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                print(f"Loaded HotpotQA from {path}  ({len(data)} examples)")
                return data
            except (json.JSONDecodeError, ValueError) as exc:
                print(f"  {path} appears corrupt ({exc}), skipping.")

    dest = "/tmp/hotpot_dev_distractor_v1_downloaded.json"
    _download_hotpotqa(dest)
    with open(dest, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    print(f"Loaded downloaded HotpotQA ({len(data)} examples)")
    return data


def filter_bridge(data: list[dict], n: int = N_EXAMPLES) -> list[dict]:
    bridge = [ex for ex in data if ex.get("type") == "bridge"]
    print(f"  Bridge questions in split: {len(bridge)}")
    return bridge[:n]


def convert_hotpotqa_example(raw: dict) -> dict:
    """Convert raw HotpotQA example to EvaluationHarness schema."""
    context_raw = raw.get("context", [])
    context_docs = []
    for entry in context_raw:
        title: str = entry[0]
        sentences: list[str] = entry[1]
        content = " ".join(s.strip() for s in sentences)
        context_docs.append({"id": title, "title": title, "content": content})

    supporting_facts = raw.get("supporting_facts", [])
    seen_titles: set[str] = set()
    gold_ids: list[str] = []
    for title, _sent_idx in supporting_facts:
        if title not in seen_titles:
            seen_titles.add(title)
            gold_ids.append(title)

    return {
        "id": raw.get("_id", ""),
        "question": raw["question"],
        "answer": raw.get("answer", ""),
        "supporting_facts": supporting_facts,
        "context": context_docs,
        "gold_ids": gold_ids,
    }


def load_hotpotqa_dataset() -> list[dict]:
    print("Loading HotpotQA bridge dataset …")
    raw_data = load_hotpotqa()
    bridge = filter_bridge(raw_data, n=N_EXAMPLES)
    dataset = [convert_hotpotqa_example(ex) for ex in bridge]
    print(f"  Converted {len(dataset)} bridge examples.\n")
    return dataset


def load_heterogeneous_v2_dataset() -> list[dict]:
    print("Generating heterogeneous benchmark v2 …")
    bench = HeterogeneousBenchmarkV2(seed=SEED)
    examples = bench.generate()
    dataset = []
    for ex in examples:
        dataset.append({
            "id":       ex["id"],
            "question": ex["question"],
            "answer":   ex["answer"],
            "context":  ex["context"],
            "gold_ids": ex["gold_ids"],
            "task_type": ex["task_type"],
        })
    print(f"  Generated {len(dataset)} examples.\n")
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# Policy & harness factories
# ─────────────────────────────────────────────────────────────────────────────

def make_ablation_policies() -> list:
    """Return all policies for the ablation study."""
    return [
        AEAHeuristicPolicy(top_k=5, coverage_threshold=0.5, max_steps=6),
        LexicalOnlyPolicy(top_k=5, max_steps=2),
        SemanticOnlyPolicy(top_k=5, max_steps=2),
        AblNoEarlyStop(top_k=5, coverage_threshold=0.5, max_steps=6),
        AblSemanticOnlySmartStop(top_k=5, coverage_threshold=0.5, max_steps=6),
        AblNoEntityHop(top_k=5, coverage_threshold=0.5, max_steps=6),
        AblAlwaysHop(top_k=5, coverage_threshold=0.5, max_steps=6),
        AblNoWorkspaceMgmt(top_k=5, coverage_threshold=0.5, max_steps=6),
    ]


def make_address_spaces() -> dict:
    return {
        "semantic": SemanticAddressSpace(model_name="all-MiniLM-L6-v2"),
        "lexical":  LexicalAddressSpace(),
        "entity":   EntityGraphAddressSpace(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_policies_on_dataset(
    policies: list,
    dataset: list[dict],
    seed: int = SEED,
    max_steps: int = 10,
    token_budget: int = 4000,
) -> dict:
    """Run every policy on the dataset and return {policy_name: result}."""
    all_results: dict = {}

    for policy in policies:
        pname = policy.name()
        print(f"  Running: {pname}")

        address_spaces = make_address_spaces()
        harness = EvaluationHarness(
            address_spaces=address_spaces,
            max_steps=max_steps,
            token_budget=token_budget,
            seed=seed,
        )

        t_start = time.perf_counter()
        result = harness.evaluate(policy, dataset)
        elapsed = time.perf_counter() - t_start

        result["runtime_seconds"] = round(elapsed, 2)
        all_results[pname] = result

        agg = result["aggregated"]
        print(
            f"    support_recall={agg['support_recall']:.4f}  "
            f"avg_ops={agg['operations_used']:.2f}  "
            f"utility@budget={agg['utility_at_budget']:.4f}  "
            f"({elapsed:.1f}s)"
        )

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Contribution analysis
# ─────────────────────────────────────────────────────────────────────────────

def _u(results: dict, policy_name: str) -> float:
    """Extract mean utility@budget for a policy."""
    if policy_name not in results:
        return float("nan")
    return results[policy_name]["aggregated"]["utility_at_budget"]


def compute_contributions(
    results: dict,
    full_aea_name: str,
    best_baseline_name: str,
) -> dict:
    """
    Compute the fraction of total improvement each component explains.

    total_improvement = U@B(full_AEA) - U@B(best_baseline)

    For each ablation:
        contribution = (U@B(full_AEA) - U@B(ablated)) / total_improvement * 100

    When total_improvement is near zero (AEA ties or loses to baseline),
    percentage contributions are numerically unstable.  In that case we
    report raw U@B deltas and mark contributions as nan with near_tie=True.
    """
    u_full = _u(results, full_aea_name)
    u_base = _u(results, best_baseline_name)
    total_improvement = u_full - u_base

    ablation_map = {
        "early_stop":          "abl_no_early_stop",
        "semantic_smart_stop": "abl_semantic_only_smart_stop",
        "no_entity_hop":       "abl_no_entity_hop",
        "always_hop":          "abl_always_hop",
        "workspace_mgmt":      "abl_no_workspace_mgmt",
    }

    contributions: dict = {}
    contributions["total_improvement"] = total_improvement
    contributions["u_full"] = u_full
    contributions["u_baseline"] = u_base
    # Raw deltas: positive means removing that component hurts AEA
    for component, abl_name in ablation_map.items():
        u_abl = _u(results, abl_name)
        contributions[f"delta_{component}"] = u_full - u_abl

    # Percentage contributions — only meaningful when |total_improvement| is large
    _MIN_IMPROVEMENT = 0.002  # 0.002 U@B minimum meaningful gap
    if abs(total_improvement) < _MIN_IMPROVEMENT:
        for key in ablation_map:
            contributions[key] = float("nan")
        contributions["near_tie"] = True
        return contributions

    contributions["near_tie"] = False
    for component, abl_name in ablation_map.items():
        u_abl = _u(results, abl_name)
        if u_abl != u_abl:  # nan check
            contributions[component] = float("nan")
        else:
            contributions[component] = (u_full - u_abl) / total_improvement * 100.0

    return contributions


def _routing_contribution(contributions: dict) -> float:
    """
    Estimate routing (selective substrate choice) contribution.

    Proxy: the gap between abl_no_entity_hop and abl_always_hop percentage
    contributions.  If always_hop is worse than no_entity_hop, selective
    routing captures the value of knowing WHEN to hop.

    For near-tie benchmarks the percentage is undefined — return 0.
    """
    if contributions.get("near_tie", False):
        return 0.0
    no_hop_contrib = contributions.get("no_entity_hop", 0.0) or 0.0
    always_hop_contrib = contributions.get("always_hop", 0.0) or 0.0
    return max(0.0, no_hop_contrib - always_hop_contrib)


# ─────────────────────────────────────────────────────────────────────────────
# Table printing
# ─────────────────────────────────────────────────────────────────────────────

_DISPLAY_ORDER = [
    ("pi_aea_heuristic",             "pi_aea_heuristic (full)"),
    ("pi_lexical",                   "pi_lexical (best HotpotQA BL)"),
    ("pi_semantic",                  "pi_semantic (best hetero BL)"),
    ("abl_no_early_stop",            "abl_no_early_stop"),
    ("abl_semantic_only_smart_stop", "abl_semantic_only_smart_stop"),
    ("abl_no_entity_hop",            "abl_no_entity_hop"),
    ("abl_always_hop",               "abl_always_hop"),
    ("abl_no_workspace_mgmt",        "abl_no_workspace_mgmt"),
]

_W_POLICY  = 34
_W_RECALL  = 13
_W_OPS     = 7
_W_UTILITY = 15
_W_DELTA   = 17


def _table_header() -> str:
    h = (
        f"| {'Policy':<{_W_POLICY}} "
        f"| {'SupportRecall':>{_W_RECALL}} "
        f"| {'AvgOps':>{_W_OPS}} "
        f"| {'Utility@Budget':>{_W_UTILITY}} "
        f"| {'Δ from Full AEA':>{_W_DELTA}} |"
    )
    sep = (
        f"| {'-'*_W_POLICY} "
        f"| {'-'*_W_RECALL} "
        f"| {'-'*_W_OPS} "
        f"| {'-'*_W_UTILITY} "
        f"| {'-'*_W_DELTA} |"
    )
    return h + "\n" + sep


def _table_row(label: str, agg: dict, delta: Optional[float]) -> str:
    delta_str = "baseline" if delta is None else f"{delta:+.4f}"
    return (
        f"| {label:<{_W_POLICY}} "
        f"| {agg['support_recall']:>{_W_RECALL}.4f} "
        f"| {agg['operations_used']:>{_W_OPS}.2f} "
        f"| {agg['utility_at_budget']:>{_W_UTILITY}.4f} "
        f"| {delta_str:>{_W_DELTA}} |"
    )


def print_benchmark_table(
    benchmark_label: str,
    results: dict,
    full_aea_name: str = "pi_aea_heuristic",
) -> None:
    u_full = _u(results, full_aea_name)
    print(f"\n[{benchmark_label}]")
    print(_table_header())
    for pname, plabel in _DISPLAY_ORDER:
        if pname not in results:
            continue
        agg = results[pname]["aggregated"]
        if pname == full_aea_name:
            delta = None
        else:
            delta = agg["utility_at_budget"] - u_full
        print(_table_row(plabel, agg, delta))


def _fmt_pct(v: float, near_tie: bool) -> str:
    """Format a percentage value or note near-tie."""
    if near_tie:
        return "n/a (near-tie)"
    if v != v:  # nan
        return "n/a"
    return f"{v:.1f}%"


def _fmt_delta(d: float) -> str:
    return f"{d:+.4f}"


def print_contribution_analysis(
    hotpot_contribs: dict,
    hetero_contribs: dict,
) -> None:
    """Print the component contribution analysis and H4 verdict."""
    h_tie = hotpot_contribs.get("near_tie", False)
    v_tie = hetero_contribs.get("near_tie", False)

    print("\nComponent Contribution Analysis:")
    print(
        f"  HotpotQA  total improvement: "
        f"{hotpot_contribs['total_improvement']:+.4f} U@B "
        f"({'near-tie — pct undefined' if h_tie else 'pct defined'})"
    )
    print(
        f"  Hetero v2 total improvement: "
        f"{hetero_contribs['total_improvement']:+.4f} U@B "
        f"({'near-tie — pct undefined' if v_tie else 'pct defined'})"
    )
    print()

    components = [
        ("early_stop",          "Early stopping",                       "delta_early_stop"),
        ("semantic_smart_stop", "Semantic-only + smart stop",           "delta_semantic_smart_stop"),
        ("no_entity_hop",       "Entity hop (abl: no-hop fallback)",    "delta_no_entity_hop"),
        ("always_hop",          "Selective hopping (abl: always-hop)",  "delta_always_hop"),
        ("workspace_mgmt",      "Workspace management",                 "delta_workspace_mgmt"),
    ]

    for pct_key, label, delta_key in components:
        hc = hotpot_contribs.get(pct_key, float("nan"))
        hv = hetero_contribs.get(pct_key, float("nan"))
        hd = hotpot_contribs.get(delta_key, float("nan"))
        vd = hetero_contribs.get(delta_key, float("nan"))
        print(
            f"  - {label}:\n"
            f"      HotpotQA  raw delta={_fmt_delta(hd)}  contrib={_fmt_pct(hc, h_tie)}\n"
            f"      Hetero v2 raw delta={_fmt_delta(vd)}  contrib={_fmt_pct(hv, v_tie)}"
        )

    # Routing contribution — uses whichever benchmark has a meaningful gap
    rc_h = _routing_contribution(hotpot_contribs)
    rc_v = _routing_contribution(hetero_contribs)
    hd_no_hop  = hotpot_contribs.get("delta_no_entity_hop", 0.0)
    hd_ah      = hotpot_contribs.get("delta_always_hop",    0.0)
    vd_no_hop  = hetero_contribs.get("delta_no_entity_hop", 0.0)
    vd_ah      = hetero_contribs.get("delta_always_hop",    0.0)
    print(
        f"  - Routing (selective substrate choice):\n"
        f"      HotpotQA  no_hop_delta={_fmt_delta(hd_no_hop)}  "
        f"always_hop_delta={_fmt_delta(hd_ah)}  routing_contrib={_fmt_pct(rc_h, h_tie)}\n"
        f"      Hetero v2 no_hop_delta={_fmt_delta(vd_no_hop)}  "
        f"always_hop_delta={_fmt_delta(vd_ah)}  routing_contrib={_fmt_pct(rc_v, v_tie)}"
    )

    # H4 verdict: use only benchmarks where pct is defined
    defined_routing = [r for r, tie in [(rc_h, h_tie), (rc_v, v_tie)] if not tie]
    if defined_routing:
        rc_avg = sum(defined_routing) / len(defined_routing)
        verdict = "YES" if rc_avg > 50 else "NO"
        scope = "HotpotQA only" if len(defined_routing) == 1 else "both benchmarks"
        print(f"\nH4 Test: Does router ablation account for >50% of improvement?")
        print(f"  Answer: [{verdict}] — routing accounts for {rc_avg:.1f}% of improvement ({scope}).")
    else:
        print(f"\nH4 Test: Does router ablation account for >50% of improvement?")
        print("  Answer: [INCONCLUSIVE] — both benchmarks are near-tie; percentage contributions undefined.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_ablations(
    seed: int = SEED,
    results_path: Optional[Path] = RESULTS_FILE,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    policies = make_ablation_policies()

    # ── HotpotQA ─────────────────────────────────────────────────────────────
    print("=" * 70)
    print("BENCHMARK 1: HotpotQA Bridge (N=100)")
    print("=" * 70)
    hotpot_dataset = load_hotpotqa_dataset()
    hotpot_results = run_policies_on_dataset(
        policies, hotpot_dataset, seed=seed
    )

    # ── Heterogeneous v2 ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BENCHMARK 2: Heterogeneous v2 (N=100)")
    print("=" * 70)
    hetero_dataset = load_heterogeneous_v2_dataset()
    hetero_results = run_policies_on_dataset(
        policies, hetero_dataset, seed=seed
    )

    # ── Report ───────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("=== ABLATION STUDY ===")
    print("=" * 70)

    print_benchmark_table("HotpotQA Bridge (N=100)", hotpot_results)
    print_benchmark_table("Heterogeneous v2 (N=100)", hetero_results)

    # Contribution analysis
    # HotpotQA: best baseline is pi_lexical (BM25 wins on HotpotQA)
    hotpot_contribs = compute_contributions(
        hotpot_results,
        full_aea_name="pi_aea_heuristic",
        best_baseline_name="pi_lexical",
    )
    # Heterogeneous: best baseline is pi_semantic
    hetero_contribs = compute_contributions(
        hetero_results,
        full_aea_name="pi_aea_heuristic",
        best_baseline_name="pi_semantic",
    )

    print_contribution_analysis(hotpot_contribs, hetero_contribs)

    # ── Save results ─────────────────────────────────────────────────────────
    payload = {
        "hotpotqa": hotpot_results,
        "heterogeneous_v2": hetero_results,
        "contribution_analysis": {
            "hotpotqa": hotpot_contribs,
            "heterogeneous_v2": hetero_contribs,
        },
    }

    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nDetailed results saved to: {results_path}")

    return payload


if __name__ == "__main__":
    run_ablations()
