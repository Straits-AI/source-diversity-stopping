"""
HotpotQA Bridge Baselines — AEA framework evaluation.

Runs all five policies (SemanticOnly, LexicalOnly, EntityOnly, Ensemble,
AEAHeuristic) on the first 100 bridge questions from the HotpotQA distractor
validation set and reports aggregated metrics.

Usage
-----
    python experiments/run_hotpotqa_baselines.py

or as a module::

    from experiments.run_hotpotqa_baselines import run_baselines
    results = run_baselines()
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

# ── Make sure project root is on sys.path when run directly ──────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── AEA imports ──────────────────────────────────────────────────────────────
from experiments.aea.address_spaces.semantic import SemanticAddressSpace
from experiments.aea.address_spaces.lexical import LexicalAddressSpace
from experiments.aea.address_spaces.entity_graph import EntityGraphAddressSpace
from experiments.aea.evaluation.harness import EvaluationHarness
from experiments.aea.policies.single_substrate import (
    SemanticOnlyPolicy,
    LexicalOnlyPolicy,
    EntityOnlyPolicy,
)
from experiments.aea.policies.ensemble import EnsemblePolicy
from experiments.aea.policies.heuristic import AEAHeuristicPolicy

# ── Constants ────────────────────────────────────────────────────────────────
HOTPOTQA_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
HOTPOTQA_LOCAL_PATHS = [
    "/tmp/hotpot_dev_distractor_v1_new.json",
    "/tmp/hotpot_dev_distractor_v1.json",
]
N_EXAMPLES = 100
SEED = 42
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_FILE = RESULTS_DIR / "hotpotqa_baselines.json"


# ── Data loading ─────────────────────────────────────────────────────────────

def _download_hotpotqa(dest_path: str) -> None:
    """Download HotpotQA distractor validation set."""
    print(f"Downloading HotpotQA from {HOTPOTQA_URL} …")
    urllib.request.urlretrieve(HOTPOTQA_URL, dest_path)
    print(f"Saved to {dest_path}")


def load_hotpotqa(local_paths: list[str] = HOTPOTQA_LOCAL_PATHS) -> list[dict]:
    """
    Load HotpotQA JSON from a local path, trying each path in order.
    Downloads from the canonical URL if none exist.
    """
    for path in local_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                print(f"Loaded HotpotQA from {path}  ({len(data)} examples)")
                return data
            except (json.JSONDecodeError, ValueError) as exc:
                print(f"  {path} appears corrupt ({exc}), skipping.")

    # None of the local paths worked — download
    dest = "/tmp/hotpot_dev_distractor_v1_downloaded.json"
    _download_hotpotqa(dest)
    with open(dest, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    print(f"Loaded downloaded HotpotQA ({len(data)} examples)")
    return data


def filter_bridge(data: list[dict], n: int = N_EXAMPLES) -> list[dict]:
    """Return the first *n* bridge-type questions."""
    bridge = [ex for ex in data if ex.get("type") == "bridge"]
    print(f"Bridge questions in split: {len(bridge)}")
    return bridge[:n]


# ── Format conversion ─────────────────────────────────────────────────────────

def convert_example(raw: dict) -> dict:
    """
    Convert a raw HotpotQA example to the EvaluationHarness input format.

    HotpotQA context schema:
        [ [title, [sentence0, sentence1, …]], … ]

    EvaluationHarness context schema:
        [ {"id": str, "title": str, "content": str}, … ]

    gold_ids are the titles of the paragraphs that contain the gold
    supporting facts (unique titles, preserving order of first occurrence).
    """
    context_raw = raw.get("context", [])

    context_docs = []
    for entry in context_raw:
        title: str = entry[0]
        sentences: list[str] = entry[1]
        content = " ".join(s.strip() for s in sentences)
        context_docs.append({
            "id": title,          # title is stable across HotpotQA splits
            "title": title,
            "content": content,
        })

    # Gold supporting-fact titles (the titles of paragraphs that have
    # at least one supporting sentence).
    supporting_facts = raw.get("supporting_facts", [])  # list of [title, sent_idx]
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


# ── Policy factory ────────────────────────────────────────────────────────────

def make_policies() -> list:
    """Instantiate all baseline policies."""
    return [
        SemanticOnlyPolicy(top_k=5, max_steps=2),
        LexicalOnlyPolicy(top_k=5, max_steps=2),
        EntityOnlyPolicy(top_k=5, max_steps=3),
        EnsemblePolicy(top_k=5, max_steps=3),
        AEAHeuristicPolicy(top_k=5, coverage_threshold=0.5, max_steps=6),
    ]


# ── Address-space factory ─────────────────────────────────────────────────────

def make_address_spaces() -> dict:
    """
    Instantiate address spaces.

    A fresh harness is built for each policy, but the address-space *objects*
    are re-created per policy run so indices are rebuilt cleanly.
    """
    return {
        "semantic": SemanticAddressSpace(model_name="all-MiniLM-L6-v2"),
        "lexical": LexicalAddressSpace(),
        "entity": EntityGraphAddressSpace(),
    }


# ── Main evaluation loop ──────────────────────────────────────────────────────

def run_baselines(
    n_examples: int = N_EXAMPLES,
    seed: int = SEED,
    results_path: Optional[Path] = RESULTS_FILE,
) -> dict:
    """
    Run all policies on the first *n_examples* HotpotQA bridge questions.

    Parameters
    ----------
    n_examples : int
        How many bridge questions to evaluate on.
    seed : int
        Random seed for reproducibility.
    results_path : Path, optional
        Where to save the JSON results.  If None, results are not saved.

    Returns
    -------
    dict
        ``{policy_name: evaluation_result_dict}``
    """
    random.seed(seed)
    np.random.seed(seed)

    # ── Load data ────────────────────────────────────────────────────────────
    raw_data = load_hotpotqa()
    bridge = filter_bridge(raw_data, n=n_examples)
    dataset = [convert_example(ex) for ex in bridge]
    print(f"Converted {len(dataset)} examples for evaluation.\n")

    policies = make_policies()
    all_results: dict = {}

    # ── Run each policy ──────────────────────────────────────────────────────
    for policy in policies:
        pname = policy.name()
        print(f"{'─' * 60}")
        print(f"Running policy: {pname}")
        print(f"{'─' * 60}")

        # Fresh address spaces and harness for each policy so indices are clean
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
        print(f"  support_recall:    {agg['support_recall']:.4f}")
        print(f"  support_precision: {agg['support_precision']:.4f}")
        print(f"  avg_operations:    {agg['operations_used']:.2f}")
        print(f"  utility@budget:    {agg['utility_at_budget']:.4f}")
        print(f"  n_errors:          {result['n_errors']}")
        print(f"  runtime:           {elapsed:.1f}s\n")

    # ── Summary table ────────────────────────────────────────────────────────
    _print_summary_table(all_results, n_examples)

    # ── Save results ─────────────────────────────────────────────────────────
    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        # Serialise: per_example traces can be large — include them in full
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump(all_results, fh, indent=2)
        print(f"\nDetailed results saved to: {results_path}")

    return all_results


def _print_summary_table(all_results: dict, n_examples: int) -> None:
    """Print a Markdown-style summary table to stdout."""
    col_widths = {
        "Policy": 22,
        "SupportRecall": 15,
        "SupportPrecision": 18,
        "AvgOps": 9,
        "Utility@Budget": 16,
    }

    header = (
        f"| {'Policy':<{col_widths['Policy']}} "
        f"| {'SupportRecall':>{col_widths['SupportRecall']}} "
        f"| {'SupportPrecision':>{col_widths['SupportPrecision']}} "
        f"| {'AvgOps':>{col_widths['AvgOps']}} "
        f"| {'Utility@Budget':>{col_widths['Utility@Budget']}} |"
    )
    sep = (
        f"| {'-' * col_widths['Policy']} "
        f"| {'-' * col_widths['SupportRecall']} "
        f"| {'-' * col_widths['SupportPrecision']} "
        f"| {'-' * col_widths['AvgOps']} "
        f"| {'-' * col_widths['Utility@Budget']} |"
    )

    print(f"\n=== HotpotQA Bridge Baselines (N={n_examples}) ===\n")
    print(header)
    print(sep)

    for pname, result in all_results.items():
        agg = result["aggregated"]
        row = (
            f"| {pname:<{col_widths['Policy']}} "
            f"| {agg['support_recall']:>{col_widths['SupportRecall']}.4f} "
            f"| {agg['support_precision']:>{col_widths['SupportPrecision']}.4f} "
            f"| {agg['operations_used']:>{col_widths['AvgOps']}.2f} "
            f"| {agg['utility_at_budget']:>{col_widths['Utility@Budget']}.4f} |"
        )
        print(row)

    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_baselines()
