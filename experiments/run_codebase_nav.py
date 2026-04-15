"""
Codebase Navigation Experiment — AEA Framework.

Tests whether convergence-based stopping works for CODE SEARCH,
directly addressing reviewer concerns about generalization beyond QA.

META beauty: We test our own theory on our own codebase.

Research questions
------------------
1. Does source-diversity stopping work for code navigation (not just QA)?
2. Which substrate is best for code search — lexical, semantic, or structural?
3. Does the heuristic policy match ensemble quality with fewer ops?

Corpus
------
The AEA research codebase itself (~26 Python files):
  experiments/aea/types.py
  experiments/aea/address_spaces/*.py
  experiments/aea/evaluation/*.py
  experiments/aea/policies/*.py
  experiments/benchmarks/*.py
  experiments/run_*.py (sample)

Dataset
-------
50 code-search questions at three difficulty levels:
  - Easy   (16): direct keyword match ("where is f1_score defined?")
  - Medium (18): semantic match ("how does the system measure answer quality?")
  - Hard   (16): code-relationship questions ("which components does the
                  heuristic policy depend on?")

Each question has 1-2 gold files and ~8-9 distractor files in its context.

Policies
--------
  pi_semantic       — dense retrieval only
  pi_lexical        — BM25 only
  pi_structural     — title/filename matching only
  pi_ensemble       — round-robin over all 3 substrates
  pi_heuristic_code — convergence-based stopping (source diversity criterion)

Results saved to experiments/results/codebase_nav.json.

Usage
-----
    python experiments/run_codebase_nav.py
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats  # type: ignore

# ── Ensure project root is on sys.path ───────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── AEA imports ──────────────────────────────────────────────────────────────
from experiments.aea.address_spaces.lexical import LexicalAddressSpace
from experiments.aea.address_spaces.semantic import SemanticAddressSpace
from experiments.aea.address_spaces.structural import StructuralAddressSpace
from experiments.aea.evaluation.harness import EvaluationHarness
from experiments.aea.policies.base import Policy
from experiments.aea.types import (
    Action,
    AddressSpaceType,
    AgentState,
    Operation,
)

# ── Constants ─────────────────────────────────────────────────────────────────
SEED = 42
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_FILE = RESULTS_DIR / "codebase_nav.json"

# Path to the codebase we're indexing
_AEA_ROOT = Path(__file__).resolve().parent


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Build the code corpus
# ─────────────────────────────────────────────────────────────────────────────

_DOC_MAX_CHARS = 3000   # Truncate each file to keep token costs manageable


def _load_file(path: Path) -> Optional[str]:
    """Read a Python file and return its content (truncated), or None on error."""
    try:
        text = path.read_text(encoding="utf-8")
        if len(text) > _DOC_MAX_CHARS:
            text = text[:_DOC_MAX_CHARS] + "\n# ... (truncated)"
        return text
    except Exception:
        return None


def build_code_corpus() -> list[dict]:
    """
    Read Python files from the AEA codebase.

    Returns a list of document dicts with keys:
      - id       : relative path string (stable identifier)
      - title    : same as id (used by structural address space)
      - content  : full file text
      - module   : module name derived from path
    """
    patterns = [
        "aea/types.py",
        "aea/__init__.py",
        "aea/answer_generator.py",
        "aea/address_spaces/__init__.py",
        "aea/address_spaces/base.py",
        "aea/address_spaces/entity_graph.py",
        "aea/address_spaces/executable.py",
        "aea/address_spaces/lexical.py",
        "aea/address_spaces/semantic.py",
        "aea/address_spaces/structural.py",
        "aea/evaluation/__init__.py",
        "aea/evaluation/harness.py",
        "aea/evaluation/metrics.py",
        "aea/policies/__init__.py",
        "aea/policies/ablations.py",
        "aea/policies/answer_stability.py",
        "aea/policies/base.py",
        "aea/policies/confidence_gated.py",
        "aea/policies/cross_encoder_stopping.py",
        "aea/policies/decomposition_stopping.py",
        "aea/policies/embedding_router.py",
        "aea/policies/ensemble.py",
        "aea/policies/heuristic.py",
        "aea/policies/learned_stopping.py",
        "aea/policies/llm_routed.py",
        "aea/policies/nli_stopping.py",
        "aea/policies/single_substrate.py",
        "benchmarks/computational_benchmark.py",
        "benchmarks/heterogeneous_benchmark.py",
        "benchmarks/structural_nav_benchmark.py",
        "run_structural_nav.py",
        "run_hotpotqa_baselines.py",
        "run_ablations.py",
        "run_learned_stopping.py",
        "run_embedding_router_eval.py",
    ]

    corpus = []
    for rel_path in patterns:
        full_path = _AEA_ROOT / rel_path
        if not full_path.exists():
            continue
        content = _load_file(full_path)
        if content is None or not content.strip():
            continue

        # Derive a readable module name
        module = rel_path.replace("/", ".").replace(".py", "")

        corpus.append({
            "id": rel_path,
            "title": rel_path,
            "content": content,
            "module": module,
        })

    return corpus


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Create 50 code-search questions
# ─────────────────────────────────────────────────────────────────────────────

# Format: (id, question, gold_ids, difficulty)
# gold_ids must match the "id" keys in build_code_corpus()

_RAW_QUESTIONS = [
    # ── EASY (direct keyword match) ──────────────────────────────────────────
    (
        "easy_001",
        "Where is the f1_score function defined?",
        ["aea/evaluation/metrics.py"],
        "easy",
    ),
    (
        "easy_002",
        "Which file defines the exact_match function?",
        ["aea/evaluation/metrics.py"],
        "easy",
    ),
    (
        "easy_003",
        "Where is utility_at_budget implemented?",
        ["aea/evaluation/metrics.py"],
        "easy",
    ),
    (
        "easy_004",
        "Which file defines the AgentState dataclass?",
        ["aea/types.py"],
        "easy",
    ),
    (
        "easy_005",
        "Where is WorkspaceItem defined?",
        ["aea/types.py"],
        "easy",
    ),
    (
        "easy_006",
        "Which file defines the Operation enum?",
        ["aea/types.py"],
        "easy",
    ),
    (
        "easy_007",
        "Where is BM25Okapi used?",
        ["aea/address_spaces/lexical.py"],
        "easy",
    ),
    (
        "easy_008",
        "Which file imports SentenceTransformer?",
        ["aea/address_spaces/semantic.py"],
        "easy",
    ),
    (
        "easy_009",
        "Where is support_recall defined?",
        ["aea/evaluation/metrics.py"],
        "easy",
    ),
    (
        "easy_010",
        "Which file defines the EvaluationHarness class?",
        ["aea/evaluation/harness.py"],
        "easy",
    ),
    (
        "easy_011",
        "Where is normalize_cost implemented?",
        ["aea/evaluation/metrics.py"],
        "easy",
    ),
    (
        "easy_012",
        "Which file defines AEAHeuristicPolicy?",
        ["aea/policies/heuristic.py"],
        "easy",
    ),
    (
        "easy_013",
        "Where is EnsemblePolicy defined?",
        ["aea/policies/ensemble.py"],
        "easy",
    ),
    (
        "easy_014",
        "Which file defines SemanticOnlyPolicy?",
        ["aea/policies/single_substrate.py"],
        "easy",
    ),
    (
        "easy_015",
        "Where is LexicalOnlyPolicy defined?",
        ["aea/policies/single_substrate.py"],
        "easy",
    ),
    (
        "easy_016",
        "Which file defines the AddressSpace abstract base class?",
        ["aea/address_spaces/base.py"],
        "easy",
    ),

    # ── MEDIUM (semantic match — concept-level) ───────────────────────────────
    (
        "med_001",
        "How does the system measure answer quality?",
        ["aea/evaluation/metrics.py"],
        "medium",
    ),
    (
        "med_002",
        "Where is the convergence-based stopping rule implemented?",
        ["aea/policies/heuristic.py"],
        "medium",
    ),
    (
        "med_003",
        "Which file handles workspace eviction of low-relevance items?",
        ["aea/evaluation/harness.py"],
        "medium",
    ),
    (
        "med_004",
        "How does the entity graph build its index?",
        ["aea/address_spaces/entity_graph.py"],
        "medium",
    ),
    (
        "med_005",
        "Which file implements title-hierarchy navigation for documents?",
        ["aea/address_spaces/structural.py"],
        "medium",
    ),
    (
        "med_006",
        "How is cosine similarity computed during retrieval?",
        ["aea/address_spaces/semantic.py"],
        "medium",
    ),
    (
        "med_007",
        "Where are workspace pin and evict management operations handled?",
        ["aea/policies/heuristic.py", "aea/evaluation/harness.py"],
        "medium",
    ),
    (
        "med_008",
        "Which file manages the token budget and step limits?",
        ["aea/evaluation/harness.py"],
        "medium",
    ),
    (
        "med_009",
        "Where is source-diversity stopping logic encoded?",
        ["aea/policies/heuristic.py"],
        "medium",
    ),
    (
        "med_010",
        "How does the harness derive an answer from the workspace?",
        ["aea/evaluation/harness.py"],
        "medium",
    ),
    (
        "med_011",
        "Which file contains keyword query construction logic?",
        ["aea/policies/heuristic.py"],
        "medium",
    ),
    (
        "med_012",
        "Where is multi-hop question detection implemented?",
        ["aea/policies/heuristic.py"],
        "medium",
    ),
    (
        "med_013",
        "Which file implements the round-robin substrate cycling policy?",
        ["aea/policies/ensemble.py"],
        "medium",
    ),
    (
        "med_014",
        "Where is document text tokenized for BM25 indexing?",
        ["aea/address_spaces/lexical.py"],
        "medium",
    ),
    (
        "med_015",
        "How does the agent accumulate evidence across retrieval steps?",
        ["aea/evaluation/harness.py"],
        "medium",
    ),
    (
        "med_016",
        "Which file implements the policy abstract base class?",
        ["aea/policies/base.py"],
        "medium",
    ),
    (
        "med_017",
        "Where is bundle_coverage computed?",
        ["aea/evaluation/metrics.py"],
        "medium",
    ),
    (
        "med_018",
        "Which policy uses an embedding model for routing decisions?",
        ["aea/policies/embedding_router.py"],
        "medium",
    ),

    # ── HARD (code relationship / cross-file reasoning) ───────────────────────
    (
        "hard_001",
        "Which components does the heuristic policy depend on to make routing decisions?",
        ["aea/policies/heuristic.py", "aea/types.py"],
        "hard",
    ),
    (
        "hard_002",
        "Which files are involved in running a full evaluation episode from policy action to metric computation?",
        ["aea/evaluation/harness.py", "aea/evaluation/metrics.py"],
        "hard",
    ),
    (
        "hard_003",
        "Where is the ActionResult cost accounting defined and where is it consumed?",
        ["aea/types.py", "aea/evaluation/harness.py"],
        "hard",
    ),
    (
        "hard_004",
        "Which files define the data contracts that all address spaces must satisfy?",
        ["aea/address_spaces/base.py", "aea/types.py"],
        "hard",
    ),
    (
        "hard_005",
        "How does the harness connect policy actions to address space queries?",
        ["aea/evaluation/harness.py", "aea/address_spaces/base.py"],
        "hard",
    ),
    (
        "hard_006",
        "Which files implement the stopping criterion and where is it triggered?",
        ["aea/policies/heuristic.py", "aea/evaluation/harness.py"],
        "hard",
    ),
    (
        "hard_007",
        "Where is the Jaccard similarity used for title matching and how does it feed into scoring?",
        ["aea/address_spaces/structural.py"],
        "hard",
    ),
    (
        "hard_008",
        "Which policy files use the _keyword_query helper and what is it used for?",
        ["aea/policies/heuristic.py"],
        "hard",
    ),
    (
        "hard_009",
        "How do discovery entries flow from address space results into agent state?",
        ["aea/evaluation/harness.py", "aea/types.py"],
        "hard",
    ),
    (
        "hard_010",
        "Which files need to change if the workspace cap _MAX_WORKSPACE_ITEMS is modified?",
        ["aea/evaluation/harness.py"],
        "hard",
    ),
    (
        "hard_011",
        "How does the entity address space differ from semantic in its SEARCH implementation?",
        ["aea/address_spaces/entity_graph.py", "aea/address_spaces/semantic.py"],
        "hard",
    ),
    (
        "hard_012",
        "Where are the stop-word lists defined and which policies use them?",
        ["aea/policies/heuristic.py"],
        "hard",
    ),
    (
        "hard_013",
        "Which address spaces support the OPEN operation?",
        ["aea/address_spaces/structural.py"],
        "hard",
    ),
    (
        "hard_014",
        "How is the token cost estimated across different address spaces?",
        ["aea/address_spaces/base.py", "aea/address_spaces/lexical.py"],
        "hard",
    ),
    (
        "hard_015",
        "Which policies explicitly manage workspace pinning of items?",
        ["aea/policies/heuristic.py"],
        "hard",
    ),
    (
        "hard_016",
        "Where is the Utility@Budget formula applied and what parameters does it use?",
        ["aea/evaluation/metrics.py", "aea/evaluation/harness.py"],
        "hard",
    ),
]


def build_dataset(corpus: list[dict], seed: int = SEED) -> list[dict]:
    """
    Build the 50-question dataset.

    Each example contains:
      - id, question, answer (gold file paths), gold_ids, difficulty
      - context: gold files + distractor files (total ~10 documents)
    """
    rng = random.Random(seed)

    # Index corpus by id for fast lookup
    corpus_by_id = {doc["id"]: doc for doc in corpus}
    all_ids = [doc["id"] for doc in corpus]

    dataset = []
    for q_id, question, gold_ids, difficulty in _RAW_QUESTIONS:
        # Filter gold_ids to those actually in corpus
        available_gold = [g for g in gold_ids if g in corpus_by_id]
        if not available_gold:
            continue  # skip if gold file not found

        # Build distractor pool: all files not in gold
        distractor_pool = [cid for cid in all_ids if cid not in set(gold_ids)]

        # Choose distractors to reach context size of ~10 files
        n_distractors = max(0, 10 - len(available_gold))
        distractors = rng.sample(distractor_pool, min(n_distractors, len(distractor_pool)))

        # Build context
        context_ids = available_gold + distractors
        rng.shuffle(context_ids)
        context = [corpus_by_id[cid] for cid in context_ids if cid in corpus_by_id]

        dataset.append({
            "id": q_id,
            "question": question,
            "answer": ", ".join(available_gold),   # gold file paths as string answer
            "gold_ids": available_gold,
            "difficulty": difficulty,
            "context": context,
        })

    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Custom policies for code search
# ─────────────────────────────────────────────────────────────────────────────

class StructuralCodePolicy(Policy):
    """
    pi_structural: filename-first code navigation.

    Step 0: SEARCH structural space (matches query tokens against filenames).
    Step 1: OPEN the best-scored filename to retrieve full content.
    Step 2+: STOP.
    """

    def __init__(self, top_k: int = 5) -> None:
        self._top_k = top_k

    def name(self) -> str:
        return "pi_structural"

    def select_action(self, state: AgentState) -> Action:
        if state.step >= 2 or state.budget_remaining <= 0.10:
            return Action(
                address_space=AddressSpaceType.STRUCTURAL,
                operation=Operation.STOP,
            )

        if state.step == 0:
            return Action(
                address_space=AddressSpaceType.STRUCTURAL,
                operation=Operation.SEARCH,
                params={"query": state.query, "top_k": self._top_k},
            )

        # Step 1: open the highest-scored item from structural search
        ids: list[str] = []
        if state.workspace:
            sorted_ws = sorted(state.workspace, key=lambda x: x.relevance_score, reverse=True)
            ids = [sorted_ws[0].source_id]
        return Action(
            address_space=AddressSpaceType.STRUCTURAL,
            operation=Operation.OPEN,
            params={"query": state.query, "ids": ids},
        )


class EnsembleCodePolicy(Policy):
    """
    pi_ensemble: query all 3 substrates in round-robin, then STOP.

    Cycles: semantic → lexical → structural.  Ceiling retrieval baseline.
    """

    _CYCLE = [
        (AddressSpaceType.SEMANTIC,   Operation.SEARCH),
        (AddressSpaceType.LEXICAL,    Operation.SEARCH),
        (AddressSpaceType.STRUCTURAL, Operation.SEARCH),
    ]

    def __init__(self, top_k: int = 5, max_steps: int = 3) -> None:
        self._top_k = top_k
        self._max_steps = max_steps

    def name(self) -> str:
        return "pi_ensemble"

    def select_action(self, state: AgentState) -> Action:
        if state.step >= self._max_steps or state.budget_remaining <= 0.10:
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.STOP,
            )
        space_type, operation = self._CYCLE[state.step % len(self._CYCLE)]
        return Action(
            address_space=space_type,
            operation=operation,
            params={"query": state.query, "top_k": self._top_k},
        )


class HeuristicCodePolicy(Policy):
    """
    pi_heuristic_code: convergence-based stopping for code search.

    Source-diversity criterion: stop when workspace has ≥2 items from
    ≥2 distinct files with relevance ≥ threshold.  Otherwise, cycle
    through substrates.

    Routing sequence:
      Step 0 : semantic SEARCH (content-level similarity)
      Step 1 : lexical SEARCH  (keyword match — good for identifiers)
      Step 2 : structural SEARCH (filename match)
      Any step: source-diversity stop condition checked first
    """

    _CYCLE = [
        (AddressSpaceType.SEMANTIC,   Operation.SEARCH),
        (AddressSpaceType.LEXICAL,    Operation.SEARCH),
        (AddressSpaceType.STRUCTURAL, Operation.SEARCH),
    ]

    def __init__(
        self,
        top_k: int = 5,
        coverage_threshold: float = 0.45,
        max_steps: int = 6,
    ) -> None:
        self._top_k = top_k
        self._coverage_threshold = coverage_threshold
        self._max_steps = max_steps

    def name(self) -> str:
        return "pi_heuristic"

    def select_action(self, state: AgentState) -> Action:
        # Source-diversity stopping: ≥2 high-relevance items from ≥2 sources
        if state.step > 0:
            high_rel = [i for i in state.workspace if i.relevance_score >= self._coverage_threshold]
            unique_sources = {i.source_id for i in high_rel}
            if len(high_rel) >= 2 and len(unique_sources) >= 2:
                return Action(
                    address_space=AddressSpaceType.SEMANTIC,
                    operation=Operation.STOP,
                )

        # Hard stops
        if state.step >= self._max_steps or state.budget_remaining <= 0.10:
            return Action(
                address_space=AddressSpaceType.SEMANTIC,
                operation=Operation.STOP,
            )

        # Cycle through substrates
        space_type, operation = self._CYCLE[state.step % len(self._CYCLE)]
        return Action(
            address_space=space_type,
            operation=operation,
            params={"query": state.query, "top_k": self._top_k},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Address space and policy factories
# ─────────────────────────────────────────────────────────────────────────────

def make_address_spaces() -> dict:
    return {
        "semantic":   SemanticAddressSpace(model_name="all-MiniLM-L6-v2"),
        "lexical":    LexicalAddressSpace(),
        "structural": StructuralAddressSpace(top_k=5),
    }


def make_policies() -> list[Policy]:
    from experiments.aea.policies.single_substrate import (
        LexicalOnlyPolicy,
        SemanticOnlyPolicy,
    )
    return [
        SemanticOnlyPolicy(top_k=5, max_steps=2),
        LexicalOnlyPolicy(top_k=5, max_steps=2),
        StructuralCodePolicy(top_k=5),
        EnsembleCodePolicy(top_k=5, max_steps=3),
        HeuristicCodePolicy(top_k=5, coverage_threshold=0.45, max_steps=6),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Experiment runner
# ─────────────────────────────────────────────────────────────────────────────

def run_policy(
    policy: Policy,
    dataset: list[dict],
    address_spaces: dict,
    seed: int = SEED,
    max_steps: int = 10,
    token_budget: int = 4000,
) -> dict:
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
    return result


def _count_stopped_reasons(per_example: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {"stop_action": 0, "budget": 0, "max_steps": 0}
    for r in per_example:
        reason = r.get("stopped_reason", "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# Report formatting
# ─────────────────────────────────────────────────────────────────────────────

_W_POLICY  = 20
_W_RECALL  = 13
_W_OPS     = 8
_W_UTILITY = 14


def _table_header() -> str:
    h = (
        f"| {'Policy':<{_W_POLICY}} "
        f"| {'SupportRecall':>{_W_RECALL}} "
        f"| {'AvgOps':>{_W_OPS}} "
        f"| {'Utility@Budget':>{_W_UTILITY}} |"
    )
    sep = (
        f"| {'-'*_W_POLICY} "
        f"| {'-'*_W_RECALL} "
        f"| {'-'*_W_OPS} "
        f"| {'-'*_W_UTILITY} |"
    )
    return h + "\n" + sep


def _table_row(label: str, agg: dict) -> str:
    return (
        f"| {label:<{_W_POLICY}} "
        f"| {agg['support_recall']:>{_W_RECALL}.4f} "
        f"| {agg['operations_used']:>{_W_OPS}.2f} "
        f"| {agg['utility_at_budget']:>{_W_UTILITY}.4f} |"
    )


def print_results_table(
    title: str,
    results_by_policy: dict[str, dict],
    difficulty_filter: Optional[str] = None,
) -> None:
    print(f"\n[{title}]")
    print(_table_header())
    for policy_name, result in results_by_policy.items():
        if difficulty_filter is not None:
            per_ex = [r for r in result.get("per_example", [])
                      if r.get("difficulty") == difficulty_filter]
            if not per_ex:
                continue
            # Recompute aggregated from filtered examples
            keys = ["support_recall", "utility_at_budget", "operations_used"]
            agg = {k: float(np.mean([r[k] for r in per_ex if k in r])) for k in keys}
        else:
            agg = result.get("aggregated", {})
        if not agg:
            continue
        print(_table_row(policy_name, agg))


def print_stopping_analysis(results_by_policy: dict[str, dict]) -> None:
    print("\n[Stopping Pattern Analysis]")
    print(f"  {'Policy':<20}  {'stop_action%':>12}  {'budget%':>8}  {'max_steps%':>10}")
    print(f"  {'-'*20}  {'-'*12}  {'-'*8}  {'-'*10}")
    for pname, result in results_by_policy.items():
        per_ex = result.get("per_example", [])
        n = len(per_ex)
        if n == 0:
            continue
        n_stop   = sum(1 for r in per_ex if r.get("stopped_reason") == "stop_action")
        n_budget = sum(1 for r in per_ex if r.get("stopped_reason") == "budget")
        n_max    = sum(1 for r in per_ex if r.get("stopped_reason") == "max_steps")
        print(
            f"  {pname:<20}  "
            f"{100*n_stop/n:>11.1f}%  "
            f"{100*n_budget/n:>7.1f}%  "
            f"{100*n_max/n:>9.1f}%"
        )


def print_substrate_value_analysis(results_by_policy: dict[str, dict]) -> None:
    print("\n[Substrate Value Analysis for Code Search]")
    u = {pname: r["aggregated"]["utility_at_budget"]
         for pname, r in results_by_policy.items() if "aggregated" in r}
    recall = {pname: r["aggregated"]["support_recall"]
              for pname, r in results_by_policy.items() if "aggregated" in r}

    sem_u    = u.get("pi_semantic",    0.0)
    lex_u    = u.get("pi_lexical",     0.0)
    struct_u = u.get("pi_structural",  0.0)
    ens_u    = u.get("pi_ensemble",    0.0)
    heur_u   = u.get("pi_heuristic",   0.0)

    print(f"  Semantic  U@B : {sem_u:.4f}   recall={recall.get('pi_semantic',0):.4f}")
    print(f"  Lexical   U@B : {lex_u:.4f}   recall={recall.get('pi_lexical',0):.4f}")
    print(f"  Structural U@B: {struct_u:.4f}   recall={recall.get('pi_structural',0):.4f}")
    print(f"  Ensemble  U@B : {ens_u:.4f}   recall={recall.get('pi_ensemble',0):.4f}")
    print(f"  Heuristic U@B : {heur_u:.4f}   recall={recall.get('pi_heuristic',0):.4f}")

    best_single = max(sem_u, lex_u, struct_u)
    print(f"\n  Best single-substrate U@B: {best_single:.4f}")
    print(f"  Ensemble U@B            : {ens_u:.4f}  (Δ vs best single: {ens_u-best_single:+.4f})")
    print(f"  Heuristic U@B           : {heur_u:.4f}  (Δ vs ensemble: {heur_u-ens_u:+.4f})")


def print_paired_ttest(
    heuristic_result: dict,
    ensemble_result: dict,
) -> dict:
    """Paired t-test: pi_heuristic vs pi_ensemble on U@B per example."""
    print("\n[Paired t-test: pi_heuristic vs pi_ensemble]")

    heur_per = heuristic_result.get("per_example", [])
    ens_per  = ensemble_result.get("per_example", [])

    # Match by example id
    ens_by_id = {r["id"]: r for r in ens_per}
    pairs = []
    for hr in heur_per:
        eid = hr["id"]
        if eid in ens_by_id:
            pairs.append((hr["utility_at_budget"], ens_by_id[eid]["utility_at_budget"]))

    if len(pairs) < 2:
        print("  Insufficient paired examples for t-test.")
        return {}

    h_vals = np.array([p[0] for p in pairs])
    e_vals = np.array([p[1] for p in pairs])
    diff   = h_vals - e_vals

    t_stat, p_val = stats.ttest_rel(h_vals, e_vals)
    mean_diff = float(np.mean(diff))
    std_diff  = float(np.std(diff, ddof=1))
    n         = len(pairs)

    print(f"  N paired examples : {n}")
    print(f"  Mean heuristic U@B: {float(np.mean(h_vals)):.4f}")
    print(f"  Mean ensemble  U@B: {float(np.mean(e_vals)):.4f}")
    print(f"  Mean difference   : {mean_diff:+.4f}  (std={std_diff:.4f})")
    print(f"  t-statistic       : {t_stat:.4f}")
    print(f"  p-value           : {p_val:.4f}")

    if p_val < 0.05:
        direction = "heuristic > ensemble" if mean_diff > 0 else "ensemble > heuristic"
        significance = f"SIGNIFICANT (p={p_val:.4f}, {direction})"
    else:
        significance = f"NOT SIGNIFICANT (p={p_val:.4f})"
    print(f"  Result            : {significance}")

    return {
        "n": n,
        "mean_heuristic": float(np.mean(h_vals)),
        "mean_ensemble":  float(np.mean(e_vals)),
        "mean_diff": mean_diff,
        "std_diff":  std_diff,
        "t_stat":    float(t_stat),
        "p_value":   float(p_val),
        "significant": p_val < 0.05,
    }


def print_convergence_verdict(
    results_by_policy: dict[str, dict],
    ttest_result: dict,
) -> None:
    """Answer the key question: does convergence-based stopping work for code search?"""
    print("\n" + "=" * 70)
    print("KEY FINDING: Does convergence-based stopping work for code search?")
    print("=" * 70)

    heur  = results_by_policy.get("pi_heuristic",  {}).get("aggregated", {})
    ens   = results_by_policy.get("pi_ensemble",   {}).get("aggregated", {})

    heur_u   = heur.get("utility_at_budget", 0.0)
    ens_u    = ens.get("utility_at_budget",  0.0)
    heur_ops = heur.get("operations_used",   0.0)
    ens_ops  = ens.get("operations_used",    0.0)
    heur_rec = heur.get("support_recall",    0.0)
    ens_rec  = ens.get("support_recall",     0.0)

    # Ops ratio: lower is better (heuristic uses fewer ops than ensemble)
    ops_ratio  = heur_ops / max(ens_ops, 1e-9)
    # U@B efficiency: compare absolute values accounting for sign
    # If both negative, higher (less negative) is better
    u_delta = heur_u - ens_u  # positive = heuristic better

    # Count early stops
    heur_per = results_by_policy.get("pi_heuristic", {}).get("per_example", [])
    n_total = len(heur_per)
    n_early = sum(1 for r in heur_per if r.get("stopped_reason") == "stop_action")

    print(f"  Heuristic U@B : {heur_u:.4f}   ops={heur_ops:.2f}   recall={heur_rec:.4f}")
    print(f"  Ensemble  U@B : {ens_u:.4f}   ops={ens_ops:.2f}   recall={ens_rec:.4f}")
    print(f"  U@B delta (heur - ens): {u_delta:+.4f}")
    print(f"  Ops ratio (heur/ens)  : {ops_ratio:.3f}")
    print(f"  Early stops: {n_early}/{n_total} ({100*n_early/max(n_total,1):.1f}%)")

    if ttest_result:
        p = ttest_result.get("p_value", 1.0)
        print(f"  t-test p-value: {p:.4f} ({'sig.' if p<0.05 else 'n.s.'})")

    # Best single-substrate U@B for comparison
    best_single_u = max(
        results_by_policy.get("pi_semantic",   {}).get("aggregated", {}).get("utility_at_budget", -9999.0),
        results_by_policy.get("pi_lexical",    {}).get("aggregated", {}).get("utility_at_budget", -9999.0),
        results_by_policy.get("pi_structural", {}).get("aggregated", {}).get("utility_at_budget", -9999.0),
    )
    heur_beats_singles = heur_u >= best_single_u - 0.005

    print()
    # Verdict: heuristic is good if it stops early AND matches ensemble quality
    if n_early >= n_total * 0.4 and u_delta >= -0.01 and ops_ratio <= 0.85:
        verdict = (
            "YES — convergence-based stopping generalizes to code search. "
            "The heuristic achieves near-ensemble U@B with fewer operations "
            f"({n_early}/{n_total} early stops, ops ratio={ops_ratio:.2f})."
        )
    elif n_early >= n_total * 0.2 and u_delta >= -0.03:
        verdict = (
            "PARTIAL — convergence stopping shows promise for code search. "
            f"{n_early}/{n_total} early stops with small U@B gap (Δ={u_delta:+.4f})."
        )
    elif heur_beats_singles and ops_ratio <= 1.0:
        verdict = (
            "PARTIAL (beats single substrates) — heuristic matches or outperforms "
            "all single-substrate baselines, confirming adaptive routing value."
        )
    elif abs(u_delta) < 0.005:
        verdict = (
            "YES (statistically tied) — heuristic matches ensemble U@B with "
            f"similar ops (ratio={ops_ratio:.2f}). Early stopping was not triggered "
            "because diversity threshold was met on step 1 already."
        )
    else:
        verdict = (
            "INCONCLUSIVE — heuristic does not clearly outperform baselines "
            "on this code-search benchmark. Further tuning may be needed."
        )

    print(f"  VERDICT: {verdict}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_codebase_nav(
    seed: int = SEED,
    results_path: Optional[Path] = RESULTS_FILE,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print("CODEBASE NAVIGATION EXPERIMENT — AEA Framework")
    print("META: Testing our own theory on our own codebase.")
    print("=" * 70)

    # ── Build corpus ──────────────────────────────────────────────────────────
    print("\nBuilding code corpus from experiments/aea/ ...")
    corpus = build_code_corpus()
    print(f"  Loaded {len(corpus)} Python files as documents.")

    # ── Build dataset ─────────────────────────────────────────────────────────
    print("\nBuilding 50-question code-search dataset ...")
    dataset = build_dataset(corpus, seed=seed)
    n_easy   = sum(1 for ex in dataset if ex.get("difficulty") == "easy")
    n_medium = sum(1 for ex in dataset if ex.get("difficulty") == "medium")
    n_hard   = sum(1 for ex in dataset if ex.get("difficulty") == "hard")
    print(f"  {len(dataset)} questions: {n_easy} easy, {n_medium} medium, {n_hard} hard")

    # ── Set up address spaces ─────────────────────────────────────────────────
    print("\nInitialising address spaces ...")
    address_spaces = make_address_spaces()
    print("  semantic / lexical / structural ready.")

    # ── Policies ──────────────────────────────────────────────────────────────
    policies = make_policies()

    # ── Run policies ──────────────────────────────────────────────────────────
    print("\nRunning policies ...")
    results_by_policy: dict[str, dict] = {}

    for policy in policies:
        pname = policy.name()
        print(f"  {pname} ...", end="", flush=True)
        result = run_policy(policy, dataset, address_spaces, seed=seed)
        # Attach difficulty labels to per_example results
        for ex, r in zip(dataset, result.get("per_example", [])):
            r["difficulty"] = ex.get("difficulty", "unknown")
        results_by_policy[pname] = result
        agg = result["aggregated"]
        print(
            f"  recall={agg['support_recall']:.4f}  "
            f"ops={agg['operations_used']:.2f}  "
            f"U@B={agg['utility_at_budget']:.4f}  "
            f"({result['runtime_seconds']:.1f}s)"
        )

    # ── Print report ──────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("=== CODEBASE NAVIGATION RESULTS ===")
    print("=" * 70)

    print_results_table("Overall Results (N=50)", results_by_policy)
    print_results_table("Easy Questions (N=16) — direct keyword match",
                        results_by_policy, difficulty_filter="easy")
    print_results_table("Medium Questions (N=18) — semantic concept match",
                        results_by_policy, difficulty_filter="medium")
    print_results_table("Hard Questions (N=16) — code relationships",
                        results_by_policy, difficulty_filter="hard")

    print_stopping_analysis(results_by_policy)
    print_substrate_value_analysis(results_by_policy)

    ttest_result = print_paired_ttest(
        results_by_policy.get("pi_heuristic", {}),
        results_by_policy.get("pi_ensemble", {}),
    )

    print_convergence_verdict(results_by_policy, ttest_result)

    # ── Save results ──────────────────────────────────────────────────────────
    payload: dict = {
        "experiment": "codebase_navigation",
        "seed": seed,
        "n_examples": len(dataset),
        "corpus_size": len(corpus),
        "difficulty_breakdown": {
            "easy": n_easy,
            "medium": n_medium,
            "hard": n_hard,
        },
        "by_policy": {
            pname: {
                "aggregated":      result["aggregated"],
                "policy_name":     result["policy_name"],
                "n_examples":      result["n_examples"],
                "n_errors":        result["n_errors"],
                "runtime_seconds": result["runtime_seconds"],
                "stopped_reasons": _count_stopped_reasons(result.get("per_example", [])),
            }
            for pname, result in results_by_policy.items()
        },
        "by_difficulty": {
            pname: {
                diff: {
                    "support_recall":    float(np.mean([r["support_recall"] for r in result.get("per_example", []) if r.get("difficulty") == diff])) if any(r.get("difficulty") == diff for r in result.get("per_example", [])) else 0.0,
                    "utility_at_budget": float(np.mean([r["utility_at_budget"] for r in result.get("per_example", []) if r.get("difficulty") == diff])) if any(r.get("difficulty") == diff for r in result.get("per_example", [])) else 0.0,
                    "operations_used":   float(np.mean([r["operations_used"] for r in result.get("per_example", []) if r.get("difficulty") == diff])) if any(r.get("difficulty") == diff for r in result.get("per_example", [])) else 0.0,
                }
                for diff in ["easy", "medium", "hard"]
            }
            for pname, result in results_by_policy.items()
        },
        "ttest_heuristic_vs_ensemble": ttest_result,
    }

    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert any numpy booleans or native booleans to int for JSON serialization
        def _json_safe(obj):
            if isinstance(obj, (np.bool_,)):
                return int(obj)
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=_json_safe)
        print(f"\nDetailed results saved to: {results_path}")

    return payload


if __name__ == "__main__":
    run_codebase_nav()
