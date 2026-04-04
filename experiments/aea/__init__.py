"""
AEA — Adaptive External Attention framework.

Core framework for the paper:
  "Agentic Attention: Harness-Level Adaptive External Attention for LLM Systems"

Version
-------
0.1.0  — Core framework: types, address spaces, evaluation harness, policies.

Quick-start
-----------
>>> from experiments.aea import types, evaluation, policies
>>> from experiments.aea.address_spaces import SemanticAddressSpace, LexicalAddressSpace
>>> from experiments.aea.evaluation import EvaluationHarness
>>> from experiments.aea.policies import AEAHeuristicPolicy

Package layout
--------------
types.py                  — shared data types (AgentState, Action, …)
address_spaces/           — semantic, lexical, entity-graph address spaces
  base.py                 — AddressSpace ABC
  semantic.py             — SemanticAddressSpace (sentence-transformers)
  lexical.py              — LexicalAddressSpace (rank_bm25)
  entity_graph.py         — EntityGraphAddressSpace (regex NER + co-occurrence)
evaluation/               — metrics and harness
  metrics.py              — exact_match, f1, support_recall, utility_at_budget, …
  harness.py              — EvaluationHarness (IMMUTABLE)
policies/                 — routing policies
  base.py                 — Policy ABC
  single_substrate.py     — SemanticOnlyPolicy, LexicalOnlyPolicy, EntityOnlyPolicy
  heuristic.py            — AEAHeuristicPolicy (adaptive hand-designed routing)
  ensemble.py             — EnsemblePolicy (query all substrates)
"""

__version__ = "0.1.0"

from . import address_spaces, evaluation, policies, types

__all__ = ["address_spaces", "evaluation", "policies", "types", "__version__"]
