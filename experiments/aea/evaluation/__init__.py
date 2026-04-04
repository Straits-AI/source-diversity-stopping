"""
AEA evaluation package.

Exports
-------
EvaluationHarness  — runs a policy on a dataset, collects metrics
metrics            — exact_match, f1_score, support_recall, …
"""

from .harness import EvaluationHarness
from . import metrics

__all__ = ["EvaluationHarness", "metrics"]
