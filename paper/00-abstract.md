# Adaptive Retrieval Routing: When Knowing What Not To Do Beats Choosing the Right Tool

## Abstract

Retrieval-augmented systems typically commit to a fixed number of retrieval operations regardless of whether they are necessary or sufficient. We study coverage-driven retrieval routing: a policy over heterogeneous substrates (semantic, lexical, entity-graph) that evaluates evidence sufficiency after each step and stops when coverage is adequate. We propose both a heuristic stopping rule and a learned stopping classifier trained on trajectory data, where the classifier discovers that diminishing marginal relevance improvement is the key stopping signal (feature importance 0.55).

On HotpotQA Bridge (N=500), the learned stopping policy achieves the highest end-to-end Utility@Budget (0.766 [95% CI: 0.715, 0.819]) with LLM answer generation, outperforming comprehensive retrieval (ensemble, 0.692 [0.638, 0.743]) at one-third the operation cost (1.23 ops vs 3.00). The hierarchy replicates on 2WikiMultiHopQA. Sensitivity analysis shows stopping-based policies dominate for any cost penalty μ ≥ 0.20. Ablation analysis confirms the mechanism: forcing unconditional retrieval escalation degrades utility by 0.115, while the learned classifier outperforms the hand-tuned heuristic without manual threshold selection.

The resulting hierarchy — learned stopping > heuristic stopping > brute-force retrieval — challenges the assumption that adaptive retrieval should optimize substrate selection, and identifies calibrated stopping as a tractable, high-impact design target for cost-efficient retrieval systems.
