# 7 Conclusion

We studied adaptive retrieval routing over heterogeneous address spaces with a focus on the tradeoff between retrieval comprehensiveness and cost efficiency. Three findings emerge.

First, **learned stopping outperforms both heuristic stopping and comprehensive retrieval** on end-to-end Utility@Budget. The learned classifier (U@B 0.766 [0.715, 0.819]) beats the hand-tuned heuristic (0.751) and the ensemble (0.692) on HotpotQA (N=500), with the hierarchy replicating on 2WikiMultiHopQA. The classifier learns from trajectory data without manual threshold selection, and its feature importance analysis reveals that **diminishing marginal relevance improvement** is the dominant stopping signal (importance 0.55).

Second, **the choice between stopping and searching depends on cost sensitivity.** Sensitivity analysis identifies a crossover at μ = 0.20: for any cost penalty above this threshold, stopping-based policies dominate; below it, comprehensive retrieval wins. This provides practitioners with concrete, actionable guidance rather than a universal recommendation.

Third, the consistent ranking — **learned stopping > heuristic stopping > brute-force retrieval** — across two benchmarks, five ablation variants, and all cost regimes μ ≥ 0.20 establishes that calibrated stopping is a tractable, high-impact design target. The gap between what a simple learned classifier achieves and what an optimal policy could achieve defines the key open challenge: training stopping policies that generalize across question distributions and retrieval substrates.
