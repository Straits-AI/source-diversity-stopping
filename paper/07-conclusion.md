# 7 Conclusion

We studied adaptive retrieval routing over heterogeneous address spaces with a focus on the tradeoff between retrieval comprehensiveness and cost efficiency. Three findings emerge.

First, **a simple coverage-driven stopping rule achieves comparable end-to-end utility to comprehensive retrieval while using 60% fewer operations.** The heuristic AEA policy (U@B 0.760) matches the ensemble (0.731) — a gap that is not statistically significant at N=100 — while completing questions in 1.21 operations versus 3.00. Sensitivity analysis shows AEA dominates when cost matters (μ ≥ 0.25) and ensemble dominates when it doesn't (μ < 0.25). The retrieval advantage is statistically validated (N=500, p < 0.0001, Cohen's d = 0.807).

Second, **LLM-based routing achieves genuine positive substrate selection** — the LLM router uses all four action types with per-question variation and achieves higher recall (0.845) than the heuristic (0.795). But this intelligence is not yet cost-efficient: 2.54 operations for marginal quality gains produce lower overall utility (0.652).

Third, the resulting hierarchy — **smart stopping > brute force > smart searching** — challenges the default assumption in adaptive retrieval. The primary value under budget constraints is knowing when to stop, not knowing what to do. This reframes the design problem: rather than optimizing substrate selection, practitioners should optimize the stopping threshold.

The gap between heuristic stopping efficiency and LLM routing intelligence defines the key open challenge: **calibrated stopping** — a policy that stops as efficiently as a simple coverage check on easy questions while routing as intelligently as an LLM on hard ones. The trajectory data and Utility@Budget framework introduced here provide the foundation for learning such a policy.
