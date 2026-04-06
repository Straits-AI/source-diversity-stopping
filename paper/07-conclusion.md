# 7 Conclusion

We studied adaptive retrieval routing over heterogeneous address spaces with a focus on the tradeoff between retrieval comprehensiveness and cost efficiency. Three findings emerge.

First, **a simple coverage-driven stopping rule outperforms both comprehensive retrieval and LLM-guided routing** on end-to-end Utility@Budget. The heuristic AEA policy (U@B 0.760) beats the ensemble (0.731) by stopping at 1.21 average operations while maintaining competitive answer quality (F1 0.630 vs 0.701). This result is statistically validated on retrieval metrics (N=500, p < 0.0001, Cohen's d = 0.807).

Second, **LLM-based routing achieves genuine positive substrate selection** — the LLM router uses all four action types with per-question variation and achieves higher recall (0.845) than the heuristic (0.795). But this intelligence is not yet cost-efficient: 2.54 operations for marginal quality gains produce lower overall utility (0.652).

Third, the resulting hierarchy — **smart stopping > brute force > smart searching** — challenges the default assumption in adaptive retrieval. The primary value under budget constraints is knowing when to stop, not knowing what to do. This reframes the design problem: rather than optimizing substrate selection, practitioners should optimize the stopping threshold.

The gap between heuristic stopping efficiency and LLM routing intelligence defines the key open challenge: **calibrated stopping** — a policy that stops as efficiently as a simple coverage check on easy questions while routing as intelligently as an LLM on hard ones. The trajectory data and Utility@Budget framework introduced here provide the foundation for learning such a policy.
