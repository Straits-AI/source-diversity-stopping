# 7 Conclusion

We studied adaptive retrieval routing over heterogeneous address spaces — semantic, lexical, and entity-graph substrates — with a focus on cost-efficient operation selection. Our coverage-driven routing policy achieves statistically significant improvements in retrieval efficiency (1.15 operations vs 2.00–3.00 for baselines, p < 0.0001, Cohen's d = 0.807) while maintaining the highest support recall on HotpotQA Bridge (N=500).

Three findings emerge from the experiments:

First, **routing avoidance dominates positive routing** in the current heuristic policy. The ability to recognize sufficient evidence and stop — rather than the ability to select the optimal next substrate — accounts for the dominant share of the cost-efficiency advantage. Forcing unconditional escalation is catastrophic; removing entity hops slightly improves performance on lexically-rich data.

Second, **the evaluation regime determines the optimal stopping threshold.** Under retrieval-only evaluation, aggressive stopping is optimal and AEA leads all policies. Under end-to-end evaluation with LLM answer generation, the ensemble policy's higher recall produces better answers despite higher cost, and AEA is second. This divergence identifies stopping threshold calibration — not substrate selection — as the key open challenge.

Third, **multi-substrate access with any routing policy outperforms single-substrate retrieval.** AEA beats all individual substrates on both evaluation regimes, confirming that heterogeneous address space support is valuable even when the routing policy is imperfect.

The gap between aggressive stopping (where heuristic policies excel) and calibrated stopping (where learned policies are needed) defines the natural next step. The trajectory data collected during this work — including per-step coverage signals and downstream answer quality — provides the training signal for closing this gap.
