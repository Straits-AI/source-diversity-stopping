# 1 Introduction

When should a retrieval system stop searching? The dominant paradigm in retrieval-augmented generation commits to a fixed number of retrieval operations — typically one or two calls to a single index — regardless of whether those operations are necessary or sufficient. This rigidity creates two failure modes: wasted computation on queries answerable from a single retrieval step, and insufficient evidence on queries requiring multiple heterogeneous sources.

Recent adaptive retrieval systems address the second failure mode. Self-RAG [Asai et al., 2024] learns when to skip retrieval. FLARE [Jiang et al., 2023] triggers retrieval on low-confidence tokens. Adaptive-RAG [Jeong et al., 2024] routes queries by estimated complexity. SmartRAG [Gao et al., 2025] jointly optimizes retrieval decisions under cost constraints. Yet these systems operate within a single retrieval modality. The question of when to *stop* searching across multiple qualitatively different retrieval substrates — semantic indexes, keyword indexes, entity graphs — has received less attention.

In this paper, we study **coverage-driven retrieval routing**: a policy that operates over multiple heterogeneous retrieval substrates, evaluating after each operation whether the current evidence is sufficient to stop or whether escalation to a different substrate is warranted. We propose both a heuristic stopping rule and a **learned stopping classifier** trained on retrieval trajectory data that discovers the optimal stopping threshold without manual tuning.

Our central empirical finding is counterintuitive. We expected the primary value of multi-substrate routing to come from *positive selection* — choosing the right substrate for each query. Instead, the dominant mechanism is **selective stopping** — knowing when evidence is sufficient. On HotpotQA Bridge (N=500), the learned stopping policy achieves the highest end-to-end Utility@Budget (0.766 [95% CI: 0.715, 0.819]) with LLM answer generation, outperforming comprehensive retrieval (ensemble, 0.692 [0.638, 0.743]) at one-third the operation cost. The classifier identifies **diminishing marginal relevance improvement** as the key stopping signal (feature importance 0.55), providing an interpretable, actionable insight. The hierarchy replicates on 2WikiMultiHopQA, and sensitivity analysis shows stopping-based policies dominate for any cost penalty μ ≥ 0.20.

Our contributions are:

1. **A coverage-driven retrieval routing framework** with both a heuristic stopping rule and a learned stopping classifier trained on trajectory data. The classifier achieves the best retrieval Utility@Budget (+17% over the heuristic) without hand-tuned thresholds (Section 3).

2. **The stopping > searching hierarchy**: under end-to-end evaluation with LLM answer generation (N=500, bootstrap CIs), learned stopping (U@B 0.766) outperforms comprehensive retrieval (0.692) and heuristic stopping (0.751). The hierarchy holds across two benchmarks with a cost-sensitivity crossover at μ = 0.20 (Section 5).

3. **An interpretable stopping signal**: the learned classifier discovers that diminishing marginal improvement in evidence relevance is the dominant feature for stopping decisions (importance 0.55), providing practitioners with a concrete, monitorable signal (Section 5).

4. **A formal framework** modeling multi-substrate retrieval as a constrained decision process with discovery/knowledge state tracking, connecting to the options framework for hierarchical action selection (Appendix A).

The remainder of the paper is organized as follows. Section 2 surveys related work. Section 3 presents the routing policy, learned classifier, and design rationale. Section 4 describes the experimental setup. Section 5 reports results, ablation analysis, and sensitivity analysis. Section 6 discusses implications, limitations, and future directions. Section 7 concludes.
