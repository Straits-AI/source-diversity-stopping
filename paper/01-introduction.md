# 1 Introduction

When should a retrieval system stop searching? The dominant paradigm in retrieval-augmented generation commits to a fixed number of retrieval operations — typically one or two calls to a single index — regardless of whether those operations are necessary or sufficient. This rigidity creates two failure modes: wasted computation on queries answerable from a single retrieval step, and insufficient evidence on queries requiring multiple heterogeneous sources.

Recent adaptive retrieval systems address the second failure mode. Self-RAG [Asai et al., 2024] learns when to skip retrieval. FLARE [Jiang et al., 2023] triggers retrieval on low-confidence tokens. Adaptive-RAG [Jeong et al., 2024] routes queries by estimated complexity. SmartRAG [Gao et al., 2025] jointly optimizes retrieval decisions under cost constraints. Yet these systems operate within a single retrieval modality. The question of when to *stop* searching across multiple qualitatively different retrieval substrates — semantic indexes, keyword indexes, entity graphs — has received less attention.

In this paper, we study **coverage-driven retrieval routing**: a policy that operates over multiple heterogeneous retrieval substrates, evaluating after each operation whether the current evidence is sufficient to stop or whether escalation to a different substrate is warranted. The policy defaults to the cheapest available operation (dense retrieval) and escalates only when a coverage gap is detected.

Our central empirical finding is counterintuitive. We expected the primary value of multi-substrate routing to come from *positive selection* — choosing the right substrate for each query. Instead, the dominant mechanism is **selective stopping** — knowing when evidence is sufficient. On HotpotQA Bridge, the policy achieves the highest end-to-end Utility@Budget (0.760), outperforming both comprehensive retrieval (ensemble, 0.731) and LLM-guided positive routing (0.652). It does this by completing questions in 1.21 average operations versus 2.00–3.00 for baselines (N=500, p < 0.0001, Cohen's d = 0.807). An LLM-based router achieves higher recall (0.845) through genuine multi-substrate reasoning but is not cost-efficient — establishing a hierarchy: **smart stopping > brute force > smart searching**.

Our contributions are:

1. **A coverage-driven retrieval routing policy** that selects among semantic, lexical, and entity-graph substrates based on workspace state — specifically, whether retrieved evidence from multiple sources meets a coverage threshold (Section 3).

2. **A controlled heterogeneous benchmark** with entity isolation and lexical overlap controls, designed to require cross-substrate navigation (Section 4).

3. **The stopping > searching hierarchy**: under budget-aware end-to-end evaluation, a simple stopping rule (U@B 0.760) beats comprehensive retrieval (0.731) and LLM-guided routing (0.652). This establishes that knowing when to stop is more valuable than knowing what to do (Section 5).

4. **A formal framework** modeling multi-substrate retrieval as a constrained decision process with discovery/knowledge state tracking, connecting to the options framework for hierarchical action selection (Appendix A).

The remainder of the paper is organized as follows. Section 2 surveys related work. Section 3 presents the routing policy and its design rationale. Section 4 describes the experimental setup. Section 5 reports results and ablation analysis. Section 6 discusses implications, limitations, and future directions. Section 7 concludes.
