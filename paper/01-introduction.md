# 1 Introduction

When should a retrieval system stop searching? The dominant paradigm in retrieval-augmented generation commits to a fixed number of retrieval operations — typically one or two calls to a single index — regardless of whether those operations are necessary or sufficient. This rigidity creates two failure modes: wasted computation on queries answerable from a single retrieval step, and insufficient evidence on queries requiring multiple heterogeneous sources.

Recent adaptive retrieval systems address the second failure mode. Self-RAG [Asai et al., 2024] learns when to skip retrieval. FLARE [Jiang et al., 2023] triggers retrieval on low-confidence tokens. Adaptive-RAG [Jeong et al., 2024] routes queries by estimated complexity. SmartRAG [Gao et al., 2025] jointly optimizes retrieval decisions under cost constraints. Yet these systems operate within a single retrieval modality. The question of when to *stop* searching across multiple qualitatively different retrieval substrates — semantic indexes, keyword indexes, entity graphs — has received less attention.

In this paper, we study **coverage-driven retrieval routing**: a policy that operates over multiple heterogeneous retrieval substrates, evaluating after each operation whether the current evidence is sufficient to stop or whether escalation to a different substrate is warranted. The policy defaults to the cheapest available operation (dense retrieval) and escalates only when a coverage gap is detected.

Our central empirical finding is counterintuitive. We expected the primary value of multi-substrate routing to come from *positive selection* — choosing the right substrate for each query. Instead, ablation analysis reveals that the dominant mechanism is **routing avoidance** — knowing when to stop and what *not* to do. On HotpotQA Bridge (N=500, bootstrap confidence intervals reported), the policy completes questions in 1.21 average operations versus 2.00 for fixed baselines, with comparable support recall. Forcing unconditional escalation to all substrates is catastrophic, while removing entity-graph hops entirely slightly *improves* performance on lexically-rich data. The value of adaptive routing, at least for heuristic policies, lies primarily in cost-efficient restraint.

Our contributions are:

1. **A coverage-driven retrieval routing policy** that selects among semantic, lexical, and entity-graph substrates based on workspace state — specifically, whether retrieved evidence from multiple sources meets a coverage threshold (Section 3).

2. **A controlled heterogeneous benchmark** with entity isolation and lexical overlap controls, designed to require cross-substrate navigation (Section 4).

3. **The routing avoidance finding**: under budget-aware evaluation, adaptive retrieval's primary value is selective stopping, not positive substrate selection. This reframes the design problem from "choose the right tool" to "know when to stop" (Section 5).

4. **A formal framework** modeling multi-substrate retrieval as a constrained decision process with discovery/knowledge state tracking, connecting to the options framework for hierarchical action selection (Appendix A).

The remainder of the paper is organized as follows. Section 2 surveys related work. Section 3 presents the routing policy and its design rationale. Section 4 describes the experimental setup. Section 5 reports results and ablation analysis. Section 6 discusses implications, limitations, and future directions. Section 7 concludes.
