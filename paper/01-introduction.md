# 1 Introduction

When should a retrieval system stop searching? The dominant paradigm in retrieval-augmented generation commits to a fixed retrieval budget — typically one or two calls to a single index — regardless of whether the retrieved evidence is sufficient or the additional operations are wasteful. Recent adaptive retrieval systems address the sufficiency side: Self-RAG [Asai et al., 2024] learns when to skip retrieval, FLARE [Jiang et al., 2023] triggers retrieval on low-confidence tokens, and Adaptive-RAG [Jeong et al., 2024] routes queries by complexity. Yet these systems operate within a single retrieval modality. The question of when to stop searching across multiple qualitatively different substrates — semantic indexes, keyword indexes, entity graphs — remains underexplored.

In this paper, we study **coverage-driven retrieval stopping** across heterogeneous substrates, using semantic and lexical retrieval as the two primary substrates and entity-graph traversal as a third substrate to test whether structured retrieval adds value. We begin with a simple structural heuristic: stop when the workspace contains evidence from two or more independent retrieval sources. On HotpotQA Bridge (N=500, end-to-end with LLM answer generation), this heuristic significantly outperforms comprehensive retrieval (paired t-test, p=0.021) at one-third the operation cost.

We then ask: **can we do better?** We implement three principled improvements — a pre-trained cross-encoder for content-aware stopping, a learned classifier on trajectory features, and LLM-based question decomposition for requirement-driven stopping — and find that **all three fail** to improve on the heuristic. Each fails for a different proximate reason, but root cause analysis reveals a common underlying explanation: the heuristic operates on a **structural signal** (source diversity) that is distribution-invariant, compositionally informative, and computationally free, while all three sophisticated alternatives introduce content-specific or distribution-specific noise that degrades stopping quality.

This failure pattern is itself the paper's central contribution. It reveals that:

1. **Evidence sufficiency in multi-hop QA is a set function** — it depends on the joint content of passage bundles, not on individual passage scores. Cross-encoders trained on single-passage ranking cannot assess it (Section 6.4.3).

2. **Workspace statistics are distribution-specific** — learned classifiers on trajectory features capture spurious correlations that break under distribution shift. The heuristic's structural signal avoids this because source diversity is distribution-invariant (Section 6.4.4).

3. **LLM decomposition introduces more noise than signal** — precision and matchability of sub-requirements are in tension, causing the policy to default to exhaustive retrieval (Section 6.4.5).

4. **Structural stopping signals connect to optimal stopping theory** — threshold rules on low-noise observables dominate value-estimation approaches when the value function is hard to learn (Section 6.4.6).

Our contributions are:

1. **A statistically validated stopping result**: a simple coverage heuristic over two primary substrates (semantic and lexical) significantly outperforms comprehensive retrieval on end-to-end evaluation (p=0.021, N=500), with the hierarchy replicating on 2WikiMultiHopQA (Section 5). We also show that entity-graph traversal, tested as a third substrate, does not contribute on either benchmark (ablation, Section 5.3).

2. **Three controlled failure analyses** showing why content-aware, learned, and decomposition-based stopping improvements fail — each illuminating a different aspect of the stopping problem (Section 5, Section 6.4).

3. **The structural signal thesis**: a unifying explanation grounded in optimal stopping theory for why distribution-invariant structural signals outperform content-specific signals in multi-substrate retrieval stopping (Section 6.4).

4. **Actionable design guidance**: practitioners should default to structural stopping signals, treat stopping as a first-class design target, and augment rather than replace simple heuristics (Section 6.4.7).
