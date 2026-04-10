# 1 Introduction

When should a retrieval system stop searching? The dominant paradigm in retrieval-augmented generation commits to a fixed retrieval budget — typically one or two calls to a single index — regardless of whether the retrieved evidence is sufficient or the additional operations are wasteful. Recent adaptive retrieval systems address the sufficiency side: Self-RAG [Asai et al., 2024] learns when to skip retrieval, FLARE [Jiang et al., 2023] triggers retrieval on low-confidence tokens, and Adaptive-RAG [Jeong et al., 2024] routes queries by complexity. Yet these systems operate within a single retrieval modality. The question of when to stop searching across multiple qualitatively different substrates — semantic indexes, keyword indexes, entity graphs — remains underexplored.

In this paper, we study **coverage-driven retrieval stopping** across heterogeneous substrates, using semantic and lexical retrieval as the two primary substrates. We begin with a simple structural heuristic: stop when the workspace contains evidence from two or more independent retrieval sources. Across three benchmark families — multi-hop QA (HotpotQA, N=1000), reasoning-intensive retrieval (BRIGHT, N=200), and diluted open-domain settings (50 paragraphs per question) — this heuristic significantly outperforms comprehensive retrieval (p < 0.003 in all cases, Cohen's d up to 0.49).

We then ask: **can we do better?** We implement five principled improvements — a pre-trained cross-encoder for content-aware stopping, an NLI-based bundle sufficiency checker, a learned classifier on trajectory features, LLM-based question decomposition, and an embedding-based router — and find that **all five fail** to improve on the heuristic. Each fails for a different proximate reason, but root cause analysis reveals a common underlying explanation: the heuristic operates on a **structural signal** (source diversity) that is distribution-invariant, compositionally informative, and computationally free, while all five alternatives introduce content-specific or distribution-specific noise that degrades stopping quality.

This failure pattern is itself the paper's central contribution. It reveals that:

1. **Evidence sufficiency in multi-hop QA is a set function** — it depends on the joint content of passage bundles, not on individual passage scores. Cross-encoders trained on single-passage ranking cannot assess it (Section 6.4.3).

2. **Even bundle-level NLI assessment fails** — multi-hop questions resist conversion to well-formed entailment hypotheses, so NLI models that correctly take the full evidence bundle as premise still cannot determine sufficiency (Section 6.4).

3. **Workspace statistics are distribution-specific** — learned classifiers on trajectory features capture spurious correlations that break under distribution shift (Section 6.4.4).

4. **Structural stopping signals connect to optimal stopping theory** — threshold rules on low-noise observables dominate value-estimation approaches when the value function is hard to learn (Section 6.4.6).

Our contributions are:

1. **A statistically validated stopping result across three benchmark families**: a simple coverage heuristic significantly outperforms comprehensive retrieval on HotpotQA (p<0.000001, d=0.379, N=1000), BRIGHT (p=0.0026, d=0.216, N=200), and open-domain settings (p<0.000001, d=0.491, N=200). The advantage is invariant to question type (bridge and comparison) and candidate set size (10 vs 50 paragraphs) (Section 5). Note that the end-to-end result (p=0.021, d=0.103) is statistically significant but with a small effect size.

2. **Five controlled failure analyses** showing why content-aware (cross-encoder, NLI), learned (gradient boosted tree (GBT)), decomposition-based (LLM), and routing-based (embedding) improvements all fail — each illuminating a different aspect of the stopping problem (Section 5.4, Section 6.4).

3. **The structural signal thesis**: a unifying explanation grounded in optimal stopping theory for why distribution-invariant structural signals outperform content-specific signals in multi-substrate retrieval stopping (Section 6.4).

4. **Actionable design guidance**: practitioners should default to structural stopping signals, treat stopping as a first-class design target, and augment rather than replace simple heuristics (Section 6.4.7).
