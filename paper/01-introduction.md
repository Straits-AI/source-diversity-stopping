# 1 Introduction

When should a multi-substrate retrieval system stop searching? This question is fundamental to cost-efficient retrieval-augmented generation, yet surprisingly underexplored. Existing adaptive retrieval systems focus on *what* to retrieve (Self-RAG [Asai et al., 2024], FLARE [Jiang et al., 2023]) or *how* to route queries (Adaptive-RAG [Jeong et al., 2024], SmartRAG [Gao et al., 2025]), but the stopping decision — whether to continue searching or return with current evidence — receives little systematic attention.

We study this problem across semantic and lexical retrieval substrates and arrive at a strong empirical result: **a one-line structural heuristic — stop when the workspace contains evidence from two or more independent sources — is Pareto-optimal** within the space of stopping mechanisms we test. It significantly outperforms comprehensive retrieval across three benchmark families (HotpotQA p<0.0001, BRIGHT p=0.003, diluted retrieval p<0.0001), and no tested alternative improves upon it.

To establish this frontier, we conduct ten controlled experiments spanning seven design categories:

**Seven content-aware alternatives** (all fail): a cross-encoder for per-passage scoring, an NLI model for bundle-level entailment, a learned classifier on trajectory features, LLM-based question decomposition, answer-stability tracking across retrieval steps, confidence-gated stopping via LLM self-assessment, and embedding-based question routing. Each fails for a different proximate reason, but root cause analysis reveals a common pattern: assessing **evidence quality** requires evaluating a set function over passage bundles that current models cannot compute reliably.

**Three structural improvements** (all converge): threshold optimization via grid search, novelty-based stopping via embedding similarity, and dual-signal stopping via relevance convergence. All three converge to identical behavior as the original heuristic because source diversity is the binding constraint — other structural signals are redundant with it.

These ten experiments identify two ceilings:

1. **A content-aware ceiling**: every content-based stopping signal tested introduces more noise (from set function approximation errors, distribution-specific correlations, parsing failures, or phrasing instability) than information. The cost of assessing evidence quality exceeds the value of the assessment.

2. **A structural ceiling**: source diversity is the maximally informative zero-cost stopping signal. Grid search over 30 threshold configurations confirms the hand-tuned 2/2/0.4 parameters are near-optimal; novelty and relevance-convergence signals are redundant.

The heuristic sits at the intersection of these ceilings — the Pareto frontier of stopping quality vs. stopping cost. This connects to classical optimal stopping theory: threshold rules on low-noise observables dominate value-estimation approaches when the value function is hard to learn.

Our contributions are:

1. **A Pareto-optimality result**: source-diversity stopping significantly outperforms comprehensive retrieval (p<0.0001, three benchmark families, Cohen's d up to 0.49) and is not improved by any of ten tested alternatives across seven design categories (Section 5).

2. **Ten controlled failure analyses** identifying the content-aware ceiling (evidence quality is an intractable set function) and the structural ceiling (source diversity is the maximal zero-cost signal) that make the heuristic Pareto-optimal (Section 5.4, 5.11, 6.4).

3. **A reframing of adaptive retrieval stopping**: from a learning problem (train a better stopping model) to a signal-selection problem (identify the right structural observable), grounded in optimal stopping theory (Section 6.4).

4. **Actionable design guidance**: practitioners should default to source-diversity stopping, invest in retrieval quality rather than stopping sophistication, and evaluate stopping mechanisms on out-of-distribution data (Section 6.4.7).
