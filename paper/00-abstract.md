# Why Simple Stopping Rules Win — and How Confidence-Gated Stopping Finally Beats Them

## Abstract

When should a retrieval system stop searching? We study this question across three benchmark families — multi-hop QA (HotpotQA, N=1000), reasoning-intensive retrieval (BRIGHT, N=200), and diluted retrieval settings (50 paragraphs per question). A simple structural heuristic — stop when the workspace contains evidence from two or more independent sources — significantly outperforms comprehensive retrieval under cost-penalized evaluation (paired t-tests, p≤0.021, Cohen's d up to 0.49).

We test six content-aware alternatives: a cross-encoder, an NLI bundle checker, a learned classifier, LLM decomposition, answer-stability tracking, and an embedding router. All six fail — each for a different reason, but with a common root cause: they attempt to assess **evidence quality** (a hard set function over passage bundles) rather than **answerer readiness** (a scalar judgment the LLM can make directly).

This analysis leads to our proposed method: **confidence-gated stopping**. After the first retrieval step, we ask the LLM once: "Can you answer this question from this evidence?" If yes, stop. If no, retrieve once more. This single binary judgment achieves the best end-to-end Utility@Budget (0.799), significantly outperforming both comprehensive retrieval (0.682, p=0.004) and matching the structural heuristic's cost efficiency (1.23 vs 1.16 operations) while improving answer quality (EM +3pp, F1 +3pp, Recall +3.5pp). The key insight: the LLM already knows whether it can answer — assessing the answerer's readiness is fundamentally easier than assessing the evidence's sufficiency.
