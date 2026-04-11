# 1 Introduction

When should a retrieval system stop searching? The dominant paradigm in retrieval-augmented generation commits to a fixed retrieval budget — typically one or two calls to a single index — regardless of whether the retrieved evidence is sufficient or the additional operations are wasteful. Recent adaptive retrieval systems address this: Self-RAG [Asai et al., 2024] learns when to skip retrieval, FLARE [Jiang et al., 2023] triggers retrieval on low-confidence tokens, and Adaptive-RAG [Jeong et al., 2024] routes queries by complexity. Yet these systems operate within a single retrieval modality. The question of when to stop searching across multiple qualitatively different substrates remains underexplored.

We study **coverage-driven retrieval stopping** across heterogeneous substrates (semantic and lexical retrieval). We begin with a simple structural heuristic: stop when the workspace contains evidence from two or more independent retrieval sources. Across three benchmark families — multi-hop QA (HotpotQA, N=1000), reasoning-intensive retrieval (BRIGHT, N=200), and diluted retrieval settings (50 paragraphs per question) — this heuristic significantly outperforms comprehensive retrieval (p < 0.003, Cohen's d up to 0.49).

We then ask: **can we do better?** We test six content-aware stopping mechanisms — a cross-encoder, an NLI bundle checker, a learned classifier, LLM decomposition, answer-stability tracking, and an embedding router — and find that **all six fail**. Root cause analysis reveals a common pattern: each tries to assess **evidence quality** (a set function over passage bundles that current models cannot reliably compute), when the easier question is **answerer readiness** (whether the LLM itself can produce a confident answer from the current evidence).

This analysis motivates our proposed method: **confidence-gated stopping**. After the first retrieval step, we ask the LLM once: "Can you answer this question from this evidence?" If yes, stop. If no, retrieve once more. This single binary judgment achieves the best end-to-end Utility@Budget (0.799), significantly outperforming comprehensive retrieval (0.682, p=0.004) while matching the structural heuristic's cost efficiency (1.23 ops vs 1.16). The method improves answer quality over the heuristic (EM +3pp, F1 +3pp) by catching cases where the structural signal incorrectly signals sufficiency but the LLM recognizes it cannot yet answer.

Our contributions are:

1. **The structural stopping baseline**: a coverage heuristic that significantly outperforms comprehensive retrieval across three benchmark families (p<0.0001, d=0.22–0.49), establishing the floor that any stopping method must beat (Section 5.1–5.9).

2. **Six controlled failure analyses** showing why content-aware stopping approaches fail — revealing that evidence quality assessment (the set function problem) is the core bottleneck (Section 5.4, Section 6.4).

3. **Confidence-gated stopping**: a method that sidesteps the set function problem by assessing answerer readiness rather than evidence quality. One LLM call, 1.23 operations, best E2E U@B (0.799), significantly beats ensemble (p=0.004) (Section 3.5, Section 5.11).

4. **The evidence-vs-readiness distinction**: a conceptual contribution clarifying that the stopping decision should assess the answerer's state (easy, scalar), not the evidence's completeness (hard, set function) (Section 6.4).
