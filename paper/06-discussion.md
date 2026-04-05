# 6 Discussion

## 6.1 The Stopping-Recall Tradeoff

The divergence between retrieval-only and end-to-end evaluation reveals a fundamental tradeoff in adaptive retrieval routing.

Under retrieval-only evaluation, where cost is penalized but answer quality is proxied by support recall alone, the optimal strategy is aggressive stopping: retrieve just enough evidence and avoid unnecessary operations. AEA excels in this regime, achieving the highest Utility@Budget through cost efficiency (1.15 operations vs 2.00–3.00 for baselines).

Under end-to-end evaluation, where a language model generates answers from the retrieved evidence, the calculus shifts. Higher recall provides the LLM with more relevant passages, improving answer quality enough to offset the additional retrieval cost. The ensemble policy — which queries all substrates unconditionally — wins in this regime despite consuming 2.6x the operations of AEA.

This tradeoff is not a failure of the adaptive routing framework but rather a calibration question: **how aggressively should the policy stop?** The current coverage threshold (2 high-relevance items from 2 sources) is too aggressive for end-to-end tasks. A learned stopping policy trained on downstream answer quality, rather than on retrieval proxy metrics, could find the optimal tradeoff point. The trajectory data collected during our experiments — operation sequences, coverage signals, per-step recall, and now downstream F1 — constitutes a natural training signal for such a policy.

## 6.2 Routing Avoidance vs. Positive Routing

The ablation analysis reveals that the current heuristic policy's value comes from knowing what *not* to do (routing avoidance) rather than from knowing what *to* do (positive substrate selection). This finding deserves careful interpretation.

It does not imply that positive routing is unimportant — oracle analysis shows 44% of questions require within-task substrate switching, and a policy that could reliably identify these questions would capture additional value. Rather, it shows that **heuristic positive routing is hard**: the current multi-hop detection patterns do not reliably distinguish questions that benefit from entity hops from those that do not. Removing entity hops entirely slightly improves performance on lexically-rich data, because the false-positive cost of unnecessary hops exceeds the true-positive benefit of necessary ones.

The practical implication is a design principle: **default to restraint.** Build retrieval systems that default to the cheapest operation and require positive evidence of a coverage gap before escalating. This is the opposite of the ensemble approach, which defaults to comprehensiveness and pays for it. Under most budget-aware evaluation regimes, restraint outperforms comprehensiveness at the single-substrate level — though as Section 5.2 shows, the advantage narrows when downstream answer quality is measured.

## 6.3 Positioning Against Existing Adaptive Retrieval Systems

Our comparison with FLARE, Self-RAG, Adaptive-RAG, IRCoT, and CRAG (Table 6 in the appendix) reveals that AEA occupies a distinct niche: it is the only system that (a) routes across qualitatively different substrate types (dense, sparse, graph), (b) includes an explicit cost model in evaluation, and (c) treats the null action (stopping) as a first-class routing decision with estimated value.

However, we note important limitations of this comparison. The systems above report downstream QA accuracy after full LLM generation on standard benchmarks with standard metrics. Our primary results use a retrieval-only proxy metric on a smaller evaluation set. Direct numerical comparison is not valid. The comparison table should be read as a positioning analysis — showing which design dimensions each system covers — rather than as a performance ranking.

## 6.4 Limitations

1. **End-to-end evaluation is preliminary.** The LLM answer generation results (N=50, free-tier model) are smaller-scale and noisier than the retrieval-only results (N=500, bootstrap CIs). The finding that ensemble beats AEA on end-to-end U@B should be validated at larger scale with a stronger model.

2. **Heuristic policy only.** The routing decisions are hand-designed rules. A learned router trained on trajectory data could address both the over-searching failures on single-substrate tasks and the under-routing failures on multi-hop tasks.

3. **Three address spaces.** Real retrieval environments include web search, tool invocation, structured databases, and code search. The cost differentials across these modalities are larger than in our setup, potentially amplifying the benefit of selective avoidance.

4. **Synthetic benchmark.** The Heterogeneous v2 benchmark is author-constructed. While entity and lexical isolation are programmatically validated, the task distribution may not reflect real-world heterogeneity.

5. **No statistical testing on end-to-end results.** The N=50 end-to-end evaluation does not include bootstrap CIs or permutation tests. This limits the confidence of comparisons in that regime.

## 6.5 Future Work

Three directions follow from the evidence:

**Learned stopping policy.** The trajectory data from all experiments — operation sequences, coverage signals, per-step recall, and downstream F1 — provides supervision for training a stopping classifier. The key question: can a learned policy find the stopping threshold that optimizes end-to-end answer quality rather than retrieval proxy metrics?

**Step-conditional routing.** Oracle trajectories show step-position preferences (semantic at step 1, entity at step 2). A router conditioned on step position could reduce false-positive entity invocations while preserving true-positive ones.

**Expanded substrates and external benchmarks.** Web search, tool execution, and structural navigation would test whether selective avoidance generalizes to larger cost differentials. BRIGHT and NoLiMa would test whether the advantage grows on reasoning-intensive tasks with low lexical overlap.
