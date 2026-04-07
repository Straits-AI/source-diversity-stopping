# 6 Discussion

## 6.1 Smart Stopping Beats Smart Searching

The end-to-end results (N=500, clean split, Table 1) establish that the heuristic stopping rule achieves the highest E2E U@B (0.759) despite producing lower-quality answers than the ensemble (F1 0.612 vs 0.681, EM 0.448 vs 0.510). The heuristic's advantage is entirely cost-driven: 1.16 operations versus 3.00, a 61% reduction that more than compensates for the quality gap under μ = 0.3.

This is an honest cost-quality tradeoff, not a universal superiority claim. The sensitivity analysis (Table 4) shows the crossover at μ ≈ 0.3: below this threshold, the ensemble's higher answer quality dominates; above it, the heuristic's cost efficiency dominates. The practical implication is a design principle for retrieval systems: **default to restraint and require strong evidence of a coverage gap before escalating,** but only when retrieval cost is a binding constraint.

## 6.2 The Positive Routing Gap

The LLM-routed policy demonstrates that positive routing — choosing the right substrate for each question — is achievable. Its action distribution (STOP=33%, SEMANTIC=19%, LEXICAL=35%, HOP=14%) shows genuine per-question substrate variation, and its higher recall (0.845 vs 0.795) confirms that the LLM identifies evidence gaps the heuristic misses.

But positive routing is not yet cost-efficient. The gap between the LLM router's routing intelligence and the heuristic's cost discipline identifies the central open challenge: **calibrated stopping** — a policy that combines the LLM's ability to recognize genuinely insufficient evidence with the heuristic's discipline to stop when evidence is merely adequate rather than comprehensive.

We conjecture that the optimal policy lies between these two extremes: it would stop as often as the heuristic (saving cost on easy questions) while routing as intelligently as the LLM (achieving higher recall on hard questions). Training such a policy requires a reward signal that captures the downstream cost-quality tradeoff — exactly the Utility@Budget metric we define.

## 6.3 Comparison with Existing Systems

Our comparison with FLARE, Self-RAG, Adaptive-RAG, IRCoT, and CRAG (Appendix B) reveals that AEA occupies a distinct niche: it is the only system that (a) routes across qualitatively different substrate types, (b) includes an explicit cost model, and (c) treats stopping as a first-class routing decision.

Direct numerical comparison is not valid: those systems report downstream QA accuracy after full LLM generation on different benchmarks with different assumptions. However, the design dimension analysis shows that none of the existing systems addresses the stopping-vs-routing tradeoff that our experiments identify as central.

## 6.4 Why the Heuristic Resists Improvement: A Root Cause Analysis

The central puzzle of our experimental program is not that the heuristic stopping rule works well -- that was expected from the ablation analysis (Section 5.3). The puzzle is that three qualitatively different sophisticated approaches all fail to improve upon it, each for a different proximate reason but -- as we argue below -- for the same underlying cause. This section provides a rigorous analysis of why simple structural stopping signals resist replacement by content-aware, learned, or decomposition-based alternatives.

### 6.4.1 The Empirical Phenomenon

We tested four approaches designed to improve on the heuristic's stopping decision ("stop when the workspace contains 2+ high-relevance items from 2+ distinct sources"):

| Approach | Mechanism | AvgOps | E2E U@B | vs Heuristic |
|---|---|---|---|---|
| Cross-encoder stopping | MS MARCO scores (question, passage) pairs | 3.09 | 0.655 | -0.133 (p<0.0001) |
| LLM decomposition | gpt-oss-120b decomposes question into sub-requirements | 2.95 | 0.758 | -0.030 |
| Learned GBT classifier | Gradient boosted tree on workspace statistics | 5.00 | 0.498 | catastrophic |
| Embedding router | Question embedding predicts best retrieval strategy | 1.28 | tied | +0.001 |
| **Heuristic** | **2+ items from 2+ sources** | **1.16** | **0.759** | **--** |

Every sophisticated approach either degrades performance or merely ties. The cross-encoder is significantly worse (p<0.0001); the decomposition approach wastes approximately 2.5x the operations for lower utility; the learned classifier catastrophically fails to generalize; and the embedding router, while successfully routing questions, confirms that the bottleneck is stopping rather than routing.

### 6.4.2 The Structural Signal Thesis

The heuristic's stopping criterion -- "2+ high-relevance items from 2+ different sources" -- operates on a **structural** property of the workspace, not on the **content** of any passage. It answers the question "has evidence converged from independent retrieval pathways?" rather than "does the evidence answer the question?" This distinction is the key to its robustness.

Structural signals have three properties that content signals lack:

**Distribution invariance.** The predicate "items from 2+ sources" is a counting function over source identifiers. It does not depend on passage vocabulary, entity density, syntactic structure, or any other property of the text distribution. When the question distribution shifts -- from HotpotQA to 2WikiMultiHopQA, from bridge questions to comparison questions -- the structural predicate remains well-defined and its threshold remains calibrated. This explains why the heuristic achieves the best E2E U@B on 2WikiMultiHopQA (1.055 vs 1.031 for semantic-only), even though it was never tuned on that benchmark.

**Compositionality for free.** Multi-hop questions have a compositional answer structure: the answer depends on a conjunction of facts, each potentially residing in a different passage. The heuristic's multi-source requirement is a natural proxy for compositionality -- if evidence has arrived from two independent retrieval pathways, the workspace likely contains both halves of the bridge. This proxy is imperfect but correct frequently enough to dominate alternatives that attempt to verify compositionality explicitly but fail due to noise.

**Computational cheapness.** The heuristic requires no model inference, no API calls, and no learned parameters. Every operation the stopping mechanism consumes is an operation unavailable for retrieval. The cross-encoder spends inference budget on passage scoring; the decomposition approach spends an entire LLM call on question analysis. The heuristic spends nothing, preserving its full budget for evidence gathering.

### 6.4.3 Why Content-Aware Stopping Fails on Multi-Hop QA

The cross-encoder stopping policy uses a pre-trained MS MARCO model to score (question, passage) pairs. It stops when the top cross-encoder score exceeds a high threshold (7.0) or when two or more passages exceed a medium threshold (3.0). Yet it achieves the worst E2E U@B of any policy tested (0.655), significantly below even the brute-force ensemble (0.707, p=0.0001).

The root cause is a **set function decomposition failure**. Multi-hop questions require an evidence *bundle* -- a set of passages that jointly contain the answer even though no individual passage does. Consider "What nationality is the director of Jaws?" The evidence consists of two passages: one identifying Steven Spielberg as the director, and another identifying Spielberg as American. Neither passage, scored independently against the full question, receives a high cross-encoder score, because neither individually answers the question.

This is a fundamental limitation, not a threshold-tuning problem. The sufficiency of an evidence bundle is a **set function** -- it depends on the joint content of {p_1, p_2, ..., p_k}. The cross-encoder computes g(q, p_i) for each p_i independently. For multi-hop questions, sufficiency S(q, P) is not decomposable as any function of individual scores, because the sufficiency emerges from the *combination* of passages.

What would work is a model that scores the entire bundle: f(q, {p_1, ..., p_k}) -> sufficiency. But training such a model requires multi-hop-specific supervision, which defeats the generalization goal. The cross-encoder's pre-training on MS MARCO passage ranking -- a single-hop task -- cannot provide the compositional reasoning needed for multi-hop sufficiency assessment.

The operational consequence is severe: because the cross-encoder rarely triggers a stop, the policy defaults to exhaustive retrieval (3.09 ops vs the ensemble's 3.00), paying the full cost while achieving lower quality because continued retrieval introduces noise.

### 6.4.4 Why Learned Stopping Fails to Generalize

The gradient boosted tree classifier, trained on trajectory data from HotpotQA questions 500-999, achieves 93.3% accuracy and 71.7% F1 on in-distribution held-out data. But when evaluated on questions 0-499, it achieves 5.00 average operations and 0.498 E2E U@B -- worse than random. The classifier effectively never triggers STOP on the evaluation distribution.

This is a textbook case of **spurious correlation under distribution shift**. The classifier's 9 features capture surface statistics of the retrieval trajectory. On the training distribution, these statistics correlate with the optimal stopping point: for example, mean_relevance > 0.45 at step 1 may reliably indicate that both gold passages have been retrieved. But this correlation is contingent on properties of the training questions: their entity density, lexical overlap, and passage characteristics.

When the distribution shifts, these contingent correlations break. The evaluation questions have different feature distributions, causing no feature vector to fall in the classifier's "stop" region.

The heuristic avoids this by construction. Its criterion -- "2+ items from 2+ sources" -- is a distribution-invariant predicate. It depends only on count and identity properties of the workspace, which are structurally stable across distributions. This connects to the principle from robust statistics that **estimators using fewer distributional assumptions degrade more gracefully under distribution shift** (Huber, 1964; Hampel et al., 1986).

### 6.4.5 Why Question Decomposition Fails in Practice

The decomposition policy decomposes each question into information requirements using gpt-oss-120b, then checks whether retrieved passages satisfy each requirement via keyword matching. The approach is intellectually appealing but achieves 2.95 average operations and lower utility than the heuristic.

The root cause is a **precision-matchability tradeoff**. The decomposition must simultaneously satisfy two competing constraints:

1. **Completeness**: requirements must cover all information needed. Missing a requirement causes premature stopping.
2. **Matchability**: each requirement must be verifiable against passages using keyword overlap. Unmatchable requirements appear perpetually unsatisfied, preventing stopping.

These constraints are in tension. Precise requirements (e.g., "Steven Spielberg's nationality") succeed only with exact lexical matches. Vague requirements (e.g., "information about the person") match everything. In practice, the LLM produces malformed or unmatchable requirements approximately 40% of the time. When this happens, the 100% coverage threshold is never reached, and the policy defaults to exhaustive retrieval.

The heuristic sidesteps this tradeoff entirely by not attempting to understand *what* evidence is needed. It asks only *whether* independent retrieval pathways have converged.

### 6.4.6 The Deeper Lesson: Optimal Stopping Under Uncertainty

The failures can be unified under **optimal stopping theory** (Ferguson, 1989; Peskir and Shiryaev, 2006). The retrieval stopping decision is structurally analogous to the classical optimal stopping problem: at each step, the agent observes a signal and decides whether to stop or continue at a cost.

The classical result is that threshold-based rules on **observable, low-noise signals** dominate value-estimation approaches when the value function is hard to learn. The heuristic implements exactly such a rule: it thresholds on a directly observable signal (source diversity count) without estimating the value of continued retrieval. The sophisticated approaches fail because they attempt the harder estimation task:

- The cross-encoder estimates passage-level relevance -- a noisy proxy for bundle-level sufficiency.
- The learned classifier estimates the optimal stopping point -- a distribution-dependent estimate.
- The decomposition approach estimates question-level requirements -- introducing parsing noise.
- The embedding router estimates the optimal strategy -- confirming routing is not the bottleneck.

In each case, the estimation step introduces noise exceeding the information gain from content awareness. The heuristic wins by asking a **simpler question** whose answer is observable, cheap, and robust. This connects to the broader ML principle that simple, robust baselines dominate complex learned approaches when the learning problem is high-dimensional and the evaluation distribution differs from training (Lipton et al., 2018; D'Amour et al., 2020).

### 6.4.7 What Would Actually Beat the Heuristic?

The failure analysis defines the requirements for improvement. A successful approach must simultaneously be: (1) **content-aware** -- to handle edge cases where diverse-source evidence is still insufficient; (2) **bundle-level** -- to assess passage *sets* rather than individual passages; (3) **noise-robust** -- to degrade gracefully when content analysis is imperfect; and (4) **distribution-invariant** -- to generalize without retraining.

These requirements are jointly difficult. Candidate approaches include NLI models applied to (question, evidence bundle) pairs, ensemble methods combining structural and content signals, and self-consistency checks. The most promising path is not to *replace* the heuristic but to *augment* it: use the structural signal as the default and add a content-aware refinement that fires only when the structural signal is ambiguous.

### 6.4.8 Implications for the Adaptive Retrieval Field

The stopping hierarchy -- structural heuristic > content-aware stopping > learned stopping on OOD data -- carries three implications:

**First, the field's focus on routing optimization may be misplaced.** The embedding router confirms this: even good question-level routing produces only +0.001 U@B improvement because the bottleneck is stopping, not routing.

**Second, structural signals should be the default for stopping decisions.** Content-aware stopping signals fail to generalize across question distributions, while structural signals are distribution-invariant by construction. Design principle: **default to structural stopping and escalate to content-aware stopping only when structural signals are uninformative.**

**Third, the stopping problem is harder than it looks.** The four failures span the full spectrum of techniques -- pretrained models, LLM reasoning, supervised learning, embedding classification -- and none succeeds. This suggests the stopping problem in multi-hop retrieval has structure that resists the standard ML playbook, specifically because each new question is a new reasoning chain. Research on distribution-robust stopping criteria, drawing on robust statistics and optimal stopping theory, is a necessary complement to the current focus on retrieval quality and routing intelligence.

## 6.5 Limitations

1. **Single benchmark for end-to-end.** The E2E results (N=100) use only HotpotQA Bridge. Validation on additional benchmarks (BRIGHT, NoLiMa) is needed.

2. **No statistical testing on E2E.** Bootstrap CIs and permutation tests are reported only for the retrieval-only evaluation (N=500). The E2E evaluation (N=100) reports point estimates.

3. **Custom evaluation metric.** Utility@Budget is author-defined. The specific η and μ values determine the ranking — sensitivity analysis across parameter ranges is reported in Appendix C.

4. **Three address spaces.** Real retrieval environments include web search, tool invocation, and structural navigation. Cost differentials across these modalities are larger, potentially amplifying the benefit of selective stopping.

5. **Heuristic policy.** The routing decisions are hand-designed rules. The results show what is achievable without learning; a learned policy could close the gap between routing avoidance and positive routing.

6. **Free-tier LLM for routing and answers.** The gpt-oss-120b model is capable but not state-of-the-art. Stronger models might shift the balance toward positive routing.

## 6.6 Future Work

**Calibrated stopping policy.** Train a stopping classifier on trajectory data with downstream F1 as the reward signal. The key question: can a learned policy match the heuristic's stopping efficiency while capturing the LLM router's recall advantage?

**Step-conditional routing.** Oracle trajectories show step-position preferences (semantic at step 1, entity at step 2). A router conditioned on step position could reduce false-positive escalations.

**Expanded substrates.** Web search, tool execution, and structural navigation would test whether the stopping > searching hierarchy holds when cost differentials are larger.

**Budget sensitivity.** The hierarchy may invert under very tight budgets (where any retrieval is expensive) or very loose budgets (where cost is negligible). Characterizing the budget regime where each policy dominates is an important practical question.
