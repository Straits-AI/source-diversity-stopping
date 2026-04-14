# 6.4 Why the Heuristic Resists Improvement: A Root Cause Analysis

The central puzzle of our experimental program is not that the heuristic stopping rule works well -- that was expected from the ablation analysis (Section 5.3). The puzzle is that three qualitatively different sophisticated approaches all fail to improve upon it, each for a different proximate reason but -- as we argue below -- for the same underlying cause. This section provides a rigorous analysis of why simple structural stopping signals resist replacement by content-aware, learned, or decomposition-based alternatives.

## 6.4.1 The Empirical Phenomenon

We tested four approaches designed to improve on the heuristic's stopping decision ("stop when the workspace contains 2+ high-relevance items from 2+ distinct sources"):

| Approach | Mechanism | AvgOps | E2E U@B | vs Heuristic |
|---|---|---|---|---|
| Cross-encoder stopping | MS MARCO scores (question, passage) pairs | 3.09 | 0.655 | -0.133 (p<0.0001) |
| LLM decomposition | gpt-oss-120b decomposes question into sub-requirements | 2.95 | 0.758 | -0.030 |
| Learned GBT classifier | Gradient boosted tree on workspace statistics | 5.00 | 0.498 | catastrophic |
| Embedding router | Question embedding predicts best retrieval strategy | 1.28 | tied | +0.001 |
| **Heuristic** | **2+ items from 2+ sources** | **1.16** | **0.759** | **--** |

Every sophisticated approach either degrades performance or merely ties. The cross-encoder is significantly worse (p<0.0001); the decomposition approach wastes approximately 2.5x the operations for lower utility; the learned classifier catastrophically fails to generalize; and the embedding router, while successfully routing questions, confirms that the bottleneck is stopping rather than routing.

## 6.4.2 The Structural Signal Thesis

The heuristic's stopping criterion -- "2+ high-relevance items from 2+ different sources" -- operates on a **structural** property of the workspace, not on the **content** of any passage. It answers the question "has evidence converged from independent retrieval pathways?" rather than "does the evidence answer the question?" This distinction is the key to its robustness.

Structural signals have three properties that content signals lack:

**Distribution invariance.** The predicate "items from 2+ sources" is a counting function over source identifiers. It does not depend on passage vocabulary, entity density, syntactic structure, or any other property of the text distribution. When the question distribution shifts -- from HotpotQA to 2WikiMultiHopQA, from bridge questions to comparison questions, from short-answer factoid queries to complex reasoning chains -- the structural predicate remains well-defined and its threshold remains calibrated. This explains why the heuristic achieves the best E2E U@B on 2WikiMultiHopQA (1.055 vs 1.031 for semantic-only), even though it was never tuned on that benchmark.

**Compositionality for free.** Multi-hop questions have a compositional answer structure: the answer depends on a conjunction of facts, each potentially residing in a different passage. The heuristic's multi-source requirement is a natural proxy for compositionality -- if evidence has arrived from two independent retrieval pathways (e.g., a semantic search that found entity A and a subsequent search that found entity B), the workspace likely contains both halves of the bridge. This proxy is imperfect (two passages from different sources may both concern entity A), but it is correct frequently enough to dominate alternatives that attempt to verify compositionality explicitly but fail due to noise.

**Computational cheapness.** The heuristic requires no model inference, no API calls, and no learned parameters. It executes in microseconds. This is not merely an engineering convenience; it is a methodological advantage. Every operation the stopping mechanism consumes is an operation unavailable for retrieval. The cross-encoder approach spends inference budget on passage scoring that could have been spent on additional retrieval; the decomposition approach spends an entire LLM call on question analysis. The heuristic spends nothing, preserving its full budget for evidence gathering.

## 6.4.3 Why Content-Aware Stopping Fails on Multi-Hop QA

The cross-encoder stopping policy uses a pre-trained MS MARCO model to score (question, passage) pairs. It stops when the top cross-encoder score exceeds a high threshold (7.0) or when two or more passages exceed a medium threshold (3.0). This approach is well-motivated: it grounds the stopping decision in the semantic relationship between the question and the retrieved evidence, rather than in workspace statistics. Yet it achieves the worst E2E U@B of any policy tested (0.655), significantly below even the brute-force ensemble (0.749).

The root cause is a **set function decomposition failure**. Multi-hop questions require an evidence *bundle* -- a set of passages that jointly contain the answer even though no individual passage does. Consider the question "What nationality is the director of Jaws?" The evidence bundle consists of two passages: one identifying Steven Spielberg as the director of Jaws, and another identifying Spielberg as American. Neither passage, scored independently against the full question, receives a high cross-encoder score, because neither passage individually answers the question. The cross-encoder scores f(q, p_1) and f(q, p_2) are both moderate; neither exceeds the high threshold, and frequently both fall below the medium threshold.

This is a fundamental limitation, not a threshold-tuning problem. The sufficiency of an evidence bundle is a **set function** -- it depends on the joint content of {p_1, p_2, ..., p_k}, not on any aggregation of individual scores. Formally, let S(q, P) denote the sufficiency of passage set P for question q. The cross-encoder computes g(q, p_i) for each p_i in P independently. The question is whether S(q, P) can be recovered from {g(q, p_i) : p_i in P}. For multi-hop questions, the answer is generally no: S is not decomposable as any function of individual scores, because the sufficiency emerges from the *combination* of passages, not from any single passage's relevance.

What would work in principle is a model that scores the entire bundle: f(q, {p_1, ..., p_k}) -> sufficiency. But training such a model requires multi-hop-specific supervision (examples of sufficient and insufficient bundles for multi-hop questions), which defeats the generalization goal. The cross-encoder's pre-training on MS MARCO passage ranking -- a single-hop, single-passage relevance task -- cannot provide the compositional reasoning needed for multi-hop sufficiency assessment.

The operational consequence is severe: because the cross-encoder rarely triggers a stop, the policy defaults to exhaustive retrieval (3.09 ops on average, compared to the ensemble's 3.00). It pays the full cost of comprehensive retrieval while achieving lower answer quality, because its continued retrieval introduces noise passages that dilute the workspace.

## 6.4.4 Why Learned Stopping Fails to Generalize

The gradient boosted tree classifier, trained on trajectory data from HotpotQA questions 500-999, achieves strong in-distribution performance (93.3% accuracy, 71.7% F1 on held-out test data from the same distribution). But when evaluated on questions 0-499, it achieves 5.00 average operations and 0.498 E2E U@B -- worse than random. The classifier effectively never triggers STOP on the evaluation distribution.

This is a textbook case of **spurious correlation under distribution shift**. The classifier's 9 features -- workspace item count, relevance statistics, source diversity, step number, and marginal improvement -- capture surface statistics of the retrieval trajectory. On the training distribution, these statistics correlate with the optimal stopping point: for example, the training questions may have a characteristic pattern where mean_relevance > 0.45 at step 1 reliably indicates that both gold passages have been retrieved. But this correlation is contingent on properties of the training questions: their entity density, the lexical overlap between questions and gold passages, the effectiveness of the semantic index on those particular passages.

When the distribution shifts to the evaluation questions, these contingent correlations break. The evaluation questions may have different entity density, different vocabulary overlap patterns, or different passage lengths, causing the feature distributions to shift. The classifier, having memorized the training distribution's feature-to-label mapping, finds that no evaluation-time feature vector falls in the "stop" region of its decision boundary.

The heuristic avoids this failure mode by construction. Its stopping criterion -- "2+ items from 2+ sources" -- is a distribution-invariant predicate. It does not depend on the values of relevance scores (which shift with the embedding model's behavior on different text), passage lengths (which vary across question distributions), or any other feature whose distribution is question-dependent. The predicate depends only on count and identity properties of the workspace, which are structurally stable across distributions.

This finding connects to a broader principle from robust statistics: **estimators that use fewer assumptions about the data-generating distribution degrade more gracefully under distribution shift** (Huber, 1964; Hampel et al., 1986). The heuristic makes almost no distributional assumptions -- only that "high-relevance" can be thresholded at 0.4 and that source diversity indicates evidence convergence. The learned classifier makes implicit distributional assumptions through every feature-split in its decision tree. Each assumption is a potential failure point under shift.

## 6.4.5 Why Question Decomposition Fails in Practice

The decomposition stopping policy decomposes each question into information requirements using gpt-oss-120b (e.g., "What nationality is the director of Jaws?" becomes ["identity of the director of Jaws", "nationality of that director"]), then checks whether retrieved passages satisfy each requirement using keyword matching. The approach is intellectually appealing: it replaces a generic structural signal with a content-specific coverage check. Yet it achieves 2.95 average operations and lower utility than the heuristic.

The root cause is a **precision-matchability tradeoff** in the decomposition step. For the approach to work, the decomposition must satisfy two competing constraints simultaneously:

1. **Completeness**: the requirements must cover all information needed to answer the question. Missing a requirement means the policy may stop before gathering enough evidence.

2. **Matchability**: each requirement must be verifiable against retrieved passages using the available matching mechanism (keyword overlap). Requirements that are semantically valid but lexically unmatchable will appear perpetually unsatisfied, causing the policy to never stop.

These constraints are in tension. Precise, specific requirements (e.g., "Steven Spielberg's nationality") are easy to verify if the passage contains "Spielberg" and "American" but miss the answer if the passage uses "the director" instead. Vague requirements (e.g., "information about the person") match everything and provide no stopping signal.

In practice, the LLM produces malformed or unmatchable requirements approximately 40% of the time. When this happens, the coverage threshold (100% of requirements satisfied) is never reached, and the policy defaults to exhaustive retrieval. This explains its high average operation count (2.95) and its failure to improve on the heuristic.

The heuristic sidesteps this tradeoff entirely by not attempting to understand *what* evidence is needed. It asks only *whether* independent retrieval pathways have converged -- a question that requires no decomposition, no matching, and no LLM inference.

## 6.4.6 The Deeper Lesson: Optimal Stopping Under Uncertainty

The failures of all four approaches can be unified under the framework of **optimal stopping theory** (Ferguson, 1989; Peskir and Shiryaev, 2006). The retrieval stopping decision is structurally analogous to the classical optimal stopping problem: at each step, the agent observes a signal and must decide whether to stop (accept the current evidence) or continue (pay a cost for potentially better evidence).

The classical result from optimal stopping theory is that threshold-based rules on **observable, low-noise signals** dominate value-estimation approaches when the value function is hard to learn. The heuristic implements exactly such a rule: it thresholds on a directly observable signal (source diversity count) without attempting to estimate the value of continued retrieval. The sophisticated approaches fail because they attempt the harder task:

- The cross-encoder estimates passage-level relevance, but this is a noisy proxy for bundle-level sufficiency.
- The learned classifier estimates the optimal stopping point, but this estimate is distribution-dependent.
- The decomposition approach estimates question-level requirements, but this estimation introduces parsing noise.
- The embedding router estimates the optimal strategy, but confirms that routing quality is not the bottleneck.

In each case, the estimation step introduces noise that exceeds the information gain from content awareness. The heuristic wins by asking a **simpler question** -- "have I seen evidence from diverse sources?" -- whose answer is observable, cheap, and robust.

This connects to the broader machine learning principle that **simple, robust baselines often dominate complex learned approaches when the learning problem is high-dimensional and the evaluation distribution differs from the training distribution** (Lipton et al., 2018; D'Amour et al., 2020). The stopping decision space is high-dimensional (the space of possible workspace states is exponential in the number of passages and features), and the evaluation distribution necessarily differs from any training distribution (new questions have new entities, new passage structures, new reasoning chains). Under these conditions, the low-dimensional, distribution-invariant heuristic is expected to outperform learned approaches.

## 6.4.7 What Would Actually Beat the Heuristic?

The failure analysis defines the requirements for a successful improvement. An approach that beats the heuristic must simultaneously be:

1. **Content-aware** (unlike the heuristic) -- to handle edge cases where diverse-source evidence is still insufficient, such as questions where both retrieved passages concern the same entity from different sources.

2. **Bundle-level** (unlike the cross-encoder) -- to assess passage *sets* rather than individual passages, capturing the compositional sufficiency that multi-hop questions require.

3. **Noise-robust** (unlike the decomposition approach) -- to degrade gracefully when the content analysis is imperfect, rather than defaulting to exhaustive retrieval on any parsing failure.

4. **Distribution-invariant** (unlike the learned classifier) -- to generalize across question types, entity distributions, and passage characteristics without retraining.

These requirements are jointly difficult to satisfy. Candidate approaches that address subsets of them include:

- **Natural language inference (NLI) models** applied to (question, evidence bundle) pairs, checking whether the bundle textually entails an answer. NLI models operate at the bundle level and are pretrained on distribution-diverse data, but they introduce their own noise (NLI accuracy on multi-hop reasoning is imperfect) and require inference cost.

- **Ensemble methods** that combine the heuristic's structural signal with a lightweight content signal. For example: "stop when 2+ sources converge AND a fast NLI check does not flag a coverage gap." This preserves the heuristic's distribution invariance as the primary stopping mechanism while adding a content-aware override for edge cases.

- **Self-consistency checks** where the LLM generates candidate answers from the current workspace and checks whether they are consistent. If multiple diverse answers are produced, the evidence is likely insufficient. This is bundle-level and distribution-invariant, but computationally expensive.

The practical implication is that the most promising path forward is not to *replace* the heuristic but to *augment* it: use the structural signal as the default stopping criterion and add a content-aware refinement that fires only when the structural signal is ambiguous (e.g., exactly 2 items from exactly 2 sources, but with low confidence).

## 6.4.8 Implications for the Adaptive Retrieval Field

The stopping hierarchy we observe -- structural heuristic > content-aware stopping > learned stopping on out-of-distribution data -- carries three implications for the broader adaptive retrieval research agenda.

**First, the field's focus on routing optimization may be misplaced.** Most adaptive retrieval work focuses on choosing *what* to retrieve (the routing problem) or *how* to retrieve it (the retrieval quality problem). Our evidence suggests that the underexplored dimension is *when to stop* retrieving, and that gains from better stopping dominate gains from better routing. The embedding router confirms this directly: even perfect question-level routing produces only marginal improvement (+0.001 U@B) because the bottleneck is the stopping decision, not the routing decision.

**Second, structural signals should be the default for stopping decisions.** The evidence shows that content-aware stopping signals (cross-encoder scores, LLM-generated requirements, learned workspace statistics) fail to generalize across question distributions, while structural signals (source diversity, retrieval pathway convergence) are distribution-invariant by construction. This suggests a design principle: **default to structural stopping and escalate to content-aware stopping only when structural signals are uninformative.**

**Third, the stopping problem is harder than it looks.** The apparent simplicity of the heuristic (a two-line predicate) masks the difficulty of improving upon it. The four failures documented here span the full spectrum of available techniques -- pretrained models, LLM reasoning, supervised learning, and embedding-based classification -- and none succeeds. This suggests that the stopping problem in multi-hop retrieval has structure that resists the standard ML playbook, specifically because the evaluation distribution is guaranteed to differ from any training distribution (each new question is a new reasoning chain). Research on distribution-robust stopping criteria, drawing on robust statistics and optimal stopping theory, is a necessary complement to the current focus on retrieval quality and routing intelligence.
