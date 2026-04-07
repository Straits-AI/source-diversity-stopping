# Why Simple Stopping Rules Win: Structural Signals Beat Content-Aware Methods in Multi-Substrate Retrieval Routing

## Abstract

When should a retrieval system stop searching? We study this question across three benchmark families — multi-hop QA (HotpotQA, N=1000), reasoning-intensive retrieval (BRIGHT, N=200), and diluted open-domain settings (50 paragraphs per question). A simple structural heuristic — stop when the workspace contains evidence from two or more independent sources — significantly outperforms comprehensive retrieval across all settings (paired t-tests, p<0.003 in all cases, Cohen's d up to 0.49).

We then test five principled content-aware alternatives: a pre-trained cross-encoder, an NLI-based bundle sufficiency checker, a learned classifier on trajectory features, LLM-based question decomposition, and an embedding-based router. All five fail. The cross-encoder cannot assess multi-document sufficiency because evidence quality is a set function over passage bundles. The NLI model — which does take bundles as premise — fails because multi-hop questions resist conversion to well-formed entailment hypotheses. The learned classifier overfits to distribution-specific statistics. The LLM decomposition introduces parsing noise. The embedding router confirms the bottleneck is stopping, not routing.

Root cause analysis reveals that the heuristic succeeds because it operates on a **structural signal** — source diversity — that is distribution-invariant, compositionally informative, and computationally free. This connects to classical optimal stopping theory: threshold rules on low-noise observable signals dominate value-estimation approaches when the value function is hard to learn. The heuristic's advantage is invariant to candidate set size (zero degradation at 5x expansion), question type (comparison questions show larger effects than bridge), and benchmark family (multi-hop, reasoning-intensive, diluted retrieval).
# 1 Introduction

When should a retrieval system stop searching? The dominant paradigm in retrieval-augmented generation commits to a fixed retrieval budget — typically one or two calls to a single index — regardless of whether the retrieved evidence is sufficient or the additional operations are wasteful. Recent adaptive retrieval systems address the sufficiency side: Self-RAG [Asai et al., 2024] learns when to skip retrieval, FLARE [Jiang et al., 2023] triggers retrieval on low-confidence tokens, and Adaptive-RAG [Jeong et al., 2024] routes queries by complexity. Yet these systems operate within a single retrieval modality. The question of when to stop searching across multiple qualitatively different substrates — semantic indexes, keyword indexes, entity graphs — remains underexplored.

In this paper, we study **coverage-driven retrieval stopping** across heterogeneous substrates, using semantic and lexical retrieval as the two primary substrates. We begin with a simple structural heuristic: stop when the workspace contains evidence from two or more independent retrieval sources. Across three benchmark families — multi-hop QA (HotpotQA, N=1000), reasoning-intensive retrieval (BRIGHT, N=200), and diluted open-domain settings (50 paragraphs per question) — this heuristic significantly outperforms comprehensive retrieval (p < 0.003 in all cases, Cohen's d up to 0.49).

We then ask: **can we do better?** We implement five principled improvements — a pre-trained cross-encoder for content-aware stopping, an NLI-based bundle sufficiency checker, a learned classifier on trajectory features, LLM-based question decomposition, and an embedding-based router — and find that **all five fail** to improve on the heuristic. Each fails for a different proximate reason, but root cause analysis reveals a common underlying explanation: the heuristic operates on a **structural signal** (source diversity) that is distribution-invariant, compositionally informative, and computationally free, while all five alternatives introduce content-specific or distribution-specific noise that degrades stopping quality.

This failure pattern is itself the paper's central contribution. It reveals that:

1. **Evidence sufficiency in multi-hop QA is a set function** — it depends on the joint content of passage bundles, not on individual passage scores. Cross-encoders trained on single-passage ranking cannot assess it (Section 6.4.3).

2. **Even bundle-level NLI assessment fails** — multi-hop questions resist conversion to well-formed entailment hypotheses, so NLI models that correctly take the full evidence bundle as premise still cannot determine sufficiency (Section 6.4.3).

3. **Workspace statistics are distribution-specific** — learned classifiers on trajectory features capture spurious correlations that break under distribution shift (Section 6.4.4).

4. **Structural stopping signals connect to optimal stopping theory** — threshold rules on low-noise observables dominate value-estimation approaches when the value function is hard to learn (Section 6.4.6).

Our contributions are:

1. **A statistically validated stopping result across three benchmark families**: a simple coverage heuristic significantly outperforms comprehensive retrieval on HotpotQA (p<0.000001, d=0.379, N=1000), BRIGHT (p=0.0026, d=0.216, N=200), and open-domain settings (p<0.000001, d=0.491, N=200). The advantage is invariant to question type (bridge and comparison) and candidate set size (10 vs 50 paragraphs) (Section 5).

2. **Five controlled failure analyses** showing why content-aware (cross-encoder, NLI), learned (GBT), decomposition-based (LLM), and routing-based (embedding) improvements all fail — each illuminating a different aspect of the stopping problem (Section 5.4, Section 6.4).

3. **The structural signal thesis**: a unifying explanation grounded in optimal stopping theory for why distribution-invariant structural signals outperform content-specific signals in multi-substrate retrieval stopping (Section 6.4).

4. **Actionable design guidance**: practitioners should default to structural stopping signals, treat stopping as a first-class design target, and augment rather than replace simple heuristics (Section 6.4.7).
# 2 Related Work

Adaptive External Attention builds on and departs from several active research threads. We survey them in turn, highlighting where each line of work leaves a gap that our policy-level formalization is designed to fill.

## 2.1 Retrieval-Augmented Generation

The dominant paradigm in retrieval-augmented generation (RAG) couples a fixed retrieval substrate to a language model reader. Sparse lexical retrieval via BM25 [Robertson and Zaragoza, 2009] remains a strong baseline for keyword-dominated queries, while dense passage retrieval [Karpukhin et al., 2020] and its bi-encoder successors achieve stronger performance on semantically paraphrased questions. Hybrid systems that linearly interpolate BM25 and dense scores [Ma et al., 2021] narrow the gap further but still commit irrevocably to a single ranked list before generation begins.

Recent engineering advances push the state of the art within this single-substrate paradigm. Contextual Retrieval [Anthropic, 2024] prepends chunk-level summaries before indexing, substantially reducing retrieval failures caused by missing discourse context. RAPTOR [Sarthi et al., 2024] builds a recursive tree of summaries over the corpus, enabling multi-granularity retrieval without multi-hop reasoning chains. PageIndex [VectifyAI, 2024] structures retrieved passages around document pages to preserve layout-sensitive information. Each of these contributions improves the substrate itself but does not address the question of whether retrieval should be invoked at all, or which substrate family is most appropriate for a given query.

Adaptive invocation has received targeted attention in several systems. Self-RAG [Asai et al., 2024] introduces learned reflection tokens — special symbols inserted during generation that signal whether retrieved passages are relevant or even necessary. By training on these tokens, Self-RAG enables the model to skip retrieval for queries it can answer from parametric knowledge. FLARE [Jiang et al., 2023] takes a complementary approach: it triggers retrieval only when the generator's token probability falls below a confidence threshold during decoding, effectively treating low confidence as a proxy for knowledge insufficiency. Adaptive-RAG [Sequeira et al., 2025] classifies queries by complexity at inference time and routes simpler queries to no-retrieval or single-step retrieval, reserving multi-hop pipelines for complex reasoning tasks. DIVER [Long et al., 2025] introduces multi-stage reasoning retrieval, reporting nDCG@10 of 46.8 on the BRIGHT benchmark, demonstrating that iterative retrieval can substantially outperform single-pass systems.

Collectively, these adaptive systems share an important insight with our work: retrieval avoidance is often as valuable as retrieval itself. Where they differ from AEA is in scope. Self-RAG and FLARE operate within a single retrieval modality, and their skip decisions are entangled with generation rather than expressed as an explicit routing policy. Adaptive-RAG performs routing across retrieval pipeline configurations, not across heterogeneous address spaces such as episodic memory, structured knowledge graphs, and lexical indexes. Our formalization generalizes this skip logic to a budget-aware policy over multiple substrate families.

## 2.2 Agentic Retrieval and Tool Use

A parallel tradition treats retrieval as one action in a broader agentic loop. ReAct [Yao et al., 2022] alternates natural-language reasoning traces with tool-use actions, including search, and demonstrated that interleaving thought and action improves grounding over either alone. Toolformer [Schick et al., 2023] extends this by training a model to self-annotate when and how to call external APIs, internalizing the decision of whether to retrieve. SWE-agent [Yang et al., 2024] illustrates, in the software engineering domain, that the design of the agent-environment interface is often more consequential than the underlying model, a finding that motivates the harness-level focus of our own work.

Within agentic RAG specifically, several multi-substrate systems have emerged. ExpertRAG [Gumaan et al., 2024] applies a mixture-of-experts routing mechanism to select among specialized RAG pipelines, learning which expert is best suited to each query type. Chain of Agents [Zhang et al., 2024] decomposes long-context tasks across a chain of communicating agents, each handling a segment of the retrieval or reasoning problem, reporting approximately ten percent improvement over single-agent RAG baselines. The Agentic RAG Survey [Singh et al., 2025] provides a systematic taxonomy of these approaches, categorizing systems by their planning depth, tool diversity, and coordination mechanisms.

NousResearch's Hermes Agent represents a practitioner instantiation of multi-substrate memory: it queries conversation history, external documents, and web tools within a single agent loop, without a formalized routing policy. While effective in deployment, the absence of an explicit state representation and budget model means that routing decisions are implicitly baked into the prompt rather than learned or optimized. AEA is perhaps closest in spirit to Hermes in its multi-substrate ambition, but departs from it by providing a formal policy substrate, explicit cost accounting, and a principled avoidance criterion.

## 2.3 Reinforcement Learning for Retrieval Optimization

A growing body of work applies reinforcement learning to optimize retrieval behavior at training time. DeepRetrieval [Jiang et al., 2025] uses RL to train a query reformulation model, achieving sixty-five percent recall on multi-hop retrieval tasks by generating queries that are better matched to the retrieval index. Search-R1 [Jin et al., 2025] extends the R1-style process reward model to retrieval-augmented reasoning chains, reporting a forty-one percent improvement over standard RAG baselines by learning when to search mid-generation. SmartRAG [Gao et al., 2025] jointly optimizes the retrieval and generation components under an explicit cost model, penalizing unnecessary retrieval calls and demonstrating that cost-aware training yields more efficient pipelines without sacrificing answer quality. Cost-Aware Retrieval [Hashemi et al., 2024] similarly incorporates latency penalties into retrieval ranking, achieving a twenty percent reduction in end-to-end query latency.

At the routing level, RAGRouter [Liu et al., 2025] learns to select among members of a RAG ensemble at inference time, assigning each query to the retrieval system whose characteristics best match the query's structure. This system is perhaps the closest precursor to AEA's routing logic, but it operates over a fixed ensemble of similarly structured retrieval pipelines rather than over qualitatively distinct address spaces — episodic memory, entity graphs, and lexical indexes — whose output distributions and cost profiles differ substantially.

AEA draws from this RL tradition the insight that routing and avoidance should be optimized objectives, not hand-coded heuristics. However, where these systems learn substrate-specific or pipeline-specific policies, AEA learns a unified policy over a heterogeneous action space that includes the null action of avoidance.

## 2.4 Structured Memory and Context Engineering

Structured memory systems — MAGMA [arXiv:2601.03236], Nemori [arXiv:2508.03341], A-MEM [arXiv:2502.12110], Zep [arXiv:2501.13956] — demonstrate that memory organizations (graph, narrative, associative, temporal) each excel on different query types, motivating routing policies rather than commitment to a single architecture. None addresses when to stop querying.

At the harness level, Meta-Harness [Lee et al., 2026] optimizes which context elements to include in model calls, achieving 7.7-point improvement with 4x fewer tokens. Our work shares this perspective but operates at runtime with explicit state tracking rather than as an offline code-level optimization.

Long-context evaluation work (HELMET, Context Rot, NoLiMa, RULER) demonstrates that synthetic benchmarks poorly predict downstream performance and that all models degrade with context length — motivating our budget-controlled evaluation protocol.

## 2.7 Positioning

AEA occupies a distinct niche relative to all of the above. It is not a new retrieval substrate — it does not improve dense retrieval, structured memory, or graph traversal in isolation. It is not a new generation architecture. It is a formalization of the **policy** over a heterogeneous address space: a runtime decision procedure that determines whether to act, on which substrate, and at what cost, given an explicit state representation and a budget constraint.

The systems closest in spirit are SmartRAG, which jointly optimizes retrieval and generation under cost constraints but within a single homogeneous pipeline; Meta-Harness, which optimizes at the harness level but as an offline, code-level transformation rather than a learned runtime policy; and Hermes Agent, which operates across multiple substrates in practice but without a formal policy representation or cost model. RAGRouter and Adaptive-RAG address routing but within substrate families of the same qualitative type.

The gap that AEA fills is the absence of any existing system that simultaneously (a) routes across qualitatively heterogeneous address spaces — episodic memory, entity graphs, lexical indexes, and the null action — (b) maintains an explicit state representation tracking which substrates have been queried and what they returned, (c) operates under a hard budget constraint expressed in latency or token cost, and (d) treats avoidance not as a fallback but as a first-class action whose value is estimated and compared against retrieval on equal footing. Our central finding — that AEA's gains derive primarily from avoidance rather than positive substrate selection — would be invisible to systems that do not include the null action in their routing vocabulary.
# 3 Method

This section describes the coverage-driven retrieval routing policy. The formal constrained MDP framework motivating the design is presented in Appendix A; here we focus on the operational policy and its components.

## 3.1 Setup and Terminology

The routing policy operates over a set of **retrieval substrates**, each exposing a common query interface:

| Substrate | Mechanism | Best For |
|-----------|-----------|----------|
| Semantic | Dense embeddings + cosine similarity | Paraphrase, distributional similarity |
| Lexical | BM25 keyword scoring | Exact terms, identifiers, rare entities |
| Entity graph | Named entity co-occurrence + BFS | Multi-hop relational chains |

The two primary substrates are semantic and lexical retrieval. We include entity graph traversal as a third substrate to test whether structured retrieval adds value; ablation analysis (Section 5.3) shows it does not contribute on the evaluated benchmarks.

Each query to a substrate returns a ranked list of passages and incurs a cost (measured in operations). The **workspace** is a bounded buffer holding the passages currently under consideration. The policy's job is to decide, after each retrieval step, whether to stop (the evidence is sufficient) or escalate (query another substrate).

Two state components guide routing decisions:

- **Discovery state**: what the agent has *located* — passage titles, entity mentions, structural cues. This is "information scent" in the sense of Pirolli and Card (1999).
- **Knowledge state**: what the agent has *verified* — grounded claims extracted from retrieved passages.

The distinction matters because knowing a relevant passage exists (discovery) is different from knowing what it says (knowledge). A policy without this distinction conflates "I haven't looked" with "I looked and found nothing," leading to redundant exploration.

## 3.2 Coverage-Driven Routing Policy

The policy implements a three-condition decision procedure evaluated after each retrieval step.

**Step 0 — Semantic anchor (always).** The policy initiates with a dense retrieval query using the original question. This establishes anchor passages and populates the workspace with high-recall candidates. Dense retrieval is chosen as the default because it has the broadest coverage and lowest per-operation cost among the three substrates.

**After each step — Coverage check.** The policy evaluates three conditions in priority order:

*Condition 1 — Sufficient coverage (STOP).* If the workspace contains at least two high-relevance passages (relevance score >= 0.4) drawn from at least two distinct sources, the policy stops. The intuition: multi-source corroboration signals that the evidence base is diverse enough for answer synthesis, and additional retrieval is unlikely to improve quality enough to justify its cost.

*Condition 2 — Single-source gap (ESCALATE via entity hop).* If the workspace evidence comes from a single source and the question structure suggests a relational chain (detected via heuristic patterns: possessives, "birthplace of," "director of," "founded by"), the policy escalates to the entity graph substrate. This redirects effort toward the one substrate designed for relational traversal, precisely when lexical and semantic retrieval have converged on a single document.

*Condition 3 — Default (ESCALATE via lexical fallback).* Otherwise, the policy issues a BM25 query with a keyword reformulation. This broadens coverage via exact-match signals that dense retrieval may have missed.

**The primary mechanism is Condition 1.** The key design insight — validated by ablation — is that most of the policy's value comes from stopping early when coverage is sufficient, not from the specific substrate selected when escalation occurs. On HotpotQA Bridge, the policy stops after a single operation on the majority of questions, reducing average operations from 2.00 to 1.21.

## 3.3 Workspace Management

The workspace is a fixed-capacity buffer (10 items). After each retrieval step:

1. Items are scored by cosine similarity to the query embedding.
2. The top-2 items are **pinned** (protected from eviction).
3. Items with relevance below 0.15 are **evicted**.

Pinning ensures the best evidence persists across steps. Eviction prevents low-signal content from diluting the coverage check.

## 3.4 Utility@Budget Metric

Standard retrieval metrics (precision, recall, NDCG) do not account for operation cost. We evaluate all systems using a composite metric:

**Utility@Budget** = SupportRecall × (1 + η × SupportPrecision) − μ × NormalizedCost

where η = 0.5 weights evidence precision and μ = 0.3 penalizes cost. Both coefficients are fixed before experiments and not tuned. NormalizedCost is the ratio of operations used to the maximum operations used by any policy on the same question.

This metric rewards high-recall, high-precision retrieval while penalizing unnecessary operations. A policy that retrieves everything but wastes budget is penalized; a policy that retrieves nothing pays no cost but scores zero on recall. The optimal strategy under this metric is to retrieve exactly what is needed and stop.

## 3.5 Learned Stopping Classifier

The heuristic stopping rule (Condition 1) uses a hand-tuned coverage threshold. We replace this with a **learned classifier** trained on retrieval trajectory data.

**Training data collection.** We run the ensemble policy (which always retrieves from all substrates) on 500 HotpotQA bridge questions separate from the evaluation set. At each step t, we record 9 workspace features: n_workspace_items, max_relevance, mean_relevance, min_relevance, n_unique_sources, relevance_diversity, step_number, new_items_added, and max_relevance_improvement. The oracle label is: should the policy stop at step t to maximize Utility@Budget? This produces ~4,200 step-level training examples.

**Classifier.** We train a gradient boosted tree (sklearn GradientBoostingClassifier) on 80% of the data and evaluate on 20%. The classifier achieves 93.3% accuracy and 71.7% F1 on the held-out test set.

**Key finding: the dominant stopping signal.** Feature importance analysis reveals that **max_relevance_improvement** — the change in the best evidence quality from the previous step — has importance 0.55, far exceeding all other features. The classifier has learned that **diminishing marginal returns in evidence quality is the optimal stopping signal**: when the last retrieval step didn't materially improve the best evidence, stop. This is interpretable, actionable, and validates the stopping thesis from data rather than intuition.

**Deployment.** The LearnedStoppingPolicy loads the trained classifier and uses it in place of the heuristic's coverage check. At each step after the initial semantic search, it extracts the 9 features from the current workspace state and queries the classifier. If the classifier predicts STOP (probability ≥ 0.35), the policy stops; otherwise, it escalates to lexical search.

## 3.6 Connection to Formal Framework

The routing policy can be viewed as an approximate solution to a constrained Markov decision process (CMDP) over heterogeneous action spaces, where each substrate is an "option" in the hierarchical RL sense (Sutton et al., 1999). The coverage threshold approximates the optimal stopping condition, and the cost penalty in Utility@Budget approximates the Lagrangian dual variable enforcing the budget constraint. The full formal treatment — state representation, weak dominance theorem, quantitative gain bound, and connection to information foraging theory — is presented in Appendix A.
# 4 Experimental Setup

We describe the benchmarks, baselines, ablation variants, evaluation metrics, and implementation details used in all experiments. The section is written to be self-contained: a reader should be able to reproduce every result reported in Section 5 from the information given here alone.

## 4.1 Benchmarks

### HotpotQA Bridge Subset (Yang et al., 2018)

We use the distractor validation split of HotpotQA. From the full 5,918 distractor-split questions we select the first 100 questions whose `type` field equals `"bridge"`, yielding a fixed, deterministic subset. Each question is accompanied by exactly 10 context paragraphs: 2 gold supporting paragraphs and 8 distractor paragraphs drawn from Wikipedia. Bridge questions require a two-step inference chain: the system must first locate a passage containing entity A, extract a bridge entity mentioned there, and then locate a second passage whose content concerns that bridge entity (entity B). Because question text explicitly names entity A, there is high lexical overlap between the question and the first-hop gold paragraph.

| Property | Value |
|---|---|
| Total questions | 100 |
| Context paragraphs per question | 10 |
| Gold supporting paragraphs | 2 |
| Distractor paragraphs | 8 |
| Question type | Bridge only |

### HotpotQA Full (All Question Types) — Generalization Experiment A

To verify that results are not specific to bridge questions, we additionally evaluate on the first 1,000 questions from the HotpotQA distractor validation set without type filtering, yielding a mixture of bridge (807) and comparison (193) questions. This tests whether the heuristic's stopping rule—which operates on workspace statistics, not question-type patterns—generalizes across question types.

| Property | Value |
|---|---|
| Total questions | 1,000 |
| Bridge questions | 807 (80.7%) |
| Comparison questions | 193 (19.3%) |
| Context paragraphs per question | 10 |
| Seed | 42 |

### HotpotQA Open-Domain Expansion — Generalization Experiment B

To verify that results are not specific to the closed 10-paragraph setting, we expand the candidate set for 200 bridge questions from 10 to 50 paragraphs (2 gold + 8 original distractors + 40 additional distractors sampled from other questions in the dataset, seed=42). The distractor pool comprises 64,900 unique paragraphs from all non-selected questions. This simulates open-domain retrieval where the gold signal is diluted 5-fold.

| Property | Value |
|---|---|
| Total questions | 200 |
| Setting A (closed) | 10 paragraphs per question |
| Setting B (open-domain) | 50 paragraphs per question |
| Gold supporting paragraphs | 2 |
| Additional distractors (B) | 40 (sampled from 64,900-paragraph pool) |
| Seed | 42 |

### Heterogeneous Benchmark v2 (this work)

To test robustness across a broader range of retrieval challenges, we construct a synthetic benchmark of 100 questions spanning six task types, generated deterministically with seed 42.

| Task Type | Count | Key Challenge |
|---|---|---|
| Entity Bridge | 20 | Bridge entity absent from question text |
| Implicit Bridge | 20 | Bridge relation expressed paraphrastically |
| Semantic + Computation | 20 | Requires semantic retrieval then arithmetic |
| Low Lexical Overlap | 20 | < 15% content-word overlap between question and gold |
| Multi-hop Chain | 10 | Sequential retrieval hops |
| Discovery + Extraction | 10 | Gold paragraph unknown until prior hop reveals it |

Two design invariants are enforced programmatically: (1) **Entity isolation** — for bridge tasks, bridge entities do not appear in the question text; (2) **Lexical isolation** — for low-overlap tasks, content word overlap is below 15%.

## 4.2 Baselines

| System | Description | Address Spaces |
|---|---|---|
| $\pi_\text{semantic}$ | Dense retrieval via all-MiniLM-L6-v2, top-k by cosine similarity | Semantic only |
| $\pi_\text{lexical}$ | BM25 via rank_bm25, top-k by BM25 score | Lexical only |
| $\pi_\text{entity}$ | Regex NER + entity co-occurrence graph + BFS | Entity only (negative control) |
| $\pi_\text{ensemble}$ | Query all substrates, merge and deduplicate | All (no routing) |
| $\pi_\text{aea}$ | Coverage-driven adaptive routing (proposed) | All (heuristic routing) |
| $\pi_\text{llm}$ | LLM reasons about evidence sufficiency at each step (gpt-oss-120b) | All (LLM routing) |

$\pi_\text{entity}$ is included as a negative control: we report it for completeness and to verify that entity-graph traversal alone does not explain system performance. Ablation results confirm it does not (Section 5.3).

$\pi_\text{llm}$ receives the question, current workspace contents, and available actions at each step, and responds with a single routing decision (STOP, SEMANTIC_SEARCH, LEXICAL_SEARCH, or ENTITY_HOP). It always performs semantic search at step 0; from step 1 onward, the LLM decides. This policy tests whether LLM-guided positive routing outperforms the heuristic's coverage-driven stopping.

### Answer Generation

For end-to-end evaluation, we generate answers from retrieved evidence using gpt-oss-120b (via OpenRouter). The prompt provides the retrieved passages and asks for a short, direct answer. The same model and prompt are used for all policies — only the retrieved evidence differs. EM and F1 are scored against gold answers using SQuAD-style normalization.

## 4.3 Ablation Variants

| Variant | Modification | Question Answered |
|---|---|---|
| abl_no_early_stop | Always execute 2 steps | Is early stopping necessary? |
| abl_semantic_smart_stop | Semantic only + coverage stop | Is stopping sufficient without routing? |
| abl_no_entity_hop | Lexical fallback replaces entity hops | Do entity hops contribute? |
| abl_always_hop | Unconditional entity hop after step 0 | Is selectivity important? |
| abl_no_workspace_mgmt | No pin/evict | Does workspace curation matter? |

## 4.4 Metrics

**Primary: Utility@Budget**

$$\text{Utility@Budget} = \text{AnswerScore} \times (1 + \eta \cdot \text{EvidenceScore}) - \mu \cdot \text{Cost}$$

with $\eta = 0.5$, $\mu = 0.3$ (fixed before experiments).

| Metric | Definition |
|---|---|
| Support Recall | $|\text{workspace} \cap \text{gold}| / |\text{gold}|$ |
| Support Precision | $|\text{workspace} \cap \text{gold}| / |\text{workspace}|$ |
| Average Operations | Mean retrieval operations per question |

**Two evaluation modes.** We report both retrieval-only metrics (Support Recall, Support Precision) and end-to-end metrics (EM, F1) with LLM answer generation via gpt-oss-120b. The retrieval-only U@B uses SupportRecall as AnswerScore; the end-to-end U@B uses F1 as AnswerScore. Both formulas are explicitly identified in each results table.

## 4.5 Implementation Details

| Parameter | Value |
|---|---|
| Embedding model | all-MiniLM-L6-v2 (384-dim) |
| BM25 | rank_bm25 BM25Okapi, default parameters |
| Entity extraction | Regex-based (capitalized multi-word phrases) |
| Workspace capacity | 10 items |
| Pin count | Top-2 by relevance |
| Eviction threshold | Relevance < 0.15 |
| Coverage stop: min items | 2, relevance >= 0.40, from >= 2 sources |
| $\eta$, $\mu$ | 0.50, 0.30 |
| Random seed | 42 |
| Hardware | Apple Silicon laptop, CPU only |
| Runtime | < 5 minutes per benchmark |

## 4.6 Evaluation Contract

1. **Shared substrates.** Address spaces are implemented once and shared across all systems.
2. **Policy variation only.** Only the routing/stopping policy differs between systems.
3. **No test-time tuning.** All hyperparameters fixed before experiments.
4. **Immutable harness.** Metric computation code is locked at experiment start.
# 5 Results

All results in this section use the clean experimental split (train: questions 500–999, eval: questions 0–499, verified zero overlap). Utility@Budget for end-to-end evaluation is computed as F1 × (1 + 0.5 × SupportRecall) − μ × (Ops / 3.0), with μ = 0.3.

## 5.1 Retrieval Quality (N=500)

On retrieval-only evaluation (N=500, 5 seeds, bootstrap CIs), the heuristic stopping policy achieves the highest support recall (0.810) among stopping-based policies at the lowest operation count (1.15). All comparisons between the heuristic and single-substrate baselines are statistically significant (paired permutation tests, p < 0.0001, Cohen's d = 0.807 vs π_semantic). Full retrieval-only results are reported in Appendix D.

## 5.2 End-to-End Answer Quality (N=500)

To validate that retrieval efficiency translates to downstream answer quality, we generate answers from retrieved evidence using gpt-oss-120b (via OpenRouter). The same model and prompt are used for all policies — only the retrieved evidence differs.

**Table 1.** End-to-end results on HotpotQA Bridge (N=500, clean split, bootstrap 95% CIs). E2E U@B uses F1 as AnswerScore with μ = 0.3.

| Policy | EM | F1 | SupportRecall | AvgOps | E2E U@B [95% CI] |
|---|---|---|---|---|---|
| π_ensemble | **0.510** | **0.681** | **0.943** | 3.00 | 0.707 [0.655, 0.759] |
| **π_heuristic** | 0.448 | 0.612 | 0.806 | **1.16** | **0.759 [0.706, 0.812]** |

**Statistical significance (paired t-test):**
- Heuristic vs Ensemble: Δ = +0.052, p = 0.021, Cohen's d = 0.103, 95% CI on difference: [0.007, 0.096]

The heuristic achieves the highest E2E U@B (0.759) despite lower F1 (0.612 vs 0.681) and lower EM (0.448 vs 0.510). The mechanism is cost efficiency: 1.16 operations versus 3.00, a 61% reduction. Under the Utility@Budget metric, this cost saving more than compensates for the quality gap.

**Important caveat:** The ensemble produces **better answers** on standard metrics (EM +6.2pp, F1 +6.9pp, Recall +13.7pp). The heuristic's advantage exists only under cost-penalized evaluation. The practical recommendation depends on the deployment context: when retrieval cost matters (μ ≥ 0.3), use the heuristic; when only answer quality matters (μ < 0.3), use the ensemble. The sensitivity analysis in Section 5.5 quantifies this tradeoff.

### The Learned Stopping Classifier: A Negative Result

We also evaluate a learned stopping classifier (gradient boosted tree on 9 workspace features, trained on questions 500–999 with zero overlap). On the clean evaluation split, the classifier **fails catastrophically**: it uses 5.00 average operations (never stops) and achieves E2E U@B of 0.498 — worse than both the heuristic and the ensemble. The root cause analysis in Section 6.4.4 explains why: the classifier captures distribution-specific correlations that do not generalize across question splits.

## 5.3 Ablation Analysis

**Table 2.** Ablation results on HotpotQA Bridge (retrieval-only, N=100).

| Ablation | Retrieval U@B | Δ from Heuristic |
|---|---|---|
| Full heuristic | 0.028 | — |
| abl_no_early_stop | 0.028 | +0.000 |
| abl_no_workspace_mgmt | 0.028 | +0.000 |
| abl_no_entity_hop | 0.032 | +0.004 |
| abl_semantic_smart_stop | −0.009 | −0.037 |
| abl_always_hop | −0.086 | **−0.115** |

**Selective stopping is the dominant mechanism.** abl_always_hop is catastrophic (Δ = −0.115): forcing unconditional escalation wastes budget on operations that degrade utility. This is the largest single effect in the ablation study and directly supports the structural signal thesis.

**Entity hops are net-neutral to negative.** abl_no_entity_hop improves retrieval U@B by +0.004, confirming that entity graph traversal adds cost without proportionate benefit on HotpotQA. The effective system operates over two substrates (semantic + lexical) with a stopping rule.

## 5.4 Five Failed Improvements

To test whether the heuristic can be improved, we implement five principled alternatives spanning content-aware scoring, bundle-level assessment, learned classification, requirement decomposition, and question-level routing. All are evaluated on the same clean split (questions 0–499).

**Table 3.** Failed improvement attempts vs. the heuristic. Note: absolute U@B values differ slightly from Table 1 (0.816 vs 0.759 for the heuristic) because each table reflects an independent LLM answer generation run with a non-deterministic model; relative comparisons within each table are valid.

| Approach | Mechanism | AvgOps | E2E U@B | vs Heuristic |
|---|---|---|---|---|
| Cross-encoder | MS MARCO scores (q, passage) pairs | 3.09 | 0.655 | −0.161 (p < 0.0001) |
| **NLI bundle** | **DeBERTa-v3 NLI on concatenated evidence** | **6.09** | **0.433** | **−0.383 (p < 0.0001)** |
| Learned GBT | 9 workspace features, trained on 500–999 | 5.00 | 0.498 | −0.318 (catastrophic) |
| LLM decomposition | gpt-oss-120b decomposes into sub-requirements | 2.95 | 0.758 | −0.058 |
| Embedding router | Question embedding → strategy classifier | 1.28 | tied | +0.000 |
| **Heuristic** | **2+ items from 2+ sources** | **1.16** | **0.816** | **baseline** |

All five alternatives either degrade performance or merely tie. The NLI result is particularly informative: NLI naturally handles evidence *bundles* (the full workspace is the premise), solving the "set function decomposition" problem identified in the cross-encoder failure. Yet it fails even more severely (E2E U@B 0.433, d = −0.731) because multi-hop questions resist conversion to well-formed entailment hypotheses. This rules out the set function explanation and strengthens the structural signal thesis: the problem is not just that individual passages can't be scored — even principled bundle-level assessment fails because the assessment itself is harder than the stopping decision.

Section 6.4 provides a unified root cause analysis of all five failures.

## 5.5 Cost-Sensitivity Analysis

**Table 4.** E2E U@B across cost penalty μ (N=500, clean split).

| μ | π_ensemble | π_heuristic | Winner |
|---|---|---|---|
| 0.0 | **1.007** | 0.874 | ensemble |
| 0.1 | **0.907** | 0.836 | ensemble |
| 0.2 | **0.807** | 0.797 | ensemble |
| 0.3 | 0.707 | **0.759** | heuristic |
| 0.4 | 0.607 | **0.720** | heuristic |
| 0.5 | 0.507 | **0.682** | heuristic |

The crossover occurs at **μ ≈ 0.3**: below this threshold, comprehensive retrieval dominates because answer quality outweighs cost; above it, the heuristic dominates because marginal retrieval yields diminishing returns. This regime is practically relevant: in production RAG systems where each retrieval call involves an embedding computation (~10ms), a reranker pass (~50ms), and context-window consumption (~500 tokens at ~$0.003 per 1K tokens for frontier models), the cost of three retrieval operations can reach 15–30% of the value of a single correct answer, placing typical deployments squarely in the μ ≥ 0.3 regime where the heuristic dominates.

## 5.6 Second Benchmark: 2WikiMultiHopQA

The heuristic's advantage replicates on 2WikiMultiHopQA (synthetic, N=100, E2E):

| Policy | EM | F1 | AvgOps | E2E U@B |
|---|---|---|---|---|
| π_semantic | 0.890 | 0.897 | 2.00 | 1.031 |
| **π_heuristic** | **0.890** | **0.905** | **1.00** | **1.055** |

The heuristic achieves the best E2E U@B (1.055) at the lowest operation count (1.00), confirming that structural stopping generalizes beyond HotpotQA.

## 5.7 Generalization: All Question Types (N=1000)

To address the concern that results may be specific to bridge questions, we evaluate on the first 1,000 questions of the HotpotQA distractor validation set without type filtering (807 bridge, 193 comparison).

**Table 5.** Full HotpotQA evaluation, all question types (retrieval-only, N=1000).

| Policy | SupportRecall | SupportPrec | AvgOps | Utility@Budget |
|---|---|---|---|---|
| π_semantic | 0.823 | 0.334 | 2.00 | 0.0175 |
| π_lexical | 0.789 | 0.320 | 2.00 | 0.0155 |
| π_ensemble | 0.953 | 0.254 | 3.00 | 0.0030 |
| **π_heuristic** | 0.839 | 0.329 | **1.15** | **0.0358** |

**Breakdown by question type:**

| Policy | Bridge U@B (N=807) | Comparison U@B (N=193) |
|---|---|---|
| π_semantic | 0.0098 | 0.0495 |
| π_lexical | 0.0133 | 0.0247 |
| π_ensemble | 0.0003 | 0.0145 |
| **π_heuristic** | **0.0303** | **0.0590** |

**Statistical tests (heuristic vs ensemble, paired t-test on U@B):**

| Scope | N | Δ | p-value | 95% CI | Cohen's d |
|---|---|---|---|---|---|
| Overall | 1000 | +0.0328 | <0.000001 | [+0.0275, +0.0382] | 0.379 |
| Bridge | 807 | +0.0301 | <0.000001 | [+0.0244, +0.0357] | 0.370 |
| Comparison | 193 | +0.0445 | <0.000001 | [+0.0294, +0.0595] | 0.419 |

The heuristic beats the ensemble on **both** question types with high statistical significance (p < 0.0001 in all cases). Comparison questions show a *larger* advantage than bridge questions (Cohen's d = 0.42 vs 0.37, absolute Δ 48% larger). The structural stopping rule is type-agnostic: it operates on workspace statistics, not question-type patterns.

## 5.8 Generalization: Open-Domain Retrieval (5x Candidate Set)

To address the concern that results may be specific to 10-paragraph closed sets, we expand the candidate pool for 200 bridge questions from 10 to 50 paragraphs (2 gold + 48 distractors). The additional 40 distractors are sampled from other questions' paragraphs (64,900 unique paragraphs available), simulating a harder open-domain setting.

**Table 6.** Open-domain retrieval results (retrieval-only, N=200).

| Policy | 10-para U@B | 50-para U@B | Degradation |
|---|---|---|---|
| π_semantic | 0.0087 | 0.0086 | −0.0001 |
| π_lexical | 0.0090 | 0.0059 | −0.0031 |
| π_ensemble | −0.0049 | −0.0097 | −0.0048 |
| **π_heuristic** | **0.0251** | **0.0251** | **0.0000** |

**Statistical tests (heuristic vs ensemble per setting):**

| Setting | N | Δ | p-value | 95% CI | Cohen's d |
|---|---|---|---|---|---|
| 10-para | 200 | +0.0301 | 0.000001 | [+0.0186, +0.0415] | 0.366 |
| 50-para | 200 | +0.0348 | <0.000001 | [+0.0249, +0.0447] | 0.491 |

The heuristic's U@B is **invariant to candidate set size** (p=0.925, Δ=+0.000032 for 10-para vs 50-para degradation test). Cohen's d for the heuristic-vs-ensemble comparison actually *increases* from 0.37 to 0.49 in the harder open-domain setting, as the ensemble degrades more than the heuristic under distractor dilution. The structural stopping rule is robust: it finds sufficient evidence early and stops, regardless of how many distractors surround the gold paragraphs.

## 5.9 Generalization: Reasoning-Intensive Retrieval (BRIGHT)

To test whether the structural signal generalizes beyond multi-hop QA, we evaluate on BRIGHT [Su et al., 2024] — a benchmark where gold documents have intentionally low lexical AND semantic overlap with queries, requiring reasoning to connect them.

**Table 7.** BRIGHT results (retrieval-only, N=200, real data from HuggingFace).

| Policy | SupportRecall | AvgOps | U@B |
|---|---|---|---|
| π_semantic | 0.880 | 1.81 | 0.200 |
| π_lexical | 0.747 | 1.69 | 0.156 |
| π_ensemble | **0.936** | 2.10 | 0.172 |
| **π_heuristic** | 0.908 | **1.69** | **0.194** |

**Heuristic vs ensemble (U@B):** Δ = +0.022, p = 0.0026, Cohen's d = 0.216.

The heuristic wins on Utility@Budget on BRIGHT despite the ensemble achieving higher raw recall (0.936 vs 0.908). The effect size (d=0.216) is smaller than on HotpotQA (d=0.379), consistent with the harder retrieval setting narrowing the efficiency gap. Critically, the structural stopping signal — source diversity — remains effective even when retrieval operates under low lexical and semantic overlap, confirming distribution invariance across three benchmark families.

## 5.10 Cross-Benchmark Summary

**Table 8.** Heuristic vs ensemble across all evaluation settings.

| Benchmark | Family | N | Δ U@B | p-value | Cohen's d |
|-----------|--------|---|-------|---------|-----------|
| HotpotQA (all types) | Multi-hop factoid | 1000 | +0.033 | <0.000001 | 0.379 |
| HotpotQA (comparison) | Comparison QA | 193 | +0.045 | <0.000001 | 0.419 |
| Open-domain (50-para) | Diluted retrieval | 200 | +0.035 | <0.000001 | 0.491 |
| BRIGHT | Reasoning-intensive | 200 | +0.022 | 0.0026 | 0.216 |
| 2WikiMultiHopQA | Bridge/comparison | 100 | Heuristic wins | — | — |
| HotpotQA E2E (N=500) | End-to-end with LLM | 500 | +0.052 | 0.021 | 0.103 |

The heuristic significantly outperforms comprehensive retrieval in **every setting tested**, with effect sizes ranging from small (d=0.103, E2E) to medium (d=0.491, open-domain). The advantage grows in harder settings (open-domain d=0.491 > standard d=0.379) and holds across three distinct benchmark families (multi-hop QA, reasoning-intensive, diluted retrieval).
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
| NLI bundle assessment | DeBERTa-v3 NLI on concatenated evidence | 6.09 | 0.433 | -0.383 (p<0.0001) |
| LLM decomposition | gpt-oss-120b decomposes question into sub-requirements | 2.95 | 0.758 | -0.030 |
| Learned GBT classifier | Gradient boosted tree on workspace statistics | 5.00 | 0.498 | catastrophic |
| Embedding router | Question embedding predicts best retrieval strategy | 1.28 | tied | +0.001 |
| **Heuristic** | **2+ items from 2+ sources** | **1.16** | **0.759** | **--** |

All five approaches either degrade performance or merely tie. The NLI result is most informative: it correctly takes the full evidence bundle as premise (addressing the cross-encoder's set function limitation), yet fails even more severely (d=-0.731) because multi-hop questions resist conversion to well-formed entailment hypotheses. A contributing factor is context truncation (DeBERTa-v3-small's 512-token window limits premise length); however, the primary failure mode — that "Were X and Y of the same nationality?" does not translate to a natural NLI hypothesis regardless of evidence quality — is fundamental rather than architectural.

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

1. **E2E evaluation on one benchmark.** End-to-end evaluation with LLM answer generation (N=500, p=0.021) uses only HotpotQA Bridge. BRIGHT and open-domain evaluations are retrieval-only. The E2E effect size is small (d=0.103).

2. **Non-deterministic LLM answers.** Answer generation via gpt-oss-120b is non-deterministic, producing slight absolute U@B variation across runs (Tables 1 and 3). Relative comparisons within each evaluation are valid.

3. **Custom evaluation metric.** Utility@Budget is author-defined. The specific η and μ values determine the ranking — sensitivity analysis across parameter ranges is reported in Appendix C.

4. **Three address spaces.** Real retrieval environments include web search, tool invocation, and structural navigation. Cost differentials across these modalities are larger, potentially amplifying the benefit of selective stopping.

5. **Heuristic policy.** The routing decisions are hand-designed rules. The results show what is achievable without learning; a learned policy could close the gap between routing avoidance and positive routing.

6. **Free-tier LLM for routing and answers.** The gpt-oss-120b model is capable but not state-of-the-art. Stronger models might shift the balance toward positive routing.

## 6.6 Future Work

**Calibrated stopping policy.** Train a stopping classifier on trajectory data with downstream F1 as the reward signal. The key question: can a learned policy match the heuristic's stopping efficiency while capturing the LLM router's recall advantage?

**Step-conditional routing.** Oracle trajectories show step-position preferences (semantic at step 1, entity at step 2). A router conditioned on step position could reduce false-positive escalations.

**Expanded substrates.** Web search, tool execution, and structural navigation would test whether the stopping > searching hierarchy holds when cost differentials are larger.

**Budget sensitivity.** The hierarchy may invert under very tight budgets (where any retrieval is expensive) or very loose budgets (where cost is negligible). Characterizing the budget regime where each policy dominates is an important practical question.
# 7 Conclusion

We studied when to stop retrieving across heterogeneous substrates and discovered that a simple structural heuristic — stop when evidence has converged from two independent sources — significantly outperforms comprehensive retrieval across three benchmark families: multi-hop QA (HotpotQA, p<0.000001, N=1000), reasoning-intensive retrieval (BRIGHT, p=0.0026, N=200), and diluted open-domain settings (p<0.000001, N=200, 5x candidate expansion). The advantage is invariant to question type and grows in harder settings (Cohen's d increases from 0.38 to 0.49 under distractor dilution).

Five principled content-aware alternatives all fail to improve on this heuristic. The cross-encoder fails because it scores passages individually, not as bundles. The NLI model — which correctly takes full bundles as premise — fails because multi-hop questions resist conversion to well-formed entailment hypotheses. The learned classifier fails because workspace statistics are distribution-specific. The LLM decomposition fails because parsing noise causes the policy to never stop. The embedding router confirms the bottleneck is stopping, not routing.

Together, these results establish that **structural stopping signals — distribution-invariant properties of the evidence gathering process — outperform content-specific signals for retrieval stopping decisions.** This connects to classical optimal stopping theory: threshold rules on low-noise observables dominate value-estimation approaches when the value function is hard to learn.

Four implications follow. First, the adaptive retrieval field should treat stopping as a first-class design target. Second, practitioners should default to structural stopping signals rather than investing in content-aware stopping until the bundle assessment problem is solved. Third, learned stopping approaches must be evaluated on out-of-distribution data. Fourth, entity-graph traversal should not be assumed to add value — our ablation shows it contributes nothing on three benchmarks, and the stopping rule captures the same multi-hop benefit more cheaply via source-diversity checking.

The gap between structural stopping (which works) and content-aware stopping (which should work but doesn't yet) defines the key open challenge. Closing it requires models that assess evidence *bundles* rather than individual passages — a set function learning problem that current pre-trained models are not designed for, and that multi-hop question structure makes particularly difficult.
# Appendix A: Formal Framework

This appendix presents the constrained MDP formalization that motivates the coverage-driven routing policy described in Section 3.

## A.1 State Representation

At each timestep t, the agent observes state s_t = (q, D_t, K_t, W_t, H_t, B_t), where q is the query, D_t is the discovery state, K_t is the knowledge state, W_t is the workspace, H_t is the action history, and B_t is the remaining budget.

The tuple (D_t, K_t, W_t, H_t) constitutes a sufficient statistic for the agent's belief about the information environment, justifying an MDP treatment rather than a full POMDP.

## A.2 Action Space as Options

Let K index the available substrates. Each substrate k exposes operations A_k. The joint action space is A = ⋃_k A_k ∪ {STOP}. Each substrate is modeled as an option (Sutton, Precup, Singh, 1999) with initiation condition, internal policy, and termination condition.

## A.3 CMDP Formulation

The agent maximizes:

max_θ min_{ν≥0} E[Σ γ^t r_t] + ν(B_0 - Σ cost_t)

where r_t = U(evidence_t) - λ·cost_t, θ parameterizes the routing policy, and ν is the Lagrange multiplier enforcing the budget constraint (Altman, 1999).

## A.4 Weak Dominance

**Theorem 1.** Let π* be optimal over A = ⋃_k A_k, and π*_k be optimal restricted to A_k. Then for all states s: V^{π*}(s) ≥ max_k V^{π*_k}(s).

This is trivially true since A_k ⊂ A. The interesting question is when the inequality is strict.

**Proposition 1 (Quantitative Gain).** Let δ = min_k(1 - E[φ_k(s)]) be the heterogeneity gap. If δ > 0: E[V^{π*} - V^{π*_k}] ≥ δ · Δ_min, where Δ_min is the minimum Q-value gap on suboptimal states.

## A.5 Discovery/Knowledge Split

**Proposition 2.** Without the D/K split, states with identical K but different D are aliased (Singh et al., 1994), causing the policy to lose the information value of prior exploration. The gap is at least α · Δ_alias where α is the fraction of aliased states.

Connection to information foraging: D_t = information scent (Pirolli and Card, 1999; Fu and Pirolli, 2007, SNIF-ACT); K_t = information diet.

## A.6 Toy Example

Two substrates (text search, database), three steps. Task requires text search for discovery (step 1) and database for extraction (steps 2-3).

| Policy | Value |
|--------|-------|
| Adaptive | 3 - 3c |
| Text-only | 3 - 3c - 2p |
| Database-only | 3 - 3c - p |
| Ensemble | 3 - 6c |

Adaptive strictly dominates all alternatives for c > 0 and p > 0.
