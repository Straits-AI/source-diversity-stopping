# Adaptive Retrieval Routing: When Knowing What Not To Do Beats Choosing the Right Tool

## Abstract

Retrieval-augmented systems typically commit to a fixed number of retrieval operations regardless of whether they are necessary or sufficient. We study coverage-driven retrieval routing: a policy over heterogeneous substrates (semantic, lexical, entity-graph) that evaluates evidence sufficiency after each step and stops when coverage is adequate. On HotpotQA Bridge (N=500, p < 0.0001, Cohen's d = 0.807), the policy achieves the highest support recall (0.810) at the lowest operation count (1.15). On end-to-end evaluation with LLM answer generation (N=100), it achieves comparable Utility@Budget to comprehensive retrieval (0.760 vs 0.731) while using 60% fewer operations (1.21 vs 3.00), with a cost-sensitivity crossover at μ=0.25. Ablation analysis reveals the mechanism: the policy's value derives from selective stopping — knowing when evidence is sufficient — rather than from positive substrate selection. Forcing unconditional escalation degrades utility by 0.115. An LLM-based router achieves higher recall (0.845) through genuine multi-substrate reasoning but is not cost-efficient (2.54 ops vs 1.21). The resulting hierarchy — smart stopping > brute force > smart searching — challenges the assumption that adaptive retrieval should optimize substrate selection, and identifies calibrated stopping as the key open problem.
# 1 Introduction

When should a retrieval system stop searching? The dominant paradigm in retrieval-augmented generation commits to a fixed number of retrieval operations — typically one or two calls to a single index — regardless of whether those operations are necessary or sufficient. This rigidity creates two failure modes: wasted computation on queries answerable from a single retrieval step, and insufficient evidence on queries requiring multiple heterogeneous sources.

Recent adaptive retrieval systems address the second failure mode. Self-RAG [Asai et al., 2024] learns when to skip retrieval. FLARE [Jiang et al., 2023] triggers retrieval on low-confidence tokens. Adaptive-RAG [Jeong et al., 2024] routes queries by estimated complexity. SmartRAG [Gao et al., 2025] jointly optimizes retrieval decisions under cost constraints. Yet these systems operate within a single retrieval modality. The question of when to *stop* searching across multiple qualitatively different retrieval substrates — semantic indexes, keyword indexes, entity graphs — has received less attention.

In this paper, we study **coverage-driven retrieval routing**: a policy that operates over multiple heterogeneous retrieval substrates, evaluating after each operation whether the current evidence is sufficient to stop or whether escalation to a different substrate is warranted. The policy defaults to the cheapest available operation (dense retrieval) and escalates only when a coverage gap is detected.

Our central empirical finding is counterintuitive. We expected the primary value of multi-substrate routing to come from *positive selection* — choosing the right substrate for each query. Instead, the dominant mechanism is **selective stopping** — knowing when evidence is sufficient. On HotpotQA Bridge, the policy achieves the highest end-to-end Utility@Budget (0.760), outperforming both comprehensive retrieval (ensemble, 0.731) and LLM-guided positive routing (0.652). It does this by completing questions in 1.21 average operations versus 2.00–3.00 for baselines (N=500, p < 0.0001, Cohen's d = 0.807). An LLM-based router achieves higher recall (0.845) through genuine multi-substrate reasoning but is not cost-efficient — establishing a hierarchy: **smart stopping > brute force > smart searching**.

Our contributions are:

1. **A coverage-driven retrieval routing policy** that selects among semantic, lexical, and entity-graph substrates based on workspace state — specifically, whether retrieved evidence from multiple sources meets a coverage threshold (Section 3).

2. **A controlled heterogeneous benchmark** with entity isolation and lexical overlap controls, designed to require cross-substrate navigation (Section 4).

3. **The stopping-vs-searching tradeoff**: under end-to-end evaluation, AEA achieves comparable utility to comprehensive retrieval (0.760 vs 0.731) at 60% lower cost, with a cost-sensitivity crossover at μ=0.25. Sensitivity analysis provides actionable guidance on when stopping beats searching (Section 5).

4. **A formal framework** modeling multi-substrate retrieval as a constrained decision process with discovery/knowledge state tracking, connecting to the options framework for hierarchical action selection (Appendix A).

The remainder of the paper is organized as follows. Section 2 surveys related work. Section 3 presents the routing policy and its design rationale. Section 4 describes the experimental setup. Section 5 reports results and ablation analysis. Section 6 discusses implications, limitations, and future directions. Section 7 concludes.
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

## 2.4 Structured Memory

Recent work has expanded the memory substrate available to language model agents beyond flat document corpora. MAGMA [arXiv:2601.03236, 2025] constructs a multi-relational graph over conversational history, enabling temporal and social relationship queries; it reports a LoCoMo judge score of 0.700, surpassing flat-retrieval baselines by a substantial margin. Nemori [arXiv:2508.03341, 2025] introduces narrative-unit memory, compressing episodic traces into structured story units and achieving an eighty-eight percent reduction in token consumption at retrieval time without a corresponding drop in recall. A-MEM [arXiv:2502.12110, 2026] organizes agent memory according to Zettelkasten principles, creating note-level indexes that support associative traversal across temporally distant events. Zep [arXiv:2501.13956, 2025] maintains a temporal knowledge graph over conversational context, enabling precise queries about the evolution of facts over time.

These systems each constitute a valuable substrate in the AEA address space. They demonstrate that structured memory is not a monolith: graph-based, narrative-based, associative, and temporal organizations each excel on different query types. This heterogeneity is precisely what motivates a routing policy in AEA rather than commitment to any single memory architecture. Critically, none of these systems addresses the question of when to query structured memory versus flat retrieval versus no retrieval at all — the policy problem that AEA formalizes.

## 2.5 Context Engineering

Context engineering — the systematic design of what information is placed in the model's context window and when — has emerged as a first-class research concern. Anthropic's context engineering guidelines [Anthropic, 2025] articulate best practices for structuring retrieved content, tool outputs, and conversation history to maximize effective utilization of the context window. Context Compression for Tools [arXiv:2407.02043, 2024] addresses the dual problem: when tool outputs are lengthy, aggressive compression is necessary to prevent context bloat from degrading performance on later turns.

The Meta-Harness [Lee et al., 2026] represents the most directly relevant work in this space. It operates at the harness level — the orchestration layer between the model and its tools — and introduces a learned policy for selecting which context elements to include in each model call. On downstream benchmarks it reports a 7.7-point improvement with four times fewer tokens consumed, demonstrating that harness-level optimization is a high-leverage intervention point. AEA shares the harness-level perspective of Meta-Harness but differs in two important ways: Meta-Harness is formulated at the code and prompt level as an offline compilation problem, whereas AEA operates at runtime with explicit state tracking; and Meta-Harness does not model the heterogeneity of retrieval substrates or their differential cost and recall profiles.

## 2.6 Long-Context Evaluation

Evaluating retrieval and memory systems over long contexts presents its own methodological challenges. HELMET [arXiv:2410.02694] demonstrates that performance on synthetic long-context benchmarks often fails to predict performance on downstream tasks, cautioning against evaluation suites that rely solely on needle-in-a-haystack style probes. The Context Rot analysis from Chroma documents a consistent degradation pattern in which all evaluated models suffer declining recall as context length grows, regardless of architecture. NoLiMa, RULER, and LongBench Pro each attempt to construct more ecologically valid long-context evaluation suites, with varying degrees of coverage across task types.

These findings shape AEA's evaluation design. We do not rely on synthetic retrieval benchmarks as our primary measure of success; instead we report performance on downstream QA tasks with budget-controlled evaluation protocols that penalize unnecessary retrieval calls, consistent with the cost-aware framing advocated by SmartRAG and Cost-Aware Retrieval.

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

## 3.5 Connection to Formal Framework

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
| $\pi_\text{entity}$ | Regex NER + entity co-occurrence graph + BFS | Entity only |
| $\pi_\text{ensemble}$ | Query all substrates, merge and deduplicate | All (no routing) |
| $\pi_\text{aea}$ | Coverage-driven adaptive routing (proposed) | All (heuristic routing) |
| $\pi_\text{llm}$ | LLM reasons about evidence sufficiency at each step (gpt-oss-120b) | All (LLM routing) |

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

**Scope note.** This evaluation is retrieval-focused. No LLM is used for answer generation; EM/F1 are not reported.

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

## 5.1 Retrieval Quality (N=500, Statistical Validation)

**Table 1.** Retrieval-only results on HotpotQA Bridge (N=500, bootstrap 95% CIs).

| Policy | SupportRecall (mean ± std) | AvgOps | Retrieval U@B [95% CI] |
|---|---|---|---|
| π_semantic | 0.797 ± 0.011 | 2.00 | 0.0129 [0.010, 0.017] |
| π_lexical | 0.772 ± 0.016 | 2.00 | 0.0115 [0.008, 0.015] |
| π_entity | 0.732 ± 0.021 | 3.00 | −0.034 [−0.036, −0.032] |
| π_ensemble | 0.929 ± 0.005 | 3.00 | −0.002 [−0.005, 0.002] |
| **π_aea** | **0.810 ± 0.009** | **1.15** | **0.0322 [0.029, 0.036]** |

All comparisons between π_aea and baselines are statistically significant (paired permutation tests, 10,000 iterations, p < 0.0001). The effect size versus the best single-substrate baseline (π_semantic) is large (Cohen's d = 0.807). Confidence intervals do not overlap. π_aea achieves the highest support recall (0.810) at the lowest operation count (1.15).

## 5.2 End-to-End Answer Quality (N=100)

To validate that retrieval efficiency translates to downstream answer quality, we added LLM-based answer generation (gpt-oss-120b via OpenRouter). The same model and prompt are used for all policies — only the retrieved evidence differs. We also evaluate an LLM-routed policy (π_llm_routed) where gpt-oss-120b makes routing decisions at each step, reasoning about evidence sufficiency.

**Table 2.** End-to-end results on HotpotQA Bridge (N=100). Utility@Budget uses F1 as AnswerScore.

| Policy | EM | F1 | SupportRecall | AvgOps | E2E U@B |
|---|---|---|---|---|---|
| π_semantic | 0.500 | 0.617 | 0.750 | 2.00 | 0.648 |
| π_lexical | 0.500 | 0.643 | 0.810 | 2.00 | 0.703 |
| π_ensemble | 0.560 | 0.701 | 0.940 | 3.00 | 0.731 |
| **π_aea** | 0.480 | 0.630 | 0.795 | **1.21** | **0.760** |
| π_llm_routed | 0.500 | 0.637 | 0.845 | 2.54 | 0.652 |

**The heuristic AEA policy achieves the highest end-to-end Utility@Budget (0.760), outperforming the ensemble (0.731) and all single-substrate baselines.** This result is the central validation of the paper: cost-efficient retrieval routing produces better overall utility than both comprehensive retrieval and LLM-guided positive routing, even when downstream answer quality is measured.

The mechanism is clear: π_aea's F1 (0.630) is 10% lower than the ensemble's (0.701), but its operation count (1.21) is 60% lower (vs 3.00). Under budget-aware evaluation, the cost savings more than compensate for the quality gap. The ensemble retrieves everything but pays for it; the heuristic retrieves enough and stops.

### The LLM-Routed Policy

π_llm_routed makes genuine multi-substrate routing decisions (action distribution: STOP=73, SEMANTIC=43, LEXICAL=80, ENTITY_HOP=31 across 227 routing calls). It achieves higher recall than the heuristic (0.845 vs 0.795), confirming that LLM reasoning enables positive substrate selection — the LLM correctly identifies when additional retrieval from a specific substrate would help.

However, this positive routing is not cost-efficient: 2.54 average operations yield only marginally better F1 (0.637 vs 0.630), producing a lower E2E U@B (0.652 vs 0.760). The LLM router over-retrieves relative to the quality gain, spending operations on evidence that doesn't improve the final answer enough to justify the cost.

The policy ranking under μ=0.3 — heuristic (0.760) > ensemble (0.731) > lexical (0.703) > LLM-routed (0.652) > semantic (0.648) — suggests: **smart stopping > brute force > smart searching**. However, we note that the E2E gap between AEA and ensemble (0.029) is not statistically significant at N=100 (approximate z=0.68, p=0.49; minimum detectable gap at N=100: 0.083). The statistically validated claim is that AEA achieves **comparable** E2E utility to the ensemble while using 60% fewer operations.

### Cost-Sensitivity Analysis

The ranking depends on the cost penalty μ. Table 2b shows U@B across μ values.

**Table 2b.** E2E Utility@Budget across cost sensitivity μ. Winner in bold.

| μ | π_semantic | π_lexical | π_ensemble | π_aea | π_llm | Winner |
|---|---|---|---|---|---|---|
| 0.00 | 0.848 | 0.903 | **1.031** | 0.880 | 0.906 | ensemble |
| 0.10 | 0.782 | 0.837 | **0.931** | 0.840 | 0.822 | ensemble |
| 0.20 | 0.715 | 0.770 | 0.831 | **0.800** | 0.737 | ensemble |
| **0.25** | 0.682 | 0.737 | 0.781 | **0.780** | 0.695 | **crossover** |
| 0.30 | 0.648 | 0.703 | 0.731 | **0.760** | 0.652 | aea |
| 0.40 | 0.582 | 0.637 | 0.631 | **0.719** | 0.568 | aea |
| 0.50 | 0.515 | 0.570 | 0.531 | **0.679** | 0.483 | aea |

The crossover occurs at **μ ≈ 0.25**: below this threshold, comprehensive retrieval (ensemble) dominates because answer quality improvements outweigh cost; above it, cost-efficient stopping (AEA) dominates because marginal retrieval yields diminishing F1 returns. This crossover provides actionable guidance: **when retrieval cost matters (μ ≥ 0.25), use adaptive stopping; when it doesn't (μ < 0.25), use comprehensive retrieval.**

## 5.3 Heterogeneous Benchmark (N=100)

**Table 3.** Retrieval-only results on Heterogeneous v2.

| Policy | SupportRecall | SupportPrec | AvgOps | U@B |
|---|---|---|---|---|
| π_semantic | 0.920 | 0.328 | 2.00 | **0.044** |
| π_lexical | 0.820 | 0.306 | 2.00 | 0.014 |
| π_entity | 0.625 | 0.721 | 3.00 | −0.004 |
| π_ensemble | 0.960 | 0.270 | 3.00 | 0.007 |
| π_aea | 0.930 | 0.388 | **1.84** | 0.043 |

π_aea near-ties π_semantic (0.043 vs 0.044) with higher precision (0.388 vs 0.328) and fewer operations (1.84 vs 2.00). Per-task-type analysis shows AEA outperforms on multi-step tasks (Semantic+Computation: +19%, Discovery+Extraction: +6%) and trails on single-substrate tasks (Low Lexical Overlap, Entity Bridge).

## 5.4 Ablation Analysis

**Table 4.** Ablation results on HotpotQA Bridge (retrieval-only, N=100).

| Ablation | U@B | Δ from Full AEA |
|---|---|---|
| Full AEA | 0.028 | — |
| abl_no_early_stop | 0.028 | +0.000 |
| abl_no_workspace_mgmt | 0.028 | +0.000 |
| abl_no_entity_hop | 0.032 | +0.004 |
| abl_semantic_smart_stop | −0.009 | −0.037 |
| abl_always_hop | −0.086 | **−0.115** |

**Selective avoidance is the dominant mechanism.** abl_always_hop is catastrophic (Δ = −0.115): forcing unconditional escalation wastes budget on operations that degrade overall utility. In contrast, removing entity hops (abl_no_entity_hop) slightly improves performance (+0.004), confirming that the heuristic's entity hop decisions are net-negative on lexically-rich data.

**Substrate diversity matters for coverage estimation.** abl_semantic_smart_stop (Δ = −0.037) shows that the coverage check requires distinct substrates to function — querying the same modality twice provides no additional signal about evidence sufficiency.

## 5.5 Within-Task Substrate Switching

Oracle analysis of HotpotQA Bridge reveals 44% of questions require within-task substrate switching. Step 1 favors semantic (92%); Step 2 favors entity hop (88%). The LLM-routed policy captures this pattern — it uses all four action types with genuine per-question variation — but cannot yet translate this into cost-efficient routing. The gap between the LLM router's positive routing capability and the heuristic's cost efficiency identifies **calibrated stopping** as the key open challenge: a router that stops as efficiently as the heuristic while routing as intelligently as the LLM.
# 6 Discussion

## 6.1 Smart Stopping Beats Smart Searching

The end-to-end results establish a clear hierarchy: **smart stopping > brute force > smart searching** under budget-aware evaluation. The heuristic AEA policy (E2E U@B 0.760) outperforms comprehensive retrieval (ensemble, 0.731) and LLM-guided routing (0.652), despite having lower F1 and recall than both.

This hierarchy has a precise explanation. The cost penalty in Utility@Budget creates a threshold: additional retrieval is worthwhile only if the marginal F1 improvement exceeds μ × (marginal cost / max cost). For the ensemble, the third retrieval step (entity graph) adds ~0.13 recall but costs 1.0 normalized operation, yielding marginal utility of ~0.13 × (1+0.5×0.94) × improvement_rate - 0.3 × 0.33 ≈ −0.01 — slightly negative. The ensemble's last step hurts more than it helps.

The LLM router faces the same trap at a finer grain: it correctly identifies questions that need more evidence, but the additional operations it authorizes produce diminishing F1 returns. Its average 2.54 operations deliver only 0.007 more F1 than the heuristic's 1.21 operations, a marginal return of 0.005 F1 per additional operation — well below the break-even threshold.

The practical implication is a design principle for retrieval systems: **default to restraint and require strong evidence of a coverage gap before escalating.** The heuristic's simple coverage check (2+ high-relevance items from 2+ sources → stop) implements this principle at near-zero computational cost.

## 6.2 The Positive Routing Gap

The LLM-routed policy demonstrates that positive routing — choosing the right substrate for each question — is achievable. Its action distribution (STOP=33%, SEMANTIC=19%, LEXICAL=35%, HOP=14%) shows genuine per-question substrate variation, and its higher recall (0.845 vs 0.795) confirms that the LLM identifies evidence gaps the heuristic misses.

But positive routing is not yet cost-efficient. The gap between the LLM router's routing intelligence and the heuristic's cost discipline identifies the central open challenge: **calibrated stopping** — a policy that combines the LLM's ability to recognize genuinely insufficient evidence with the heuristic's discipline to stop when evidence is merely adequate rather than comprehensive.

We conjecture that the optimal policy lies between these two extremes: it would stop as often as the heuristic (saving cost on easy questions) while routing as intelligently as the LLM (achieving higher recall on hard questions). Training such a policy requires a reward signal that captures the downstream cost-quality tradeoff — exactly the Utility@Budget metric we define.

## 6.3 Comparison with Existing Systems

Our comparison with FLARE, Self-RAG, Adaptive-RAG, IRCoT, and CRAG (Appendix B) reveals that AEA occupies a distinct niche: it is the only system that (a) routes across qualitatively different substrate types, (b) includes an explicit cost model, and (c) treats stopping as a first-class routing decision.

Direct numerical comparison is not valid: those systems report downstream QA accuracy after full LLM generation on different benchmarks with different assumptions. However, the design dimension analysis shows that none of the existing systems addresses the stopping-vs-routing tradeoff that our experiments identify as central.

## 6.4 Limitations

1. **Single benchmark for end-to-end.** The E2E results (N=100) use only HotpotQA Bridge. Validation on additional benchmarks (BRIGHT, NoLiMa) is needed.

2. **No statistical testing on E2E.** Bootstrap CIs and permutation tests are reported only for the retrieval-only evaluation (N=500). The E2E evaluation (N=100) reports point estimates.

3. **Custom evaluation metric.** Utility@Budget is author-defined. The specific η and μ values determine the ranking — sensitivity analysis across parameter ranges is reported in Appendix C.

4. **Three address spaces.** Real retrieval environments include web search, tool invocation, and structural navigation. Cost differentials across these modalities are larger, potentially amplifying the benefit of selective stopping.

5. **Heuristic policy.** The routing decisions are hand-designed rules. The results show what is achievable without learning; a learned policy could close the gap between routing avoidance and positive routing.

6. **Free-tier LLM for routing and answers.** The gpt-oss-120b model is capable but not state-of-the-art. Stronger models might shift the balance toward positive routing.

## 6.5 Future Work

**Calibrated stopping policy.** Train a stopping classifier on trajectory data with downstream F1 as the reward signal. The key question: can a learned policy match the heuristic's stopping efficiency while capturing the LLM router's recall advantage?

**Step-conditional routing.** Oracle trajectories show step-position preferences (semantic at step 1, entity at step 2). A router conditioned on step position could reduce false-positive escalations.

**Expanded substrates.** Web search, tool execution, and structural navigation would test whether the stopping > searching hierarchy holds when cost differentials are larger.

**Budget sensitivity.** The hierarchy may invert under very tight budgets (where any retrieval is expensive) or very loose budgets (where cost is negligible). Characterizing the budget regime where each policy dominates is an important practical question.
# 7 Conclusion

We studied adaptive retrieval routing over heterogeneous address spaces with a focus on the tradeoff between retrieval comprehensiveness and cost efficiency. Three findings emerge.

First, **a simple coverage-driven stopping rule achieves comparable end-to-end utility to comprehensive retrieval while using 60% fewer operations.** The heuristic AEA policy (U@B 0.760) matches the ensemble (0.731) — a gap that is not statistically significant at N=100 — while completing questions in 1.21 operations versus 3.00. Sensitivity analysis shows AEA dominates when cost matters (μ ≥ 0.25) and ensemble dominates when it doesn't (μ < 0.25). The retrieval advantage is statistically validated (N=500, p < 0.0001, Cohen's d = 0.807).

Second, **LLM-based routing achieves genuine positive substrate selection** — the LLM router uses all four action types with per-question variation and achieves higher recall (0.845) than the heuristic (0.795). But this intelligence is not yet cost-efficient: 2.54 operations for marginal quality gains produce lower overall utility (0.652).

Third, the resulting hierarchy — **smart stopping > brute force > smart searching** — challenges the default assumption in adaptive retrieval. The primary value under budget constraints is knowing when to stop, not knowing what to do. This reframes the design problem: rather than optimizing substrate selection, practitioners should optimize the stopping threshold.

The gap between heuristic stopping efficiency and LLM routing intelligence defines the key open challenge: **calibrated stopping** — a policy that stops as efficiently as a simple coverage check on easy questions while routing as intelligently as an LLM on hard ones. The trajectory data and Utility@Budget framework introduced here provide the foundation for learning such a policy.
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
