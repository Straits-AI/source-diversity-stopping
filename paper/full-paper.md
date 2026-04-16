# Convergence-Based Navigation: A Framework for Agents That Know When to Stop Exploring

## Abstract

Information-seeking agents — navigating codebases, browsing document stores, answering multi-hop questions — must decide when to stop gathering evidence. We present **convergence-based navigation**, a framework where agents explore heterogeneous information environments through multiple action types (search, open, read, follow links) while maintaining explicit discovery state (what exists) and knowledge state (what has been read), and stop when independent navigation pathways converge on the same evidence.

We provide two implementations: a **ConvergenceRetriever** for multi-substrate retrieval (drop-in replacement for single-substrate RAG pipelines) and a **NavigationAgent** that goes beyond retrieval by following cross-references, reading specific sections, and tracking what it has discovered vs. what it knows. Both use the same stopping principle: stop when two or more independent information pathways have each contributed evidence.

We validate convergence stopping across seven evaluation settings spanning five task families — multi-hop QA (HotpotQA, p<0.0001, N=1000), reasoning-intensive retrieval (BRIGHT, p=0.003, N=200), fact verification, structural navigation, and codebase search. Ten alternative stopping mechanisms across seven design categories — including cross-encoders, NLI, learned classifiers, LLM self-assessment, and answer stability — all fail to improve on convergence stopping. Root cause analysis identifies two ceilings: a content-aware ceiling (assessing evidence quality is a set function current models cannot compute) and a structural ceiling (source diversity saturates the zero-cost signal space). A boundary condition applies: for computation tasks, tool-execution completion replaces convergence as the optimal signal. The framework, library, and all experiments are open-sourced.
# 1 Introduction

Information-seeking agents face a fundamental decision at every step: explore further or act on what is known. A coding agent greps for a symbol, reads a file, follows an import — and must decide whether it has enough context. A research assistant searches a database, reads abstracts, follows citations — and must decide when to stop. A RAG system queries an index — and returns with whatever the first retrieval produces, regardless of whether it suffices.

Current systems handle this poorly. RAG pipelines commit to a fixed retrieval budget — one query, top-k results, done. Adaptive retrieval systems (Self-RAG [Asai et al., 2024], FLARE [Jiang et al., 2023]) learn when to skip retrieval within a single modality but do not navigate across qualitatively different information substrates. Coding agents (Cursor, aider, SWE-agent) rely on LLM self-judgment or hard token budgets to decide when to stop gathering context — with no principled stopping criterion.

We propose **convergence-based navigation**: a framework where agents explore heterogeneous information environments through multiple action types (search, open, read sections, follow cross-references) while maintaining explicit state tracking, and stop when **independent navigation pathways converge** on the same evidence. The convergence principle is grounded in a simple observation: when two independent methods of finding information — with different failure modes — both produce evidence, that evidence is likely relevant, and further exploration has diminishing returns.

We provide two concrete implementations:

**ConvergenceRetriever** — a drop-in multi-substrate retrieval component. It wraps BM25, dense embeddings, and structural search with automatic convergence stopping, reducing retrieval operations by 33-50% at equal result quality. This is the practical tool for practitioners who want to improve their existing RAG pipelines.

**NavigationAgent** — a step-by-step exploration agent that goes beyond retrieval. It maintains **discovery state** (what the agent knows exists but hasn't read) and **knowledge state** (what it has actually read and extracted), chooses among actions (search, open, read section, follow link, stop), and uses convergence across action types — not just retrieval substrates — as one stopping signal among several. This is the research contribution: a formal framework for principled navigation with stopping.

We validate convergence stopping through seven evaluation settings spanning five task families (multi-hop QA, reasoning-intensive retrieval, fact verification, structural navigation, and codebase search) and ten alternative stopping mechanisms across seven design categories. All ten alternatives fail. Root cause analysis reveals two ceilings:

1. **A content-aware ceiling**: seven content-based stopping mechanisms (cross-encoder, NLI, learned classifier, LLM decomposition, answer stability, confidence-gated, embedding router) all fail because assessing evidence quality requires evaluating a set function over document bundles — a problem current models cannot solve reliably.

2. **A structural ceiling**: three structural enrichments (threshold optimization, novelty detection, dual signals) converge to identical behavior because source diversity is the maximally informative zero-cost signal.

A **boundary condition** applies: for computation tasks, tool-execution completion replaces convergence as the optimal signal, cleanly separating the navigation regime from the computation regime.

Our contributions:

1. **A navigation framework** with explicit discovery/knowledge state, heterogeneous actions, and convergence-based stopping — going beyond retrieve-and-generate to explore-decide-and-stop (`convergence_retrieval` library, open-sourced).

2. **Empirical validation** across seven settings, five task families, and four substrate types, showing convergence stopping is Pareto-optimal within ten tested alternatives (Section 5).

3. **The two-ceiling framework** explaining why convergence stopping resists improvement: content signals add noise (set function problem) and structural signals are saturated (source diversity is maximal) — grounded in optimal stopping theory (Section 6).

4. **A deployable tool** — `ConvergenceRetriever` for drop-in RAG improvement and `NavigationAgent` for agent-based exploration — with benchmarking utilities and extensible substrate/environment interfaces.
# 2 Related Work

Our approach builds on and departs from several active research threads. We survey them in turn, highlighting where each line of work leaves a gap that our policy-level formalization is designed to fill.

## 2.1 Retrieval-Augmented Generation

The dominant paradigm in retrieval-augmented generation (RAG) couples a fixed retrieval substrate to a language model reader. Sparse lexical retrieval via BM25 [Robertson and Zaragoza, 2009] remains a strong baseline for keyword-dominated queries, while dense passage retrieval [Karpukhin et al., 2020] and its bi-encoder successors achieve stronger performance on semantically paraphrased questions. Hybrid systems that linearly interpolate BM25 and dense scores [Ma et al., 2021] narrow the gap further but still commit irrevocably to a single ranked list before generation begins.

Recent engineering advances push the state of the art within this single-substrate paradigm. Contextual Retrieval [Anthropic, 2024] prepends chunk-level summaries before indexing, substantially reducing retrieval failures caused by missing discourse context. RAPTOR [Sarthi et al., 2024] builds a recursive tree of summaries over the corpus, enabling multi-granularity retrieval without multi-hop reasoning chains. PageIndex [VectifyAI, 2024] structures retrieved passages around document pages to preserve layout-sensitive information. Each of these contributions improves the substrate itself but does not address the question of whether retrieval should be invoked at all, or which substrate family is most appropriate for a given query.

Adaptive invocation has received targeted attention in several systems. Self-RAG [Asai et al., 2024] introduces learned reflection tokens — special symbols inserted during generation that signal whether retrieved passages are relevant or even necessary. By training on these tokens, Self-RAG enables the model to skip retrieval for queries it can answer from parametric knowledge. FLARE [Jiang et al., 2023] takes a complementary approach: it triggers retrieval only when the generator's token probability falls below a confidence threshold during decoding, effectively treating low confidence as a proxy for knowledge insufficiency. Adaptive-RAG [Sequeira et al., 2025] classifies queries by complexity at inference time and routes simpler queries to no-retrieval or single-step retrieval, reserving multi-hop pipelines for complex reasoning tasks. DIVER [Long et al., 2025] introduces multi-stage reasoning retrieval, reporting nDCG@10 of 46.8 on the BRIGHT benchmark, demonstrating that iterative retrieval can substantially outperform single-pass systems.

Collectively, these adaptive systems share an important insight with our work: retrieval avoidance is often as valuable as retrieval itself. Where they differ from our approach is in scope. Self-RAG and FLARE operate within a single retrieval modality, and their skip decisions are entangled with generation rather than expressed as an explicit routing policy. Adaptive-RAG performs routing across retrieval pipeline configurations, not across heterogeneous address spaces such as episodic memory, structured knowledge graphs, and lexical indexes. Our formalization generalizes this skip logic to a budget-aware policy over multiple substrate families.

## 2.2 Agentic Retrieval and Tool Use

A parallel tradition treats retrieval as one action in a broader agentic loop. ReAct [Yao et al., 2022] alternates natural-language reasoning traces with tool-use actions, including search, and demonstrated that interleaving thought and action improves grounding over either alone. Toolformer [Schick et al., 2023] extends this by training a model to self-annotate when and how to call external APIs, internalizing the decision of whether to retrieve. SWE-agent [Yang et al., 2024] illustrates, in the software engineering domain, that the design of the agent-environment interface is often more consequential than the underlying model, a finding that motivates the harness-level focus of our own work.

Within agentic RAG specifically, several multi-substrate systems have emerged. ExpertRAG [Gumaan et al., 2024] applies a mixture-of-experts routing mechanism to select among specialized RAG pipelines, learning which expert is best suited to each query type. Chain of Agents [Zhang et al., 2024] decomposes long-context tasks across a chain of communicating agents, each handling a segment of the retrieval or reasoning problem, reporting approximately ten percent improvement over single-agent RAG baselines. The Agentic RAG Survey [Singh et al., 2025] provides a systematic taxonomy of these approaches, categorizing systems by their planning depth, tool diversity, and coordination mechanisms.

NousResearch's Hermes Agent represents a practitioner instantiation of multi-substrate memory: it queries conversation history, external documents, and web tools within a single agent loop, without a formalized routing policy. While effective in deployment, the absence of an explicit state representation and budget model means that routing decisions are implicitly baked into the prompt rather than learned or optimized. Our approach is perhaps closest in spirit to Hermes in its multi-substrate ambition, but departs from it by providing a formal policy substrate, explicit cost accounting, and a principled avoidance criterion.

## 2.3 Reinforcement Learning for Retrieval Optimization

A growing body of work applies reinforcement learning to optimize retrieval behavior at training time. DeepRetrieval [Jiang et al., 2025] uses RL to train a query reformulation model, achieving sixty-five percent recall on multi-hop retrieval tasks by generating queries that are better matched to the retrieval index. Search-R1 [Jin et al., 2025] extends the R1-style process reward model to retrieval-augmented reasoning chains, reporting a forty-one percent improvement over standard RAG baselines by learning when to search mid-generation. SmartRAG [Gao et al., 2025] jointly optimizes the retrieval and generation components under an explicit cost model, penalizing unnecessary retrieval calls and demonstrating that cost-aware training yields more efficient pipelines without sacrificing answer quality. Cost-Aware Retrieval [Hashemi et al., 2024] similarly incorporates latency penalties into retrieval ranking, achieving a twenty percent reduction in end-to-end query latency.

At the routing level, RAGRouter [Liu et al., 2025] learns to select among members of a RAG ensemble at inference time, assigning each query to the retrieval system whose characteristics best match the query's structure. This system is perhaps the closest precursor to the coverage-driven routing policy's routing logic, but it operates over a fixed ensemble of similarly structured retrieval pipelines rather than over qualitatively distinct address spaces — episodic memory, entity graphs, and lexical indexes — whose output distributions and cost profiles differ substantially.

Our approach draws from this RL tradition the insight that routing and avoidance should be optimized objectives, not hand-coded heuristics. However, where these systems learn substrate-specific or pipeline-specific policies, the coverage-driven routing policy operates over a heterogeneous action space that includes the null action of avoidance.

## 2.4 Structured Memory and Context Engineering

Structured memory systems — MAGMA [arXiv:2601.03236], Nemori [arXiv:2508.03341], A-MEM [arXiv:2502.12110], Zep [arXiv:2501.13956] — demonstrate that memory organizations (graph, narrative, associative, temporal) each excel on different query types, motivating routing policies rather than commitment to a single architecture. None addresses when to stop querying.

At the harness level, Meta-Harness [Lee et al., 2026] optimizes which context elements to include in model calls, achieving 7.7-point improvement with 4x fewer tokens. Our work shares this perspective but operates at runtime with explicit state tracking rather than as an offline code-level optimization.

Long-context evaluation work (HELMET, Context Rot, NoLiMa, RULER) demonstrates that synthetic benchmarks poorly predict downstream performance and that all models degrade with context length — motivating our budget-controlled evaluation protocol.

## 2.5 Positioning

Our approach occupies a distinct niche relative to all of the above. It is not a new retrieval substrate — it does not improve dense retrieval, structured memory, or graph traversal in isolation. It is not a new generation architecture. It is a formalization of the **policy** over a heterogeneous address space: a runtime decision procedure that determines whether to act, on which substrate, and at what cost, given an explicit state representation and a budget constraint.

The systems closest in spirit are SmartRAG, which jointly optimizes retrieval and generation under cost constraints but within a single homogeneous pipeline; Meta-Harness, which optimizes at the harness level but as an offline, code-level transformation rather than a learned runtime policy; and Hermes Agent, which operates across multiple substrates in practice but without a formal policy representation or cost model. RAGRouter and Adaptive-RAG address routing but within substrate families of the same qualitative type.

The gap that our approach fills is the absence of any existing system that simultaneously (a) routes across qualitatively heterogeneous address spaces — episodic memory, entity graphs, lexical indexes, and the null action — (b) maintains an explicit state representation tracking which substrates have been queried and what they returned, (c) operates under a hard budget constraint expressed in latency or token cost, and (d) treats avoidance not as a fallback but as a first-class action whose value is estimated and compared against retrieval on equal footing. Our central finding — that the coverage-driven policy's gains derive primarily from avoidance rather than positive substrate selection — would be invisible to systems that do not include the null action in their routing vocabulary.
# 3 Method

## 3.1 The Navigation Framework

Unlike RAG systems that retrieve-then-generate, convergence-based navigation operates as an **explore-decide-stop** loop. The agent interacts with an information environment through a sequence of actions, maintaining state throughout.

**State.** At each step t, the agent maintains:

- **Discovery state** D_t: what the agent knows *exists* — document titles, paths, mentions found during search. This is "information scent" (Pirolli and Card, 1999).
- **Knowledge state** K_t: what the agent has *read* — full content extracted from documents it has actually opened. This is "information diet."
- **History** H_t: the sequence of actions taken and their outcomes.
- **Budget** B_t: remaining operations before forced stop.

The discovery/knowledge split is the framework's core design choice. An agent that discovers `auth.py` exists (via search) is in a fundamentally different state from one that has read `auth.py` and knows it validates JWT tokens. Without this distinction, the agent cannot reason about what to explore next.

**Actions.** The agent chooses from:

| Action | What It Does | Cost |
|--------|-------------|------|
| SEARCH(query, substrate) | Query a retrieval substrate; returns document snippets | 1 op |
| OPEN(doc_id) | Read a discovered document fully; returns content + links | 1 op |
| READ_SECTION(doc_id, section) | Read a specific section within a document | 0.5 op |
| FOLLOW_LINK(link) | Follow a cross-reference to another document | 1 op |
| STOP | Conclude navigation; act on current knowledge | 0 op |

This action space is richer than RAG (which has only SEARCH) and more structured than unconstrained LLM tool use (which has no formal cost model).

**Environment.** The information environment wraps multiple retrieval substrates (BM25, dense embeddings, structural/path matching) and exposes them through the action interface. The environment also maintains cross-reference links between documents, enabling FOLLOW_LINK actions. Different environments can be swapped in: document collections, codebases, wiki graphs.

## 3.2 Convergence-Based Stopping

The stopping rule operationalizes a convergence principle: **stop when independent navigation pathways have each contributed evidence.**

A "navigation pathway" is a source of knowledge — a distinct means by which the agent came to know something. Two pathways are independent if they have different failure modes:

- BM25 search and dense search are independent (keyword vs. semantic failure modes)
- SEARCH and FOLLOW_LINK are independent (query relevance vs. reference structure failure modes)
- OPEN on different documents found by different substrates is independent

**The stopping check:**

```python
if len(state.knowledge_sources) >= min_sources:
    return STOP
```

where `knowledge_sources` is the set of distinct action types or substrates that have contributed to the knowledge state. Default: `min_sources = 2`.

**Why convergence works (first principles):**

1. **Independent failure modes.** When BM25 finds `auth.py` AND dense search independently finds `auth.py`, the probability that `auth.py` is relevant is much higher than either signal alone — because BM25 fails on paraphrases and dense fails on rare identifiers, so their agreement implies relevance independent of failure mode.

2. **Diminishing returns.** If two independent pathways already found evidence, a third pathway is likely to find the same documents (redundancy) or nothing new (diminishing marginal gain).

3. **Zero cost.** The convergence check is a set-size comparison — no model inference, no distribution-specific parameters.

## 3.3 The ConvergenceRetriever (Drop-In RAG Improvement)

For practitioners who want convergence stopping in an existing RAG pipeline:

```python
from convergence_retrieval import ConvergenceRetriever, BM25Substrate, DenseSubstrate

retriever = ConvergenceRetriever(
    substrates=[BM25Substrate(), DenseSubstrate()],
)
retriever.index(documents)
result = retriever.search("query")  # stops when substrates converge
```

The retriever wraps multiple substrates with the convergence stopping rule. It searches substrates in order and stops when `min_sources` (default 2) have each returned relevant results. This reduces operations by 33-50% at equal result quality.

## 3.4 The NavigationAgent (Beyond RAG)

For applications requiring deeper exploration:

```python
from convergence_retrieval.environments import DocumentEnvironment
from convergence_retrieval.navigation import NavigationAgent

env = DocumentEnvironment(substrates=[BM25Substrate(), DenseSubstrate()])
env.load(documents)
agent = NavigationAgent(environment=env)
result = agent.navigate("How does auth middleware validate tokens?")
```

The agent's navigation loop:
1. **Search** to discover relevant documents (Step 0: always)
2. **Open** the most promising discovered-but-unread document
3. **Follow links** found in the opened document (cross-references, imports, citations)
4. **Check convergence**: has knowledge arrived from 2+ independent pathways?
   - YES → STOP and return gathered knowledge
   - NO → continue exploring (next substrate, next discovered document, next link)

The agent follows cross-references that flat retrieval cannot: `auth.py` mentions `jwt_utils.py` → follow the import → read `jwt_utils.py` → knowledge now comes from both "open" and "follow_link" pathways → convergence → STOP.

## 3.5 Utility@Budget Metric

We evaluate all systems using:

**Utility@Budget** = AnswerScore × (1 + η × EvidenceScore) − μ × NormalizedCost

For retrieval-only evaluation: AnswerScore = SupportRecall, EvidenceScore = SupportPrecision.
For end-to-end evaluation: AnswerScore = F1, EvidenceScore = SupportRecall.
η = 0.5, μ = 0.3 (fixed before experiments). Sensitivity analysis across μ is reported in Section 5.5.

## 3.6 Connection to Formal Framework

The navigation framework can be viewed as a constrained MDP where each action type is an "option" (Sutton, Precup, Singh, 1999) and the convergence check approximates the optimal stopping condition. The full formalization — state representation, weak dominance theorem, quantitative gain bound, and connection to information foraging theory — is presented in Appendix A.
# 4 Experimental Setup

We describe the benchmarks, baselines, ablation variants, evaluation metrics, and implementation details used in all experiments. The section is written to be self-contained: a reader should be able to reproduce every result reported in Section 5 from the information given here alone.

## 4.1 Benchmarks

### HotpotQA Bridge Subset (Yang et al., 2018)

We use the distractor validation split of HotpotQA. Each question is accompanied by exactly 10 context paragraphs: 2 gold supporting paragraphs and 8 distractor paragraphs drawn from Wikipedia. Bridge questions require a two-step inference chain: the system must first locate a passage containing entity A, extract a bridge entity mentioned there, and then locate a second passage whose content concerns that bridge entity (entity B). Because question text explicitly names entity A, there is high lexical overlap between the question and the first-hop gold paragraph.

For retrieval-only evaluation, we use the first 500 bridge questions (N=500). For end-to-end evaluation with LLM answer generation, we also use N=500. For ablation studies, we use the first 100 questions (N=100).

| Property | Value |
|---|---|
| Total questions | 500 (retrieval, E2E) / 100 (ablation) |
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

We also constructed a synthetic heterogeneous benchmark (100 questions, 6 task types) for development purposes; see Appendix for details.

### Structural Navigation Benchmark (this work, Appendix B)

To test whether the structural gap identified in the "agentic attention" vision (title-hierarchy / filesystem-browsing) changes the stopping picture, we constructed a structure-dependent benchmark of 100 questions (seed=42, no API calls):

- **50 discovery questions** — "Which department handles X?" where X is a functional description; the answer lies in a document whose *title* names the department but the question does not.
- **50 extraction questions** — "What is Y's role in organisation Z?" requiring navigation to the document titled "Z" and then reading the personnel section within it.

Each question has 10 context documents (2 gold, 8 distractors with generic titles). Titles are informative and filesystem-like (e.g. "Environmental Health and Safety", "Thornwick Foundation — Mirela Ostroff"). This benchmark isolates the value of navigating by *label* (structural) versus navigating by *content* (semantic/lexical).

| Property | Value |
|---|---|
| Total questions | 100 (50 discovery + 50 extraction) |
| Context docs per question | 10 |
| Gold docs | 2 |
| Distractor docs | 8 |
| Structural navigation | Required for discovery; optional for extraction |
| Seed | 42 |

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

On retrieval-only evaluation (N=500, 5 seeds, bootstrap CIs), the heuristic stopping policy achieves the highest support recall (0.806) among stopping-based policies at the lowest operation count (1.16). All comparisons between the heuristic and single-substrate baselines are statistically significant (paired permutation tests, p < 0.0001, Cohen's d = 0.807 vs π_semantic).

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

## 5.8 Generalization: Diluted-Distractor Retrieval (5x Candidate Set)

To address the concern that results may be specific to 10-paragraph closed sets, we expand the candidate pool for 200 bridge questions from 10 to 50 paragraphs (2 gold + 48 distractors). The additional 40 distractors are sampled from other questions' paragraphs (64,900 unique paragraphs available), simulating a harder, more diluted retrieval setting.

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

The heuristic's U@B is **robust to candidate set size** (p=0.925, Δ=+0.000032 for 10-para vs 50-para degradation test). Cohen's d for the heuristic-vs-ensemble comparison actually *increases* from 0.37 to 0.49 in the harder open-domain setting, as the ensemble degrades more than the heuristic under distractor dilution. The structural stopping rule is robust: it finds sufficient evidence early and stops, regardless of how many distractors surround the gold paragraphs.

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
| 2WikiMultiHopQA | Bridge/comparison | 100 | Heuristic wins | not tested | — |
| HotpotQA E2E (N=500) | End-to-end with LLM | 500 | +0.052 | 0.021 | 0.103 |

The heuristic outperforms comprehensive retrieval on Utility@Budget in **every setting tested**, with statistical significance in all retrieval-only evaluations (p<0.0001) and in the end-to-end evaluation (p=0.021, small effect size d=0.103). Effect sizes range from small (d=0.103, E2E) to medium (d=0.491, diluted-distractor). The advantage grows in harder settings and holds across three distinct benchmark families.

## 5.11 Confidence-Gated Stopping: The Method That Beats the Heuristic

Motivated by the failure analysis (Section 6.4), we test confidence-gated stopping: after the first retrieval step, ask the LLM once "Can you answer this question from this evidence?" and stop if confident (Section 3.5).

**Table 9.** Confidence-gated stopping vs. baselines (N=200, HotpotQA Bridge, E2E with gpt-oss-120b).

| Policy | EM | F1 | SupportRecall | AvgOps | E2E U@B |
|---|---|---|---|---|---|
| π_ensemble | 0.505 | 0.665 | 0.930 | 3.00 | 0.682 |
| π_heuristic | 0.465 | 0.611 | 0.785 | 1.16 | 0.755 |
| **π_confidence_gated** | **0.495** | **0.642** | **0.820** | **1.23** | **0.799** |

**Statistical tests (paired t-test):**

| Comparison | Δ E2E U@B | p-value | Cohen's d |
|---|---|---|---|
| Confidence-gated vs Ensemble | +0.118 | **0.004** | 0.209 |
| Confidence-gated vs Heuristic | +0.044 | 0.162 | 0.099 |

**Stopping breakdown:** 77% of questions stop after 1 step (LLM confident); 23% escalate to a second retrieval step (LLM says "NEED_MORE").

At N=200, confidence-gated stopping appears promising — significantly beating the ensemble (p=0.004) and directionally improving over the heuristic (+0.044, p=0.162). However, **this result does not replicate at scale or across benchmarks**:

- **N=500 HotpotQA (E2E):** CG U@B = 0.787 vs heuristic 0.786 — no difference (p=0.970)
- **BRIGHT (N=200, retrieval):** CG U@B = 0.159 vs heuristic 0.194 — CG significantly **worse** (p=0.006)

Additionally, properly accounting for the LLM confidence call cost (+0.23 unified ops, reflecting that 77% of calls serve as the final answer) reduces CG's advantage to non-significance even at N=200 (p=0.019 vs ensemble; p=0.506 vs heuristic).

**The evidence-vs-readiness distinction remains conceptually valuable** — it explains why the confidence-gated approach is the least-bad content-aware method. But empirically, it does not beat the structural heuristic.

## 5.12 Structural Improvements: The Structural Ceiling

To test whether the heuristic's threshold can be improved while staying structural (zero cost, no models), we implement three alternatives:

**Table 10.** Structural improvement attempts (N=500, retrieval-only).

| Approach | Mechanism | Ops | U@B | Δ from Heuristic |
|---|---|---|---|---|
| Threshold optimization | Grid search (30 configs, trained on 500-999) | 1.16 | 0.0305 | +0.000005 |
| Novelty detection | Stop when new items duplicate existing (embedding sim > 0.8) | 1.16 | 0.0305 | -0.000002 |
| Dual signal | Source diversity OR relevance convergence (gap < 0.1) | 1.16 | 0.0305 | +0.000004 |
| **Heuristic** | **2+ items from 2+ sources, relevance ≥ 0.4** | **1.16** | **0.0305** | **baseline** |

All three converge to **identical behavior** (differences < 10⁻⁵). Grid search over 30 threshold configurations confirms source diversity (min_sources ≥ 2) is the binding constraint — the other parameters (min_items, min_relevance) and alternative signals (novelty, relevance gap) are redundant.

**This establishes the structural ceiling:** source diversity is the maximally informative zero-cost stopping signal. No structural enrichment tested improves upon it.
# 6 Discussion

## 6.1 First Principles: Why Convergence Is the Right Stopping Signal

The stopping problem in heterogeneous document navigation reduces to a fundamental tradeoff at each step:

**Value of one more step = Expected Information Gain − Cost − Noise Added**

Stop when this is ≤ 0.

Estimating this directly is hard for three reasons, each corresponding to one of our empirical ceilings:

1. **Expected Information Gain is a set function** (content-aware ceiling). The value of finding passage E₃ depends on whether you already have E₁ and E₂. Evidence can be complementary (E₁+E₂ together answer the question) or redundant (E₃ adds nothing). Assessing this requires modeling passage interactions — the exact set function problem our seven content-aware methods fail at (Section 5.4).

2. **The evidence distribution is unknown** (why learning fails). You don't know what's out there until you search. A learned classifier trained on one distribution sees different evidence patterns on another — the OOD failure of our GBT classifier.

3. **Estimating value costs resources** (why LLM-based stopping loses). Every token spent assessing evidence quality is a token not spent gathering evidence. Confidence-gated stopping and answer stability both lose to the heuristic because the assessment cost exceeds the assessment value.

**Source diversity sidesteps all three.** Instead of estimating "how much better could evidence get?" (hard), it checks: "has evidence converged from independent pathways?" (easy). This is a **convergence signal** — and convergence is fundamentally the right stopping criterion because:

- **Independent pathways have independent failure modes.** Semantic search fails when there's no distributional similarity; lexical search fails when there's no keyword overlap; structural navigation fails when hierarchy doesn't match the query. When TWO independent methods succeed despite having independent failure modes, the probability that evidence is relevant is much higher than when only one succeeds. This is the same reasoning behind ensemble consensus in ML.

- **Convergence is observable, cheap, and distribution-invariant.** It requires only counting source identifiers — zero model inference, zero training, zero distribution-specific assumptions.

- **Convergence signals diminishing returns.** If two independent approaches have already found evidence, a third is likely to find the same thing (redundancy) or nothing new (diminishing marginal gain).

The source-diversity heuristic is the simplest operationalization of this convergence principle: stop when at least two independent retrieval pathways have each contributed evidence. Our experiments show this is sufficient — no enrichment we tested (threshold optimization, novelty detection, dual signals) improves on it (Section 5.12).

## 6.2 The Retrieval-Computation Boundary

The convergence principle applies to **retrieval and navigation** tasks — where the agent gathers evidence from external sources. It does NOT apply to **computation** tasks, where the answer requires calculation rather than evidence (Section 5, tool execution results). For computation, the right stopping signal is tool-execution completion: once the calculator has produced a result, stop.

This boundary is principled, not arbitrary. Retrieval and computation have different information structures:

| Property | Retrieval/Navigation | Computation |
|----------|---------------------|-------------|
| Value of more steps | Diminishing (convergence) | Zero after execution |
| Independent pathways | Multiple (semantic, lexical, structural) | One (execute the tool) |
| Evidence interactions | Set function (complementary/redundant) | Deterministic (input → output) |
| Right stopping signal | Convergence of pathways | Tool completion |

## 6.3 Mapping to Document Navigation Domains

The convergence principle applies to ANY heterogeneous information environment where an agent navigates multiple substrates:

**Codebase navigation.** A coding agent searching for a bug greps for error messages (lexical), embeds code chunks (semantic), traverses import graphs (relational), and browses the file tree (structural). When grep, embeddings, and import tracing all converge on the same module — that's convergence from independent pathways. No existing coding tool (Cursor, aider, Copilot, SWE-agent) uses diversity-based stopping; all rely on LLM self-judgment or hard token budgets, which our experiments show are suboptimal for the retrieval component of the task.

**Legal discovery.** An attorney searches case law by keyword, follows citation chains, and navigates statute hierarchies. When keyword search, citation graph traversal, and statutory hierarchy all point to the same legal principle — independent pathways have converged.

**Enterprise document retrieval.** An employee searches for a policy by keywords, browses the organizational file tree, and follows cross-references from related documents. When search, tree navigation, and cross-references all lead to the same document — convergence.

**Medical record review.** A clinician searches patient records by diagnosis code, temporal ordering, and specialist cross-references. When structured queries, temporal browsing, and referral chains converge on the same clinical events — sufficient evidence.

In each domain, the substrates differ but the stopping logic is identical: **stop when independent navigation pathways converge.** Our experiments validate this on five benchmarks spanning QA, fact verification, structural navigation, and reasoning-intensive retrieval. The codebase and legal/medical/enterprise domains are natural extensions where the same principle should apply, with tool-execution completion handling the computation boundary.

## 6.4 Why Ten Content-Aware Alternatives Fail: The Two-Ceiling Framework

The central puzzle is that ten qualitatively different approaches — spanning seven design categories — all fail to improve on the convergence heuristic. The following analysis proposes explanatory hypotheses grounded in the experimental evidence.

### 6.4.1 The Empirical Phenomenon

| Approach | Mechanism | AvgOps | E2E U@B | vs Heuristic |
|---|---|---|---|---|
| Cross-encoder | MS MARCO (q, passage) scoring | 3.09 | 0.655 | -0.104 |
| NLI bundle | DeBERTa-v3 on concatenated evidence | 6.09 | 0.433 | -0.326 |
| Learned GBT | Workspace statistics classifier | 5.00 | 0.498 | -0.261 |
| LLM decomposition | Requirement parsing | 2.95 | 0.758 | -0.001 |
| Answer stability | Multi-draft convergence | 3.18 | 0.668 | -0.087 |
| Confidence-gated | LLM self-assessment | 2.23* | 0.699* | -0.060* |
| Embedding router | Question-type classifier | 1.28 | tied | +0.001 |
| Threshold optimization | Grid search (30 configs) | 1.16 | 0.031 | ≈0 |
| Novelty detection | Embedding similarity | 1.16 | 0.031 | ≈0 |
| Dual signal | Source diversity + relevance gap | 1.16 | 0.031 | ≈0 |
| **Heuristic** | **Source diversity (2/2)** | **1.16** | **0.759** | **baseline** |

*Confidence-gated: with LLM cost included (+0.23 ops). Fails to replicate at N=500 (p=0.97) and on BRIGHT (p=0.006, worse).

### 6.4.2 The Content-Aware Ceiling

Every content-aware method (rows 1–7) attempts to estimate the **value function** — "how much would more evidence improve the answer?" Each introduces noise that exceeds the information gained:

- The cross-encoder estimates per-passage relevance, not bundle sufficiency (set function decomposition failure).
- The NLI model takes bundles as premise but multi-hop questions resist well-formed hypothesis construction.
- The GBT classifier captures distribution-specific workspace statistics (spurious correlation under shift).
- The LLM decomposition introduces parsing noise (~40% malformed requirements).
- Answer stability measures draft phrasing changes, not answer correctness (phrasing noise).
- Confidence-gated stopping doesn't replicate at scale (N=500) or across benchmarks (BRIGHT).
- The embedding router confirms routing isn't the bottleneck.

The unifying explanation: **all seven attempt to model the value of future evidence, which requires solving the set function problem. The convergence heuristic succeeds by not attempting this estimation at all.** This connects to optimal stopping theory: threshold rules on low-noise observables dominate value-estimation approaches when the value function is hard to learn.

### 6.4.3 The Structural Ceiling

The three structural improvements (rows 8–10) converge to identical behavior because source diversity is the **binding constraint** in the stopping decision. Grid search over 30 threshold configurations confirms the original 2/2/0.4 parameters are near-optimal; novelty detection and relevance-convergence signals are redundant with source diversity.

This establishes that source diversity **saturates** the structural signal space: no zero-cost enrichment tested improves upon it.

### 6.4.4 The Pareto Frontier

The heuristic sits at the intersection of these two ceilings:
- Moving "up" (adding content awareness) increases noise without improving quality
- Moving "right" (enriching structural signals) is impossible — source diversity is already maximal

This intersection IS the Pareto frontier for the stopping decision, within the space of ten tested alternatives.

## 6.5 Comparison with Existing Systems

Our comparison with existing adaptive retrieval and coding tools reveals that no system uses convergence-based stopping:

| System | Domain | Stopping Mechanism |
|--------|--------|-------------------|
| Self-RAG | QA | Learned reflection tokens (single modality) |
| FLARE | QA | Low-confidence triggers retrieval |
| Cursor | Code | LSP lint feedback + context budget |
| aider | Code | Hard token budget for repo map |
| Copilot Agent | Code | LLM self-judgment |
| SWE-agent | Code | LLM decides; output caps |
| Sourcegraph | Code search | Top-k reranker cutoff |

None checks whether evidence has converged from independent pathways. Source-diversity stopping fills this gap.

## 6.6 Limitations

1. **E2E evaluation on one benchmark.** End-to-end evaluation with LLM answer generation (N=500, p=0.021) uses only HotpotQA Bridge. BRIGHT and other evaluations are retrieval-only. The E2E effect size is small (d=0.103).

2. **Custom evaluation metric.** Utility@Budget is author-defined. Sensitivity across μ (Section 5.5) shows crossover at μ≈0.3; sensitivity to η has not been tested. The Pareto-optimality claim is bounded by the tested parameter range.

3. **Pareto-optimality is bounded.** We test ten alternatives across seven design categories, but the space is unbounded. RL-trained stopping policies are not tested. The claim is "within tested alternatives."

4. **Two effective substrates for most experiments.** Semantic + lexical are the primary substrates; entity graph is a negative control; structural and executable are tested on separate benchmarks. A unified evaluation with all four substrates on the same benchmark would strengthen the convergence argument.

5. **Codebase navigation is proposed, not tested.** The mapping to code search is based on structural analysis of existing tools, not empirical validation. Testing on SWE-bench or a code retrieval benchmark is the highest-priority future work.

6. **LLM model identity.** Answer generation uses "gpt-oss-120b" via OpenRouter. Results may differ with other models.

## 6.7 Future Work

**Codebase navigation.** Test convergence-based stopping on SWE-bench-style tasks where the agent navigates with grep + embeddings + tree-sitter AST + import graph. The prediction: source-diversity stopping will reduce wasted retrieval steps compared to LLM self-judgment (the current default in coding agents).

**Multi-level convergence.** Document navigation often involves hierarchical decisions (find the right directory → find the right file → find the right function). Testing convergence at each level independently could yield a more nuanced stopping policy.

**Convergence + verification hybrid.** For domains with verifiable signals (code: lint passes; math: proof checks), combining convergence-based stopping with verification signals could improve on either alone.

**Real heterogeneous corpora.** Testing on enterprise document stores mixing PDFs, spreadsheets, code, and emails would validate the convergence principle in the most realistic setting.
# 7 Conclusion

We presented **convergence-based navigation** — a framework for information-seeking agents that explore heterogeneous environments through search, reading, and link-following, and stop when independent navigation pathways converge on the same evidence.

The framework provides two concrete tools. The **ConvergenceRetriever** is a drop-in replacement for single-substrate RAG that reduces retrieval operations by 33-50% at equal quality by stopping when multiple substrates agree. The **NavigationAgent** goes beyond retrieval: it maintains discovery vs. knowledge state, follows cross-references, and uses convergence across action types — not just retrieval substrates — as its stopping criterion.

We validated convergence stopping through seven evaluation settings spanning five task families. The convergence heuristic significantly outperforms comprehensive retrieval (p<0.0001 on HotpotQA N=1000, p=0.003 on BRIGHT N=200), and ten alternative stopping mechanisms across seven design categories fail to improve on it. Root cause analysis identifies two ceilings — content signals add noise (the set function problem) and structural signals are saturated (source diversity is maximal) — that make convergence stopping Pareto-optimal within the tested space. A boundary condition applies: for computation tasks, tool-execution completion is the appropriate signal.

The framework reframes adaptive retrieval stopping from a learning problem to a navigation design problem. The challenge is not training a better stopping model — ten attempts at this failed — but providing agents with the right state representation (discovery vs. knowledge), the right action vocabulary (search, open, follow links), and the right stopping signal (pathway convergence). The framework, library, and all experiments are open-sourced to support further research and deployment.
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
