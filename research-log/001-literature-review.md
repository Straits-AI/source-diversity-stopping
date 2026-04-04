# Literature Review

**Date:** 2026-04-04
**Phase:** 1
**Iteration:** 0
**Status:** completed

## Context
Following Phase 0 setup, conducted Deep-intensity literature review targeting 30-50 papers. Dispatched 4 parallel searchers across 8 topic clusters. Found ~63 total papers (48 new + 15 previously cited).

## Literature Map

### 1. Classical & Improved Retrieval

| Paper | Year | Key Contribution | Relevance |
|-------|------|-----------------|-----------|
| Contextual Retrieval (Anthropic) | 2024 | Enriches chunks with document-specific context before indexing | 4/5 |
| RAPTOR (Sarthi et al.) | 2024 | Recursive abstractive tree retrieval over summary hierarchies | 4/5 |
| PageIndex (VectifyAI) | 2024 | Hierarchical tree index with reasoning-based retrieval, no vectors | 4/5 |
| CSQE (Li et al., arXiv:2402.18031) | 2024 | Corpus-steered query expansion grounding LLM expansions in actual corpus | 4/5 |
| ColBERT-Att (arXiv:2603.25248) | 2026 | Late-interaction with explicit attention weights for ranking | 3/5 |

**What works:** Enriching chunks with context (Contextual Retrieval), hierarchical organization (RAPTOR, PageIndex), and corpus-grounded query expansion (CSQE) all improve over flat dense retrieval. PageIndex shows vectors are not necessary for effective retrieval.

**What's missing:** All systems commit to a single retrieval substrate at design time. None adaptively switch between approaches based on query or document characteristics.

---

### 2. Adaptive & Active Retrieval

| Paper | Year | Key Contribution | Relevance |
|-------|------|-----------------|-----------|
| Self-RAG (Asai et al., arXiv:2310.11511) | 2024 | Learned reflection tokens for adaptive retrieve/skip decisions | 5/5 |
| FLARE (Jiang et al., arXiv:2305.06983) | 2023 | Forward-looking active retrieval during generation | 5/5 |
| Adaptive-RAG (Sequeira et al., arXiv:2407.21712) | 2025 | Query complexity classifier routes to no/single/multi-step retrieval | 4/5 |
| DIVER (Long et al., arXiv:2508.07995) | 2025 | Multi-stage reasoning-intensive retrieval with iterative query expansion. nDCG@10=46.8 on BRIGHT | 5/5 |

**What works:** Adaptive timing (Self-RAG, FLARE) and complexity-aware routing (Adaptive-RAG) both outperform fixed retrieval. DIVER shows iterative expansion helps on reasoning-intensive tasks.

**What's missing:** These systems adapt *when* to retrieve and *how much* to retrieve, but not *which substrate* to use. The action space is limited to retrieve/skip within a single address space.

---

### 3. Multi-hop & Structured Retrieval

| Paper | Year | Key Contribution | Relevance |
|-------|------|-----------------|-----------|
| HopRAG (Zhang et al., arXiv:2502.12442) | 2025 | Graph-structured passage retrieval with retrieve-reason-prune. +76.78% answer metric vs conventional RAG | 5/5 |
| KG-IRAG (arXiv:2503.14234) | 2025 | Iterative KG retrieval for temporal/logical dependencies | 5/5 |
| RT-RAG (Liu et al., arXiv:2601.11255) | 2026 | Tree decomposition for multi-hop, +7% F1 over SOTA | 3/5 |
| MultiHop-RAG (Tang & Yang, arXiv:2401.15391) | 2024 | Benchmark showing SOTA performs poorly on multi-hop | 3/5 |
| Chain of Retrieval (arXiv:2507.10057) | 2024 | Multi-aspect parallel queries (topic, method, dataset, venue) | 4/5 |

**What works:** Graph-structured retrieval (HopRAG) and iterative KG traversal (KG-IRAG) substantially improve multi-hop reasoning. Multi-aspect querying (Chain of Retrieval) shows parallel address spaces help.

**What's missing:** Each system is locked to one graph structure. None can choose between graph traversal and other strategies (lexical search, tool execution) based on the task.

---

### 4. Agentic Retrieval & Tool Use

| Paper | Year | Key Contribution | Relevance |
|-------|------|-----------------|-----------|
| ReAct (Yao et al.) | 2022 | Interleaved reasoning and acting | 5/5 |
| Toolformer (Schick et al.) | 2023 | Self-supervised tool-use learning | 4/5 |
| SWE-agent (Yang et al.) | 2024 | Agent-computer interface design matters for performance | 5/5 |
| Agentic RAG Survey (Singh et al., arXiv:2501.09136) | 2025 | Comprehensive survey of agentic RAG patterns | 5/5 |
| ExpertRAG (Gumaan et al., arXiv:2504.08744) | 2024 | MoE routing between internal knowledge and external retrieval | 5/5 |
| Chain of Agents (Zhang et al., arXiv:2406.02818) | 2024 | Multi-agent sequential context processing, +10% over RAG and full-context | 5/5 |

**What works:** Action interleaving (ReAct), interface design (SWE-agent), MoE routing (ExpertRAG), and multi-agent collaboration (Chain of Agents) all show gains from treating retrieval as an agentic problem.

**What's missing:** These systems treat tool use as the main action space. None formalize the broader workspace management problem (pin, evict, compress, expand). The Agentic RAG Survey catalogs patterns but notes standardized metrics are lacking.

---

### 5. RL & Optimization for Retrieval

| Paper | Year | Key Contribution | Relevance |
|-------|------|-----------------|-----------|
| OPEN-RAG | 2024 | RL-optimized retrieval against downstream reward | 4/5 |
| MMOA-RAG | 2024 | Multi-module optimization for RAG | 4/5 |
| DeepRetrieval (Jiang et al., arXiv:2503.00223) | 2025 | RL-trained query generation, 65% recall vs 25% prior SOTA | 5/5 |
| SmartRAG (Gao et al., arXiv:2410.18141) | 2025 | Joint RL optimization of retrieve-decision + query-rewrite + answer-gen with cost-aware rewards | 5/5 |
| Search-R1 (Jin et al., arXiv:2503.09516) | 2025 | RL for interleaved search during reasoning, +41% over RAG | 5/5 |
| Knowledgeable-r1 (Hwang et al., arXiv:2506.05154) | 2025 | RL for balancing parametric vs contextual knowledge, +17% on counterfactual | 5/5 |
| RAG-DDR (Liu et al., arXiv:2410.13509) | 2025 | Differentiable data rewards for end-to-end RAG optimization | 5/5 |
| R3-RAG (Li et al., arXiv:2505.23794) | 2025 | Step-by-step reasoning and retrieval interleaving via RL | 3/5 |
| RPO (Yan et al., arXiv:2501.13726) | 2025 | Retrieval preference optimization for robust RAG | 3/5 |
| LarPO (Jin et al., arXiv:2502.03699) | 2025 | LLM alignment reframed as retriever optimization, +39% AlpacaEval2 | 4/5 |
| InfoFlow (Li et al., arXiv:2510.26575) | 2024 | Reward density optimization for deep search via subproblem decomposition | 4/5 |
| Cost-Aware Retrieval (Hashemi et al., arXiv:2510.15719) | 2024 | Adaptive retrieval depth with cost-aware RL, 20% latency reduction | 4/5 |
| RAGRouter (Liu et al., arXiv:2505.23052) | 2025 | Learned routing across RAG model ensemble, +3.6% over best single | 4/5 |
| Retro* (Ye et al., arXiv:2509.24869) | 2024 | LLM-as-ranker with test-time scaling on BRIGHT | 3/5 |

**What works:** RL-based optimization of retrieval policies shows strong gains across the board. SmartRAG demonstrates joint optimization with cost awareness. Search-R1 and DeepRetrieval show massive improvements from learned query policies. Cost-Aware Retrieval shows budget-aware RL is practical.

**What's missing:** All systems optimize retrieval within a single paradigm (search, QA, ranking). None learn policies over heterogeneous address spaces simultaneously. No system combines memory traversal, document navigation, tool execution, and search in a single learned policy.

---

### 6. Structured Memory for Agents

| Paper | Year | Key Contribution | Relevance |
|-------|------|-----------------|-----------|
| MAGMA (arXiv:2601.03236) | 2025 | Multi-graph memory with adaptive traversal policy. LoCoMo judge=0.700, LongMemEval=61.2% | 5/5 |
| Agentic Memory (arXiv:2601.01885) | 2025 | Unified LTM/STM management as tool-based agent actions | 5/5 |
| A-MEM (arXiv:2502.12110) | 2026 | Zettelkasten-based dynamic memory with interconnected knowledge networks | 4/5 |
| MemoryBank (arXiv:2305.10250) | 2023 | Continuous memory evolution with contradiction resolution | 4/5 |
| Nemori (arXiv:2508.03341) | 2025 | Self-organizing memory via Event Segmentation Theory. 88% fewer tokens, outperforms on LoCoMo/LongMemEval | 5/5 |
| MemFactory (arXiv:2603.29493) | 2026 | Unified modular framework for memory-augmented agents | 3/5 |
| Zep (arXiv:2501.13956) | 2025 | Temporal knowledge graph bridging conversations and structured data | 4/5 |

**What works:** MAGMA and Nemori show structured memory with traversal policies beats flat retrieval on long-horizon tasks. Agentic Memory shows memory operations as agent actions is viable. Zep bridges unstructured and structured data.

**What's missing:** All systems optimize memory in isolation. None integrate memory with document browsing, tool execution, or external search in a unified policy. MAGMA's own paper notes evaluation is concentrated on conversational/agentic settings rather than heterogeneous environments.

---

### 7. Context Engineering & Management

| Paper | Year | Key Contribution | Relevance |
|-------|------|-----------------|-----------|
| Anthropic Context Engineering | 2025 | Context curation, compaction, and handoff as system design | 5/5 |
| Meta-Harness (Lee et al., arXiv:2603.28052) | 2026 | End-to-end harness optimization as agentic search. +7.7pt classification, 4x fewer tokens | 5/5 |
| Adaptive Context Compression (arXiv:2603.29193) | 2026 | Importance-aware memory selection with dynamic budget allocation | 4/5 |
| In-Context Former (arXiv:2406.13618) | 2024 | Linear-time context compression via cross-attention digest tokens | 3/5 |
| Context Compression for Tools (arXiv:2407.02043) | 2024 | Specialized compression preserving tool-relevant information | 4/5 |
| Context Engineering for OSS (arXiv:2510.21413) | 2025 | Practical context engineering for software agents | 3/5 |

**What works:** Meta-Harness shows that treating the harness itself as an optimization target yields large gains. Context compression with tool-awareness (arXiv:2407.02043) shows selective retention for different use cases.

**What's missing:** Meta-Harness optimizes harness code, not runtime context operations. No system learns an online policy for context management (pin, evict, compress, expand) during task execution.

---

### 8. Long-Context Models & Evaluation

| Paper | Year | Key Contribution | Relevance |
|-------|------|-----------------|-----------|
| MSA (arXiv:2603.23516) | 2025 | Memory Sparse Attention scaling to 100M tokens | 4/5 |
| RULER (arXiv:2404.06654) | 2024 | Controlled long-context stress tests | 5/5 |
| NoLiMa (arXiv:2502.05167) | 2025 | Long-context eval beyond literal matching | 5/5 |
| LongBench Pro (arXiv:2601.02872) | 2026 | Realistic long-context tasks 8K-256K tokens | 4/5 |
| HELMET (Princeton NLP, arXiv:2410.02694) | 2024 | Holistic 7-category eval: synthetic tasks don't predict downstream | 5/5 |
| Context Rot (Chroma Research) | 2024 | All 18 frontier models degrade with more tokens. 30%+ accuracy drops at mid-context | 5/5 |
| LoCoBench (Huang et al., arXiv:2509.09614) | 2024 | Long-context SE benchmark, 10K-1M tokens, 17 metrics | 4/5 |
| 100-LongBench (Yang et al., arXiv:2505.19293) | 2025 | Existing benchmarks confound baseline knowledge with long-context ability | 5/5 |
| MMNeedle (Wang et al., arXiv:2406.11230) | 2024 | Multimodal needle-in-haystack, 10M+ tokens | 3/5 |

**What works:** Long context helps but degrades predictably (Context Rot, RULER, NoLiMa). HELMET shows task categories don't correlate — synthetic performance doesn't predict real tasks.

**What's missing:** No benchmark tests whether adaptive context operations improve over brute-force long context. All evaluations assume a fixed context strategy.

---

### 9. QA Benchmarks & Evidence Evaluation

| Paper | Year | Key Contribution | Relevance |
|-------|------|-----------------|-----------|
| BRIGHT (arXiv:2407.12883) | 2024 | Reasoning-intensive retrieval benchmark | 5/5 |
| HotpotQA (arXiv:1809.09600) | 2018 | Multi-hop QA with supporting facts | 4/5 |
| MuSiQue (arXiv:2108.00573) | 2021 | Shortcut-resistant multi-hop QA | 4/5 |
| LongMemEval (Wu et al., arXiv:2410.10813) | 2025 | 5 memory abilities, 115K-1.5M tokens. Different strategies needed per ability | 4/5 |
| GroUSE (arXiv:2409.06595) | 2025 | Meta-evaluation: 7 RAG failure modes, 144 unit tests | 4/5 |
| ConSens (arXiv:2505.00065) | 2025 | Context grounding metric via perplexity contrast | 3/5 |

---

### 10. Information Foraging & Decision Theory

| Paper | Year | Key Contribution | Relevance |
|-------|------|-----------------|-----------|
| Proactive Info Gathering (Huang et al., arXiv:2507.21389) | 2025 | POMDP formulation of information gathering, +18% over o3-mini | 5/5 |
| ReAtt (Jiang et al., arXiv:2212.02027) | 2022 | Retrieval as attention within a single transformer | 4/5 |
| Learning ICL Examples (Su et al., arXiv:2307.07164) | 2024 | Reward-driven ICL example retrieval | 4/5 |

---

## Gap Analysis

### Gap 1: No unified policy over heterogeneous address spaces
Every paper optimizes within a single substrate. Self-RAG adapts *when* to retrieve. SmartRAG jointly optimizes retrieval *decisions*. MAGMA traverses *memory graphs*. SWE-agent shows *interface* matters. But NO paper proposes a single policy that routes across memory, documents, tools, search, and workspace operations. **This is the core gap our work fills.**

### Gap 2: Discovery vs. knowledge state is unmodeled
No paper explicitly separates "knowing something exists" from "knowing its contents." The proactive info gathering work (POMDP) is closest but addresses clarification questions, not document exploration. **Our discovery/knowledge state formalization is novel.**

### Gap 3: Evidence bundles are not scored as sets
All systems score individual passages/chunks. None evaluate whether a *set* of evidence pieces jointly satisfies a question's requirements (definition + number + exception + reference). **Bundle scoring is a methodological contribution.**

### Gap 4: No cost-normalized evaluation across substrates
Cost-Aware Retrieval and SmartRAG consider cost within single substrates. No benchmark evaluates Utility@Budget across memory + documents + tools + search simultaneously. **Our Utility@Budget metric across regimes is novel.**

### Gap 5: No heterogeneous environment benchmark
LoCoMo/LongMemEval test memory. BRIGHT/NoLiMa test retrieval. LoCoBench tests code. No benchmark tests mixed corpora (PDFs + spreadsheets + code + APIs + conversations) with mixed tasks requiring adaptive substrate selection. **Our custom benchmark fills this gap.**

---

## Baselines to Beat

| System | Benchmark | Metric | Value |
|--------|-----------|--------|-------|
| MAGMA | LoCoMo | Judge score | 0.700 |
| MAGMA | LongMemEval | Avg accuracy | 61.2% |
| Nemori | LoCoMo/LongMemEval | Tokens | 88% fewer than full context |
| DIVER | BRIGHT | nDCG@10 | 46.8 |
| DeepRetrieval | Publication search | Recall | 65.07% |
| Search-R1 | QA (Qwen2.5-7B) | Accuracy | +41% over RAG |
| Chain of Agents | QA/summarization | Accuracy | +10% over RAG and full-context |
| HopRAG | Multi-hop QA | Answer metric | +76.78% over conventional RAG |
| SmartRAG | Knowledge QA | Accuracy | Joint > separate optimization |
| Meta-Harness | Text classification | Accuracy | +7.7pt, 4x fewer tokens |

---

## Proposed Research Directions

### Direction A: Harness-Level Agentic Attention Policy (PRIMARY)
**Gap addressed:** #1, #2, #4
**What:** Build a harness-level policy π(a|s) that routes across heterogeneous address spaces (memory graph, document tree, lexical index, vector index, tool APIs, workspace) with explicit discovery/knowledge state tracking and cost-aware stopping.
**Why it could work:** SmartRAG shows joint optimization wins. Meta-Harness shows harness-level optimization yields large gains. Chain of Agents shows multi-strategy approaches beat fixed pipelines. The RL-for-retrieval cluster (DeepRetrieval, Search-R1, Knowledgeable-r1) shows learned retrieval policies are highly effective.
**Risk:** Complexity of the full action space may make learning hard. Mitigation: start with heuristic policy, learn incrementally.

### Direction B: Discovery-vs-Knowledge Evaluation Framework (SECONDARY)
**Gap addressed:** #2, #5
**What:** Build a benchmark that explicitly tests discovery operations (locating relevant sources) separately from knowledge operations (extracting and grounding answers). Mixed corpora: PDFs, spreadsheets, code, APIs, conversations.
**Why it matters:** No existing benchmark tests what our method uniquely provides. Without it, gains could be attributed to "just using more tools."
**Risk:** Benchmark construction is labor-intensive. Mitigation: semi-automated generation with human validation.

### Direction C: Evidence Bundle Scoring (SUPPORTING)
**Gap addressed:** #3
**What:** Score evidence as sets rather than individual chunks. A question may need {definition, number, exception, reference} — evaluate whether the bundle covers the need.
**Why it matters:** Bundle scoring is what makes the stopping rule meaningful. Without it, the system can't know when it has "enough."
**Risk:** Defining bundle requirements per question is hard. Mitigation: use LLM-based requirement decomposition.

---

## Recommendation

**Pursue Direction A as the primary research direction**, with Direction B as a necessary evaluation contribution and Direction C as a component within the method. This matches the plan from Phase 0 and is strongly supported by the literature evidence.

The literature review confirms:
1. The RL-for-retrieval explosion (2024-2025) validates that learned retrieval policies work.
2. No existing system combines multiple address spaces under a single policy.
3. The harness/context-engineering framing (Meta-Harness, Anthropic) is gaining traction but lacks a formal method.
4. The evaluation gap (no heterogeneous benchmark) is both a risk and an opportunity.

## Next Steps
- Phase 2: Formalize hypothesis with mathematical justification
- Focus on H4 ("biggest gains from routing, not substrate") as the central testable claim
