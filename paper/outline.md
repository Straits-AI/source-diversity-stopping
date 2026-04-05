# Paper Outline

## Title
**Adaptive External Attention: Cost-Efficient Retrieval Through Selective Operation Avoidance Over Heterogeneous Address Spaces**

*Alternative:* Agentic Attention: When Knowing What Not To Do Beats Knowing What To Do in Retrieval-Augmented Systems

## Key Claim (revised from Phase 5 analysis)
Harness-level adaptive attention over heterogeneous address spaces improves retrieval-augmented QA primarily through **cost-efficient operation selection** — knowing when to stop and what not to do — rather than through positive substrate discovery. Selective routing avoidance yields up to 67% better Utility@Budget than fixed pipelines.

---

## Abstract (150-300 words)
- Problem: Fixed retrieval pipelines commit to a single addressing primitive regardless of task characteristics
- Gap: No system learns a unified policy over heterogeneous address spaces with cost-aware operation selection
- Method: AEA — harness-level policy with coverage-driven stopping over semantic, lexical, and entity-graph substrates
- Key result: +67% Utility@Budget on HotpotQA through cost efficiency; ablation reveals routing avoidance (knowing what NOT to do) is more valuable than positive substrate selection
- Significance: Reframes adaptive retrieval from "choose the right tool" to "avoid unnecessary operations"

## 1. Introduction
- Motivate: RAG systems commit to one retrieval primitive; the real problem is a policy over context operations
- State the gap: no harness-level policy over heterogeneous address spaces
- Present revised thesis: adaptive attention = cost-efficient operation selection
- Contributions (numbered):
  1. Formalization of retrieval as harness-level external attention over heterogeneous address spaces with discovery/knowledge state tracking
  2. AEA prototype with coverage-driven routing policy
  3. A heterogeneous evaluation benchmark designed to require cross-substrate navigation
  4. Finding: routing avoidance (selective non-action) dominates positive routing in current heuristic policies
- Outline structure

## 2. Related Work
Organized by technique family:

### 2.1 Retrieval-Augmented Generation
- Classical RAG: BM25, dense retrieval, hybrid
- Improved: Contextual Retrieval, RAPTOR, PageIndex
- Adaptive: Self-RAG, FLARE, Adaptive-RAG

### 2.2 Agentic Retrieval & Tool Use
- ReAct, Toolformer, SWE-agent
- Agentic RAG Survey
- Chain of Agents, ExpertRAG
- NousResearch Hermes Agent (ad hoc multi-substrate without formalization)

### 2.3 RL for Retrieval Optimization
- DeepRetrieval, SmartRAG, Search-R1
- Knowledgeable-r1, RAG-DDR
- Cost-Aware Retrieval

### 2.4 Structured Memory
- MAGMA, Nemori, A-MEM
- MemoryBank, Zep

### 2.5 Context Engineering
- Meta-Harness
- Anthropic's context engineering
- Context compression for tools

### 2.6 Long-Context Evaluation
- RULER, NoLiMa, HELMET, Context Rot
- Key finding: synthetic tasks don't predict downstream (HELMET)

### Positioning
Our work differs from all above: we formalize the POLICY over heterogeneous substrates, not any individual substrate. Closest: SmartRAG (joint optimization but single pipeline), Meta-Harness (harness optimization but code-level, not runtime). The Hermes Agent implements ad hoc multi-substrate memory without the formal harness-level policy we propose.

## 3. Method

### 3.1 Problem Formulation
- Information environment E
- State s_t = (q, D_t, K_t, W_t, H_t, B_t)
- Discovery vs knowledge state distinction
- Action space as union of address-space-specific operations
- CMDP formulation with Lagrangian relaxation

### 3.2 Address Spaces
- Semantic (vector), Lexical (BM25), Entity Graph
- Common interface: query(state, operation, params) → result + cost
- Options framework: each space is an option with internal policy

### 3.3 Coverage-Driven Routing Policy
- Step 0: Semantic entry (always)
- Step 1+: Coverage-driven decision
  - If 2+ high-relevance items from different sources → STOP
  - If evidence from single source + multi-hop question → entity HOP
  - Otherwise → lexical fallback
- Workspace management: pin top-k, evict low-relevance

### 3.4 Theoretical Justification
- Weak dominance theorem (superset argument)
- Strict dominance condition with quantitative bound
- Toy example (2-substrate, 3-step)
- Connection to options framework and information foraging theory

## 4. Experimental Setup

### 4.1 Benchmarks
- HotpotQA bridge subset (100 questions) — lexically rich multi-hop
- Heterogeneous v2 (100 synthetic questions, 6 task types) — controlled substrate requirements
- MuSiQue (100 questions) — shortcut-resistant multi-hop [if results available]

### 4.2 Baselines
- BM25 (lexical only)
- Dense retrieval (semantic only)
- Entity graph only
- Ensemble (all substrates, no routing)
- AEA heuristic (adaptive routing)

### 4.3 Ablations
- No early stop
- Semantic only + smart stop
- No entity hops
- Always hop
- No workspace management

### 4.4 Metrics
- Primary: Utility@Budget = AnswerScore × (1+η·EvidenceScore) − μ·Cost
- Support recall, support precision
- Average operations per question
- η=0.5, μ=0.3 (fixed before experiments)

### 4.5 Implementation
- Base LLM: not used for answer generation (retrieval-only evaluation)
- sentence-transformers all-MiniLM-L6-v2 for embeddings
- BM25 via rank_bm25
- Entity extraction via regex NER
- Seeds: 42

## 5. Results

### 5.1 Main Results
- Table: all policies × all benchmarks
- AEA wins on HotpotQA (+67% U@B over BM25)
- AEA near-ties semantic on heterogeneous v2
- Per-task-type breakdown showing where AEA wins/loses

### 5.2 Ablation Analysis
- Table: all ablations × both benchmarks
- Key finding: abl_always_hop catastrophic (-0.1146)
- Key finding: abl_no_entity_hop actually improves on HotpotQA
- Component contribution analysis

### 5.3 Routing Avoidance as the Dominant Mechanism
- The AEA's advantage comes from NOT doing expensive operations
- Coverage-driven stopping reduces ops from 2.00 to 1.21
- Selective avoidance worth 0.1146 U@B (always_hop ablation)
- Comparison with ensemble: brute force loses to selective avoidance

### 5.4 Within-Task Substrate Switching
- PoC validation: 44% switching rate on real HotpotQA
- Oracle step analysis: A₁ (semantic) at step 1, A₂ (entity) at step 2
- But current heuristic rarely exploits this productively

## 6. Discussion

### 6.1 Reframing Adaptive Retrieval
- Original thesis: "choose the right substrate"
- Evidence-based thesis: "avoid unnecessary operations"
- This is still a routing decision — negative selection is selection
- Implications for system design: default to cheapest operation, only escalate when coverage is insufficient

### 6.2 Limitations
- No LLM answer generation (retrieval-only evaluation)
- Heuristic policy only (no learned router)
- Limited to 3 address spaces (no tool execution, no structural navigation)
- Synthetic benchmark may not reflect real-world heterogeneity
- Single seed, no statistical significance testing across seeds
- HotpotQA is well-studied and may not generalize

### 6.3 When Does Adaptive Routing Help Most?
- When the action space includes expensive operations that are often unnecessary
- When tasks vary in which substrate is optimal (heterogeneous workloads)
- When budget constraints make cost efficiency critical
- NOT when a single substrate dominates (e.g., BM25 on lexically-rich tasks)

### 6.4 Future Work (evidence-based only)
- Learned router trained on trajectory data (could unlock positive routing)
- More address spaces (tool execution, structural navigation, web search)
- Full pipeline with LLM answer generation
- External benchmarks: BRIGHT, NoLiMa, RULER
- Budget-aware RL optimization

## 7. Conclusion
- Contributions: formalization + method + benchmark + finding
- Main result: adaptive attention through selective operation avoidance
- Practical implication: build retrieval systems that know when to stop, not just what to do
- The gap between routing avoidance and routing optimization suggests a fruitful direction for learned policies

## References
~40 papers from literature review

---

## Section → Research Log Mapping

| Section | Source |
|---------|--------|
| Abstract, Introduction | 000-setup.md, 005-analysis-iter-0.md |
| Related Work | 001-literature-review.md |
| Method (3.1-3.2) | 002-hypothesis.md |
| Method (3.3) | heuristic.py source |
| Method (3.4) | 002-hypothesis.md (theorems) |
| Experimental Setup | 004-experiment-plan.md |
| Results (5.1) | 004a, 004b, 004c logs |
| Results (5.2) | 004d ablation study |
| Results (5.3-5.4) | 005-analysis-iter-0.md, 003-poc logs |
| Discussion | 005-analysis-iter-0.md |

## Parallelizable Writing Groups

**Group 1 (parallel):** Related Work (2), Methodology (3), Experimental Setup (4)
**Group 2 (after Group 1):** Results (5), Discussion (6)
**Group 3 (after Group 2):** Introduction (1), Abstract, Conclusion (7)
