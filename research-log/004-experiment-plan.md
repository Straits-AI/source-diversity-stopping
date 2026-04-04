# Experiment Plan

**Date:** 2026-04-05
**Phase:** 4
**Iteration:** 0
**Status:** in-progress

## Context
Phase 3 PoC validated the core mechanism: 44% within-task substrate switching on real HotpotQA, adaptive routing achieves best Utility@Budget. Ready for full experiments.

---

## 1. Experiment Architecture

### 1.1 Base LLM (Fixed)
- **Model:** Claude 3.5 Sonnet (via Anthropic API)
- **Rationale:** Strong reasoning, tool use, long context. Held fixed across ALL experiments — only the harness-level policy varies.

### 1.2 AEA Prototype Modules

**Router** — Selects which address space to consult next.
- Heuristic v1: Rule-based (query type → address space mapping)
- Learned v1: Small classifier trained on logged trajectories

**Operator Selector** — Selects what to do in the chosen address space.
- Actions: {search, preview, open, expand, compress, evict, hop, tool_call, stop}

**Workspace Manager** — Maintains active context window.
- Pin: keep critical evidence across steps
- Evict: remove low-relevance items when budget is tight
- Compress: summarize stale items

**Evidence Bundle Scorer** — Evaluates evidence set completeness.
- Decomposes question into requirements
- Scores: bundle_coverage = requirements_satisfied / total_requirements

**Stopping Rule** — Decides when to return answer.
- Stop when bundle_coverage ≥ τ AND confidence ≥ σ, or budget exhausted

### 1.3 Address Spaces

| Space | Type | Implementation |
|-------|------|---------------|
| A_vec | Geometric | FAISS index + sentence-transformers embeddings |
| A_lex | Lexical | BM25 via rank_bm25 |
| A_struct | Structural | Document tree navigator (section/page hierarchy) |
| A_entity | Relational | Entity co-occurrence graph + BFS hops |
| A_tool | Executable | Python code execution, SQL queries |
| A_memory | Contextual | Working memory slots (pinned evidence) |

### 1.4 Evaluation Regimes

**Regime A: Memory-Centric** (compare with MAGMA)
- LoCoMo, LongMemEval
- Address spaces: A_vec, A_entity, A_memory
- Measures: judge score, accuracy, token efficiency

**Regime B: Retrieval & Long-Context** (stress tests)
- HotpotQA (bridge subset), MuSiQue
- Address spaces: A_vec, A_lex, A_entity
- Measures: EM, F1, support recall, Utility@Budget

**Regime C: Heterogeneous Environment** (custom benchmark)
- Custom discovery-vs-knowledge tasks (to be built)
- Address spaces: ALL (A_vec, A_lex, A_struct, A_entity, A_tool, A_memory)
- Measures: Utility@Budget, discovery efficiency, substrate selection accuracy

---

## 2. Run Schedule

### Phase 4a: Baselines (sequential)

| Run ID | System | Regime | Est. Time | Est. API Cost |
|--------|--------|--------|-----------|--------------|
| base-bm25 | BM25 top-k retrieval | B | 1 hour | $5 |
| base-dense | Dense retrieval (FAISS) | B | 1 hour | $5 |
| base-hybrid | BM25 + dense + reranker | B | 2 hours | $10 |
| base-react | ReAct agent (search tool) | B | 2 hours | $30 |
| base-fullctx | Full context (no retrieval) | B | 1 hour | $20 |
| base-selfrag | Self-RAG style (adaptive retrieve/skip) | B | 2 hours | $30 |

**Checkpoint:** Baseline results must match literature within reasonable margin before proceeding.

### Phase 4b: Core AEA Experiment (sequential)

| Run ID | System | Regime | Est. Time | Est. API Cost |
|--------|--------|--------|-----------|--------------|
| aea-heuristic-B | AEA heuristic policy | B | 3 hours | $40 |
| aea-heuristic-C | AEA heuristic policy | C | 3 hours | $40 |

**Checkpoint:** AEA must beat best baseline on Utility@Budget before proceeding to ablations.

### Phase 4c: Ablation Studies (after checkpoint)

| Run ID | What's Ablated | Regime | Purpose |
|--------|---------------|--------|---------|
| abl-no-router | Fixed single substrate | B, C | Test H4: is routing the main contributor? |
| abl-no-struct | Remove A_struct | C | Is structural navigation necessary? |
| abl-no-entity | Remove A_entity | B | Is entity graph necessary? |
| abl-no-tool | Remove A_tool | C | Is executable addressing necessary? |
| abl-no-bundle | Individual chunk scoring | B, C | Is bundle scoring necessary? |
| abl-no-workspace | No pin/evict/compress | B, C | Is workspace management necessary? |
| abl-no-dk | No discovery/knowledge split | B, C | Is D/K state necessary? |
| abl-ensemble | Query all substrates always | B, C | Is intelligent routing better than brute force? |

### Phase 4d: Scaling & Robustness (after ablations)

| Run ID | What Varies | Purpose |
|--------|------------|---------|
| scale-budget-low | 50% token budget | Cost sensitivity |
| scale-budget-high | 200% token budget | Diminishing returns? |
| seed-1 through seed-5 | Random seeds [42,123,456,789,1024] | Reproducibility |

---

## 3. Implementation Order

The practical implementation follows this sequence:

### Step 1: Evaluation Infrastructure
- Download/prepare HotpotQA distractor split (done), MuSiQue
- Build evaluation harness: EM, F1, support precision/recall, bundle coverage, Utility@Budget
- Build cost tracker: tokens, latency, operations count
- This is IMMUTABLE once built (per evaluation contract)

### Step 2: Address Space Implementations
- A_vec: FAISS index builder + query interface
- A_lex: BM25 index builder + query interface  
- A_entity: Entity extraction + graph builder + BFS query
- A_struct: Document tree builder (for custom benchmark)
- A_tool: Python/SQL execution sandbox
- A_memory: Working memory with pin/evict/compress operations
- Each space exposes a common interface: query(state) → results + cost

### Step 3: Baseline Implementations
- BM25/dense/hybrid: standard retrieval → LLM answer
- ReAct: LLM + search tool in a loop
- Full context: stuff everything into context
- Self-RAG style: LLM decides whether to retrieve

### Step 4: AEA Heuristic Policy
- Router rules: query type classification → address space selection
- Operator selection: fixed operation per address space
- Workspace manager: simple LRU eviction + pinning of high-score evidence
- Bundle scorer: LLM-based requirement decomposition + coverage check
- Stopping rule: coverage threshold

### Step 5: Run Baselines (Phase 4a)
### Step 6: Run Core AEA (Phase 4b)  
### Step 7: Checkpoint with User
### Step 8: Run Ablations (Phase 4c)
### Step 9: Run Scaling/Robustness (Phase 4d)

---

## 4. Custom Benchmark Design (Regime C)

### Task Types

| Task | Address Spaces Required | Discovery/Knowledge Test |
|------|------------------------|------------------------|
| Cross-format QA | A_vec + A_struct + A_tool | Find file → navigate structure → extract answer |
| Computation grounding | A_vec + A_tool | Find text context → execute calculation → verify |
| Multi-source synthesis | A_vec + A_lex + A_entity | Gather evidence from 3+ sources → synthesize |
| Discovery-only | A_struct | Identify which file/section, don't extract content |
| Knowledge-only | A_vec | Given known location, extract specific information |
| Budget-constrained | ALL | Answer under strict 500-token budget |

### Corpus
- 5 PDFs (financial reports, policy documents)
- 3 code repositories (Python, with docstrings)
- 2 spreadsheets (CSV with structured data)
- 1 knowledge graph (entity relationships)
- 1 conversational history (20-turn dialogue)

Target: 100 questions, 30-60% requiring cross-substrate navigation.

---

## 5. Adaptation Rules

After each run, apply these decision rules:

**Baseline doesn't match literature:** Debug retrieval pipeline. Do NOT proceed until baseline is reasonable.

**AEA loses to best baseline on Utility@Budget:**
- If loses on accuracy: check router is selecting appropriate substrates
- If loses on cost: check for unnecessary routing overhead (FM2)
- If loses on both: fundamental issue, loop back to hypothesis

**Single ablation causes >50% of total improvement to disappear:**
- That component is the main contributor → revise the narrative
- If it's the router: H4 is supported
- If it's a single substrate: H4 is not supported (honest negative)

**Ensemble matches AEA on Utility@Budget:**
- The routing contribution is minimal → the contribution is multi-substrate access, not intelligent routing
- Revise paper framing accordingly

---

## 6. Estimated Total Resources

| Resource | Estimate |
|----------|----------|
| API calls (Anthropic) | ~$300-500 total |
| Local compute | Laptop sufficient for indices + heuristic policy |
| Dataset preparation | 1-2 days |
| Baseline implementation | 2-3 days |
| AEA prototype | 3-5 days |
| Custom benchmark | 2-3 days |
| Experiment runs | 3-5 days |
| Analysis | 2-3 days |
| **Total Phase 4** | **~3-4 weeks** |

## Next Steps
- ~~Step 1: Evaluation infrastructure~~ — **done** (`experiments/aea/evaluation/`)
- ~~Step 2: Address space implementations (A_vec, A_lex, A_entity)~~ — **done** (`experiments/aea/address_spaces/`)
- ~~Step 4: AEA Heuristic Policy~~ — **done** (`experiments/aea/policies/heuristic.py`)
- Step 3: Baseline implementations (BM25/dense/hybrid/ReAct/full-context/Self-RAG)
- Step 5: Run Baselines (Phase 4a)
- Step 6: Run Core AEA (Phase 4b)
- Step 7: Checkpoint with User

**Framework status (2026-04-04):**
The `experiments/aea/` package is complete and end-to-end tested.
All five policies (SemanticOnly, LexicalOnly, EntityOnly, AEAHeuristic, Ensemble)
run through the immutable EvaluationHarness and produce correct per-example and
aggregated metrics including Utility@Budget.
