# Agentic Attention: Harness-Level Adaptive External Attention for LLM Systems

## Research Summary

This project investigates whether a harness-level policy that adaptively chooses among multiple address spaces and context operations can outperform fixed retrieval pipelines on grounded QA under a fixed budget.

**Core claim:** Retrieval should be formalized as external attention allocation over an information environment, not as one-shot chunk selection. The real problem is a policy over context operations (retrieve, pin, expand, compress, evict, refresh, delegate) across heterogeneous address spaces (vectors, terms, file trees, knowledge graphs, SQL, APIs, memory slots).

## Method: Adaptive External Attention (AEA)

A harness-level policy that selects the next best address-operation pair:

```
(a_t, x_t) = argmax_{a,x} Value(q, s_t, a, x) - Cost(a, x)
```

Where the system state tracks discovery state, knowledge state, active workspace, interaction history, and remaining budget.

### Core Modules
- **Router** — chooses which address space to consult next
- **Operator selector** — chooses what to do (preview, retrieve, expand, compress, tool-call, stop)
- **Workspace manager** — maintains active context, summaries, pinned constraints
- **Evidence bundle scorer** — scores sets of evidence, not individual chunks
- **Stopping rule** — stops when support coverage and consistency are sufficient

## Evaluation

Three evaluation regimes:
1. **Memory-centric** (LoCoMo, LongMemEval) — comparison with MAGMA
2. **Retrieval/long-context** (BRIGHT, NoLiMa, RULER, LongBench Pro) — stress tests
3. **Heterogeneous environment** (custom benchmark) — mixed corpora and tasks

## Project Structure

```
research-log/           # One .md per research event
experiments/            # Code, scripts, configs
experiments/poc/        # Proof-of-concept code (Phase 3)
experiments/aea/        # Core AEA framework (Phase 4)
  types.py              # Shared data types: AgentState, Action, EvidenceBundle, …
  address_spaces/       # Retrieval backends
    base.py             # AddressSpace ABC
    semantic.py         # Dense retrieval (sentence-transformers)
    lexical.py          # BM25 keyword retrieval (rank_bm25)
    entity_graph.py     # Entity co-occurrence graph (regex NER + BFS)
  evaluation/           # Immutable evaluation infrastructure
    metrics.py          # EM, F1, support_recall, utility_at_budget, …
    harness.py          # EvaluationHarness
  policies/             # Routing policies
    base.py             # Policy ABC
    single_substrate.py # π_semantic, π_lexical, π_entity baselines
    heuristic.py        # π_aea_heuristic (adaptive hand-designed routing)
    ensemble.py         # π_ensemble (query all substrates)
experiments/configs/    # Configuration files
data/                   # Datasets, intermediate results
paper/                  # Living document sections
paper/figures/          # Generated plots and diagrams
prompts/                # Subagent prompt templates
```

## Training Progression

1. **Heuristic policy** — hand-designed routing rules
2. **Learned scoring** — router and bundle scorer from logged trajectories
3. **Budget-aware RL** — optimize downstream utility directly

## Experiments

### Phase 0 PoC: Substrate Switching Validation

#### Run 1 — Hardcoded examples (2026-04-04)

**Location:** `experiments/poc/substrate_switching_poc.py`

**Goal:** Validate the core assumption that within-task substrate switching occurs on real multi-hop tasks and that an adaptive policy outperforms single-substrate baselines.

**Design:**
- Dataset: 20 bridge-type questions (HotpotQA format, hardcoded examples)
- Address space A₁: Semantic search via `all-MiniLM-L6-v2` embeddings + cosine similarity
- Address space A₂: Entity link hop via regex-based NER + co-occurrence graph
- Three policies: `π_semantic` (always A₁), `π_graph` (always A₂), `π_heuristic` (adaptive: A₁ for entry, A₂ for bridge hops)
- Oracle analysis: per-step substrate attribution based on which substrate actually finds gold docs

**Key Results:**

| Policy | SupportRecall | StepsToFirst | TotalOps |
|---|---|---|---|
| π_semantic | 0.875 | 1.00 | 2.15 |
| π_graph | 1.000 | 1.05 | 5.00 |
| π_heuristic | 1.000 | 1.00 | 2.00 |

Oracle: **3/20 (15%)** of questions required substrate switching. Low rate due to "easy bridge" data artefact — both gold entity names appear directly in the question, making semantic search alone sufficient.

**Raw results:** `experiments/poc/poc_results.json`

---

#### Run 2 — Real HotpotQA distractor validation set (2026-04-04)

**Location:** `experiments/poc/substrate_switching_real_hotpotqa.py`

**Goal:** Re-run with authentic HotpotQA bridge questions (first 50 of 5918 bridge-type examples) to get a more accurate switching rate.

**Design:**
- Dataset: HotpotQA distractor split, validation, bridge-type — 50 questions (out of 5,918 available)
- Same A₁/A₂ implementations and three policies as Run 1
- Additional metric: Utility@Budget with η=0.5, μ=0.3

**Key Results:**

| Policy | SupportRecall | StepsToFirst | TotalOps | Utility@Budget |
|---|---|---|---|---|
| π_semantic | 0.630 | 1.18 | 2.52 | 0.1890 |
| π_graph | 0.930 | 1.46 | 5.80 | 0.1750 |
| π_heuristic | 0.930 | 1.38 | 3.04 | **0.3130** |

Oracle: **22/50 (44.0%)** of questions required substrate switching.
- 50% solvable by A₁ alone, 6% by A₂ alone, 44% require both.
- Step 1 oracle: A₁=92%, A₂=8%. Step 2 oracle: A₁=12%, A₂=88%.

**Interpretation:**
- Switching rate of 44% on real data — falls squarely in the revised 40-60% expectation.
- `π_heuristic` matches `π_graph`'s recall (0.930) while using ~48% fewer operations (3.04 vs 5.80), yielding a substantially better Utility@Budget (0.313 vs 0.175).
- `π_semantic` undershoots on recall (0.630) because it cannot navigate the implicit bridge hop that characterizes authentic HotpotQA bridge questions.
- The A₁→A₂ step pattern is confirmed: oracle uses semantic search to enter, entity hop to bridge.

**Raw results:** `experiments/poc/poc_results_real_hotpotqa.json`

**Dependencies:** `experiments/poc/requirements.txt` (`sentence-transformers>=2.2.0`, `datasets>=2.0.0`, `numpy>=1.21.0`)

## Experiments

### HotpotQA Bridge Baselines (Phase 4a)

100 bridge questions, distractor split.  BM25 is the strongest single-substrate baseline (Utility@Budget=0.0169); AEA heuristic is close behind (0.0163).  High lexical overlap makes BM25 near-optimal here — consistent with H2.

### Heterogeneous Benchmark (Phase 4b — Regime C)

**File:** `experiments/benchmarks/heterogeneous_benchmark.py`
**Runner:** `experiments/run_heterogeneous_benchmark.py`
**Results:** `experiments/results/heterogeneous_benchmark.json`

100 synthetic questions across 6 task types designed so no single address space can solve all tasks:

| Task Type | N | Gold Docs | Key Challenge |
|---|---|---|---|
| Entity Bridge | 20 | 2 | Person→country→capital; bridge entity not in question |
| Implicit Bridge | 20 | 2 | Movie→director (implicit)→award |
| Semantic + Computation | 20 | 2 | Find both companies, compare revenue numerically |
| Low Lexical Overlap | 20 | 1 | Synonyms/paraphrases defeat BM25 |
| Multi-Hop Chain | 10 | 3 | Three-hop A→B→C |
| Discovery + Extraction | 10 | 2 | Index paragraph → department budget |

**Results (N=100):**

| Policy | SupportRecall | SupportPrec | AvgOps | Utility@Budget |
|--------|--------------|-------------|--------|----------------|
| π_semantic | 0.9183 | 0.3340 | 2.00 | 0.0270 |
| π_lexical | 0.9100 | 0.3360 | 2.00 | 0.0187 |
| π_entity | 0.7733 | 0.6197 | 3.00 | 0.0065 |
| π_ensemble | 0.9600 | 0.2952 | 3.00 | 0.0113 |
| π_aea_heuristic | 0.9333 | 0.3825 | 2.17 | 0.0177 |

AEA substrate switching: 57/100 questions used multiple substrates; AEA outperformed best single-substrate on 27/100 questions; avg 1.57 substrates per question.

## Status

Phase 0: Research setup — complete.
Phase 1: Literature review — complete (63 papers, 5 gaps identified).
Phase 2: Hypothesis formation — complete (RIGOROUS after theory review).
Phase 3: PoC validation — complete (Run 1: mechanism confirmed; Run 2: 44% switching rate on real HotpotQA).
Phase 4: Full experiments — in progress.
  - Core AEA framework implemented (`experiments/aea/`): types, three address spaces,
    immutable evaluation harness, five policies (semantic-only, lexical-only,
    entity-only, AEA heuristic, ensemble).
  - Phase 4a: HotpotQA baselines — complete.
  - Phase 4b (Regime C): Heterogeneous benchmark — complete (100 synthetic questions,
    6 task types, all 5 policies evaluated).
  - Next: MuSiQue / BRIGHT-style runs, LLM-based answer generation for EM/F1.
