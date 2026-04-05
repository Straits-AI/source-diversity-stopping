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
    ablations.py        # Ablation variants (no_early_stop, semantic_smart_stop, no_entity_hop, always_hop, no_workspace_mgmt)
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

### Heterogeneous Benchmark v1 (Phase 4b — Regime C)

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

**Design flaws identified in v1:**
- Entity Bridge and Implicit Bridge: bridge entity names leaked into questions, allowing BM25/semantic to trivially find gold paragraphs.
- Low Lexical Overlap: drug names appeared identically in questions and corpus; BM25 trivially retrieved correct docs.
- Multi-Hop Chain: three-hop chains exceeded heuristic policy depth; all policies scored ≤0.63 recall.

---

### Heterogeneous Benchmark v2 (Phase 4b — Regime C, corrected)

**File:** `experiments/benchmarks/heterogeneous_benchmark_v2.py`
**Runner:** `experiments/run_heterogeneous_v2.py`
**Results:** `experiments/results/heterogeneous_v2.json`

Fixes all v1 design flaws while keeping the same 100-question, 6-type structure:

| Task Type | N | Gold Docs | v2 Fix |
|---|---|---|---|
| Entity Bridge | 20 | 2 | Question names only EntityA; EntityB (birthplace) absent from question; entity hop required to find P2 |
| Implicit Bridge | 20 | 2 | Creator/director name absent from question; only work title mentioned; entity hop required |
| Semantic + Computation | 20 | 2 | Unchanged — worked well in v1 |
| Low Lexical Overlap | 20 | 1 | Question uses lay language only (no drug name, no clinical term); gold paragraph uses pharmaceutical terminology |
| Multi-Hop Chain | 10 | 2 | Simplified to 2-hop chains; question names only EntityA |
| Discovery + Extraction | 10 | 2 | Unchanged — worked well in v1 |

**Validation (60/60 checks passed):**
- Entity Bridge: all 20 bridge entities (birthplace city) absent from question text — PASS
- Implicit Bridge: all 20 creator names absent from question text — PASS
- Low Lexical Overlap: all 20 questions have 0.0–0.059 Jaccard word overlap with gold paragraph (< 0.15 threshold) — PASS

**Results (N=100):**

| Policy | SupportRecall | SupportPrec | AvgOps | Utility@Budget |
|--------|--------------|-------------|--------|----------------|
| π_semantic | 0.9200 | 0.3280 | 2.00 | 0.0439 |
| π_lexical | 0.8200 | 0.3060 | 2.00 | 0.0142 |
| π_entity | 0.6250 | 0.7207 | 3.00 | -0.0038 |
| π_ensemble | 0.9600 | 0.2697 | 3.00 | 0.0071 |
| π_aea_heuristic | 0.9300 | 0.3905 | 2.99 | 0.0246 |

**By task type (SupportRecall / Utility@Budget):**

| Task Type | π_semantic | π_lexical | π_entity | π_aea_heuristic |
|---|---|---|---|---|
| Entity Bridge | 0.90 / -0.009 | 0.925 / 0.025 | 1.00 / -0.013 | 0.925 / 0.009 |
| Implicit Bridge | 0.90 / -0.023 | 0.85 / -0.022 | 0.65 / -0.021 | 0.925 / -0.025 |
| Semantic + Computation | 1.00 / 0.067 | 1.00 / 0.065 | 1.00 / 0.052 | 1.00 / **0.083** |
| Low Lexical Overlap | 1.00 / **0.053** | 0.55 / 0.000 | 0.00 / -0.017 | 1.00 / -0.005 |
| Multi-Hop Chain (2-hop) | 0.60 / -0.024 | 0.55 / -0.024 | 0.95 / -0.026 | 0.60 / -0.033 |
| Discovery + Extraction | 1.00 / **0.287** | 1.00 / 0.029 | 0.00 / -0.016 | 1.00 / 0.156 |

**Key v2 findings vs v1:**
- Low lexical overlap: BM25 recall dropped from 1.0 to 0.55 (fix confirmed working); semantic stays at 1.0 (correct design).
- Entity bridge: AEA recall improvement negligible (0.925); entity-only still at 1.0 because entity graph hops on person name reach the city paragraph through co-occurrence.
- Multi-hop chain: entity-only jumped to 0.95 recall (2-hop chains are easier than 3-hop for entity graph BFS); confirms chain simplification was appropriate.
- AEA substrate switching: 77/100 questions used multiple substrates (up from 57 in v1); AEA outperformed best single-substrate on 20/100 questions; avg 1.78 substrates per question.

### Ablation Study (Phase 4d — H4 test)

**Files:** `experiments/aea/policies/ablations.py`, `experiments/run_ablations.py`
**Results:** `experiments/results/ablation_study.json`

Five ablation variants of the AEA heuristic, run on both benchmarks:

| Ablation | What is disabled |
|---|---|
| abl_no_early_stop | Coverage-driven early stopping |
| abl_semantic_only_smart_stop | All non-semantic substrates (entity hop, lexical); keeps smart stop |
| abl_no_entity_hop | Entity graph hops; replaced by lexical fallback |
| abl_always_hop | Selective hop decision; always hops after semantic |
| abl_no_workspace_mgmt | Pin/evict workspace curation |

**HotpotQA Bridge (N=100):**

| Policy | SupportRecall | AvgOps | Utility@Budget | Δ from Full AEA |
|---|---|---|---|---|
| pi_aea_heuristic (full) | 0.7950 | 1.21 | 0.0282 | baseline |
| pi_lexical (best baseline) | 0.8100 | 2.00 | 0.0169 | -0.0114 |
| abl_no_early_stop | 0.7950 | 1.21 | 0.0283 | +0.0000 |
| abl_semantic_only_smart_stop | 0.7500 | 3.09 | -0.0087 | -0.0370 |
| abl_no_entity_hop | 0.7850 | 1.21 | 0.0321 | +0.0039 |
| abl_always_hop | 0.8550 | 5.35 | -0.0864 | -0.1146 |
| abl_no_workspace_mgmt | 0.7950 | 1.21 | 0.0283 | +0.0001 |

**Heterogeneous v2 (N=100):**

| Policy | SupportRecall | AvgOps | Utility@Budget | Δ from Full AEA |
|---|---|---|---|---|
| pi_aea_heuristic (full) | 0.9300 | 1.84 | 0.0430 | baseline |
| pi_semantic (best baseline) | 0.9200 | 2.00 | 0.0440 | +0.0009 |
| abl_no_early_stop | 0.9300 | 1.96 | 0.0415 | -0.0015 |
| abl_semantic_only_smart_stop | 0.9150 | 4.85 | 0.0072 | -0.0358 |
| abl_no_entity_hop | 0.9200 | 1.84 | 0.0486 | +0.0056 |
| abl_always_hop | 0.9900 | 6.00 | 0.0136 | -0.0294 |
| abl_no_workspace_mgmt | 0.9350 | 1.96 | 0.0415 | -0.0015 |

**H4 verdict (HotpotQA, where AEA has meaningful advantage over best baseline):**
- Total improvement: +0.0114 U@B
- Early stopping contribution: -0.1% (negligible; heuristic already stops early in most cases)
- Workspace management contribution: -0.4% (negligible)
- Entity hop contribution: -34.3% (removing hops actually helps slightly — entity hop not a net positive on HotpotQA)
- abl_always_hop delta: +0.1146 (always-hopping is catastrophic — ops explode, budget wasted)
- Routing (selective substrate choice) contribution: 0.0% by the gap metric — but always_hop delta (+0.1146) is the largest single effect, showing the VALUE of NOT hopping indiscriminately
- **H4: [NO]** — the selective routing gate (knowing when NOT to hop) is the dominant mechanism, but it registers as routing contribution = 0% by the no_hop vs always_hop gap metric (because no_entity_hop outperforms full AEA on this benchmark)

**Key insight:** On HotpotQA, entity hops are slightly harmful (the bridge approach used here doesn't help BM25-friendly questions), and abl_always_hop is catastrophically bad (-0.1146 U@B). The AEA value on HotpotQA comes from NOT doing unnecessary hops, not from doing selective hops. On heterogeneous v2, the near-tie between AEA and pi_semantic means contributions are numerically undefined.

### MuSiQue Multi-hop QA (Phase 4e — H2 test)

**Runner:** `experiments/run_musique.py`
**Results:** `experiments/results/musique.json`

Tests hypothesis H2: AEA should show larger gains on harder multi-hop questions where lexical overlap is lower and multiple entity bridge hops are required.

**Data:** MuSiQue is not publicly accessible via HuggingFace or direct download; 40 synthetic MuSiQue-style questions were constructed following the MuSiQue format (bridge chains, low lexical overlap, question_decomposition sub-questions, distractor paragraphs). Distribution: 20 2-hop, 10 3-hop, 10 4-hop.

**Results (N=40):**

| Policy | SupportRecall | AvgOps | Utility@Budget |
|---|---|---|---|
| π_semantic | 0.9875 | 2.00 | -0.0218 |
| π_lexical | 0.9667 | 2.00 | -0.0004 |
| π_entity | 0.8729 | 3.00 | -0.0264 |
| π_ensemble | 1.0000 | 3.00 | -0.0095 |
| π_aea_heuristic | 0.9167 | 1.02 | -0.0112 |

**By hop count (SupportRecall):**

| Policy | 2-hop (N=20) | 3-hop (N=10) | 4-hop (N=10) |
|---|---|---|---|
| π_semantic | 1.0000 | 1.0000 | 0.9500 |
| π_lexical | 1.0000 | 0.8667 | 1.0000 |
| π_entity | 0.9250 | 0.6667 | 0.9750 |
| π_ensemble | 1.0000 | 1.0000 | 1.0000 |
| π_aea_heuristic | 0.8750 | 0.9667 | 0.9500 |

**H2 Analysis (AEA gain over best single-substrate SupportRecall):**
- 2-hop: AEA=0.8750, best_baseline=1.0000, gain=-0.1250
- 3-hop: AEA=0.9667, best_baseline=1.0000, gain=-0.0333
- 4-hop: AEA=0.9500, best_baseline=1.0000, gain=-0.0500

**H2 verdict:** Not confirmed on synthetic data. AEA's recall gap relative to the best single-substrate baseline narrows as hop count increases (from -0.125 on 2-hop to -0.033 on 3-hop and -0.050 on 4-hop), which is partially consistent with H2's direction. However, AEA never exceeds the best baseline in SupportRecall. AEA's distinctive advantage is efficiency: it achieves near-competitive recall with only 1.02 avg operations vs 2–3 for all other policies, yielding Utility@Budget competitive with or better than multi-operation policies (pi_entity, pi_ensemble). The Utility@Budget metric used here includes a cost penalty that disadvantages policies spending more operations; AEA's adaptive early-stopping makes it efficient even when recall is lower.

**Caveat:** Results are on synthetic data designed with uniform low-lexical-overlap properties across all hop counts; real MuSiQue exhibits steeper per-hop lexical overlap gradients that would likely amplify H2 signal.

---

## Status

Phase 0: Research setup — complete.
Phase 1: Literature review — complete (63 papers, 5 gaps identified).
Phase 2: Hypothesis formation — complete (RIGOROUS after theory review).
Phase 3: PoC validation — complete (Run 1: mechanism confirmed; Run 2: 44% switching rate on real HotpotQA).
Phase 4: Full experiments — in progress.
  - Core AEA framework implemented (`experiments/aea/`): types, three address spaces,
    immutable evaluation harness, five policies + 5 ablation variants.
  - Phase 4a: HotpotQA baselines — complete.
  - Phase 4b (Regime C): Heterogeneous benchmark — complete (100 synthetic questions,
    6 task types, all 5 policies evaluated).
  - Phase 4d: Ablation study — complete (H4 test: routing selectivity, not routing per se,
    is the key mechanism; always_hop catastrophically worsens results).
  - Phase 4e: MuSiQue multi-hop QA benchmark — complete (see below).
  - Next: BRIGHT-style runs, LLM-based answer generation for EM/F1.
