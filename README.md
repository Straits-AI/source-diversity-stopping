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
experiments/poc/        # Proof-of-concept code
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

**Location:** `experiments/poc/substrate_switching_poc.py`

**Goal:** Validate the core assumption that within-task substrate switching occurs on real multi-hop tasks and that an adaptive policy outperforms single-substrate baselines.

**Design:**
- Dataset: 20 bridge-type questions (HotpotQA format, hardcoded examples)
- Address space A₁: Semantic search via `all-MiniLM-L6-v2` embeddings + cosine similarity
- Address space A₂: Entity link hop via regex-based NER + co-occurrence graph
- Three policies: `π_semantic` (always A₁), `π_graph` (always A₂), `π_heuristic` (adaptive: A₁ for entry, A₂ for bridge hops)
- Oracle analysis: per-step substrate attribution based on which substrate actually finds gold docs

**Key Results (2026-04-04):**

| Policy | SupportRecall | StepsToFirst | TotalOps |
|---|---|---|---|
| π_semantic | 0.875 | 1.00 | 2.15 |
| π_graph | 1.000 | 1.05 | 5.00 |
| π_heuristic | 1.000 | 1.00 | 2.00 |

Oracle: **3/20 (15%)** of questions required substrate switching.  
Step 1 oracle: A₁=100%. Step 2 oracle: A₂=100% (only for switch cases).

**Interpretation:** `π_heuristic` achieves full recall (1.000) at the same speed as semantic search (StepsToFirst=1.00) and at less than half the operations cost of `π_graph` (2.00 vs 5.00). The 15% switching rate is lower than the 60-80% hypothesis — this is partially a data artefact (examples where both gold paragraphs share entity tokens with the question, making them both recoverable in one semantic step). The core mechanism is validated on the 3 bridge cases that require switching; the heuristic correctly handles them at no cost penalty.

**Next steps:** Re-run with HotpotQA distractor split (full download) to get authentic bridge questions with more implicit bridge entities. Expect switching rate to increase substantially with harder examples.

**Raw results:** `experiments/poc/poc_results.json`  
**Dependencies:** `experiments/poc/requirements.txt` (`sentence-transformers>=2.2.0`, `datasets>=2.0.0`, `numpy>=1.21.0`)

## Status

Phase 0: Research setup — complete.
Phase 1: Literature review — complete (63 papers, 5 gaps identified).
Phase 2: Hypothesis formation — complete (RIGOROUS after theory review).
Phase 3: PoC validation — complete (mechanism confirmed, switching rate 15% on easy data).
Phase 4: Full experiments — pending.
