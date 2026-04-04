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

## Status

Phase 0: Research setup — complete.
Phase 1: Literature review — pending.
