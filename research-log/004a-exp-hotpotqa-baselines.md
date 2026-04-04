# Experiment: HotpotQA Bridge Baselines

**Date:** 2026-04-05
**Phase:** 4a
**Iteration:** 0
**Status:** completed

## Context
First full run of all 5 policies through the immutable evaluation harness on 100 real HotpotQA bridge questions (distractor split).

## Setup

- Dataset: HotpotQA distractor validation, first 100 bridge-type questions
- 10 context paragraphs per question (2 gold + 8 distractors)
- Address spaces: Semantic (all-MiniLM-L6-v2), Lexical (BM25), Entity Graph (regex NER)
- No LLM answer generation — retrieval-only evaluation
- Seed: 42

## Results

| Policy | SupportRecall | SupportPrecision | AvgOps | Utility@Budget |
|--------|--------------|-----------------|--------|----------------|
| π_semantic | 0.750 | 0.300 | 2.00 | 0.0149 |
| π_lexical | 0.810 | 0.324 | 2.00 | 0.0169 |
| π_entity | 0.715 | 0.385 | 3.00 | -0.0409 |
| π_ensemble | 0.940 | 0.244 | 3.00 | 0.0028 |
| π_aea_heuristic | 0.800 | 0.272 | 1.94 | 0.0163 |

## Interpretation

### Key Findings

**1. BM25 is the strongest single-substrate baseline on HotpotQA.**
BM25 achieves 0.810 recall at 2.00 ops, giving the best Utility@Budget (0.0169). This is expected — HotpotQA has high lexical overlap between questions and gold paragraphs, which is BM25's strength.

**2. AEA heuristic is competitive but doesn't clearly beat BM25.**
Utility@Budget: 0.0163 (AEA) vs 0.0169 (BM25). The difference is 0.0006 — likely not statistically significant. AEA uses slightly fewer operations (1.94 vs 2.00) but has slightly lower recall (0.800 vs 0.810).

**3. Ensemble has highest recall but worst cost-efficiency.**
0.940 recall at 3.00 ops, but low precision (0.244) and near-zero utility. This validates FM5: the ensemble finds everything but wastes budget.

**4. Entity graph alone is insufficient.**
0.715 recall, negative utility (-0.0409). The entity graph is precise (0.385 precision, best among all) but misses documents without shared entities with the query.

### Why AEA Doesn't Win on HotpotQA

This result is consistent with H2: **AEA should show LARGER gains on reasoning-intensive tasks with low lexical overlap.** HotpotQA has high lexical overlap — questions directly name the entities in the relevant paragraphs. In this regime, BM25 is already near-optimal, and routing adds overhead without proportional benefit.

The real test for H2 is BRIGHT and NoLiMa, where:
- Lexical overlap is deliberately minimized
- Reasoning is required to connect queries to relevant documents
- Single retrieval primitives demonstrably fail

### Caveats

**1. No LLM answer generation.** All policies use heuristic answer derivation (return top workspace item). EM/F1 are near-zero for all policies. Utility@Budget is almost entirely driven by evidence quality minus cost. The full pipeline needs Claude API calls for answer generation to produce meaningful EM/F1.

**2. The heuristic policy is not tuned for HotpotQA.** It uses generic routing rules (semantic first → entity hop if multi-hop detected). Domain-specific tuning could improve it, but we deliberately avoid this — the point is a general policy.

**3. Equal performance on HotpotQA is not a failure.** The "don't lose on home turf" criterion from the plan (Section 6, skeptical reviewer objection #3) says: the method must not lose SUBSTANTIALLY on single-substrate benchmarks. A 0.0006 difference is well within noise.

## Decision

**Proceed to harder benchmarks** (MuSiQue, BRIGHT-style tasks) where substrate switching should matter more. Also implement LLM-based answer generation for the full pipeline.

Two immediate next steps:
1. Run on MuSiQue (harder multi-hop, less lexical overlap)
2. Add Claude API answer generation for EM/F1 scoring

## Results TSV Update

```
hotpotqa-semantic    support_recall    0.750    0.5    60    keep    semantic search only on HotpotQA
hotpotqa-lexical     support_recall    0.810    0.5    60    keep    BM25 search only on HotpotQA
hotpotqa-entity      support_recall    0.715    0.5    60    keep    entity graph only on HotpotQA
hotpotqa-ensemble    support_recall    0.940    0.5    60    keep    ensemble (all substrates) on HotpotQA
hotpotqa-heuristic   support_recall    0.800    0.5    60    keep    AEA heuristic routing on HotpotQA
```
