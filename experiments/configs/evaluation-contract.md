# Evaluation Contract

**Date:** 2026-04-04
**Status:** active

## Mutable (can modify during experiments)

- Harness-level policy (router, operator selector, stopping rule)
- Workspace manager logic (pin, evict, compress strategies)
- Evidence bundle scoring model
- Address space implementations (how each space is queried)
- Hyperparameters: routing thresholds, budget allocation weights, compression ratios
- Training logic for learned router/scorer

## Immutable (read-only — NEVER modify to improve metrics)

- Evaluation harness and scoring scripts
- Dataset loading and preprocessing
- Metric computation (EM, F1, judge scores, support precision/recall, etc.)
- Dataset splits (train/val/test)
- Random seeds for reproducibility
- Benchmark task definitions (BRIGHT, NoLiMa, RULER, LongBench Pro, HotpotQA, MuSiQue, LoCoMo, LongMemEval)
- Cost measurement methodology (token counting, latency measurement)
- Budget constraints per evaluation run

## Primary Metric

**Utility@Budget:**

```
Utility@Budget = AnswerScore × η·EvidenceScore − μ·Cost
```

Where:
- AnswerScore: EM or F1 or judge score (task-dependent)
- EvidenceScore: support precision × support recall × bundle coverage
- Cost: normalized(tokens + latency + tool_calls)
- η, μ: weighting coefficients (fixed before experiments begin)

## Secondary Metrics

| Family | Metrics |
|--------|---------|
| Answer quality | EM, F1, task-specific accuracy, judge score |
| Evidence quality | support precision, support recall, page/section hit rate, bundle coverage |
| Discovery efficiency | steps to first relevant evidence, file hit@k, section hit@k, success under limited exploration budget |
| Cost | tokens consumed, wall-clock latency, tool calls, compute cost |
| Calibration | accuracy when system says "insufficient evidence", over-answering rate |
| Robustness | performance vs lexical mismatch, distractor density, corpus heterogeneity, context length |
| Policy quality | source-selection accuracy, correct-address-space selection rate, unnecessary-tool-call rate, over-expansion rate, stale-context retention rate |

## Immutable Constants

- Base LLM: one model, fixed across all experiments (to be determined — likely Claude 3.5 Sonnet or GPT-4o)
- Tool layer: fixed API interface
- Corpus parser: fixed document processing pipeline
- Hardware budget: fixed per-run token/latency caps
- Random seeds: [42, 123, 456, 789, 1024] for 5-seed evaluation

## Evaluation Regimes

### Regime A: Memory-centric long-horizon reasoning
- Benchmarks: LoCoMo, LongMemEval
- Primary metric: judge score, accuracy
- Purpose: apples-to-apples comparison with MAGMA

### Regime B: Retrieval and long-context stress
- Benchmarks: BRIGHT, NoLiMa, RULER, LongBench Pro
- Primary metric: EM, F1, accuracy
- Purpose: test adaptive routing under retrieval difficulty and context scale

### Regime C: Heterogeneous environment (custom)
- Benchmarks: custom discovery-vs-knowledge benchmark
- Primary metric: Utility@Budget
- Purpose: test the unique contribution — routing across mixed substrates

## Violation Protocol

If any immutable component is accidentally modified:
1. Discard all results after the modification
2. Restore from git history
3. Re-run affected experiments
4. Document the incident in research log
