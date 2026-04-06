# Review Panel: Paper v5

**Date:** 2026-04-06
**Phase:** 6
**Status:** 3 reviews received

## Scores Summary

| Criterion | R1 (IR) | R2 (Systems) | R3 (QA) | Mean |
|-----------|---------|-------------|---------|------|
| Soundness | 4 | 4 | 4 | 4.0 |
| Presentation | 6 | 7 | 6 | 6.3 |
| Contribution | 5 | 5 | 5 | 5.0 |
| Overall | 4 | 4 | 4 | 4.0 |
| Recommendation | Borderline Reject | Reject (weak) | Reject | Reject |
| Confidence | 4/5 | 4/5 | 4/5 | 4.0 |

## Consensus Issues (raised by all 3 reviewers)

1. **No E2E statistical significance** — learned vs ensemble p=0.054, all CIs overlap
2. **Custom metric determines ranking** — U@B with mu=0.3 is above the crossover; ensemble wins on standard EM/F1
3. **Entity graph adds nothing** — ablation improves when removed; "3 heterogeneous substrates" is overclaim
4. **2Wiki doesn't replicate learned > heuristic** — learned (0.989) < heuristic (1.055)
5. **Learned classifier methodology concerns** — threshold tuned on test set, oracle labels use same metric

## Critical New Issue (R1)

**Statistical reporting discrepancy:** R1 reports computing p=0.0014 from the saved data file (e2e_n500.json) vs p=0.054 reported in paper. Either the data file doesn't match the reported numbers, or there's a bug. Must be investigated.

**Potential train/test overlap (R1):** Training uses questions 200-700, evaluation uses first 500. Questions 200-499 may appear in both. Must be verified.

## Consensus Strengths (acknowledged by all 3)

1. Core insight is genuinely useful and actionable
2. Honest statistical reporting and transparent limitations
3. Clean experimental controls (evaluation contract)
4. Interpretable stopping signal (max_relevance_improvement)
5. Well-structured sensitivity analysis

## Path to Acceptance (synthesized from all 3)

1. **Report standard metrics as primary** — EM/F1 plus Pareto frontier, U@B as secondary
2. **Fix train/test contamination** — verify no overlap between classifier training and eval
3. **Honest 2-substrate framing** — drop entity graph or show it helps on some benchmark
4. **Achieve significance or reframe** — increase N, use multiple benchmarks, or frame as "comparable at lower cost" trend
5. **Ground mu in real costs** — cite cost-sensitive evaluation literature, or report full Pareto curve
6. **Resolve number inconsistencies** — Discussion cites old numbers, scope note contradicts E2E eval
7. **Drop or reduce CMDP** — it does no work
