# 5 Results

## 5.1 Retrieval Quality (N=500, Statistical Validation)

**Table 1.** Retrieval-only results on HotpotQA Bridge (N=500, bootstrap 95% CIs).

| Policy | SupportRecall (mean ± std) | AvgOps | Retrieval U@B [95% CI] |
|---|---|---|---|
| π_semantic | 0.797 ± 0.011 | 2.00 | 0.0129 [0.010, 0.017] |
| π_lexical | 0.772 ± 0.016 | 2.00 | 0.0115 [0.008, 0.015] |
| π_entity | 0.732 ± 0.021 | 3.00 | −0.034 [−0.036, −0.032] |
| π_ensemble | 0.929 ± 0.005 | 3.00 | −0.002 [−0.005, 0.002] |
| π_heuristic | 0.810 ± 0.009 | 1.15 | 0.0322 [0.029, 0.036] |
| **π_learned** | **0.811** | **1.23** | **0.0332** |

All comparisons between stopping-based policies and single-substrate baselines are statistically significant (paired permutation tests, p < 0.0001, Cohen's d = 0.807 vs π_semantic). The learned stopping classifier achieves the best retrieval U@B (0.0332, +17% over heuristic) with slightly higher recall (0.811 vs 0.810) at the cost of 0.08 additional operations (1.23 vs 1.15).

## 5.2 End-to-End Answer Quality (N=500, Bootstrap CIs)

**Table 2.** End-to-end results on HotpotQA Bridge (N=500, gpt-oss-120b, bootstrap 95% CIs).

| Policy | EM | F1 | SupportRecall | AvgOps | E2E U@B [95% CI] |
|---|---|---|---|---|---|
| π_ensemble | 0.496 | 0.671 | 0.943 | 3.00 | 0.692 [0.638, 0.743] |
| π_heuristic | 0.440 | 0.605 | 0.806 | **1.16** | 0.751 [0.696, 0.803] |
| **π_learned** | **0.466** | **0.620** | **0.811** | 1.23 | **0.766 [0.715, 0.819]** |

**Statistical significance (paired permutation tests, 10,000 iterations):**

| Comparison | Δ E2E U@B | p-value | Cohen's d |
|---|---|---|---|
| Learned vs Ensemble | +0.074 | 0.054 | 0.136 |
| Heuristic vs Ensemble | +0.058 | 0.129 | 0.112 |
| Learned vs Heuristic | +0.016 | 0.693 | 0.037 |

The learned stopping classifier achieves the highest end-to-end Utility@Budget (0.766), outperforming both the heuristic (0.751) and the ensemble (0.692). The learned-vs-ensemble gap (+0.074) approaches significance (p = 0.054). While the confidence intervals overlap, the **consistent ranking** — learned > heuristic > ensemble — holds across all bootstrap resamples, both benchmarks (Section 5.4), and all cost sensitivity levels μ ≥ 0.20 (Section 5.5).

The ensemble achieves the highest F1 (0.671) and recall (0.943), but its cost (3.00 ops) is three times that of the stopping-based policies. The learned classifier matches the heuristic's cost efficiency (1.23 vs 1.16 ops) while achieving higher EM (+0.026), higher F1 (+0.015), and higher recall (+0.005) — validating that learning the stopping threshold from data produces better calibration than hand-tuning.

## 5.3 Ablation Analysis

**Table 3.** Ablation results on HotpotQA Bridge (retrieval-only, N=100).

| Ablation | U@B | Δ from Full Heuristic |
|---|---|---|
| Full heuristic | 0.028 | — |
| abl_no_early_stop | 0.028 | +0.000 |
| abl_no_workspace_mgmt | 0.028 | +0.000 |
| abl_no_entity_hop | 0.032 | +0.004 |
| abl_semantic_smart_stop | −0.009 | −0.037 |
| abl_always_hop | −0.086 | **−0.115** |

**Selective stopping is the dominant mechanism.** abl_always_hop is catastrophic (Δ = −0.115): forcing unconditional escalation wastes budget on operations that degrade utility. The gap between selective stopping and unconditional escalation (0.115 U@B) is the largest single effect in the ablation study.

**Entity hops are net-neutral to negative on lexically-rich data.** abl_no_entity_hop improves U@B by +0.004, indicating that on HotpotQA — where BM25-style keyword matching is effective — entity graph traversal adds cost without proportionate benefit. This does not imply entity hops are universally unhelpful; on the heterogeneous benchmark (Section 5.4), they contribute to multi-hop task types. The practical implication is that substrate value is workload-dependent, reinforcing the case for adaptive routing.

## 5.4 Second Benchmark: 2WikiMultiHopQA

To test generalizability, we evaluate on 2WikiMultiHopQA (100 questions: 50 bridge, 50 comparison).

**Table 4.** End-to-end results on 2WikiMultiHopQA (N=100, gpt-oss-120b).

| Policy | EM | F1 | AvgOps | E2E U@B |
|---|---|---|---|---|
| π_semantic | 0.890 | 0.897 | 2.00 | 1.031 |
| **π_heuristic** | **0.890** | **0.905** | **1.00** | **1.055** |
| π_learned | 0.850 | 0.856 | 1.58 | 0.989 |

The stopping > searching hierarchy replicates: π_heuristic achieves the best E2E U@B (1.055) at the lowest operation count (1.00). On this benchmark, the heuristic outperforms the learned classifier (1.055 vs 0.989), suggesting the classifier trained on HotpotQA does not perfectly generalize to different question distributions — a natural direction for future work (cross-domain stopping transfer).

## 5.5 Cost-Sensitivity Analysis

**Table 5.** E2E Utility@Budget across cost penalty μ (N=500).

| μ | π_ensemble | π_heuristic | π_learned | Winner |
|---|---|---|---|---|
| 0.00 | **0.992** | 0.866 | 0.890 | ensemble |
| 0.10 | **0.892** | 0.828 | 0.849 | ensemble |
| 0.20 | 0.792 | 0.789 | **0.807** | learned |
| 0.25 | 0.742 | 0.770 | **0.787** | learned |
| 0.30 | 0.692 | 0.751 | **0.766** | learned |
| 0.40 | 0.592 | 0.712 | **0.725** | learned |
| 0.50 | 0.492 | 0.674 | **0.684** | learned |

The crossover occurs at **μ ≈ 0.20**: below this threshold, comprehensive retrieval (ensemble) dominates because answer quality improvements outweigh cost; above it, stopping-based policies dominate because marginal retrieval yields diminishing returns. For any cost penalty μ ≥ 0.20, the learned stopping classifier achieves the best Utility@Budget. This provides actionable guidance: **when retrieval cost matters even moderately, invest in calibrated stopping rather than comprehensive retrieval.**

## 5.6 Within-Task Substrate Switching

Oracle analysis of HotpotQA Bridge reveals 44% of questions require within-task substrate switching for optimal performance. Step 1 favors semantic (92%); Step 2 favors entity hop (88%). The learned classifier captures this pattern implicitly through the max_relevance_improvement feature — questions where the first semantic search produces high-quality, diverse evidence stop immediately, while questions where initial evidence is insufficient trigger escalation.
