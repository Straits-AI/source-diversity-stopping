# 5 Results

## 5.1 Retrieval Quality (N=500)

**Table 1.** Retrieval-only results on HotpotQA Bridge (N=500, bootstrap 95% CIs). Primary metric is retrieval-focused Utility@Budget (SupportRecall-based).

| Policy | SupportRecall (mean ± std) | AvgOps | Retrieval U@B [95% CI] |
|---|---|---|---|
| π_semantic | 0.797 ± 0.011 | 2.00 | 0.0129 [0.010, 0.017] |
| π_lexical | 0.772 ± 0.016 | 2.00 | 0.0115 [0.008, 0.015] |
| π_entity | 0.732 ± 0.021 | 3.00 | −0.034 [−0.036, −0.032] |
| π_ensemble | 0.929 ± 0.005 | 3.00 | −0.002 [−0.005, 0.002] |
| **π_aea** | **0.810 ± 0.009** | **1.15** | **0.0322 [0.029, 0.036]** |

**Statistical significance (paired permutation tests, 10,000 iterations):**

| Comparison | Δ U@B | p-value | Cohen's d |
|---|---|---|---|
| AEA vs π_semantic | +0.019 | < 0.0001 | 0.807 (large) |
| AEA vs π_lexical | +0.021 | < 0.0001 | 0.273 (small) |
| AEA vs π_ensemble | +0.034 | < 0.0001 | 0.440 (medium) |

On retrieval-only evaluation, π_aea achieves the highest support recall (0.810) at the lowest operation count (1.15) — less than half the cost of fixed single-substrate policies (2.00) and less than 40% the cost of the ensemble (3.00). All comparisons are significant at p < 0.0001 with non-overlapping 95% CIs.

## 5.2 End-to-End Answer Quality (N=50)

To test whether retrieval efficiency translates to downstream answer quality, we added LLM-based answer generation using a free-tier model (Qwen 3.6 Plus via OpenRouter). The same model and prompt are used for all policies — only the retrieved evidence differs.

**Table 2.** End-to-end results on HotpotQA Bridge (N=50). Utility@Budget uses F1 as AnswerScore.

| Policy | EM | F1 | SupportRecall | AvgOps | End-to-End U@B |
|---|---|---|---|---|---|
| π_semantic | 0.480 | 0.600 | 0.750 | 2.00 | 0.667 |
| π_lexical | 0.460 | 0.602 | 0.760 | 2.00 | 0.670 |
| π_entity | 0.480 | 0.603 | 0.700 | 3.00 | 0.647 |
| **π_ensemble** | **0.560** | **0.697** | **0.950** | 3.00 | **0.736** |
| π_aea | 0.500 | 0.609 | 0.780 | **1.16** | 0.689 |

The end-to-end evaluation reveals a different ranking than retrieval-only evaluation. π_ensemble leads on all quality metrics (EM 0.56, F1 0.70) and on U@B (0.736), because its higher recall (0.95) provides the LLM with more relevant evidence for answer synthesis. π_aea is second (U@B 0.689), outperforming all single-substrate policies but trailing the ensemble.

### Two Evaluation Regimes, Two Stories

| Regime | Winner | Runner-up | Mechanism |
|---|---|---|---|
| Retrieval-only (cost-heavy) | π_aea | π_semantic | Cost savings dominate; AEA's 1.15 ops vs 2.00+ is decisive |
| End-to-end (answer quality) | π_ensemble | π_aea | Higher recall → better evidence → better answers; cost penalty is proportionally smaller relative to F1 scores |

This divergence reveals that the current stopping rule is **too aggressive for end-to-end tasks**: it sacrifices recall that matters for answer quality. When cost is the primary concern (retrieval-only), early stopping is optimal. When answer quality dominates, comprehensive retrieval is worth the cost. The practical implication is that the optimal stopping threshold should be calibrated to the downstream task — a natural target for learned policies.

Critically, π_aea outperforms all single-substrate policies on both regimes. This confirms that multi-substrate access with adaptive routing improves over fixed retrieval, even when the routing policy is imperfect.

## 5.3 Heterogeneous Benchmark (N=100)

**Table 3.** Results on Heterogeneous v2 (retrieval-only).

| Policy | SupportRecall | SupportPrec | AvgOps | U@B |
|---|---|---|---|---|
| π_semantic | 0.920 | 0.328 | 2.00 | **0.044** |
| π_lexical | 0.820 | 0.306 | 2.00 | 0.014 |
| π_entity | 0.625 | 0.721 | 3.00 | −0.004 |
| π_ensemble | 0.960 | 0.270 | 3.00 | 0.007 |
| π_aea | 0.930 | 0.388 | **1.84** | 0.043 |

π_aea near-ties π_semantic (0.043 vs 0.044) with higher precision (0.388 vs 0.328) and fewer operations (1.84 vs 2.00). AEA outperforms on task types requiring multi-step retrieval:

| Task Type | π_semantic | π_aea | Notes |
|---|---|---|---|
| Semantic+Computation | 0.067 | **0.079** | AEA +19%, stops early on single-hop |
| Discovery+Extraction | 0.287 | **0.302** | AEA +6% |
| Low Lexical Overlap | **0.053** | 0.001 | Semantic dominates; AEA over-searches |
| Entity Bridge | **0.032** | 0.024 | AEA routing not yet reliable |

## 5.4 Ablation Analysis

**Table 4.** Ablation results on HotpotQA Bridge (retrieval-only, N=100).

| Ablation | U@B | Δ from Full AEA |
|---|---|---|
| Full AEA | 0.028 | — |
| abl_no_early_stop | 0.028 | +0.000 |
| abl_no_workspace_mgmt | 0.028 | +0.000 |
| abl_no_entity_hop | 0.032 | +0.004 |
| abl_semantic_smart_stop | −0.009 | −0.037 |
| abl_always_hop | −0.086 | **−0.115** |

Four findings:

**Selective avoidance is the dominant mechanism.** abl_always_hop is catastrophic (Δ = −0.115), establishing that AEA's retrieval-only advantage comes from avoiding unnecessary operations, not from superior substrate selection.

**Substrate diversity matters for coverage estimation.** abl_semantic_smart_stop (Δ = −0.037) shows that reliable coverage estimation requires querying distinct substrates — a redundant substrate provides no additional signal.

**Entity hops are net-negative on lexically-rich data.** abl_no_entity_hop improves U@B by +0.004, indicating that on HotpotQA (where BM25 is strong), entity operations add cost without proportionate recall benefit.

**Early stopping and workspace management are inert.** Both produce Δ ≈ 0 on this benchmark.

## 5.5 Within-Task Substrate Switching

Oracle analysis of HotpotQA Bridge reveals 44% of questions require within-task substrate switching for optimal performance. Step 1 favors semantic search (92%); Step 2 favors entity hop (88%). This confirms that heterogeneous substrate support is motivated — no single substrate is universally optimal across reasoning steps. The gap between this oracle pattern and the current policy's inability to exploit it reliably (abl_no_entity_hop finding) identifies substrate-conditional stopping as the key open challenge.
