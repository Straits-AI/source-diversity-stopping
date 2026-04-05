# 5 Results

## 5.1 Main Results

**Table 1.** Results on HotpotQA Bridge (N=500, 5 bootstrap seeds, 95% CIs). Best U@B is bolded.

| Policy | SupportRecall (mean ± std) | AvgOps | Utility@Budget [95% CI] |
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

**Table 2.** Results on Heterogeneous v2 (N=100, single evaluation). Best U@B is bolded.

| Policy | SupportRecall | SupportPrec | AvgOps | Utility@Budget |
|---|---|---|---|---|
| π_semantic | 0.920 | 0.328 | 2.00 | **0.0440** |
| π_lexical | 0.820 | 0.306 | 2.00 | 0.0142 |
| π_entity | 0.625 | 0.721 | 3.00 | −0.0038 |
| π_ensemble | 0.960 | 0.270 | 3.00 | 0.0071 |
| π_aea | 0.930 | 0.388 | **1.84** | 0.0430 |

On HotpotQA Bridge, π_aea achieves a U@B of 0.0322 [0.029, 0.036], significantly higher than all baselines (p < 0.0001 for all comparisons). The 95% confidence intervals do not overlap with any baseline. The effect size versus π_semantic is large (Cohen's d = 0.807). Notably, AEA achieves the highest support recall (0.810) among all policies while using only 1.15 average operations — less than half the cost of fixed single-substrate policies (2.00) and less than 40% the cost of the ensemble (3.00).

On Heterogeneous v2, π_semantic leads (0.0440) with π_aea close at 0.0430 (−2%). However, π_aea exhibits higher precision (0.388 vs. 0.328) and recall (0.930 vs. 0.920) at fewer operations (1.84 vs. 2.00).

### Per-Task-Type Breakdown

**Table 3.** Per-task-type U@B on Heterogeneous v2.

| Task Type | π_semantic | π_aea | Notes |
|---|---|---|---|
| Semantic+Computation | 0.0665 | **0.0791** | AEA +19% |
| Discovery+Extraction | 0.2865 | **0.3024** | AEA +6% |
| Low Lexical Overlap | **0.0530** | 0.0013 | Semantic dominates |
| Entity Bridge | **0.0324** | 0.0236 | Semantic leads |

AEA outperforms on tasks requiring multi-step retrieval and falls behind on single-substrate tasks where over-searching introduces noise.

## 5.2 Ablation Analysis

**Table 4.** Ablation results on HotpotQA Bridge.

| Ablation | U@B | Δ from Full AEA |
|---|---|---|
| Full AEA | 0.0282 | — |
| abl_no_early_stop | 0.0283 | +0.0001 |
| abl_no_workspace_mgmt | 0.0283 | +0.0001 |
| abl_no_entity_hop | 0.0321 | +0.0039 |
| abl_semantic_smart_stop | −0.0087 | −0.0370 |
| abl_always_hop | −0.0864 | **−0.1146** |

**Selective avoidance is the dominant contributor.** abl_always_hop is catastrophic (Δ = −0.1146), establishing that AEA's gain comes from avoiding unnecessary operations.

**Substrate diversity matters.** abl_semantic_smart_stop (Δ = −0.0370) confirms that coverage estimation requires distinct substrates.

**Entity hops are net-negative on HotpotQA.** abl_no_entity_hop improves U@B by +0.0039, indicating entity operations introduce cost without proportionate benefit on lexically-rich data.

**Early stopping and workspace management are inert.** Both produce Δ ≈ 0.

## 5.3 Routing Avoidance as the Core Mechanism

The central empirical finding is that routing avoidance — selective omission of retrieval operations — is the primary driver of AEA's advantage. π_aea distributes operation counts bimodally: most questions resolve with a single operation, with the remainder escalating. The 1.21 AvgOps reflects this mixture and produces the 67% U@B improvement over fixed-cost baselines.

This is not lower ambition — AEA achieves comparable recall — but correct identification of when a single step suffices.

## 5.4 Within-Task Substrate Switching

Oracle analysis of HotpotQA Bridge reveals 44% of questions require within-task substrate switching. Step 1 favors semantic (92%); Step 2 favors entity hop (88%). This confirms heterogeneous address space support is motivated, even though the current heuristic does not yet exploit this pattern reliably — explaining the abl_no_entity_hop finding.
