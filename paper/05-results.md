# 5 Results

## 5.1 Main Results

Table 1 and Table 2 report the primary evaluation metrics across all five policies on HotpotQA Bridge (N=100) and Heterogeneous v2 (N=100), respectively.

**Table 1.** Results on HotpotQA Bridge (N=100). Best U@B is bolded.

| Policy | SupportRecall | SupportPrec | AvgOps | Utility@Budget |
|---|---|---|---|---|
| π_semantic | 0.750 | 0.300 | 2.00 | 0.0149 |
| π_lexical | 0.810 | 0.324 | 2.00 | 0.0169 |
| π_entity | 0.715 | 0.385 | 3.00 | −0.0409 |
| π_ensemble | 0.940 | 0.244 | 3.00 | 0.0028 |
| π_aea | 0.795 | 0.296 | **1.21** | **0.0283** |

**Table 2.** Results on Heterogeneous v2 (N=100). Best U@B is bolded.

| Policy | SupportRecall | SupportPrec | AvgOps | Utility@Budget |
|---|---|---|---|---|
| π_semantic | 0.920 | 0.328 | 2.00 | **0.0440** |
| π_lexical | 0.820 | 0.306 | 2.00 | 0.0142 |
| π_entity | 0.625 | 0.721 | 3.00 | −0.0038 |
| π_ensemble | 0.960 | 0.270 | 3.00 | 0.0071 |
| π_aea | 0.930 | 0.388 | **1.84** | 0.0430 |

On HotpotQA Bridge, π_aea achieves a U@B of 0.0283, a 67% improvement over π_lexical (0.0169). This gain is driven by cost efficiency: AEA completes questions in 1.21 operations versus 2.00 for all fixed policies. The ensemble policy, despite the highest recall (0.940), is penalized to near-zero U@B by its 3.00 AvgOps.

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
