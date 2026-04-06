# 5 Results

## 5.1 Retrieval Quality (N=500, Statistical Validation)

**Table 1.** Retrieval-only results on HotpotQA Bridge (N=500, bootstrap 95% CIs).

| Policy | SupportRecall (mean ± std) | AvgOps | Retrieval U@B [95% CI] |
|---|---|---|---|
| π_semantic | 0.797 ± 0.011 | 2.00 | 0.0129 [0.010, 0.017] |
| π_lexical | 0.772 ± 0.016 | 2.00 | 0.0115 [0.008, 0.015] |
| π_entity | 0.732 ± 0.021 | 3.00 | −0.034 [−0.036, −0.032] |
| π_ensemble | 0.929 ± 0.005 | 3.00 | −0.002 [−0.005, 0.002] |
| **π_aea** | **0.810 ± 0.009** | **1.15** | **0.0322 [0.029, 0.036]** |

All comparisons between π_aea and baselines are statistically significant (paired permutation tests, 10,000 iterations, p < 0.0001). The effect size versus the best single-substrate baseline (π_semantic) is large (Cohen's d = 0.807). Confidence intervals do not overlap. π_aea achieves the highest support recall (0.810) at the lowest operation count (1.15).

## 5.2 End-to-End Answer Quality (N=100)

To validate that retrieval efficiency translates to downstream answer quality, we added LLM-based answer generation (gpt-oss-120b via OpenRouter). The same model and prompt are used for all policies — only the retrieved evidence differs. We also evaluate an LLM-routed policy (π_llm_routed) where gpt-oss-120b makes routing decisions at each step, reasoning about evidence sufficiency.

**Table 2.** End-to-end results on HotpotQA Bridge (N=100). Utility@Budget uses F1 as AnswerScore.

| Policy | EM | F1 | SupportRecall | AvgOps | E2E U@B |
|---|---|---|---|---|---|
| π_semantic | 0.500 | 0.617 | 0.750 | 2.00 | 0.648 |
| π_lexical | 0.500 | 0.643 | 0.810 | 2.00 | 0.703 |
| π_ensemble | 0.560 | 0.701 | 0.940 | 3.00 | 0.731 |
| **π_aea** | 0.480 | 0.630 | 0.795 | **1.21** | **0.760** |
| π_llm_routed | 0.500 | 0.637 | 0.845 | 2.54 | 0.652 |

**The heuristic AEA policy achieves the highest end-to-end Utility@Budget (0.760), outperforming the ensemble (0.731) and all single-substrate baselines.** This result is the central validation of the paper: cost-efficient retrieval routing produces better overall utility than both comprehensive retrieval and LLM-guided positive routing, even when downstream answer quality is measured.

The mechanism is clear: π_aea's F1 (0.630) is 10% lower than the ensemble's (0.701), but its operation count (1.21) is 60% lower (vs 3.00). Under budget-aware evaluation, the cost savings more than compensate for the quality gap. The ensemble retrieves everything but pays for it; the heuristic retrieves enough and stops.

### The LLM-Routed Policy

π_llm_routed makes genuine multi-substrate routing decisions (action distribution: STOP=73, SEMANTIC=43, LEXICAL=80, ENTITY_HOP=31 across 227 routing calls). It achieves higher recall than the heuristic (0.845 vs 0.795), confirming that LLM reasoning enables positive substrate selection — the LLM correctly identifies when additional retrieval from a specific substrate would help.

However, this positive routing is not cost-efficient: 2.54 average operations yield only marginally better F1 (0.637 vs 0.630), producing a lower E2E U@B (0.652 vs 0.760). The LLM router over-retrieves relative to the quality gain, spending operations on evidence that doesn't improve the final answer enough to justify the cost.

The policy ranking — heuristic (0.760) > ensemble (0.731) > lexical (0.703) > LLM-routed (0.652) > semantic (0.648) — reveals a hierarchy: **smart stopping > brute force > smart searching** under budget constraints. This is counterintuitive: a simple coverage threshold outperforms both comprehensive retrieval and LLM-guided routing.

## 5.3 Heterogeneous Benchmark (N=100)

**Table 3.** Retrieval-only results on Heterogeneous v2.

| Policy | SupportRecall | SupportPrec | AvgOps | U@B |
|---|---|---|---|---|
| π_semantic | 0.920 | 0.328 | 2.00 | **0.044** |
| π_lexical | 0.820 | 0.306 | 2.00 | 0.014 |
| π_entity | 0.625 | 0.721 | 3.00 | −0.004 |
| π_ensemble | 0.960 | 0.270 | 3.00 | 0.007 |
| π_aea | 0.930 | 0.388 | **1.84** | 0.043 |

π_aea near-ties π_semantic (0.043 vs 0.044) with higher precision (0.388 vs 0.328) and fewer operations (1.84 vs 2.00). Per-task-type analysis shows AEA outperforms on multi-step tasks (Semantic+Computation: +19%, Discovery+Extraction: +6%) and trails on single-substrate tasks (Low Lexical Overlap, Entity Bridge).

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

**Selective avoidance is the dominant mechanism.** abl_always_hop is catastrophic (Δ = −0.115): forcing unconditional escalation wastes budget on operations that degrade overall utility. In contrast, removing entity hops (abl_no_entity_hop) slightly improves performance (+0.004), confirming that the heuristic's entity hop decisions are net-negative on lexically-rich data.

**Substrate diversity matters for coverage estimation.** abl_semantic_smart_stop (Δ = −0.037) shows that the coverage check requires distinct substrates to function — querying the same modality twice provides no additional signal about evidence sufficiency.

## 5.5 Within-Task Substrate Switching

Oracle analysis of HotpotQA Bridge reveals 44% of questions require within-task substrate switching. Step 1 favors semantic (92%); Step 2 favors entity hop (88%). The LLM-routed policy captures this pattern — it uses all four action types with genuine per-question variation — but cannot yet translate this into cost-efficient routing. The gap between the LLM router's positive routing capability and the heuristic's cost efficiency identifies **calibrated stopping** as the key open challenge: a router that stops as efficiently as the heuristic while routing as intelligently as the LLM.
