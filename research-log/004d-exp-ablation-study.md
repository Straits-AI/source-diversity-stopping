# Experiment: Ablation Study — H4 Test

**Date:** 2026-04-04
**Phase:** 4d
**Status:** completed

## Objective

Test hypothesis H4: "Router ablation accounts for >50% of total improvement."

Each ablation removes exactly one mechanism from the AEA heuristic policy and measures the U@B drop (or gain) relative to the full system.

## Ablation Variants

| Ablation name | Mechanism disabled |
|---|---|
| abl_no_early_stop | Coverage-driven early stopping; always runs to max_steps |
| abl_semantic_only_smart_stop | All non-semantic substrates; keeps smart stop; tests whether stopping rule alone explains gain |
| abl_no_entity_hop | Entity graph HOP; replaced with lexical fallback; tests entity hop contribution |
| abl_always_hop | Selective hop decision; always hops after semantic; tests value of selective routing |
| abl_no_workspace_mgmt | Pin/evict workspace curation; keeps all items |

## Files

- **Ablation policy classes:** `experiments/aea/policies/ablations.py`
- **Runner:** `experiments/run_ablations.py`
- **Results:** `experiments/results/ablation_study.json`

## Setup

- Same harness config as prior experiments: max_steps=10, token_budget=4000, seed=42.
- Ablation policies use same top_k=5, coverage_threshold=0.5, max_steps=6 as full AEA.
- Both benchmarks run: HotpotQA Bridge (N=100) + Heterogeneous v2 (N=100).

## Results

### HotpotQA Bridge (N=100)

| Policy | SupportRecall | AvgOps | Utility@Budget | Δ from Full AEA |
|--------|--------------|--------|----------------|-----------------|
| pi_aea_heuristic (full) | 0.7950 | 1.21 | 0.0282 | baseline |
| pi_lexical (best baseline) | 0.8100 | 2.00 | 0.0169 | -0.0114 |
| pi_semantic | 0.7500 | 2.00 | 0.0149 | -0.0134 |
| abl_no_early_stop | 0.7950 | 1.21 | 0.0283 | +0.0000 |
| abl_semantic_only_smart_stop | 0.7500 | 3.09 | -0.0087 | -0.0370 |
| abl_no_entity_hop | 0.7850 | 1.21 | 0.0321 | +0.0039 |
| abl_always_hop | 0.8550 | 5.35 | -0.0864 | -0.1146 |
| abl_no_workspace_mgmt | 0.7950 | 1.21 | 0.0283 | +0.0001 |

Total improvement (full AEA over pi_lexical): +0.0114 U@B

### Heterogeneous v2 (N=100)

| Policy | SupportRecall | AvgOps | Utility@Budget | Δ from Full AEA |
|--------|--------------|--------|----------------|-----------------|
| pi_aea_heuristic (full) | 0.9300 | 1.84 | 0.0430 | baseline |
| pi_lexical | 0.8200 | 2.00 | 0.0142 | -0.0289 |
| pi_semantic (best baseline) | 0.9200 | 2.00 | 0.0440 | +0.0009 |
| abl_no_early_stop | 0.9300 | 1.96 | 0.0415 | -0.0015 |
| abl_semantic_only_smart_stop | 0.9150 | 4.85 | 0.0072 | -0.0358 |
| abl_no_entity_hop | 0.9200 | 1.84 | 0.0486 | +0.0056 |
| abl_always_hop | 0.9900 | 6.00 | 0.0136 | -0.0294 |
| abl_no_workspace_mgmt | 0.9350 | 1.96 | 0.0415 | -0.0015 |

Total improvement (full AEA over pi_semantic): -0.0009 U@B (near-tie — percentage contributions undefined).

## Contribution Analysis

### HotpotQA (where AEA has a meaningful +0.0114 U@B advantage over best baseline)

| Component | Raw delta | % contribution |
|---|---|---|
| Early stopping | -0.0000 | -0.1% (negligible) |
| Semantic smart stop | +0.0370 | 324.8% |
| Entity hop (no-hop fallback) | -0.0039 | -34.3% |
| Selective hop gate (always-hop) | +0.1146 | 1007.3% |
| Workspace management | -0.0001 | -0.4% (negligible) |
| Routing (selective gate) | 0.0% (by gap metric) | — |

Note: negative contribution means removing the component **improves** AEA — i.e., entity hops hurt on HotpotQA.

### Heterogeneous v2 (near-tie — % contributions undefined)

Raw deltas show the same directional pattern:
- abl_always_hop delta: +0.0294 (always-hopping wastes budget)
- abl_semantic_only_smart_stop delta: +0.0358 (substrate diversity matters; stop rule alone is insufficient)
- abl_no_entity_hop: -0.0056 (no-hop fallback slightly outperforms full AEA; entity hops not net-positive here either)
- abl_no_early_stop: +0.0015 (early stopping slightly hurts on hetero v2 — heuristic stops too early on some tasks)
- abl_no_workspace_mgmt: +0.0015 (workspace curation neutral)

## H4 Verdict

**H4: [NO]** — routing in the sense of "selective substrate selection" does not account for >50% of improvement using the standard proxy metric (no_entity_hop_pct − always_hop_pct).

However, the deeper insight is more nuanced:

1. **The dominant mechanism on HotpotQA is budget efficiency, not substrate diversity.** The abl_always_hop ablation is catastrophically bad (−0.1146 U@B, ops=5.35 vs 1.21), showing that the value of the AEA heuristic on HotpotQA is almost entirely in *not* spending ops on entity hops that don't help, rather than in routing to entity hops that do help.

2. **Entity hops are slightly harmful on HotpotQA** (abl_no_entity_hop outperforms full AEA by +0.0039). BM25 is near-optimal for HotpotQA's high-lexical-overlap bridge questions; adding entity hops only wastes budget.

3. **On heterogeneous v2, AEA barely ties pi_semantic** (0.0430 vs 0.0440). The AEA heuristic's routing rules were not designed for the v2 task patterns — in particular, it over-searches on low_lexical_overlap (5.25 avg ops vs 2.0 for semantic-only) and doesn't reliably trigger on the v2 multi-hop template.

4. **abl_semantic_only_smart_stop is consistently the second-worst policy** (only abl_always_hop is worse on HotpotQA) — showing substrate diversity *does* matter, but the current entity hop routing is net-negative on these benchmarks.

## Interpretation

The H4 test is not cleanly falsifiable with the current setup because:

- On HotpotQA, the AEA gain comes from routing efficiency (stopping early, not hopping when not needed), not from positive substrate selection.
- On heterogeneous v2, AEA does not outperform the best single substrate, so the total improvement denominator is negative, making contributions undefined.

The ablation study instead reveals:

- **Budget efficiency** (knowing when to stop) and **routing conservatism** (knowing when NOT to hop) are the primary drivers on HotpotQA.
- **Substrate diversity** (the smart stop ablation being catastrophically bad) shows that having semantic + entity + lexical options is important — but the current heuristic doesn't exploit them well.
- **Workspace management** and **early stopping** are not significant contributors in either direction.

## Next Steps

1. Improve heuristic routing for hetero v2 (especially low_lexical_overlap and multi_hop_chain patterns).
2. Investigate whether entity hops can be made net-positive on HotpotQA by improving hop quality (entity extraction precision).
3. Run ablations on a third benchmark where AEA has a larger advantage, to get cleaner contribution percentages.
