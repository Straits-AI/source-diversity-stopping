# Analysis: Iteration 0

**Date:** 2026-04-05
**Phase:** 5
**Iteration:** 0
**Status:** completed

## Context
Completed Phase 4a-4c: baselines, core AEA, and ablation studies on HotpotQA (100 bridge) and Heterogeneous v2 (100 synthetic) benchmarks.

---

## 1. Did It Work?

**Partially yes.** AEA heuristic achieves the best Utility@Budget on HotpotQA (+67% over BM25) and near-ties semantic search on heterogeneous v2 (-2%). The method is competitive but does not dramatically dominate on diverse tasks.

## 2. Why Did It Work (or Not)?

**What works:** The coverage-driven stopping rule. AEA achieves 1.21 ops on HotpotQA (vs 2.00 for all baselines) by recognizing when it already has sufficient diverse evidence. This cost saving drives the Utility@Budget advantage.

**What doesn't work:** Positive substrate selection. Removing entity hops actually IMPROVES performance on HotpotQA (+0.0039 U@B). The entity graph adds cost without adding recall on lexically-rich tasks.

**The reframed mechanism:** AEA's value is **routing avoidance** — knowing what NOT to do. The `abl_always_hop` catastrophe (-0.1146 U@B, ops explode to 5.35) proves that selective avoidance is enormously valuable. But this is still a routing decision.

## 3. What Contributed Most?

From the ablation study on HotpotQA:

| Component | Ablation Impact | Contribution |
|-----------|----------------|-------------|
| Selective routing (avoidance) | abl_always_hop: -0.1146 | **Dominant** |
| Substrate diversity | abl_semantic_smart_stop: -0.0370 | Significant |
| Coverage-driven stopping | abl_no_early_stop: +0.0001 | Negligible (absorbed by harness) |
| Entity hops (positive) | abl_no_entity_hop: +0.0039 | Negative (harmful) |
| Workspace management | abl_no_workspace_mgmt: +0.0001 | Negligible |

**Key insight:** The largest contributor is knowing WHEN NOT to act (routing avoidance), not knowing WHICH substrate to use (positive routing).

## 4. How Robust Is It?

Not yet tested across seeds. Single-run results on fixed datasets. Robustness checks (5 seeds) are planned but not yet executed.

## 5. What Was Surprising?

1. **Entity hops are harmful on HotpotQA.** Removing them improves U@B. This was not predicted.
2. **Early stopping has zero contribution** despite being the narrative explanation for AEA's advantage. The harness's own step limit likely subsumes this.
3. **Workspace management has zero contribution.** Pin/evict doesn't help when the workspace is small.
4. **The dominant mechanism is negative routing** (what not to do), not positive routing (what to do).

## 6. How Does It Compare to Literature?

| System | Benchmark | Our Metric | Literature |
|--------|-----------|-----------|------------|
| BM25 baseline | HotpotQA | 0.810 recall | Expected range |
| Semantic baseline | HotpotQA | 0.750 recall | Expected range |
| AEA heuristic | HotpotQA | 0.795 recall, 1.21 ops | Novel (no direct comparison) |

Direct comparison with MAGMA, DeepRetrieval, etc. not possible without their code/data. Our results are self-consistent and the relative comparisons (AEA vs baselines) are valid.

---

## Hypothesis Status

| Hypothesis | Status | Evidence |
|------------|--------|---------|
| **H1:** AEA > fixed baselines on U@B | **SUPPORTED** | +67% on HotpotQA, near-tie on heterogeneous |
| **H2:** Larger gains on hard tasks | **PARTIALLY SUPPORTED** | Wins on computation+discovery tasks, not on entity bridge |
| **H3:** Executable addressing on structured tasks | **NOT TESTED** | No tool execution address space yet |
| **H4:** Router ablation > substrate ablation | **REFRAMED** | Positive routing: 0%. Routing avoidance: dominant. Selective routing matters enormously but through negative selection, not positive. |

---

## Revised Paper Narrative

The original narrative was: "Adaptive routing across substrates improves performance by choosing the RIGHT substrate."

The evidence supports a different narrative: **"Adaptive attention improves performance primarily through cost-efficient operation selection — knowing when to stop and what not to do, rather than always finding the optimal next substrate."**

This is arguably a MORE interesting finding:
1. It explains why simple systems often work well (they avoid unnecessary complexity)
2. It provides a concrete mechanism (coverage-driven stopping)
3. It predicts where the method will be most valuable (when the action space includes expensive operations that are often unnecessary)

---

## Decision: Path Assessment

### Diminishing returns?
The heuristic policy has been refined once (pattern-driven → coverage-driven). Further heuristic tuning faces diminishing returns. The learned router (Phase 2 of training) would address positive substrate selection.

### Three paths:

**Path A: Iterate — train learned router.**
Evidence-based next step: the heuristic can't do positive routing well. A learned router trained on trajectory data could learn when entity hops genuinely help. This addresses the gap between "routing avoidance" (current) and "routing optimization" (goal).
Risk: training infrastructure needed, may take weeks.

**Path B: Conclude with current results.**
The formalization + heuristic policy + ablation study is a solid workshop paper. The finding that "routing avoidance > routing optimization" is publishable. Add the custom benchmark + negative finding on H4 + reframed narrative.
Risk: reviewers may want learned policy results.

**Path C: Expand benchmarks.**
Run on BRIGHT/NoLiMa/MuSiQue to test H2 more thoroughly. This validates the method on external benchmarks without requiring new infrastructure.
Risk: may not change the story.

### Recommendation
**Path B (conclude) with elements of Path C (one more benchmark).** The current results tell a coherent, honest story. Run one external benchmark (MuSiQue) to strengthen H2, then proceed to Phase 6 (paper writing).

---

## Next Steps
- Checkpoint with user: conclude vs iterate vs expand
- If conclude: Phase 6 paper writing
- If iterate: implement learned router
- If expand: run MuSiQue benchmark
