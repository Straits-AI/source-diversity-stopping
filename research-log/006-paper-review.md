# Paper Review: Draft v1

**Date:** 2026-04-05
**Phase:** 6
**Status:** NEEDS_REVISION

## Reviewer Assessment

| Criterion | Score | Key Issue |
|-----------|-------|-----------|
| Claims & Evidence | 2/5 | N=100, single seed, no significance tests, custom metric inflation |
| Technical Soundness | 2/5 | CMDP formalization does no work; gap between theory and heuristic |
| Novelty & Positioning | 3/5 | Routing avoidance insight is interesting; missing adaptive retrieval comparisons |
| Clarity & Writing | 3/5 | Good abstract; title too long; heavy notation for simple system |
| Significance | 2/5 | Preliminary results; no downstream evaluation |

**Overall: NEEDS_REVISION** — "Promising workshop contribution or early-stage preprint, not main-conference submission."

## Blocking Issues

### 1. Statistical Rigor
- Run 5 seeds minimum, report mean ± std
- Increase N to full HotpotQA dev set (7,405 bridge) or justify subsample
- Bootstrap confidence intervals or permutation tests
- Report component metrics as primary, composite U@B as secondary

### 2. Missing Baseline Comparisons
- Must compare against: FLARE, Self-RAG, Adaptive-RAG (at minimum qualitatively)
- Add citations: IRCoT (Trivedi et al., 2023), CRAG (Yan et al., 2024)

### 3. No Downstream Evaluation
- Add LLM answer generation with EM/F1 on HotpotQA
- Or reframe as purely retrieval contribution (not RAG)

## Important Issues

### 4. Formalization-Implementation Gap
- CMDP is decorative — derive the stopping rule from it, or move to appendix
- Foreground the heuristic as the actual contribution

### 5. Presentation
- Shorten title: "Adaptive Retrieval Routing: When Stopping Beats Switching"
- Reduce notation — heavy CMDP/options framework for an if-then-else policy
- Define "harness-level," "address space," "substrate"

### 6. Ablation Design
- abl_always_hop is a strawman — add complexity-classifier ablation
- Investigate WHY entity hops hurt on HotpotQA

## Structural Suggestion
Restructure around the stopping insight:
1. Motivation: When does multi-substrate retrieval help?
2. Method: Coverage-driven stopping (foreground heuristic)
3. Formalization: CMDP (appendix or condensed)
4. Experiments: Full HotpotQA + downstream
5. Analysis: Routing avoidance as the primary finding

## Next Steps for Revision
Priority 1: 5 seeds + larger N (can do now)
Priority 2: Add FLARE/Self-RAG comparison table (qualitative if not implementable)
Priority 3: LLM answer generation (requires API calls)
Priority 4: Restructure paper around stopping insight
