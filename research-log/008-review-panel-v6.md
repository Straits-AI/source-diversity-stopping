# Review Panel: Paper v6

**Date:** 2026-04-07
**Status:** 3 reviews received — all improved from v1

## Scores

| Criterion | R1 (IR) | R2 (Systems) | R3 (QA) | Mean | v1 Mean |
|-----------|---------|-------------|---------|------|---------|
| Soundness | 5 | 4 | 5 | 4.7 | 4.0 |
| Presentation | 6 | 5 | 7 | 6.0 | 6.3 |
| Contribution | 6 | 5 | 5 | 5.3 | 5.0 |
| Overall | 5 | 5 | 5 | 5.0 | 4.0 |
| Recommendation | B.Reject | Reject(weak) | B.Reject | — | Reject |

**+1.0 overall improvement.** All reviewers moved toward acceptance.

## Unanimous Praise

All 3 reviewers explicitly praise:
- Root cause analysis (Section 6.4) — "genuinely excellent" (R1), "strongest part" (R3), "intellectually interesting" (R2)
- Honest reporting of classifier failure and contamination discovery
- Reframing from "method" to "phenomenon + explanation"
- p=0.021 verified and real by all 3

## The Single Blocking Issue (unanimous)

**Stale numbers from contaminated runs in Sections 5 and 6.1.**

All 3 reviewers found the same problem: Table 2 shows old contaminated results (learned U@B=0.766, heuristic=0.751) while Section 6.4 shows clean results (learned=0.498, heuristic=0.759). Four different U@B values for the heuristic appear across sections.

This is a **purely editorial fix** — no new experiments needed. Just update all tables to use clean-split numbers.

## Path to Acceptance (consensus)

R1: "If a clean, consistent version were produced... solid borderline accept"
R3: "If the authors address points (a) and (b)... willing to move to weak accept"
R2: "The paper should be accepted once the authors recompute ALL numbers from the clean data split"

### Required Fixes
1. Reconcile ALL numbers to clean split (e2e_n500_clean.json)
2. Remove Table 2's contaminated learned classifier results
3. Fix scope note in Section 4.4 (contradicts E2E eval)
4. Distinguish retrieval U@B formula from E2E U@B formula
5. Qualify "outperforms" — note ensemble wins on EM/F1

### Remaining Concerns (non-blocking per reviewers)
- Entity graph adds nothing (drop or acknowledge as 2-substrate system)
- Narrow scope (10-paragraph closed sets, 2 similar benchmarks)
- Custom metric (but sensitivity analysis helps)
- Small effect size (d≈0.10)
