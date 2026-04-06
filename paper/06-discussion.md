# 6 Discussion

## 6.1 Smart Stopping Beats Smart Searching

The end-to-end results establish a clear hierarchy: **smart stopping > brute force > smart searching** under budget-aware evaluation. The heuristic AEA policy (E2E U@B 0.760) outperforms comprehensive retrieval (ensemble, 0.731) and LLM-guided routing (0.652), despite having lower F1 and recall than both.

This hierarchy has a precise explanation. The cost penalty in Utility@Budget creates a threshold: additional retrieval is worthwhile only if the marginal F1 improvement exceeds μ × (marginal cost / max cost). For the ensemble, the third retrieval step (entity graph) adds ~0.13 recall but costs 1.0 normalized operation, yielding marginal utility of ~0.13 × (1+0.5×0.94) × improvement_rate - 0.3 × 0.33 ≈ −0.01 — slightly negative. The ensemble's last step hurts more than it helps.

The LLM router faces the same trap at a finer grain: it correctly identifies questions that need more evidence, but the additional operations it authorizes produce diminishing F1 returns. Its average 2.54 operations deliver only 0.007 more F1 than the heuristic's 1.21 operations, a marginal return of 0.005 F1 per additional operation — well below the break-even threshold.

The practical implication is a design principle for retrieval systems: **default to restraint and require strong evidence of a coverage gap before escalating.** The heuristic's simple coverage check (2+ high-relevance items from 2+ sources → stop) implements this principle at near-zero computational cost.

## 6.2 The Positive Routing Gap

The LLM-routed policy demonstrates that positive routing — choosing the right substrate for each question — is achievable. Its action distribution (STOP=33%, SEMANTIC=19%, LEXICAL=35%, HOP=14%) shows genuine per-question substrate variation, and its higher recall (0.845 vs 0.795) confirms that the LLM identifies evidence gaps the heuristic misses.

But positive routing is not yet cost-efficient. The gap between the LLM router's routing intelligence and the heuristic's cost discipline identifies the central open challenge: **calibrated stopping** — a policy that combines the LLM's ability to recognize genuinely insufficient evidence with the heuristic's discipline to stop when evidence is merely adequate rather than comprehensive.

We conjecture that the optimal policy lies between these two extremes: it would stop as often as the heuristic (saving cost on easy questions) while routing as intelligently as the LLM (achieving higher recall on hard questions). Training such a policy requires a reward signal that captures the downstream cost-quality tradeoff — exactly the Utility@Budget metric we define.

## 6.3 Comparison with Existing Systems

Our comparison with FLARE, Self-RAG, Adaptive-RAG, IRCoT, and CRAG (Appendix B) reveals that AEA occupies a distinct niche: it is the only system that (a) routes across qualitatively different substrate types, (b) includes an explicit cost model, and (c) treats stopping as a first-class routing decision.

Direct numerical comparison is not valid: those systems report downstream QA accuracy after full LLM generation on different benchmarks with different assumptions. However, the design dimension analysis shows that none of the existing systems addresses the stopping-vs-routing tradeoff that our experiments identify as central.

## 6.4 Limitations

1. **Single benchmark for end-to-end.** The E2E results (N=100) use only HotpotQA Bridge. Validation on additional benchmarks (BRIGHT, NoLiMa) is needed.

2. **No statistical testing on E2E.** Bootstrap CIs and permutation tests are reported only for the retrieval-only evaluation (N=500). The E2E evaluation (N=100) reports point estimates.

3. **Custom evaluation metric.** Utility@Budget is author-defined. The specific η and μ values determine the ranking — sensitivity analysis across parameter ranges is reported in Appendix C.

4. **Three address spaces.** Real retrieval environments include web search, tool invocation, and structural navigation. Cost differentials across these modalities are larger, potentially amplifying the benefit of selective stopping.

5. **Heuristic policy.** The routing decisions are hand-designed rules. The results show what is achievable without learning; a learned policy could close the gap between routing avoidance and positive routing.

6. **Free-tier LLM for routing and answers.** The gpt-oss-120b model is capable but not state-of-the-art. Stronger models might shift the balance toward positive routing.

## 6.5 Future Work

**Calibrated stopping policy.** Train a stopping classifier on trajectory data with downstream F1 as the reward signal. The key question: can a learned policy match the heuristic's stopping efficiency while capturing the LLM router's recall advantage?

**Step-conditional routing.** Oracle trajectories show step-position preferences (semantic at step 1, entity at step 2). A router conditioned on step position could reduce false-positive escalations.

**Expanded substrates.** Web search, tool execution, and structural navigation would test whether the stopping > searching hierarchy holds when cost differentials are larger.

**Budget sensitivity.** The hierarchy may invert under very tight budgets (where any retrieval is expensive) or very loose budgets (where cost is negligible). Characterizing the budget regime where each policy dominates is an important practical question.
