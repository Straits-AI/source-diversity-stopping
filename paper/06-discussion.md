# 6 Discussion

## 6.1 Reframing Adaptive Retrieval as Negative Selection

The motivating thesis was that an adaptive agent should learn to select the right substrate. The evidence suggests a more precise characterization: the primary value of adaptation lies in identifying which operations are unnecessary and avoiding them.

This reframing has practical implications. Under the original framing, a router must learn a positive assignment function mapping queries to substrates. Under the revised framing, the router's primary responsibility is a simpler binary judgment: is the current coverage sufficient, or must the agent escalate? Default behavior is the cheapest operation plus halt; escalation is the exception.

Negative selection is substantially easier to learn and audit. A coverage sufficiency check can be grounded in observable signals — passage overlap, confidence scores, entity resolution — whereas positive substrate selection requires implicit reasoning about modality-query affinity.

This perspective clarifies why π_ensemble fails despite its high recall: it applies positive selection of all substrates unconditionally. Its recall of 0.940 is highest, yet U@B is nearly zero (0.0028), because the cost of comprehensive coverage is not recoverable under budget-aware evaluation. Adaptive retrieval under resource constraints is fundamentally a problem of restraint, not coverage maximization.

## 6.2 When Adaptive Routing Provides the Greatest Benefit

Adaptive routing provides the greatest benefit when the action space contains expensive operations that are frequently unnecessary. This condition holds on HotpotQA, where entity-hop traversal is costly and only required for a subset, yielding 67% U@B improvement. It holds less strongly on Heterogeneous v2, where semantic retrieval is already a strong default.

Adaptive routing is additionally most beneficial under heterogeneous workloads where no single substrate dominates across task types. The per-task breakdown confirms this: AEA leads on multi-step tasks while falling behind on single-substrate tasks.

Conversely, adaptive routing provides minimal value when a single substrate dominates uniformly and when budget constraints are not binding.

## 6.3 Limitations

Several limitations constrain generalizability:

1. **Retrieval-only evaluation.** No downstream answer generation stage. U@B does not measure whether retrieved passages enable correct answers.

2. **Heuristic policy only.** Routing decisions are hand-designed rules, not learned from data. A learned router could address the over-searching failures on Low Lexical Overlap tasks.

3. **Three address spaces.** Real environments include web search, tool invocation, structured databases. Additional modalities may produce different routing dynamics.

4. **Synthetic benchmark.** Heterogeneous v2 is constructed, not naturally occurring. External benchmarks (BRIGHT, NoLiMa) are needed for broader validation.

5. **Single seed, no significance testing.** N=100 per benchmark. Results should be interpreted with appropriate caution.

6. **MuSiQue unavailable.** The third benchmark used synthetic fallback data (N=40), limiting external validation.

## 6.4 Future Work

Several directions follow from the evidence:

**Learned router.** Trajectory data from evaluation provides natural supervision for training a coverage sufficiency classifier, addressing over-searching and under-routing failures.

**Step-conditional routing.** Oracle trajectories show strong step-position preferences (semantic at Step 1, entity at Step 2). A step-conditioned router may reduce unnecessary entity invocations.

**Expanded address spaces.** Web retrieval, tool-augmented search, and structural navigation would test whether selective avoidance generalizes. Cost differentials are larger across these modalities, potentially amplifying U@B benefits.

**Full pipeline evaluation.** Coupling retrieval with an LLM reader and measuring end-to-end answer accuracy would validate whether U@B improvements translate downstream.

**Budget-aware RL.** Training the policy directly on a reward that penalizes operation cost is the natural formulation for joint optimization.
