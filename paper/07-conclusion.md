# 7 Conclusion

We studied when to stop retrieving across heterogeneous substrates and arrived at three findings.

First, **a simple structural heuristic significantly outperforms comprehensive retrieval** across three benchmark families (HotpotQA p<0.000001, BRIGHT p=0.003, diluted-distractor p<0.000001). The advantage is robust across question types and grows in harder settings (Cohen's d increases from 0.38 to 0.49 under distractor dilution).

Second, **six content-aware stopping alternatives all fail** — cross-encoder, NLI, learned classifier, LLM decomposition, answer-stability tracking, and embedding router. Root cause analysis reveals a common pattern: all six attempt to assess **evidence quality** (a set function over passage bundles that current models cannot reliably compute). The failures span per-passage scoring, bundle-level NLI, distribution-specific statistics, parsing noise, draft-phrasing noise, and routing misdirection — establishing that evidence quality assessment is the core bottleneck in adaptive retrieval stopping.

Third, **confidence-gated stopping resolves this bottleneck** by reframing the question from "is my evidence sufficient?" to "can I answer?" One LLM call after the first retrieval step achieves the best end-to-end Utility@Budget (0.799), significantly outperforming comprehensive retrieval (0.682, p=0.004) at the same cost as the structural heuristic (1.23 vs 1.16 ops). The method succeeds because it assesses **answerer readiness** — a scalar judgment the LLM can make directly — rather than evidence completeness, sidestepping the set function problem entirely.

The conceptual contribution is the **evidence-vs-readiness distinction**: stopping decisions should assess the answerer's state (easy), not the evidence's completeness (hard). This connects to a broader principle in agent design: when the value of continued action is hard to estimate from external signals, the agent's own output confidence is a more reliable stopping criterion.
