# 7 Conclusion

We studied when to stop retrieving across heterogeneous substrates and established that source-diversity stopping — a one-line heuristic that checks whether evidence has arrived from two or more independent retrieval pathways — is Pareto-optimal within a broad space of alternatives.

The evidence comes from three directions. First, the heuristic significantly outperforms comprehensive retrieval across three benchmark families (HotpotQA p<0.0001 N=1000, BRIGHT p=0.003 N=200, diluted retrieval p<0.0001 N=200), with advantages that are robust across question types and grow in harder settings (Cohen's d 0.38→0.49).

Second, seven content-aware stopping alternatives — spanning per-passage scoring, bundle-level NLI, learned classification, LLM decomposition, answer-stability tracking, confidence-gated self-assessment, and embedding-based routing — all fail. Root cause analysis identifies a common bottleneck: evidence quality is a set function over passage bundles that current models cannot reliably compute. Each method introduces more noise (from approximation errors, distribution shift, parsing failures, or phrasing instability) than information.

Third, three structural improvements — threshold optimization, novelty detection, and dual-signal stopping — converge to identical behavior. Grid search over 30 configurations confirms source diversity is the binding constraint; other structural signals are redundant.

These results identify two ceilings: a content-aware ceiling (content signals add more noise than value) and a structural ceiling (source diversity is maximally informative at zero cost). The heuristic sits at their intersection — the Pareto frontier.

The implication for practitioners is clear: invest in retrieval quality, not stopping sophistication. For researchers, the result reframes adaptive stopping from a learning problem to a signal-selection problem: the challenge is not training a better stopping model but finding a structural observable that exceeds source diversity in information content without exceeding it in noise. Until such a signal is identified, the one-line heuristic is the method of choice.
