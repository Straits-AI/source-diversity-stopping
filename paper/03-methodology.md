# 3 Method

This section describes the coverage-driven retrieval routing policy. The formal constrained MDP framework motivating the design is presented in Appendix A; here we focus on the operational policy and its components.

## 3.1 Setup and Terminology

The routing policy operates over a set of **retrieval substrates**, each exposing a common query interface:

| Substrate | Mechanism | Best For |
|-----------|-----------|----------|
| Semantic | Dense embeddings + cosine similarity | Paraphrase, distributional similarity |
| Lexical | BM25 keyword scoring | Exact terms, identifiers, rare entities |
| Entity graph | Named entity co-occurrence + BFS | Multi-hop relational chains |

The two primary substrates are semantic and lexical retrieval. We include entity graph traversal as a third substrate to test whether structured retrieval adds value; ablation analysis (Section 5.3) shows it does not contribute on the evaluated benchmarks.

Each query to a substrate returns a ranked list of passages and incurs a cost (measured in operations). The **workspace** is a bounded buffer holding the passages currently under consideration. The policy's job is to decide, after each retrieval step, whether to stop (the evidence is sufficient) or escalate (query another substrate).

Two state components guide routing decisions:

- **Discovery state**: what the agent has *located* — passage titles, entity mentions, structural cues. This is "information scent" in the sense of Pirolli and Card (1999).
- **Knowledge state**: what the agent has *verified* — grounded claims extracted from retrieved passages.

The distinction matters because knowing a relevant passage exists (discovery) is different from knowing what it says (knowledge). A policy without this distinction conflates "I haven't looked" with "I looked and found nothing," leading to redundant exploration.

## 3.2 Coverage-Driven Routing Policy

The policy implements a three-condition decision procedure evaluated after each retrieval step.

**Step 0 — Semantic anchor (always).** The policy initiates with a dense retrieval query using the original question. This establishes anchor passages and populates the workspace with high-recall candidates. Dense retrieval is chosen as the default because it has the broadest coverage and lowest per-operation cost among the three substrates.

**After each step — Coverage check.** The policy evaluates three conditions in priority order:

*Condition 1 — Sufficient coverage (STOP).* If the workspace contains at least two high-relevance passages (relevance score >= 0.4) drawn from at least two distinct sources, the policy stops. The intuition: multi-source corroboration signals that the evidence base is diverse enough for answer synthesis, and additional retrieval is unlikely to improve quality enough to justify its cost.

*Condition 2 — Single-source gap (ESCALATE via entity hop).* If the workspace evidence comes from a single source and the question structure suggests a relational chain (detected via heuristic patterns: possessives, "birthplace of," "director of," "founded by"), the policy escalates to the entity graph substrate. This redirects effort toward the one substrate designed for relational traversal, precisely when lexical and semantic retrieval have converged on a single document.

*Condition 3 — Default (ESCALATE via lexical fallback).* Otherwise, the policy issues a BM25 query with a keyword reformulation. This broadens coverage via exact-match signals that dense retrieval may have missed.

**The primary mechanism is Condition 1.** The key design insight — validated by ablation — is that most of the policy's value comes from stopping early when coverage is sufficient, not from the specific substrate selected when escalation occurs. On HotpotQA Bridge, the policy stops after a single operation on the majority of questions, reducing average operations from 2.00 to 1.21.

## 3.3 Workspace Management

The workspace is a fixed-capacity buffer (10 items). After each retrieval step:

1. Items are scored by cosine similarity to the query embedding.
2. The top-2 items are **pinned** (protected from eviction).
3. Items with relevance below 0.15 are **evicted**.

Pinning ensures the best evidence persists across steps. Eviction prevents low-signal content from diluting the coverage check.

## 3.4 Utility@Budget Metric

Standard retrieval metrics (precision, recall, NDCG) do not account for operation cost. We evaluate all systems using a composite metric:

**Utility@Budget** = SupportRecall × (1 + η × SupportPrecision) − μ × NormalizedCost

where η = 0.5 weights evidence precision and μ = 0.3 penalizes cost. Both coefficients are fixed before experiments and not tuned. NormalizedCost is the ratio of operations used to the maximum operations used by any policy on the same question.

This metric rewards high-recall, high-precision retrieval while penalizing unnecessary operations. A policy that retrieves everything but wastes budget is penalized; a policy that retrieves nothing pays no cost but scores zero on recall. The optimal strategy under this metric is to retrieve exactly what is needed and stop.

## 3.5 Confidence-Gated Stopping (Testing the Evidence-vs-Readiness Hypothesis)

The content-aware approaches tested in Section 5.4 all fail because they try to assess **evidence quality** — whether the retrieved passages are sufficient to answer the question. This is fundamentally a set function over passage bundles: the sufficiency of {p₁, p₂, ..., pₖ} depends on their joint content in ways that cannot be decomposed from individual scores.

To test whether this bottleneck can be bypassed, we implement a simpler question: instead of asking "is my evidence good enough?", ask **"can I answer this?"** The LLM — which will ultimately generate the answer — is the most direct judge of its own readiness.

**Confidence-gated stopping** works as follows:

1. **Step 0:** Semantic search (same as all policies).
2. **Step 1:** Present the workspace evidence to the LLM with the prompt:
   ```
   Evidence: {workspace passages}
   Question: {question}
   Can you answer this question from the evidence above?
   If YES: respond with just the answer (1-5 words).
   If NO: respond with exactly "NEED_MORE".
   ```
3. **If the LLM responds with an answer** (confident) → STOP. (77% of questions in our evaluation.)
4. **If the LLM responds "NEED_MORE"** (uncertain) → execute one lexical search, then STOP. (23% of questions.)

**Total cost:** 1–2 retrieval operations + exactly 1 LLM call. This matches the structural heuristic's cost efficiency (1.23 vs 1.16 ops) while adding content-aware judgment.

**Why this succeeds where others fail:** The confidence-gated approach sidesteps the set function problem entirely. It does not try to assess evidence quality from outside — it asks the answerer to assess its own state from inside. The LLM's response encodes an implicit bundle-level sufficiency judgment: by attempting to answer, it determines whether the evidence supports a confident response without explicitly modeling passage interactions.

**Relation to the structural heuristic:** Confidence-gated stopping can be viewed as an augmentation of the structural heuristic rather than a replacement. On easy questions (77%), both policies stop after one step — the LLM's confidence aligns with the structural signal. On the remaining 23%, the LLM identifies cases where the structural signal would incorrectly indicate sufficiency (evidence from multiple sources but not actually answering the question) and escalates.

## 3.6 Connection to Formal Framework

The routing policy can be viewed as an approximate solution to a constrained Markov decision process (CMDP) over heterogeneous action spaces, where each substrate is an "option" in the hierarchical RL sense (Sutton et al., 1999). The coverage threshold approximates the optimal stopping condition, and the cost penalty in Utility@Budget approximates the Lagrangian dual variable enforcing the budget constraint. The full formal treatment — state representation, weak dominance theorem, quantitative gain bound, and connection to information foraging theory — is presented in Appendix A.
