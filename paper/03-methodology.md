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

## 3.5 Learned Stopping Classifier

The heuristic stopping rule (Condition 1) uses a hand-tuned coverage threshold. We replace this with a **learned classifier** trained on retrieval trajectory data.

**Training data collection.** We run the ensemble policy (which always retrieves from all substrates) on 500 HotpotQA bridge questions separate from the evaluation set. At each step t, we record 9 workspace features: n_workspace_items, max_relevance, mean_relevance, min_relevance, n_unique_sources, relevance_diversity, step_number, new_items_added, and max_relevance_improvement. The oracle label is: should the policy stop at step t to maximize Utility@Budget? This produces ~4,200 step-level training examples.

**Classifier.** We train a gradient boosted tree (sklearn GradientBoostingClassifier) on 80% of the data and evaluate on 20%. The classifier achieves 93.3% accuracy and 71.7% F1 on the held-out test set.

**Key finding: the dominant stopping signal.** Feature importance analysis reveals that **max_relevance_improvement** — the change in the best evidence quality from the previous step — has importance 0.55, far exceeding all other features. The classifier has learned that **diminishing marginal returns in evidence quality is the optimal stopping signal**: when the last retrieval step didn't materially improve the best evidence, stop. This is interpretable, actionable, and validates the stopping thesis from data rather than intuition.

**Deployment.** The LearnedStoppingPolicy loads the trained classifier and uses it in place of the heuristic's coverage check. At each step after the initial semantic search, it extracts the 9 features from the current workspace state and queries the classifier. If the classifier predicts STOP (probability ≥ 0.35), the policy stops; otherwise, it escalates to lexical search.

## 3.6 Connection to Formal Framework

The routing policy can be viewed as an approximate solution to a constrained Markov decision process (CMDP) over heterogeneous action spaces, where each substrate is an "option" in the hierarchical RL sense (Sutton et al., 1999). The coverage threshold approximates the optimal stopping condition, and the cost penalty in Utility@Budget approximates the Lagrangian dual variable enforcing the budget constraint. The full formal treatment — state representation, weak dominance theorem, quantitative gain bound, and connection to information foraging theory — is presented in Appendix A.
