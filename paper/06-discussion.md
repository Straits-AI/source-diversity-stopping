# 6 Discussion

## 6.1 First Principles: Why Convergence Is the Right Stopping Signal

The stopping problem in heterogeneous document navigation reduces to a fundamental tradeoff at each step:

**Value of one more step = Expected Information Gain − Cost − Noise Added**

Stop when this is ≤ 0.

Estimating this directly is hard for three reasons, each corresponding to one of our empirical ceilings:

1. **Expected Information Gain is a set function** (content-aware ceiling). The value of finding passage E₃ depends on whether you already have E₁ and E₂. Evidence can be complementary (E₁+E₂ together answer the question) or redundant (E₃ adds nothing). Assessing this requires modeling passage interactions — the exact set function problem our seven content-aware methods fail at (Section 5.4).

2. **The evidence distribution is unknown** (why learning fails). You don't know what's out there until you search. A learned classifier trained on one distribution sees different evidence patterns on another — the OOD failure of our GBT classifier.

3. **Estimating value costs resources** (why LLM-based stopping loses). Every token spent assessing evidence quality is a token not spent gathering evidence. Confidence-gated stopping and answer stability both lose to the heuristic because the assessment cost exceeds the assessment value.

**Source diversity sidesteps all three.** Instead of estimating "how much better could evidence get?" (hard), it checks: "has evidence converged from independent pathways?" (easy). This is a **convergence signal** — and convergence is fundamentally the right stopping criterion because:

- **Independent pathways have independent failure modes.** Semantic search fails when there's no distributional similarity; lexical search fails when there's no keyword overlap; structural navigation fails when hierarchy doesn't match the query. When TWO independent methods succeed despite having independent failure modes, the probability that evidence is relevant is much higher than when only one succeeds. This is the same reasoning behind ensemble consensus in ML.

- **Convergence is observable, cheap, and distribution-invariant.** It requires only counting source identifiers — zero model inference, zero training, zero distribution-specific assumptions.

- **Convergence signals diminishing returns.** If two independent approaches have already found evidence, a third is likely to find the same thing (redundancy) or nothing new (diminishing marginal gain).

The source-diversity heuristic is the simplest operationalization of this convergence principle: stop when at least two independent retrieval pathways have each contributed evidence. Our experiments show this is sufficient — no enrichment we tested (threshold optimization, novelty detection, dual signals) improves on it (Section 5.12).

## 6.2 The Retrieval-Computation Boundary

The convergence principle applies to **retrieval and navigation** tasks — where the agent gathers evidence from external sources. It does NOT apply to **computation** tasks, where the answer requires calculation rather than evidence (Section 5, tool execution results). For computation, the right stopping signal is tool-execution completion: once the calculator has produced a result, stop.

This boundary is principled, not arbitrary. Retrieval and computation have different information structures:

| Property | Retrieval/Navigation | Computation |
|----------|---------------------|-------------|
| Value of more steps | Diminishing (convergence) | Zero after execution |
| Independent pathways | Multiple (semantic, lexical, structural) | One (execute the tool) |
| Evidence interactions | Set function (complementary/redundant) | Deterministic (input → output) |
| Right stopping signal | Convergence of pathways | Tool completion |

## 6.3 Mapping to Document Navigation Domains

The convergence principle applies to ANY heterogeneous information environment where an agent navigates multiple substrates:

**Codebase navigation.** A coding agent searching for a bug greps for error messages (lexical), embeds code chunks (semantic), traverses import graphs (relational), and browses the file tree (structural). When grep, embeddings, and import tracing all converge on the same module — that's convergence from independent pathways. No existing coding tool (Cursor, aider, Copilot, SWE-agent) uses diversity-based stopping; all rely on LLM self-judgment or hard token budgets, which our experiments show are suboptimal for the retrieval component of the task.

**Legal discovery.** An attorney searches case law by keyword, follows citation chains, and navigates statute hierarchies. When keyword search, citation graph traversal, and statutory hierarchy all point to the same legal principle — independent pathways have converged.

**Enterprise document retrieval.** An employee searches for a policy by keywords, browses the organizational file tree, and follows cross-references from related documents. When search, tree navigation, and cross-references all lead to the same document — convergence.

**Medical record review.** A clinician searches patient records by diagnosis code, temporal ordering, and specialist cross-references. When structured queries, temporal browsing, and referral chains converge on the same clinical events — sufficient evidence.

In each domain, the substrates differ but the stopping logic is identical: **stop when independent navigation pathways converge.** Our experiments validate this on five benchmarks spanning QA, fact verification, structural navigation, and reasoning-intensive retrieval. The codebase and legal/medical/enterprise domains are natural extensions where the same principle should apply, with tool-execution completion handling the computation boundary.

## 6.4 Why Ten Content-Aware Alternatives Fail: The Two-Ceiling Framework

The central puzzle is that ten qualitatively different approaches — spanning seven design categories — all fail to improve on the convergence heuristic. The following analysis proposes explanatory hypotheses grounded in the experimental evidence.

### 6.4.1 The Empirical Phenomenon

| Approach | Mechanism | AvgOps | E2E U@B | vs Heuristic |
|---|---|---|---|---|
| Cross-encoder | MS MARCO (q, passage) scoring | 3.09 | 0.655 | -0.104 |
| NLI bundle | DeBERTa-v3 on concatenated evidence | 6.09 | 0.433 | -0.326 |
| Learned GBT | Workspace statistics classifier | 5.00 | 0.498 | -0.261 |
| LLM decomposition | Requirement parsing | 2.95 | 0.758 | -0.001 |
| Answer stability | Multi-draft convergence | 3.18 | 0.668 | -0.087 |
| Confidence-gated | LLM self-assessment | 2.23* | 0.699* | -0.060* |
| Embedding router | Question-type classifier | 1.28 | tied | +0.001 |
| Threshold optimization | Grid search (30 configs) | 1.16 | 0.031 | ≈0 |
| Novelty detection | Embedding similarity | 1.16 | 0.031 | ≈0 |
| Dual signal | Source diversity + relevance gap | 1.16 | 0.031 | ≈0 |
| **Heuristic** | **Source diversity (2/2)** | **1.16** | **0.759** | **baseline** |

*Confidence-gated: with LLM cost included (+0.23 ops). Fails to replicate at N=500 (p=0.97) and on BRIGHT (p=0.006, worse).

### 6.4.2 The Content-Aware Ceiling

Every content-aware method (rows 1–7) attempts to estimate the **value function** — "how much would more evidence improve the answer?" Each introduces noise that exceeds the information gained:

- The cross-encoder estimates per-passage relevance, not bundle sufficiency (set function decomposition failure).
- The NLI model takes bundles as premise but multi-hop questions resist well-formed hypothesis construction.
- The GBT classifier captures distribution-specific workspace statistics (spurious correlation under shift).
- The LLM decomposition introduces parsing noise (~40% malformed requirements).
- Answer stability measures draft phrasing changes, not answer correctness (phrasing noise).
- Confidence-gated stopping doesn't replicate at scale (N=500) or across benchmarks (BRIGHT).
- The embedding router confirms routing isn't the bottleneck.

The unifying explanation: **all seven attempt to model the value of future evidence, which requires solving the set function problem. The convergence heuristic succeeds by not attempting this estimation at all.** This connects to optimal stopping theory: threshold rules on low-noise observables dominate value-estimation approaches when the value function is hard to learn.

### 6.4.3 The Structural Ceiling

The three structural improvements (rows 8–10) converge to identical behavior because source diversity is the **binding constraint** in the stopping decision. Grid search over 30 threshold configurations confirms the original 2/2/0.4 parameters are near-optimal; novelty detection and relevance-convergence signals are redundant with source diversity.

This establishes that source diversity **saturates** the structural signal space: no zero-cost enrichment tested improves upon it.

### 6.4.4 The Pareto Frontier

The heuristic sits at the intersection of these two ceilings:
- Moving "up" (adding content awareness) increases noise without improving quality
- Moving "right" (enriching structural signals) is impossible — source diversity is already maximal

This intersection IS the Pareto frontier for the stopping decision, within the space of ten tested alternatives.

## 6.5 Comparison with Existing Systems

Our comparison with existing adaptive retrieval and coding tools reveals that no system uses convergence-based stopping:

| System | Domain | Stopping Mechanism |
|--------|--------|-------------------|
| Self-RAG | QA | Learned reflection tokens (single modality) |
| FLARE | QA | Low-confidence triggers retrieval |
| Cursor | Code | LSP lint feedback + context budget |
| aider | Code | Hard token budget for repo map |
| Copilot Agent | Code | LLM self-judgment |
| SWE-agent | Code | LLM decides; output caps |
| Sourcegraph | Code search | Top-k reranker cutoff |

None checks whether evidence has converged from independent pathways. Source-diversity stopping fills this gap.

## 6.6 Limitations

1. **E2E evaluation on one benchmark.** End-to-end evaluation with LLM answer generation (N=500, p=0.021) uses only HotpotQA Bridge. BRIGHT and other evaluations are retrieval-only. The E2E effect size is small (d=0.103).

2. **Custom evaluation metric.** Utility@Budget is author-defined. Sensitivity across μ (Section 5.5) shows crossover at μ≈0.3; sensitivity to η has not been tested. The Pareto-optimality claim is bounded by the tested parameter range.

3. **Pareto-optimality is bounded.** We test ten alternatives across seven design categories, but the space is unbounded. RL-trained stopping policies are not tested. The claim is "within tested alternatives."

4. **Two effective substrates for most experiments.** Semantic + lexical are the primary substrates; entity graph is a negative control; structural and executable are tested on separate benchmarks. A unified evaluation with all four substrates on the same benchmark would strengthen the convergence argument.

5. **Codebase navigation is proposed, not tested.** The mapping to code search is based on structural analysis of existing tools, not empirical validation. Testing on SWE-bench or a code retrieval benchmark is the highest-priority future work.

6. **LLM model identity.** Answer generation uses "gpt-oss-120b" via OpenRouter. Results may differ with other models.

## 6.7 Future Work

**Codebase navigation.** Test convergence-based stopping on SWE-bench-style tasks where the agent navigates with grep + embeddings + tree-sitter AST + import graph. The prediction: source-diversity stopping will reduce wasted retrieval steps compared to LLM self-judgment (the current default in coding agents).

**Multi-level convergence.** Document navigation often involves hierarchical decisions (find the right directory → find the right file → find the right function). Testing convergence at each level independently could yield a more nuanced stopping policy.

**Convergence + verification hybrid.** For domains with verifiable signals (code: lint passes; math: proof checks), combining convergence-based stopping with verification signals could improve on either alone.

**Real heterogeneous corpora.** Testing on enterprise document stores mixing PDFs, spreadsheets, code, and emails would validate the convergence principle in the most realistic setting.
