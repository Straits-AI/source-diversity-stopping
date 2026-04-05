# 3. Methodology

## 3.1 Problem Formulation

We formalize adaptive retrieval over heterogeneous address spaces as a constrained Markov decision process (CMDP). At each timestep $t$, the agent observes a state

$$s_t = (q,\; D_t,\; K_t,\; W_t,\; H_t,\; B_t),$$

where $q$ is the natural-language query issued by the user; $D_t$ denotes the **discovery state**, a structured record of all documents, passages, and entities that have been located but not yet fully verified; $K_t$ denotes the **knowledge state**, comprising claims that have been confirmed against retrieved evidence; $W_t$ is the **workspace**, a bounded buffer holding items currently under consideration; $H_t$ is the **action history**, recording the sequence of operations and their outcomes; and $B_t \in \mathbb{R}_{\geq 0}$ is the remaining **budget**, expressed in token-equivalent cost units. Together, the tuple $(D_t, K_t, W_t, H_t)$ constitutes a sufficient statistic for the agent's belief about the information environment: given this tuple, the conditional distribution over future rewards is independent of any earlier observations not already captured therein. This sufficiency property justifies a full MDP treatment — rather than maintaining an explicit belief distribution over latent world states — and is a standing assumption of the framework.

**Discovery and knowledge states.** The separation of $D_t$ from $K_t$ is deliberate and theoretically motivated. $D_t$ encodes *information scent* in the sense of Pirolli and Card (1999): proximal cues — titles, snippets, entity co-occurrences — that signal where valuable content is likely to reside without guaranteeing its quality. $K_t$ encodes the *information diet*: the verified, claim-level propositions that have been consolidated from retrieved content and are available for answer synthesis. A policy that collapses $D_t$ and $K_t$ into a single state variable suffers from **state aliasing** (Singh et al., 1994): two situations that differ in the depth of prior exploration become indistinguishable, causing the policy to forgo operations whose value derives precisely from resolving that uncertainty. Proposition 2 in Section 3.4 makes this loss precise.

**Action space.** Let $\mathcal{K} = \{1, \ldots, K\}$ index the available address spaces. Each address space $k$ exposes a finite set $\mathcal{A}_k$ of typed operations (e.g., SEARCH, PREVIEW, HOP). The joint action space is

$$\mathcal{A} = \bigcup_{k \in \mathcal{K}} \mathcal{A}_k \;\cup\; \{\textsc{Stop}\},$$

where $\textsc{Stop}$ terminates retrieval and triggers answer synthesis. Each address space is modeled as a **temporally extended action** or *option* in the sense of Sutton, Precup, and Singh (1999): it encapsulates an initiation condition, an internal execution policy, and a termination condition. This framing allows AEA to reason about entire retrieval sub-episodes rather than individual primitive steps, matching the granularity at which costs are meaningfully assessed.

**Reward and constraint.** Invoking action $a_t \in \mathcal{A}$ at state $s_t$ yields an immediate reward

$$r_t = U(\text{evidence}_t) - \lambda \cdot \text{cost}_t,$$

where $U(\cdot)$ measures the marginal utility of newly acquired evidence — operationalized via bundle coverage, defined formally in Section 3.5 — and $\text{cost}_t$ is the token-equivalent expense of $a_t$. The scalar $\lambda > 0$ is a fixed cost-sensitivity coefficient. The agent maximizes expected discounted return subject to a budget constraint:

$$\max_{\theta}\; \min_{\nu \geq 0}\; \mathbb{E}\!\left[\sum_{t=0}^{T} \gamma^t r_t\right] + \nu \!\left(B_0 - \sum_{t=0}^{T} \text{cost}_t\right),$$

where $\theta$ parameterizes the routing policy $\pi_\theta$, $\nu \geq 0$ is the Lagrange multiplier enforcing the budget constraint $\sum_t \text{cost}_t \leq B_0$, and $\gamma \in (0,1]$ is the discount factor. This CMDP formulation captures the fundamental tension in adaptive retrieval: each additional operation may improve answer quality but consumes finite budget. The dual variable $\nu$ adjusts dynamically to reflect how binding the constraint is, and its role in the AEA heuristic policy is approximated through the early-stopping rule described in Section 3.3.

---

## 3.2 Address Spaces

AEA operates over three heterogeneous address spaces, each exposing a common interface and each suited to a distinct class of retrieval sub-task.

**Common interface.** Every address space $k$ implements the function

$$\texttt{query}(s_t,\; \textit{operation},\; \textit{params}) \;\longrightarrow\; \texttt{ActionResult}(\textit{items},\; \textit{cost}_{\text{tokens}},\; \textit{cost}_{\text{latency}}),$$

where $\textit{items}$ is a ranked list of retrieved objects (passages, entities, or triples), $\textit{cost}_{\text{tokens}}$ is the number of tokens consumed by encoding, scoring, or generation steps, and $\textit{cost}_{\text{latency}}$ is wall-clock time in milliseconds. This uniform interface decouples the routing policy from implementation details of individual substrates, allowing new address spaces to be registered without modifying the harness.

**Semantic address space.** Documents are encoded offline using a sentence-transformers model (Reimers and Gurevych, 2019) into fixed-dimensional dense vectors. At query time, the query $q$ is encoded with the same encoder and ranked against the corpus by cosine similarity. Two operations are exposed: SEARCH($q$, $k$), which returns the top-$k$ passages by similarity score, and PREVIEW($\textit{id}$), which returns the full text of a specific document. The semantic space excels at surface-paraphrase retrieval — cases where the query and the answer-bearing passage share distributional meaning but not lexical form.

**Lexical address space.** Documents are indexed using BM25 (Robertson and Zaragoza, 2009), implemented via the `rank_bm25` library. The sole exposed operation is SEARCH($q_\text{lex}$, $k$), where $q_\text{lex}$ is a focused keyword reformulation of the original query. BM25 complements the semantic space by rewarding exact token overlap, making it effective for queries that contain rare proper nouns, identifiers, or technical terms that dense encoders may conflate with semantically similar but factually distinct strings.

**Entity graph address space.** Named entities are extracted from the corpus using regex-based heuristics, and co-occurrence edges are constructed between entities that appear within a fixed token window. The resulting graph $G = (V, E)$, where $V$ is the entity set and $E$ encodes co-occurrence strength, supports two operations: SEARCH($e$), which retrieves passages mentioning entity $e$, and HOP($e$, $d$), which performs breadth-first search from entity $e$ to depth $d$ and returns passages associated with discovered neighbors. The entity graph is the natural substrate for multi-hop queries that require traversing relational chains — for example, "the director of the studio that produced film $X$" — which neither dense nor lexical retrieval resolves reliably in a single step.

**Assumption (substrate complementarity).** The three address spaces are assumed to be *complementarily specialized*: on any query, the substrate that achieves highest marginal utility varies in a manner that cannot be determined by query surface form alone. This assumption motivates the adaptive routing policy; if one substrate were uniformly dominant, a fixed strategy would be optimal and no adaptive mechanism would be needed.

---

## 3.3 Coverage-Driven Routing Policy

AEA implements a deterministic heuristic policy $\pi_\text{AEA}$ that approximates the solution to the CMDP of Section 3.1. The policy is organized as an ordered decision procedure executed after each retrieval step.

**Step 0 — Semantic anchor.** The policy always initiates with a semantic SEARCH. This step is unconditional: it establishes a set of *anchor documents* that define the initial relevance landscape and populate $D_t$ with high-recall candidates. The semantic search is treated as a fixed cost that is paid regardless of subsequent decisions, analogous to an obligatory initialization in an options framework.

**Step 1 and beyond — Coverage-driven decision.** After each completed operation, the policy evaluates the current workspace $W_t$ against three conditions, tested in priority order.

*Condition 1 (Sufficient coverage — STOP):* If $W_t$ contains at least two high-relevance items — items whose relevance score exceeds a threshold $\tau_\text{high}$ — drawn from at least two distinct source substrates, the policy issues $\textsc{Stop}$. This condition encodes the intuition that multi-source corroboration is a reliable proxy for answer completeness: evidence that converges from independent substrates is unlikely to reflect a shared retrieval artifact.

*Condition 2 (Single-source multi-hop — entity HOP):* If the current evidence in $W_t$ derives from a single source substrate and the query structure suggests a relational chain (detected heuristically by the presence of possessive or relational phrases), the policy issues a HOP operation on the most central entity in $D_t$. This condition redirects effort toward the entity graph precisely when lexical and semantic substrates have been exhausted on a single modality, and when the remaining information gap is most likely relational in character.

*Condition 3 (Default — lexical fallback):* Otherwise, the policy issues a lexical SEARCH with a keyword reformulation of $q$ constructed by extracting salient noun phrases from $q$ and from the highest-scoring item in $W_t$. This fallback broadens coverage by exploiting exact-match signals that the semantic anchor may have missed.

**Primary mechanism: routing avoidance.** A critical design insight is that the dominant source of cost savings in $\pi_\text{AEA}$ is **early stopping** under Condition 1, not positive substrate selection under Conditions 2 and 3. The policy avoids unnecessary operations by recognizing when the current workspace is already sufficient for answer synthesis, rather than by consistently directing queries to the single optimal substrate. This reframes AEA as a mechanism of *selective operation avoidance* over heterogeneous address spaces: the adaptive value lies in knowing when to stop, with substrate routing serving as a secondary mechanism that governs what to do when continuation is warranted.

**Workspace management.** The workspace $W_t$ is maintained as a fixed-capacity buffer. After each retrieval step, items are scored by a weighted combination of relevance (cosine similarity to $q$) and recency (inverse position in $H_t$). The top-2 items by this score are pinned; items whose relevance falls below a threshold $\tau_\text{evict} < \tau_\text{high}$ are evicted. Pinning ensures that the best-known evidence is always available to Condition 1; eviction prevents the workspace from accumulating low-signal content that could mask the signal of genuinely relevant items.

---

## 3.4 Theoretical Justification

We provide three formal results that underpin the design choices of AEA.

**Theorem 1 (Weak Dominance).** *Let $\pi^*$ denote any optimal policy for the CMDP of Section 3.1, and let $\pi^*_k$ denote an optimal policy restricted to address space $k$ alone. Then for all states $s$,*

$$V^{\pi^*}(s) \;\geq\; \max_{k \in \mathcal{K}}\; V^{\pi^*_k}(s).$$

*Proof sketch.* Any single-substrate policy $\pi^*_k$ is a member of the feasible set over which $\pi^*$ optimizes. Since $\pi^*$ maximizes over a weakly larger set, the inequality follows immediately. The bound is tight when one substrate dominates all others uniformly across the state space. $\square$

Theorem 1 is deliberately modest: it establishes that heterogeneous routing cannot be harmful in principle, without quantifying the gain. The following proposition provides a quantitative lower bound.

**Proposition 1 (Quantitative Gain).** *Let $\delta \in [0,1]$ denote the heterogeneity gap — the fraction of query instances for which the identity of the best-performing substrate differs from the overall best single substrate. Let $\Delta_\text{min} > 0$ denote the minimum per-instance value difference between the best and second-best substrate on those instances. Then*

$$V^{\pi^*}(s_0) - \max_{k} V^{\pi^*_k}(s_0) \;\geq\; \delta \cdot \Delta_\text{min}.$$

This bound is informative only when $\delta > 0$ (genuine heterogeneity exists) and $\Delta_\text{min}$ is non-negligible.

**Toy example.** Consider a two-substrate ($K=2$: text search and SQL database) three-step task in which the correct answer requires locating a person's name (step 1, best resolved by text search) and retrieving associated numerical data (steps 2-3, best resolved by SQL). Let $c$ denote the per-step retrieval cost and $p > 0$ the per-step penalty incurred when querying the wrong substrate. Value estimates under four policies are:

| Policy | Value |
|---|---|
| Adaptive (AEA) | $3 - 3c$ |
| Text-only | $3 - 3c - 2p$ |
| Database-only | $3 - 3c - p$ |
| Ensemble (both substrates at every step) | $3 - 6c$ |

For small $c$ and any $p > 0$, adaptive strictly dominates text-only and database-only. Adaptive dominates ensemble whenever $3c > 0$, which holds as long as retrieval is not free.

**Proposition 2 (D/K Split prevents state aliasing).** *Let $s = (q, D, K, W, H, B)$ and $s' = (q, D', K, W, H, B)$ with $D \neq D'$. A policy that collapses $D$ and $K$ into a single state variable treats $s$ and $s'$ as identical and must assign them the same action. If the optimal action differs between $s$ and $s'$ — which occurs whenever the information scent in $D$ versus $D'$ points to different substrates — the collapsed policy suffers positive regret relative to $\pi^*$.*

This is an instance of the state-aliasing phenomenon identified by Singh et al. (1994): insufficient state representation forces suboptimal action selection.

---

## 3.5 Utility@Budget Metric

Standard retrieval metrics — precision, recall, NDCG — do not account for budget consumption, and answer-quality metrics alone do not penalize unnecessary operations. We therefore introduce **Utility@Budget**, a composite metric that jointly rewards accurate, well-supported answers while penalizing cost:

$$\text{Utility@Budget} \;=\; \text{AnswerScore} \times (1 + \eta \cdot \text{EvidenceScore}) \;-\; \mu \cdot \text{Cost},$$

where $\eta = 0.5$ and $\mu = 0.3$ are coefficients fixed prior to any experimental evaluation. AnswerScore $\in [0,1]$ measures the proportion of gold supporting facts present in the final workspace. EvidenceScore $\in [0,1]$ measures the precision of the workspace contents with respect to gold supporting facts. Cost is the total token-equivalent expense of all retrieval operations issued for the query, normalized to $[0,1]$.

The multiplicative coupling between AnswerScore and EvidenceScore reflects the judgment that evidence quality is valuable only insofar as it accompanies a correct answer: high EvidenceScore with zero AnswerScore contributes nothing. The additive cost term penalizes budget consumption independently of answer quality.

**Bundle coverage.** EvidenceScore is operationalized as **bundle coverage**:

$$\text{EvidenceScore} \;=\; \frac{|\text{requirements\_satisfied}|}{|\text{requirements\_total}|},$$

where requirements_total is the set of information requirements identified for a given query, and requirements_satisfied is the subset thereof for which at least one retrieved item provides confirming evidence.
