# Hypothesis Formation

**Date:** 2026-04-04
**Phase:** 2
**Iteration:** 1 (revised after theory review)
**Status:** completed (RIGOROUS — 7/8 resolved, 1 partially resolved non-blocking)

## Context
Literature review (001) confirmed 5 key gaps. Direction A approved: harness-level agentic attention policy over heterogeneous address spaces.

---

## 1. Hypothesis

### Claim (falsifiable)

**A harness-level policy π_θ(a|s) that adaptively routes across K heterogeneous address spaces with explicit discovery/knowledge state tracking achieves strictly higher Utility@Budget than any policy restricted to a single address space, when tasks require information from multiple substrates or when the optimal substrate varies across reasoning steps.**

More specifically:

**H1:** AEA achieves higher Utility@Budget than fixed retrieval pipelines under equal cost budgets.

**H2:** AEA shows larger gains on tasks with low lexical overlap, multi-hop dependencies, and strong document structure (BRIGHT, NoLiMa) than on simple paraphrase retrieval.

**H3:** On structured-data and exact-computation tasks, the policy learns to prefer executable addressing (SQL, code, APIs) over text retrieval.

**H4 (central):** Ablating the router (fixing to a single address space) causes greater performance degradation than improving any single substrate. The gain comes from *routing*, not from any individual substrate.

### Independent Variables
- Policy type: {heuristic routing, learned routing, fixed single-substrate (baselines)}
- Address spaces available: {full set, subsets for ablation}
- Action space: {full operations, restricted operations for ablation}
- State representation: {with D/K split, without D/K split}

### Dependent Variables
- Utility@Budget (primary composite metric)
- Answer quality: EM, F1, judge score
- Evidence quality: support precision, support recall, bundle coverage
- Discovery efficiency: steps to first relevant evidence, file hit@k
- Cost: tokens consumed, wall-clock latency, tool calls

### Controls (held constant)
- Base LLM: one model, fixed across all experiments
- Corpus/documents: fixed per benchmark
- Evaluation harness: immutable per evaluation contract
- Random seeds: [42, 123, 456, 789, 1024]
- Per-run token and latency budgets

### Expected Effect
- H1: 10-20% improvement on Utility@Budget over best fixed pipeline
- H2: 2-3x larger gain on reasoning-intensive benchmarks vs. paraphrase tasks
- H3: >90% of actions on structured-data tasks use executable addressing
- H4: Router ablation accounts for >50% of total improvement

---

## 2. Mathematical/Theoretical Justification

### 2.1 Formal Setup

Model the information environment as a budgeted decision process.

**Observable state:** s_t = (q, D_t, K_t, W_t, H_t, B_t)
- q: query or current subgoal
- D_t: discovery state (what the agent knows exists and where)
- K_t: knowledge state (grounded, verified claims)
- W_t: active workspace (context window contents)
- H_t: interaction history
- B_t: remaining budget

**Key modeling choice — approximate full observability:** The underlying information environment is partially observable (the agent doesn't know what's in files it hasn't opened). However, the tuple (D_t, K_t, W_t, H_t) acts as a **sufficient statistic for the belief state**: D_t tracks what has been discovered, K_t tracks what has been verified, and H_t records the full interaction trace. Given these, the agent's belief about the environment is deterministic — it knows exactly what it has and hasn't explored. This justifies using MDP-style value functions rather than POMDP belief-space optimization, following the standard approach in information-gathering problems where the agent's observation history fully determines its epistemic state (see Araya-López et al., 2010, "A POMDP Extension with Belief-Dependent Rewards").

**Action space as options:** A = ⋃_{k=1}^{K} A_k where A_k is the set of operations available in address space k. Following the options framework (Sutton, Precup, Singh, 1999), each address space k defines an **option** ω_k = (I_k, π_k, β_k) — an initiation set, internal policy, and termination condition. The router is a **policy over options**, selecting which address space to engage. This factorization is critical: it reduces the flat action space (11 operations × many addresses) to a two-level hierarchy (select option, then follow internal policy), which has well-studied convergence guarantees for hierarchical RL.

Specifically: A = {semantic_search, lexical_search, descend, preview, open, expand, compress, evict, graph_hop, tool_call, stop}

**Transition:** s_{t+1} ~ P(·|s_t, a_t) — deterministic workspace updates, stochastic information returns.

**Reward:** r_t = U(evidence_t) - λ₁·tokens_t - λ₂·latency_t - λ₃·tool_cost_t - λ₄·context_noise_t

**Constrained objective (CMDP):** The budget constraint creates a constrained MDP (Altman, 1999). We use the Lagrangian relaxation:

max_θ min_{ν≥0} E_π[Σ_t γ^t r_t] + ν·(B_0 - Σ_t cost_t)

where ν is a learned dual variable that adaptively controls the cost-quality tradeoff. In practice, ν is tuned on validation data before experiments begin (not optimized per-task), making the final objective:

max_θ E_π[Σ_t γ^t (r_t - ν·cost_t)]

This reduces to the unconstrained case with a specific cost penalty, but the CMDP framing clarifies that ν should be set to satisfy the budget constraint in expectation, not arbitrarily.

### 2.2 Superset Dominance Theorem

**Theorem 1 (Weak Dominance).** Let π*_k be the optimal policy restricted to address space A_k, and π* be the optimal policy over A = ⋃_k A_k. Then for all states s:

V^{π*}(s) ≥ max_k V^{π*_k}(s)

**Proof.** For any state s_t:

V^{π*}(s_t) = max_{a ∈ A} [r(s_t, a) + γ E[V^{π*}(s_{t+1})]]
             ≥ max_{a ∈ A_k} [r(s_t, a) + γ E[V^{π*}(s_{t+1})]]    (since A_k ⊂ A)
             ≥ max_{a ∈ A_k} [r(s_t, a) + γ E[V^{π*_k}(s_{t+1})]]   (since V^{π*} ≥ V^{π*_k} inductively)
             = V^{π*_k}(s_t)

Taking the max over k: V^{��*}(s) ≥ max_k V^{π*_k}(s). □

**Note:** This is trivially true. The interesting question is strict inequality and whether the gap is achievable by a learned policy.

**Bridging π* and π_θ:** Theorem 1 concerns the optimal policy π*, but our system uses a learned approximation π_θ. The options framework provides the bridge. Since each address space k is an option with its own internal policy π_k, the router only needs to learn a policy over K options — a much smaller decision space than the flat action space. The SMDP (semi-Markov decision process) convergence results for policy-over-options (Sutton, Precup, Singh, 1999) guarantee that the hierarchical policy converges to the optimal option-selection policy given sufficient exploration, provided the option set is fixed. Our heuristic → learned → RL training progression ensures stable option definitions before optimizing the router.

### 2.3 Strict Dominance Condition

**Proposition 1 (Quantitative Gain from Heterogeneity).** Define φ_k(s) = 1[argmax_{a∈A} Q(s,a) ∈ A_k] as the indicator that the optimal action at state s belongs to address space k. Let ρ be the state distribution induced by π*. Define the **heterogeneity gap**:

δ = min_k (1 - E_{s~ρ}[φ_k(s)])

This measures how often the *best* single address space is suboptimal. If δ > 0 (no single space is always optimal), then for all k:

E_{s~ρ}[V^{π*}(s) - V^{π*_k}(s)] ≥ δ · Δ_min

where Δ_min = min_{s ∈ S_k} (Q^*(s, a^*) - max_{a∈A_k} Q^*(s, a)) is the minimum Q-value gap on states where address space k is suboptimal.

**Proof sketch.** For each k, the set S_k = {s : argmax_a Q(s,a) ∉ A_k} has ρ(S_k) ≥ δ by definition of the heterogeneity gap. On S_k, the unrestricted policy gains at least Δ_min per state. Therefore E[V^{π*} - V^{π*_k}] ≥ δ · Δ_min. □

**Interpretation:** The gain from heterogeneous routing is proportional to (a) how often the best single space is wrong (δ), and (b) how much worse the second-best action is (Δ_min). Both are empirically measurable. If δ is small or Δ_min is small, routing provides marginal benefit — and we should report that honestly.

### 2.4 Empirical Evidence That the Strict Condition Holds

The strict condition requires that the optimal address space varies across states. Five independent lines of evidence support this:

**(E1) HELMET [arXiv:2410.02694]:** Seven task categories show low cross-category correlation. Systems strong at one category are not reliably strong at others. This implies different tasks favor different retrieval strategies, i.e., φ_k varies across task distributions.

**(E2) LongMemEval [arXiv:2410.10813]:** Five distinct memory abilities (information extraction, multi-session reasoning, temporal reasoning, knowledge updates, abstention) each require different retrieval strategies. Optimizations effective for one ability (e.g., time-aware expansion for temporal reasoning) don't help others.

**(E3) BRIGHT [arXiv:2407.12883] + NoLiMa [arXiv:2502.05167]:** Reasoning-intensive retrieval tasks where lexical overlap is absent defeat standard dense and sparse retrievers. Combined with the fact that these same retrievers excel on paraphrase-heavy tasks, this shows the optimal retrieval primitive is task-dependent.

**(E4) SWE-agent [arXiv:2210.03629 context]:** Agent-computer interface design (action space, observation format) materially changes task success rate. This shows the action space itself — not just the policy within it — is a critical design variable.

**(E5) SmartRAG [ICLR 2025, arXiv:2410.18141]:** Joint RL optimization of three operations (retrieve-decision, query-rewrite, answer-generation) outperforms separately optimized modules. This demonstrates that coordination across operations within a single task yields measurable gains.

**Critical gap in evidence:** E1-E3 show *between-task* variation in optimal strategy. Our hypothesis also claims *within-task* variation (different steps of the same task require different address spaces). This is not yet empirically established by the literature and must be validated in the PoC (Phase 3).

### 2.5 Toy Example: Within-Task Substrate Switching

To make the strict gain concrete and address the anti-stacking concern, consider a minimal analytical example.

**Setup:** Two address spaces: A₁ (text search) and A₂ (database query). Three-step task.

**Task:** "What percentage of employees in Department X have salary above the company median?"

**Step 1 (s₀):** D₀=∅, K₀=∅. Agent doesn't know where Department X is defined.
- Optimal: text_search("Department X") → discovers dept_id, schema location
- Alternative: sql_query without schema → fails or returns wrong table
- A₁ is optimal. Discovery state updates: D₁ = {dept_id=X, salary_table=employees}

**Step 2 (s₁):** D₁={dept_id, table}, K₁=∅. Agent knows where data lives but hasn't extracted it.
- Optimal: sql_query("SELECT salary FROM employees WHERE dept=X") → returns salary list
- Alternative: text_search for salary data → gets unstructured prose, unreliable for computation
- A₂ is optimal. Knowledge state updates: K₂ = {salaries=[...]}

**Step 3 (s₂):** D₂={dept_id, table}, K₂={salaries}, W={salary data}
- Optimal: sql_query("SELECT COUNT(CASE WHEN salary > (SELECT AVG...) ...")
- Alternative: compute in LLM reasoning → error-prone on large lists
- A₂ is optimal. K₃ = {answer=47%}

**Value computation (simplified, undiscounted, 3 steps):**

Let reward per correct step = +1, cost per action = c, wrong-substrate penalty = -p.

| Policy | Step 1 | Step 2 | Step 3 | Total Value |
|--------|--------|--------|--------|-------------|
| π* (adaptive) | A₁ (+1-c) | A₂ (+1-c) | A₂ (+1-c) | 3 - 3c |
| π*₁ (text only) | A₁ (+1-c) | A₁ (+1-c-p) | A₁ (+1-c-p) | 3 - 3c - 2p |
| π*₂ (DB only) | A₂ (+1-c-p) | A₂ (+1-c) | A₂ (+1-c) | 3 - 3c - p |
| π_ensemble (both) | A₁∧A₂ (+1-2c) | A₁∧A₂ (+1-2c) | A₁∧A₂ (+1-2c) | 3 - 6c |

**Strict gains:**
- V(π*) - V(π*₁) = 2p > 0 (text-only pays penalty on steps 2-3)
- V(π*) - V(π*₂) = p > 0 (DB-only pays penalty on step 1)
- V(π*) - V(π_ensemble) = 3c > 0 (ensemble pays double cost every step)

**Key observations:**
1. Within-task substrate switching is necessary: step 1 needs A₁, steps 2-3 need A₂.
2. The adaptive policy strictly dominates all alternatives: single-substrate (by p) and ensemble (by 3c).
3. The discovery state D₁ (knowing the schema exists) enables the optimal action at step 2. Without D/K tracking, the agent cannot distinguish "I know where the data is" from "I haven't looked yet."
4. The ensemble baseline is dominated because it wastes budget querying unnecessary substrates.

This example is constructive: it proves within-task variation exists for tasks combining discovery (text) and computation (structured data). The PoC must verify this pattern holds on real tasks at scale.

### 2.6 Why the Discovery/Knowledge Split Is Necessary

Consider two states with identical knowledge but different discovery:

- s₁ = (q, D={file_x_exists, section_y_relevant}, K=∅, W=∅, B)
- s₂ = (q, D=∅, K=∅, W=∅, B)

Optimal actions differ:
- At s₁: open(file_x, section_y) — exploit known location, cost: ~1 operation
- At s₂: semantic_search(q) — explore to discover, cost: ~N operations

Without the D/K split, s₁ and s₂ are indistinguishable (both have K=∅). The policy loses the information value of prior exploration and must re-discover what it already found. This is a case of **state aliasing** — when the agent's state representation conflates states that require different optimal actions, the resulting policy is provably suboptimal (Singh, Jaakkola, and Jordan, 1994, "Reinforcement Learning with Soft State Aggregation").

**Proposition 2 (Information Value of Discovery State).** Let π_DK be the optimal policy with (D_t, K_t) in the state, and π_K be the optimal policy with only K_t. The D/K split resolves state aliasing: states with identical K but different D require different optimal actions (exploit known location vs. explore unknown territory — see toy example in Section 2.5, steps 1 vs 2). By the state aliasing result, if the fraction of aliased states under π_K is α > 0, then:

E[V^{π_DK}] - E[V^{π_K}] ≥ α · Δ_alias

where Δ_alias is the average Q-value gap on aliased states.

**Connection to information foraging theory:** Pirolli & Card (1999) distinguish "information scent" (signals about where valuable information might be) from "information diet" (what has been consumed). Our D/K split formalizes this: D_t is the scent (discovered locations), K_t is the diet (grounded knowledge). Fu & Pirolli (2007) operationalize this in the SNIF-ACT computational model, which is the closest prior computational formalism to our state representation.

**Connection to POMDP literature:** The Proactive Info Gathering paper [arXiv:2507.21389] formalizes information gathering as a POMDP and shows RL-trained policies for uncertainty reduction outperform o3-mini by 18%. Our formulation extends this from clarification questions to multi-substrate exploration.

### 2.6 Budget-Aware Optimality

Under budget constraint B_t, the agent faces an exploration-exploitation tradeoff at each step:

**Explore:** Query a new address space (high information gain, uncertain return, budget consumed)
**Exploit:** Use known relevant source (lower risk, targeted extraction, budget consumed)
**Stop:** Return answer with current evidence (no additional cost)

The optimal stopping time τ* satisfies:

E[U(answer_τ*, evidence_τ*)] - λ·cost_τ* ≥ E[U(answer_{τ*+1}, evidence_{τ*+1})] - λ·cost_{τ*+1}

i.e., the marginal utility of one more step is non-positive. This is a standard optimal stopping result.

**Key insight:** The stopping rule depends on the *bundle* of evidence, not individual pieces. This requires a critical assumption: **the utility function U is non-separable across evidence pieces.** Concretely, U(e₁, e₂, ..., eₙ) ≠ Σᵢ u(eᵢ) — the value of a definition + a number + an exception together exceeds the sum of their individual values when the question requires all three. This non-separability is what makes bundle scoring necessary. If U were additive, individual chunk scoring would suffice.

We define **bundle coverage** as: given a question q that decomposes into requirements R = {r₁, ..., rₘ} (e.g., definition, number, exception, citation), bundle_coverage = |{rᵢ : ∃eⱼ ∈ bundle satisfying rᵢ}| / m. The stopping condition becomes: stop when bundle_coverage ≥ τ AND confidence ≥ σ, where τ and σ are pre-set thresholds.

---

## 3. Failure Modes

### FM1: Policy learning too hard
**Risk:** Action space (11 operations × many addresses) may be too large for effective learning.
**Condition for trigger:** Learned policy doesn't outperform heuristic policy after training.
**Mitigation:** Start with heuristic, use curriculum learning, restrict action space initially.
**Severity:** Medium — heuristic policy may still validate the thesis even if learned policy is marginal.

### FM2: Routing overhead dominates
**Risk:** Computing which address space to consult adds latency/tokens that exceed the routing benefit.
**Condition for trigger:** AEA with full routing loses to simple baselines on Utility@Budget despite winning on raw accuracy.
**Mitigation:** Cost-aware rewards penalize unnecessary routing. Lightweight router (small classifier, not full LLM call).
**Severity:** High — if routing is too expensive, the method is impractical.

### FM3: Single substrate dominates
**Risk:** One address space (e.g., dense retrieval) handles 90%+ of states optimally, making routing marginal.
**Condition for trigger:** Ablation of router shows <5% degradation; one substrate ablation shows >80% of total degradation.
**Mitigation:** Custom heterogeneous benchmark designed to require cross-substrate routing. If this fails on standard benchmarks, the contribution is the benchmark + the negative finding.
**Severity:** High for the method claim, but an honest negative result is still publishable if the formalization and benchmark are strong.

### FM4: Discovery state hard to maintain
**Risk:** Agent can't reliably track what it has discovered vs. what it knows, making D/K split noisy.
**Condition for trigger:** With-D/K and without-D/K ablation show no significant difference.
**Mitigation:** Explicit structured state with verified entries (not implicit). Treat D_t as a typed log.
**Severity:** Medium — the method can still work without D/K if routing alone is sufficient.

### FM5: Ensemble effect explains gains (not routing)
**Risk:** Simply querying ALL substrates every step (trivial ensemble) and merging results captures most of the benefit without intelligent routing.
**Condition for trigger:** π_ensemble (query all, merge) matches π_θ on answer quality while paying only modestly more cost.
**Mitigation:** π_ensemble must be an explicit baseline. The toy example (Section 2.5) shows the ensemble is dominated by 3c (triple cost). If the cost penalty is small relative to the quality gain, the ensemble becomes competitive. The Utility@Budget metric explicitly penalizes cost, making the comparison fair.
**Severity:** High — if the ensemble wins, the routing contribution is the wrong framing; the real contribution would be "multi-substrate access" not "intelligent routing."

### FM6: Within-task strategy variation doesn't exist
**Risk:** For any given task, the optimal address space is constant across all steps (contradicting the need for adaptive routing within a task).
**Condition for trigger:** Oracle analysis of training traces shows >90% of steps use the same address space.
**Mitigation:** If between-task variation is sufficient, the method reduces to a task-level classifier (simpler but still useful). Custom benchmark should include tasks that require within-task variation.
**Severity:** Medium — reduces the contribution but doesn't invalidate it entirely.

### What would disprove H4?
If a single-substrate baseline (e.g., MAGMA-only or dense-retrieval-only) achieves equal or higher Utility@Budget than the full heterogeneous policy across ALL three evaluation regimes, then H4 is disproved.

### What would be inconclusive?
If the full system wins but ablations show >80% of the gain comes from one substrate, then H4 ("routing matters most") is not supported, even if the system works.

---

## 4. Metrics and Thresholds

### Primary Metric
**Utility@Budget** = AnswerScore × η·EvidenceScore − μ·Cost

Where:
- AnswerScore ∈ [0,1]: EM or F1 or judge score (benchmark-dependent)
- EvidenceScore ∈ [0,1]: support_precision × support_recall × bundle_coverage
- Cost �� [0,1]: normalized(tokens + latency + tool_calls)
- η = 0.5 (evidence weight — set before experiments, not tuned)
- μ = 0.3 (cost penalty — set before experiments, not tuned)

### Success Thresholds

| Hypothesis | Success | Failure | Metric |
|------------|---------|---------|--------|
| H1 | p < 0.05, >5% improvement | <2% improvement | Utility@Budget vs best fixed baseline |
| H2 | Gain on BRIGHT/NoLiMa > 2× gain on paraphrase tasks | Equal gains across task types | Relative improvement ratio |
| H3 | >70% executable actions on structured tasks | <30% executable actions | Action distribution |
| H4 | Router ablation causes >50% degradation | Router ablation causes <20% degradation | Ablation impact ratio |

### Statistical Requirements
- 5 random seeds per experiment
- Paired t-test or Wilcoxon signed-rank for significance
- 95% confidence intervals on all reported metrics
- Cohen's d for effect size

---

## 5. Anti-Stacking Check

**Is this a genuine conceptual reframing?**

| Check | Assessment |
|-------|-----------|
| Can be explained without "combine" or "integrate"? | YES: "The system learns to direct attention to the most informative region of the information environment, choosing which address space to consult and what operation to perform." |
| Proposes a new way of thinking? | YES: Retrieval as external attention allocation, not relevance scoring. The address spaces are interchangeable substrates; the contribution is the allocation policy. |
| New conceptual primitive? | YES: Discovery/knowledge state split formalizes "scent vs diet" from information foraging theory. |
| Would removing any component produce "just stacking"? | If you remove the policy and just expose all substrates: that's just a tool collection (stacking). The policy IS the contribution. |

**Verdict: PASSES anti-stacking check.**

---

## 6. Self-Critique

### What I'm confident about
- The superset dominance argument is mathematically sound (trivially true).
- The strict condition is supported by 5 independent evidence lines.
- The formalization as a budgeted POMDP is well-grounded in prior work.
- The anti-stacking check passes: this is genuinely about the policy, not the substrates.

### What I'm less confident about
- **Within-task variation:** The evidence for *between-task* substrate variation is strong (HELMET, LongMemEval). But the evidence for *within-task* variation (different steps of the same task needing different substrates) is thin. This MUST be validated in the PoC.
- **Bundle scoring formalization:** The stopping rule requires evaluating evidence bundles, but I haven't provided a concrete mathematical definition of "bundle coverage." This needs work.
- **Learned policy convergence:** The action space is large. Whether RL can learn an effective policy here is an open question. The heuristic-first strategy mitigates this but doesn't resolve it.

### What a skeptical reviewer would say
1. "The superset argument is obvious. Show me the empirical gap is meaningful, not just theoretically possible."
   → Fair. This is what Phase 4 experiments must demonstrate.

2. "The D/K split adds complexity. Is it really necessary, or is interaction history H_t sufficient?"
   → Valid concern. The ablation plan includes D/K removal. If H_t suffices, the contribution is simpler but the system still works.

3. "You're comparing against baselines that weren't designed for heterogeneous environments. Of course a system designed for mixed tasks wins on a mixed-task benchmark."
   → This is the strongest objection. Mitigation: include standard single-substrate benchmarks (Regimes A and B) where baselines are on home turf. The method must not lose substantially on single-substrate tasks.

---

## 7. Theory Review (Iteration 0)

**Assessment:** NEEDS_REVISION (DONE_WITH_CONCERNS)

### Issues Identified by Reviewer

1. **π* to π_θ gap:** Theorem proves nothing about the learned policy. Approximation error could swallow the theoretical advantage.
   → **Fixed:** Added options framework connection (Section 2.1). Address spaces as options; router as policy-over-options. Hierarchical RL convergence guarantees apply. Added bridging argument (Section 2.2).

2. **MDP/POMDP inconsistency:** Bellman equation was for MDP but problem is POMDP.
   → **Fixed:** Argued (D_t, K_t, W_t, H_t) is a sufficient statistic for belief state (Section 2.1). Justified MDP-style value functions.

3. **No concrete toy example:** Within-task variation was claimed but not demonstrated.
   → **Fixed:** Added 2-substrate, 3-step analytical example with closed-form values (Section 2.5). Shows adaptive policy strictly dominates single-substrate and ensemble.

4. **Binary heterogeneity condition:** Needed quantitative bound.
   → **Fixed:** Proposition 1 now provides bound: gain ≥ δ · Δ_min where δ = heterogeneity gap (Section 2.3).

5. **CMDP structure:** Budget constraint not properly handled.
   → **Fixed:** Added Lagrangian relaxation formulation with learned dual variable ν (Section 2.1). Cited Altman (1999).

6. **Ensemble as alternative explanation:** Not addressed.
   → **Fixed:** Added FM5 (ensemble baseline). Toy example shows ensemble dominated by 3c cost.

7. **Proposition 2 ("provably") without proof:**
   → **Fixed:** Cited state aliasing result (Singh, Jaakkola, Jordan 1994) and added quantitative bound.

8. **Bundle coverage undefined:**
   → **Fixed:** Added explicit definition: coverage = fraction of question requirements satisfied by evidence bundle.

### Additional References Added
- Sutton, Precup, Singh (1999) — Options framework for hierarchical RL
- Altman (1999) — Constrained MDPs
- Singh, Jaakkola, Jordan (1994) — State aliasing
- Fu & Pirolli (2007) — SNIF-ACT computational model
- Araya-López et al. (2010) — POMDP with belief-dependent rewards

### Remaining Items for Phase 3 (PoC)
- **Within-task variation must be empirically validated.** The toy example is constructive but synthetic. The PoC must show real tasks exhibit substrate switching.
- **δ and Δ_min must be measured.** The quantitative bound is only useful if we can estimate these empirically.
- **Ensemble baseline must be implemented and compared.** This is the strongest alternative explanation.

### Reviewer's Alternative Suggestion (noted, not adopted)
The reviewer suggested framing as **online portfolio selection** (Cover, 1991) with Exp3 regret bounds. This gives a cleaner theoretical result (sublinear regret relative to best fixed mixture). We note this as a potential alternative formulation for a revised version but proceed with the CMDP/options framework for now, as it more directly maps to the implementation (heuristic → learned → RL progression).

## Next Steps
- ~~Re-dispatch theory reviewer on revised hypothesis~~
- **Re-review result: RIGOROUS** (7/8 resolved, 1 partially resolved non-blocking)
- **Proceed to Phase 3 (PoC validation)**
- Phase 3 must specifically validate:
  1. Within-task substrate variation on real tasks
  2. Sufficient exploration for SMDP convergence
  3. Whether δ varies by state region
