# Research Setup

**Date:** 2026-04-04
**Phase:** 0
**Iteration:** 0
**Status:** completed

## Idea DNA

### Problem
Fixed retrieval pipelines (including structured memory systems like MAGMA) optimize within a single information substrate but do not adaptively route across heterogeneous sources under budget constraints. Current RAG answers "which chunks should I fetch?" when the real problem is choosing the next best address-operation pair across memory, documents, tools, and search.

### Assumptions

**Explicit (from researcher):**
- A harness-level policy choosing address-operation pairs across heterogeneous address spaces will outperform fixed retrieval pipelines under equal cost budgets.
- MAGMA-like structured memory is a strong substrate but not the whole solution — the control layer above it matters more.
- The biggest gains will come on reasoning-intensive and long-context tasks with low lexical overlap, multi-hop dependencies, and strong document structure.

**Inferred:**
- The largest performance gains come from routing decisions (which substrate to use next) rather than from improvements within any single substrate.
- Executable addressing will dominate text retrieval on exact-computation and structured-data tasks.
- A discovery/knowledge state split is essential: knowing a file exists is different from knowing its contents.

### Novelty Claim
Formalize retrieval as harness-level external attention allocation over heterogeneous address spaces. The policy chooses among memory traversal, document navigation, tool execution, search, and workspace operations — not chunks. Show that workspace control dominates single-substrate optimization on mixed workloads.

### Domain
ML / NLP / Information Retrieval / AI Agent Systems

### Success Criteria
1. AEA outperforms fixed retrieval pipelines on utility-under-budget across all three evaluation regimes.
2. AEA outperforms memory-only policies (MAGMA) when tasks require cross-substrate routing.
3. Ablations demonstrate that routing decisions contribute more than any single substrate improvement.
4. Results hold across memory-centric (LoCoMo, LongMemEval), retrieval-centric (BRIGHT, NoLiMa, RULER, LongBench Pro), and heterogeneous benchmarks.

### Scope Constraints
- Research intensity: Deep (30-50 papers, comprehensive ablations, custom benchmark)
- Budget: Flexible
- Compute: API-based LLM backbone + local retrieval infrastructure
- Timeline: ~4 months (Months 1-4 as outlined in plan)

## Literature Sources

Available search tools for this session:
- Web search (via WebSearch/WebFetch)
- Scholar Gateway MCP (if authenticated)
- Context7 documentation search

Research intensity: **Deep** — targeting 30-50 papers total.

## Compute Environment

**LLM backbone:** API-based (Anthropic Claude or OpenAI GPT-4 class)
**Local infrastructure:** Retrieval indices (FAISS, BM25), graph DB, workspace manager
**Router training:** Consumer GPU sufficient (RTX 3090/4090 or equivalent)
**Estimated API cost:** $1,000-3,000 for full study

*Note: User to confirm exact local machine specs.*

## Key Design Decisions

1. **MAGMA as baseline, not target.** MAGMA strengthens the memory substrate; our contribution is the harness-level policy above it.
2. **Phased training.** Heuristic policy → learned scoring → budget-aware RL. No premature end-to-end RL.
3. **Three evaluation regimes.** Memory-centric, retrieval-centric, heterogeneous — each tests different aspects.
4. **Hold memory substrate fixed in first paper.** Vary only the attention policy to isolate the contribution.
5. **Custom benchmark.** Build a discovery-vs-knowledge benchmark with mixed corpora types.

## Next Steps
- Phase 1: Deep literature review (30-50 papers)
- Set up evaluation contract
- Initialize results tracking
