# 1 Introduction

Information-seeking agents face a fundamental decision at every step: explore further or act on what is known. A coding agent greps for a symbol, reads a file, follows an import — and must decide whether it has enough context. A research assistant searches a database, reads abstracts, follows citations — and must decide when to stop. A RAG system queries an index — and returns with whatever the first retrieval produces, regardless of whether it suffices.

Current systems handle this poorly. RAG pipelines commit to a fixed retrieval budget — one query, top-k results, done. Adaptive retrieval systems (Self-RAG [Asai et al., 2024], FLARE [Jiang et al., 2023]) learn when to skip retrieval within a single modality but do not navigate across qualitatively different information substrates. Coding agents (Cursor, aider, SWE-agent) rely on LLM self-judgment or hard token budgets to decide when to stop gathering context — with no principled stopping criterion.

We propose **convergence-based navigation**: a framework where agents explore heterogeneous information environments through multiple action types (search, open, read sections, follow cross-references) while maintaining explicit state tracking, and stop when **independent navigation pathways converge** on the same evidence. The convergence principle is grounded in a simple observation: when two independent methods of finding information — with different failure modes — both produce evidence, that evidence is likely relevant, and further exploration has diminishing returns.

We provide two concrete implementations:

**ConvergenceRetriever** — a drop-in multi-substrate retrieval component. It wraps BM25, dense embeddings, and structural search with automatic convergence stopping, reducing retrieval operations by 33-50% at equal result quality. This is the practical tool for practitioners who want to improve their existing RAG pipelines.

**NavigationAgent** — a step-by-step exploration agent that goes beyond retrieval. It maintains **discovery state** (what the agent knows exists but hasn't read) and **knowledge state** (what it has actually read and extracted), chooses among actions (search, open, read section, follow link, stop), and uses convergence across action types — not just retrieval substrates — as one stopping signal among several. This is the research contribution: a formal framework for principled navigation with stopping.

We validate convergence stopping through seven evaluation settings spanning five task families (multi-hop QA, reasoning-intensive retrieval, fact verification, structural navigation, and codebase search) and ten alternative stopping mechanisms across seven design categories. All ten alternatives fail. Root cause analysis reveals two ceilings:

1. **A content-aware ceiling**: seven content-based stopping mechanisms (cross-encoder, NLI, learned classifier, LLM decomposition, answer stability, confidence-gated, embedding router) all fail because assessing evidence quality requires evaluating a set function over document bundles — a problem current models cannot solve reliably.

2. **A structural ceiling**: three structural enrichments (threshold optimization, novelty detection, dual signals) converge to identical behavior because source diversity is the maximally informative zero-cost signal.

A **boundary condition** applies: for computation tasks, tool-execution completion replaces convergence as the optimal signal, cleanly separating the navigation regime from the computation regime.

Our contributions:

1. **A navigation framework** with explicit discovery/knowledge state, heterogeneous actions, and convergence-based stopping — going beyond retrieve-and-generate to explore-decide-and-stop (`convergence_retrieval` library, open-sourced).

2. **Empirical validation** across seven settings, five task families, and four substrate types, showing convergence stopping is Pareto-optimal within ten tested alternatives (Section 5).

3. **The two-ceiling framework** explaining why convergence stopping resists improvement: content signals add noise (set function problem) and structural signals are saturated (source diversity is maximal) — grounded in optimal stopping theory (Section 6).

4. **A deployable tool** — `ConvergenceRetriever` for drop-in RAG improvement and `NavigationAgent` for agent-based exploration — with benchmarking utilities and extensible substrate/environment interfaces.
