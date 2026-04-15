# 1 Introduction

Information-seeking agents — whether navigating codebases, browsing enterprise documents, or answering multi-hop questions — face a common problem: they must search across multiple heterogeneous substrates (semantic search, keyword matching, structural navigation, graph traversal) and decide **when to stop**. Navigation and retrieval are two faces of this problem: navigation determines *where to look next* in the information space, retrieval determines *what to extract*, and stopping determines *when evidence is sufficient*. The interplay among these decisions — across qualitatively different information substrates — is the core challenge of heterogeneous document navigation.

This stopping decision is ubiquitous yet understudied. In codebase navigation, a coding agent greps for symbols, embeds code chunks, traverses import graphs, and checks LSP diagnostics — but no existing tool (Cursor, aider, Copilot, SWE-agent) has a principled criterion for when to stop gathering context. In retrieval-augmented generation, adaptive systems like Self-RAG [Asai et al., 2024] and FLARE [Jiang et al., 2023] learn when to retrieve but operate within a single modality. The question of when to stop searching across *multiple qualitatively different substrates* remains open.

We study this problem across six evaluation settings spanning four substrate types (semantic, lexical, structural, executable) and three task families (QA, fact verification, computation). We arrive at a strong empirical result: **a one-line structural heuristic — stop when evidence has arrived from two or more independent retrieval pathways — is Pareto-optimal** within ten tested alternatives. No alternative improves quality without increasing cost, and no alternative reduces cost without reducing quality.

We establish this through three lines of evidence:

**Line 1: The heuristic dominates comprehensive retrieval.** Across five benchmarks — multi-hop QA (HotpotQA, p<0.0001, N=1000), reasoning-intensive retrieval (BRIGHT, p=0.003, N=200), fact verification (FEVER-style, p≈0, N=200), structural navigation (p<0.001, N=100), and diluted retrieval (p<0.0001, N=200, 5x candidate expansion) — the heuristic significantly outperforms retrieving from all substrates. The advantage is robust across question types and grows in harder settings (Cohen's d from 0.22 to 0.49).

**Line 2: Seven content-aware stopping mechanisms fail.** A cross-encoder, NLI bundle checker, learned classifier, LLM decomposition, answer-stability tracker, confidence-gated self-assessment, and embedding router all fail to improve on the heuristic. Root cause analysis reveals a common bottleneck: assessing evidence quality requires evaluating a **set function** over document bundles — a problem that current models cannot solve reliably. Each method introduces more noise (from approximation errors, distribution shift, parsing failures, or phrasing instability) than information.

**Line 3: Three structural improvements converge.** Threshold optimization, novelty-based stopping, and dual-signal stopping all produce identical behavior, confirming source diversity is the **maximally informative zero-cost signal** — the structural ceiling.

A **boundary condition** applies: for computation tasks, tool-execution completion replaces source diversity as the optimal stopping signal. This cleanly separates the retrieval regime (where navigation and evidence gathering dominate) from the computation regime (where tool execution dominates).

Our contributions:

1. **The heterogeneous document navigation stopping problem** — a general formulation encompassing codebase search, enterprise retrieval, and QA as instances of the same stopping decision over multiple substrates (Section 3).

2. **Source-diversity stopping as a Pareto-optimal structural signal** — validated across five benchmarks, four substrate types, and three task families, with ten alternatives tested (Section 5).

3. **The two-ceiling framework** — content-aware ceiling (noise > information) and structural ceiling (source diversity is maximal) — explaining why the heuristic resists improvement, grounded in optimal stopping theory (Section 6.4).

4. **The retrieval-computation boundary** — source diversity is optimal for retrieval/navigation; tool-execution completion is optimal for computation (Section 5).

5. **Mapping to codebase navigation** — where the same substrates (grep, embeddings, AST, import graphs) and stopping problem arise but no existing tool uses diversity-based stopping (Section 6).
