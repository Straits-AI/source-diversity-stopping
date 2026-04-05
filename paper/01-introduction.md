# 1 Introduction

Retrieval-augmented generation (RAG) has become the standard approach for grounding language model outputs in external knowledge. Yet the dominant paradigm commits to a single retrieval primitive — typically dense passage retrieval or BM25 — regardless of the query's characteristics, the corpus structure, or the cost of retrieval operations. This rigidity creates a fundamental mismatch: some queries are best served by semantic similarity search, others by exact keyword matching, and still others by multi-hop entity traversal, but the system has no mechanism for choosing among them.

Recent work has begun to address this limitation. Self-RAG [Asai et al., 2024] learns when to skip retrieval entirely. Adaptive-RAG [Sequeira et al., 2025] routes queries by complexity. SmartRAG [Gao et al., 2025] jointly optimizes retrieval and generation under cost constraints. Yet each of these systems operates within a single retrieval modality or over a homogeneous pipeline ensemble. No existing system provides a unified policy that routes across qualitatively heterogeneous address spaces — semantic indexes, lexical indexes, entity graphs, and the null action — under an explicit budget constraint with structured state tracking.

In this paper, we formalize retrieval as **adaptive external attention allocation** over heterogeneous address spaces. We model the retrieval problem as a constrained Markov decision process (CMDP) in which the agent maintains a structured state comprising discovery knowledge (what information sources have been located), verified knowledge (what claims have been grounded), and a bounded workspace. At each step, the agent selects an address-space-operation pair or stops, with the objective of maximizing answer-supporting evidence quality minus retrieval cost.

We introduce **Adaptive External Attention (AEA)**, a harness-level routing policy that implements this formalization. The AEA heuristic policy uses a coverage-driven decision procedure: it initiates with semantic search to establish anchor documents, then evaluates whether the current workspace provides sufficient multi-source evidence to stop. If coverage is insufficient and the query structure suggests a relational chain, the policy escalates to entity-graph traversal; otherwise, it falls back to lexical retrieval.

Our central empirical finding challenges the intuitive framing of adaptive retrieval as "choosing the right tool." Across two benchmarks — HotpotQA Bridge (100 multi-hop questions) and a controlled Heterogeneous benchmark (100 synthetic questions across 6 task types) — the dominant mechanism driving AEA's advantage is **routing avoidance**: knowing when to stop and what not to do, rather than consistently identifying the optimal next substrate. On HotpotQA, AEA achieves a 67% improvement in Utility@Budget over the strongest fixed baseline by completing questions in an average of 1.21 operations versus 2.00. Ablation analysis confirms that forcing the policy to always escalate (abl_always_hop) is catastrophic (−0.1146 U@B), while removing entity hops entirely slightly improves performance (+0.0039), establishing that selective avoidance — not positive substrate selection — is the primary value driver.

Our contributions are:

1. **Formalization.** We formalize retrieval as harness-level external attention allocation over heterogeneous address spaces, with explicit discovery/knowledge state tracking and a CMDP budget constraint (Section 3).

2. **Method.** We introduce AEA, a coverage-driven routing policy that selects among semantic, lexical, and entity-graph substrates based on workspace state rather than query surface patterns (Section 3.3).

3. **Benchmark.** We construct a heterogeneous evaluation benchmark with controlled entity isolation and lexical overlap, designed to require cross-substrate navigation (Section 4.1).

4. **Finding.** We show that adaptive retrieval's primary value under budget constraints is routing avoidance — selective operation omission — rather than positive substrate selection. This reframes the design problem from "choose the right tool" to "know when to stop" (Section 5.3).

The remainder of the paper is organized as follows. Section 2 surveys related work. Section 3 presents the formal framework and AEA policy. Section 4 describes the experimental setup. Section 5 reports results and ablation analysis. Section 6 discusses implications, limitations, and future directions. Section 7 concludes.
