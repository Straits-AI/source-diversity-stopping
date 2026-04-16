# 3 Method

## 3.1 The Navigation Framework

Unlike RAG systems that retrieve-then-generate, convergence-based navigation operates as an **explore-decide-stop** loop. The agent interacts with an information environment through a sequence of actions, maintaining state throughout.

**State.** At each step t, the agent maintains:

- **Discovery state** D_t: what the agent knows *exists* — document titles, paths, mentions found during search. This is "information scent" (Pirolli and Card, 1999).
- **Knowledge state** K_t: what the agent has *read* — full content extracted from documents it has actually opened. This is "information diet."
- **History** H_t: the sequence of actions taken and their outcomes.
- **Budget** B_t: remaining operations before forced stop.

The discovery/knowledge split is the framework's core design choice. An agent that discovers `auth.py` exists (via search) is in a fundamentally different state from one that has read `auth.py` and knows it validates JWT tokens. Without this distinction, the agent cannot reason about what to explore next.

**Actions.** The agent chooses from:

| Action | What It Does | Cost |
|--------|-------------|------|
| SEARCH(query, substrate) | Query a retrieval substrate; returns document snippets | 1 op |
| OPEN(doc_id) | Read a discovered document fully; returns content + links | 1 op |
| READ_SECTION(doc_id, section) | Read a specific section within a document | 0.5 op |
| FOLLOW_LINK(link) | Follow a cross-reference to another document | 1 op |
| STOP | Conclude navigation; act on current knowledge | 0 op |

This action space is richer than RAG (which has only SEARCH) and more structured than unconstrained LLM tool use (which has no formal cost model).

**Environment.** The information environment wraps multiple retrieval substrates (BM25, dense embeddings, structural/path matching) and exposes them through the action interface. The environment also maintains cross-reference links between documents, enabling FOLLOW_LINK actions. Different environments can be swapped in: document collections, codebases, wiki graphs.

## 3.2 Convergence-Based Stopping

The stopping rule operationalizes a convergence principle: **stop when independent navigation pathways have each contributed evidence.**

A "navigation pathway" is a source of knowledge — a distinct means by which the agent came to know something. Two pathways are independent if they have different failure modes:

- BM25 search and dense search are independent (keyword vs. semantic failure modes)
- SEARCH and FOLLOW_LINK are independent (query relevance vs. reference structure failure modes)
- OPEN on different documents found by different substrates is independent

**The stopping check:**

```python
if len(state.knowledge_sources) >= min_sources:
    return STOP
```

where `knowledge_sources` is the set of distinct action types or substrates that have contributed to the knowledge state. Default: `min_sources = 2`.

**Why convergence works (first principles):**

1. **Independent failure modes.** When BM25 finds `auth.py` AND dense search independently finds `auth.py`, the probability that `auth.py` is relevant is much higher than either signal alone — because BM25 fails on paraphrases and dense fails on rare identifiers, so their agreement implies relevance independent of failure mode.

2. **Diminishing returns.** If two independent pathways already found evidence, a third pathway is likely to find the same documents (redundancy) or nothing new (diminishing marginal gain).

3. **Zero cost.** The convergence check is a set-size comparison — no model inference, no distribution-specific parameters.

## 3.3 The ConvergenceRetriever (Drop-In RAG Improvement)

For practitioners who want convergence stopping in an existing RAG pipeline:

```python
from convergence_retrieval import ConvergenceRetriever, BM25Substrate, DenseSubstrate

retriever = ConvergenceRetriever(
    substrates=[BM25Substrate(), DenseSubstrate()],
)
retriever.index(documents)
result = retriever.search("query")  # stops when substrates converge
```

The retriever wraps multiple substrates with the convergence stopping rule. It searches substrates in order and stops when `min_sources` (default 2) have each returned relevant results. This reduces operations by 33-50% at equal result quality.

## 3.4 The NavigationAgent (Beyond RAG)

For applications requiring deeper exploration:

```python
from convergence_retrieval.environments import DocumentEnvironment
from convergence_retrieval.navigation import NavigationAgent

env = DocumentEnvironment(substrates=[BM25Substrate(), DenseSubstrate()])
env.load(documents)
agent = NavigationAgent(environment=env)
result = agent.navigate("How does auth middleware validate tokens?")
```

The agent's navigation loop:
1. **Search** to discover relevant documents (Step 0: always)
2. **Open** the most promising discovered-but-unread document
3. **Follow links** found in the opened document (cross-references, imports, citations)
4. **Check convergence**: has knowledge arrived from 2+ independent pathways?
   - YES → STOP and return gathered knowledge
   - NO → continue exploring (next substrate, next discovered document, next link)

The agent follows cross-references that flat retrieval cannot: `auth.py` mentions `jwt_utils.py` → follow the import → read `jwt_utils.py` → knowledge now comes from both "open" and "follow_link" pathways → convergence → STOP.

## 3.5 Utility@Budget Metric

We evaluate all systems using:

**Utility@Budget** = AnswerScore × (1 + η × EvidenceScore) − μ × NormalizedCost

For retrieval-only evaluation: AnswerScore = SupportRecall, EvidenceScore = SupportPrecision.
For end-to-end evaluation: AnswerScore = F1, EvidenceScore = SupportRecall.
η = 0.5, μ = 0.3 (fixed before experiments). Sensitivity analysis across μ is reported in Section 5.5.

## 3.6 Connection to Formal Framework

The navigation framework can be viewed as a constrained MDP where each action type is an "option" (Sutton, Precup, Singh, 1999) and the convergence check approximates the optimal stopping condition. The full formalization — state representation, weak dominance theorem, quantitative gain bound, and connection to information foraging theory — is presented in Appendix A.
