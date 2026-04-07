# 4 Experimental Setup

We describe the benchmarks, baselines, ablation variants, evaluation metrics, and implementation details used in all experiments. The section is written to be self-contained: a reader should be able to reproduce every result reported in Section 5 from the information given here alone.

## 4.1 Benchmarks

### HotpotQA Bridge Subset (Yang et al., 2018)

We use the distractor validation split of HotpotQA. From the full 5,918 distractor-split questions we select the first 100 questions whose `type` field equals `"bridge"`, yielding a fixed, deterministic subset. Each question is accompanied by exactly 10 context paragraphs: 2 gold supporting paragraphs and 8 distractor paragraphs drawn from Wikipedia. Bridge questions require a two-step inference chain: the system must first locate a passage containing entity A, extract a bridge entity mentioned there, and then locate a second passage whose content concerns that bridge entity (entity B). Because question text explicitly names entity A, there is high lexical overlap between the question and the first-hop gold paragraph.

| Property | Value |
|---|---|
| Total questions | 100 |
| Context paragraphs per question | 10 |
| Gold supporting paragraphs | 2 |
| Distractor paragraphs | 8 |
| Question type | Bridge only |

### HotpotQA Full (All Question Types) — Generalization Experiment A

To verify that results are not specific to bridge questions, we additionally evaluate on the first 1,000 questions from the HotpotQA distractor validation set without type filtering, yielding a mixture of bridge (807) and comparison (193) questions. This tests whether the heuristic's stopping rule—which operates on workspace statistics, not question-type patterns—generalizes across question types.

| Property | Value |
|---|---|
| Total questions | 1,000 |
| Bridge questions | 807 (80.7%) |
| Comparison questions | 193 (19.3%) |
| Context paragraphs per question | 10 |
| Seed | 42 |

### HotpotQA Open-Domain Expansion — Generalization Experiment B

To verify that results are not specific to the closed 10-paragraph setting, we expand the candidate set for 200 bridge questions from 10 to 50 paragraphs (2 gold + 8 original distractors + 40 additional distractors sampled from other questions in the dataset, seed=42). The distractor pool comprises 64,900 unique paragraphs from all non-selected questions. This simulates open-domain retrieval where the gold signal is diluted 5-fold.

| Property | Value |
|---|---|
| Total questions | 200 |
| Setting A (closed) | 10 paragraphs per question |
| Setting B (open-domain) | 50 paragraphs per question |
| Gold supporting paragraphs | 2 |
| Additional distractors (B) | 40 (sampled from 64,900-paragraph pool) |
| Seed | 42 |

### Heterogeneous Benchmark v2 (this work)

To test robustness across a broader range of retrieval challenges, we construct a synthetic benchmark of 100 questions spanning six task types, generated deterministically with seed 42.

| Task Type | Count | Key Challenge |
|---|---|---|
| Entity Bridge | 20 | Bridge entity absent from question text |
| Implicit Bridge | 20 | Bridge relation expressed paraphrastically |
| Semantic + Computation | 20 | Requires semantic retrieval then arithmetic |
| Low Lexical Overlap | 20 | < 15% content-word overlap between question and gold |
| Multi-hop Chain | 10 | Sequential retrieval hops |
| Discovery + Extraction | 10 | Gold paragraph unknown until prior hop reveals it |

Two design invariants are enforced programmatically: (1) **Entity isolation** — for bridge tasks, bridge entities do not appear in the question text; (2) **Lexical isolation** — for low-overlap tasks, content word overlap is below 15%.

## 4.2 Baselines

| System | Description | Address Spaces |
|---|---|---|
| $\pi_\text{semantic}$ | Dense retrieval via all-MiniLM-L6-v2, top-k by cosine similarity | Semantic only |
| $\pi_\text{lexical}$ | BM25 via rank_bm25, top-k by BM25 score | Lexical only |
| $\pi_\text{entity}$ | Regex NER + entity co-occurrence graph + BFS | Entity only (negative control) |
| $\pi_\text{ensemble}$ | Query all substrates, merge and deduplicate | All (no routing) |
| $\pi_\text{aea}$ | Coverage-driven adaptive routing (proposed) | All (heuristic routing) |
| $\pi_\text{llm}$ | LLM reasons about evidence sufficiency at each step (gpt-oss-120b) | All (LLM routing) |

$\pi_\text{entity}$ is included as a negative control: we report it for completeness and to verify that entity-graph traversal alone does not explain system performance. Ablation results confirm it does not (Section 5.3).

$\pi_\text{llm}$ receives the question, current workspace contents, and available actions at each step, and responds with a single routing decision (STOP, SEMANTIC_SEARCH, LEXICAL_SEARCH, or ENTITY_HOP). It always performs semantic search at step 0; from step 1 onward, the LLM decides. This policy tests whether LLM-guided positive routing outperforms the heuristic's coverage-driven stopping.

### Answer Generation

For end-to-end evaluation, we generate answers from retrieved evidence using gpt-oss-120b (via OpenRouter). The prompt provides the retrieved passages and asks for a short, direct answer. The same model and prompt are used for all policies — only the retrieved evidence differs. EM and F1 are scored against gold answers using SQuAD-style normalization.

## 4.3 Ablation Variants

| Variant | Modification | Question Answered |
|---|---|---|
| abl_no_early_stop | Always execute 2 steps | Is early stopping necessary? |
| abl_semantic_smart_stop | Semantic only + coverage stop | Is stopping sufficient without routing? |
| abl_no_entity_hop | Lexical fallback replaces entity hops | Do entity hops contribute? |
| abl_always_hop | Unconditional entity hop after step 0 | Is selectivity important? |
| abl_no_workspace_mgmt | No pin/evict | Does workspace curation matter? |

## 4.4 Metrics

**Primary: Utility@Budget**

$$\text{Utility@Budget} = \text{AnswerScore} \times (1 + \eta \cdot \text{EvidenceScore}) - \mu \cdot \text{Cost}$$

with $\eta = 0.5$, $\mu = 0.3$ (fixed before experiments).

| Metric | Definition |
|---|---|
| Support Recall | $|\text{workspace} \cap \text{gold}| / |\text{gold}|$ |
| Support Precision | $|\text{workspace} \cap \text{gold}| / |\text{workspace}|$ |
| Average Operations | Mean retrieval operations per question |

**Two evaluation modes.** We report both retrieval-only metrics (Support Recall, Support Precision) and end-to-end metrics (EM, F1) with LLM answer generation via gpt-oss-120b. The retrieval-only U@B uses SupportRecall as AnswerScore; the end-to-end U@B uses F1 as AnswerScore. Both formulas are explicitly identified in each results table.

## 4.5 Implementation Details

| Parameter | Value |
|---|---|
| Embedding model | all-MiniLM-L6-v2 (384-dim) |
| BM25 | rank_bm25 BM25Okapi, default parameters |
| Entity extraction | Regex-based (capitalized multi-word phrases) |
| Workspace capacity | 10 items |
| Pin count | Top-2 by relevance |
| Eviction threshold | Relevance < 0.15 |
| Coverage stop: min items | 2, relevance >= 0.40, from >= 2 sources |
| $\eta$, $\mu$ | 0.50, 0.30 |
| Random seed | 42 |
| Hardware | Apple Silicon laptop, CPU only |
| Runtime | < 5 minutes per benchmark |

## 4.6 Evaluation Contract

1. **Shared substrates.** Address spaces are implemented once and shared across all systems.
2. **Policy variation only.** Only the routing/stopping policy differs between systems.
3. **No test-time tuning.** All hyperparameters fixed before experiments.
4. **Immutable harness.** Metric computation code is locked at experiment start.
