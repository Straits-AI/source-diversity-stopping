> Note: PoC files referenced below have been moved to deprecated/poc/

# Experiment: Heterogeneous Benchmark (Regime C)

**Date:** 2026-04-04
**Phase:** 4b
**Iteration:** 0
**Status:** completed

## Context

HotpotQA baselines (Phase 4a) showed BM25 matching AEA (Utility@Budget 0.0169 vs 0.0163), explained by high lexical overlap.  H2 predicts AEA gains should be larger on tasks with low lexical overlap, multi-hop dependencies, and structural diversity.  This experiment tests H2 directly using a purpose-built synthetic benchmark where each task type was designed to defeat at least one single-substrate policy.

## Benchmark Design

**File:** `experiments/benchmarks/heterogeneous_benchmark.py`
**Runner:** `experiments/run_heterogeneous_benchmark.py`

100 questions across 6 task types, corpus of 10 paragraphs per question (2–3 gold + 7–8 distractors).  All generation is seeded deterministically (seed=42).

| Task Type | N | Gold Docs | Defeating Strategy |
|---|---|---|---|
| entity_bridge | 20 | 2 | Gold para 1 names country, not capital; para 2 names capital; question asks for capital via person |
| implicit_bridge | 20 | 2 | Director name only in movie para; award only in director para; question only names movie |
| semantic_computation | 20 | 2 | Both company paras found by semantic search; answer requires numerical comparison |
| low_lexical_overlap | 20 | 1 | Question uses "medication"/"medical condition", corpus uses "drug"/"disease" — BM25 penalised |
| multi_hop_chain | 10 | 3 | Three-hop chain A→B→C; only entity following can reach all three |
| discovery_extraction | 10 | 2 | Index paragraph lists all depts; dept-specific paragraph has budget; must discover, then extract |

Distractors are template-generated with shared entity names or same domain but without the answer components.

## Setup

- 5 policies: SemanticOnly, LexicalOnly, EntityOnly, Ensemble, AEAHeuristic
- Address spaces: Semantic (all-MiniLM-L6-v2), Lexical (BM25), Entity Graph (regex NER)
- Harness: max_steps=10, token_budget=4000, seed=42
- No LLM answer generation (heuristic derivation from top workspace item)

## Results

### Overall (N=100)

| Policy | SupportRecall | SupportPrecision | AvgOps | Utility@Budget |
|--------|--------------|-----------------|--------|----------------|
| π_semantic | 0.9183 | 0.3340 | 2.00 | 0.0270 |
| π_lexical | 0.9100 | 0.3360 | 2.00 | 0.0187 |
| π_entity | 0.7733 | 0.6197 | 3.00 | 0.0065 |
| π_ensemble | 0.9600 | 0.2952 | 3.00 | 0.0113 |
| π_aea_heuristic | 0.9333 | 0.3825 | 2.17 | 0.0177 |

### By Task Type

**Entity Bridge (N=20)**

| Policy | SupportRecall | AvgOps | Utility@Budget |
|--------|--------------|--------|----------------|
| π_semantic | 0.9250 | 2.00 | -0.0242 |
| π_lexical | 1.0000 | 2.00 | -0.0241 |
| π_entity | 1.0000 | 3.00 | -0.0279 |
| π_ensemble | 1.0000 | 3.00 | -0.0307 |
| π_aea_heuristic | 1.0000 | 3.00 | -0.0361 |

**Implicit Bridge (N=20)**

| Policy | SupportRecall | AvgOps | Utility@Budget |
|--------|--------------|--------|----------------|
| π_semantic | 1.0000 | 2.00 | -0.0240 |
| π_lexical | 0.7500 | 2.00 | -0.0241 |
| π_entity | 0.6500 | 3.00 | -0.0229 |
| π_ensemble | 1.0000 | 3.00 | -0.0311 |
| π_aea_heuristic | 1.0000 | 2.20 | -0.0263 |

**Semantic + Computation (N=20)**

| Policy | SupportRecall | AvgOps | Utility@Budget |
|--------|--------------|--------|----------------|
| π_semantic | 1.0000 | 2.00 | 0.0665 |
| π_lexical | 1.0000 | 2.00 | 0.0652 |
| π_entity | 1.0000 | 3.00 | 0.0450 |
| π_ensemble | 1.0000 | 3.00 | 0.0565 |
| π_aea_heuristic | 1.0000 | 1.30 | **0.0839** |

**Low Lexical Overlap (N=20)**

| Policy | SupportRecall | AvgOps | Utility@Budget |
|--------|--------------|--------|----------------|
| π_semantic | 1.0000 | 2.00 | -0.0151 |
| π_lexical | 1.0000 | 2.00 | 0.0735 |
| π_entity | 0.9000 | 3.00 | 0.0592 |
| π_ensemble | 1.0000 | 3.00 | 0.0655 |
| π_aea_heuristic | 1.0000 | 1.85 | 0.0063 |

**Multi-Hop Chain (N=10)**

| Policy | SupportRecall | AvgOps | Utility@Budget |
|--------|--------------|--------|----------------|
| π_semantic | 0.3333 | 2.00 | -0.0235 |
| π_lexical | 0.6000 | 2.00 | -0.0233 |
| π_entity | 0.6333 | 3.00 | -0.0255 |
| π_ensemble | 0.6000 | 3.00 | -0.0312 |
| π_aea_heuristic | 0.3333 | 3.00 | -0.0350 |

**Discovery + Extraction (N=10)**

| Policy | SupportRecall | AvgOps | Utility@Budget |
|--------|--------------|--------|----------------|
| π_semantic | 1.0000 | 2.00 | **0.2865** |
| π_lexical | 1.0000 | 2.00 | 0.0290 |
| π_entity | 0.0000 | 3.00 | -0.0163 |
| π_ensemble | 1.0000 | 3.00 | 0.0236 |
| π_aea_heuristic | 1.0000 | 2.00 | 0.1562 |

### Substrate Switching Analysis (AEA Heuristic)

- Questions where AEA used multiple substrates: **57/100**
- Questions where AEA outperformed best single-substrate: **27/100**
- Average substrates used per question by AEA: **1.57**

## Interpretation

### Key Findings

**1. No single substrate dominates across all task types.**
Entity alone is best on some entity bridge questions (1.0 recall) but fails entirely on discovery/extraction (0.0 recall).  Semantic is best overall (0.9183 recall) but that recall is pulled up by easy task types (semantic_computation, discovery_extraction at 1.0).

**2. AEA shows strongest advantage on semantic_computation.**
AEA heuristic achieves Utility@Budget=0.0839 on semantic_computation tasks vs 0.0665 (semantic) and 0.0652 (lexical).  The heuristic recognises these as simple single-hop queries and stops early (AvgOps=1.30), yielding a large cost saving while maintaining 1.0 recall.

**3. Low lexical overlap does NOT defeat BM25 as designed.**
BM25 achieves 1.0 recall on low_lexical_overlap tasks.  Investigation: the drug names (Zolparin, Veranex, …) appear identically in both question and gold paragraph.  BM25 keys on the drug name rather than the generic nouns ("medication" vs "drug"), and since the drug name is the main entity, BM25 still retrieves correctly.  This is a benchmark design flaw: the synonym replacement only applies to the generic nouns, not the key entity name.  Fix: use only paraphrases in questions with no shared named entities with the corpus.

**4. Multi-hop chain is the hardest task type for all policies.**
Best recall is entity (0.6333) and lexical (0.6000).  AEA and semantic both at 0.3333.  The three-hop chain design works — no single substrate reliably reaches all three gold paragraphs.  AEA does not improve here because the heuristic routing (semantic first → entity hop if multi-hop detected) only does one entity hop step; three-hop chains need deeper graph traversal.

**5. Discovery + extraction heavily favours semantic.**
Semantic achieves 0.2865 Utility@Budget, far above lexical (0.0290) and entity (0.0 recall).  The index paragraph ("Municipal Services Directory listing all departments") is well-matched by semantic similarity to the question mentioning the department's responsibility.

**6. Substrate switching at 57% — above the 44% seen on HotpotQA.**
Consistent with H2: harder, more heterogeneous tasks prompt more routing decisions.

### Benchmark Calibration Assessment

The target was: each single-substrate policy solves <60% of tasks.  Results by task type:

- **Entity bridge:** lexical/entity/ensemble all at 1.0 recall — too easy for lexical (entity names appear in corpus).
- **Implicit bridge:** semantic 1.0 recall — too easy for dense retrieval (director name co-occurs with movie in same embedding cluster).
- **Semantic computation:** all substrates at 1.0 recall — the task difficulty is in computation, not retrieval.
- **Low lexical overlap:** BM25 still 1.0 recall (see Finding 3).
- **Multi-hop chain:** all <0.7 recall — calibrated correctly.
- **Discovery extraction:** only entity fails.

Overall, this benchmark is harder than HotpotQA for entity-only approaches but the retrieval challenge is insufficient to distinguish semantic from lexical on most task types.  The key flaw is that most gold paragraphs share the main named entity with the question, so both BM25 and semantic converge to high recall.

### Benchmark Design Lessons

1. **Named entity overlap must be controlled.** If the key entity (drug name, movie title, company name) appears identically in question and corpus, BM25 trivially retrieves the right doc.  For genuine lexical-overlap control, questions must use only synonymic or paraphrastic reformulations of ALL content words, not just generic nouns.

2. **Implicit bridge needs vocabulary firewall.** Movie para must NOT contain the award keyword; director para must NOT contain the movie title.  Current generation meets this requirement but semantic similarity still connects them (director + film domain share an embedding cluster).

3. **Multi-hop chain works well as a difficulty ceiling.**  All policies underperform; deeper entity traversal is needed.

4. **Discovery + extraction as designed is a semantic search problem**, not a discovery/knowledge distinction problem.  The index paragraph is found easily by semantic search on responsibility keywords.  True discovery challenge would require the index to be in a non-semantic format (table, code, structured list not easily embedded).

## Decision

**Proceed with benchmark improvements:**

1. Fix low_lexical_overlap task type: remove entity name from question; use only paraphrase of the condition name.
2. Increase multi-hop chain depth from 3 to 4 hops; increase N from 10 to 20.
3. Replace discovery_extraction with a structural navigation task (requires A_struct, not yet implemented).
4. Add LLM answer generation for EM/F1 scoring to see if routing improves answer accuracy, not just evidence recall.

The current benchmark (v1) provides useful per-type breakdowns and confirms substrate switching at 57% on heterogeneous tasks.  The refined benchmark (v2) should show clearer differentiation between single-substrate and adaptive policies.
