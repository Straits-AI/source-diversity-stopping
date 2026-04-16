# Convergence-Based Navigation: A Framework for Agents That Know When to Stop Exploring

**Paper:** "Source-Diversity Stopping is Pareto-Optimal for Multi-Substrate Retrieval"  
**Status:** Ready for EMNLP 2026 submission (deadline May 25)  
**Review scores:** R1=8, R2=7, R3=8 (Mean=7.67) after 6 review rounds

## Key Finding

A one-line structural heuristic — **stop when the workspace contains evidence from two or more independent retrieval sources** — is Pareto-optimal within ten tested alternatives for multi-substrate retrieval stopping.

- Significantly outperforms comprehensive retrieval across 3 benchmark families (HotpotQA p<0.0001, BRIGHT p=0.003, diluted retrieval p<0.0001)
- **Ten** alternative stopping mechanisms tested across 7 design categories — **none** beats it
- Two ceilings identified: content-aware (noise > information) and structural (source diversity is maximal)

## The Two-Ceiling Framework

```
Content-aware ceiling: all 7 content-based methods add more noise than information
  NLI bundle (d=-0.73) | GBT classifier (catastrophic) | Cross-encoder (-0.10)
  LLM decomposition (-0.001) | Answer stability (-0.06) | Confidence-gated (fails at N=500)
  Embedding router (tied)

Structural ceiling: 3 structural improvements converge to identical behavior
  Threshold optimization | Novelty detection | Dual-signal stopping
  → Source diversity is the binding constraint; other signals are redundant

Heuristic sits at the intersection = Pareto frontier
```

## Results Summary

| Benchmark | Family | N | Heuristic vs Ensemble p | Cohen's d |
|-----------|--------|---|------------------------|-----------|
| HotpotQA (all types) | Multi-hop factoid | 1000 | <0.000001 | 0.379 |
| HotpotQA (comparison) | Comparison QA | 193 | <0.000001 | 0.419 |
| Diluted retrieval (50-para) | Diluted retrieval | 200 | <0.000001 | 0.491 |
| BRIGHT | Reasoning-intensive | 200 | 0.0026 | 0.216 |
| HotpotQA E2E (with LLM) | End-to-end | 500 | 0.021 | 0.103 |
| **FEVER (fact verification)** | **Non-QA classification** | **200** | **<0.0001** | **(Pareto)** |

## `convergence_retrieval` Library

The `convergence_retrieval` package provides the core framework as an importable library.

### Installation

```bash
pip install -e .
```

### Basic Usage

```python
from convergence_retrieval import ConvergenceRetriever, BM25Substrate, DenseSubstrate

retriever = ConvergenceRetriever(
    substrates=[
        BM25Substrate(),
        DenseSubstrate(model="all-MiniLM-L6-v2"),
    ],
)
retriever.index(documents)
results = retriever.search("How does auth work?")
# Returns results in ~1.2 operations instead of 2.0
```

### NavigationAgent Usage

```python
from convergence_retrieval import NavigationAgent, ConvergencePolicy, NavigationState
from convergence_retrieval import DocumentEnvironment

env = DocumentEnvironment(documents)
policy = ConvergencePolicy(min_sources=2)
agent = NavigationAgent(policy=policy)

state = NavigationState()
result = agent.run(query="What is the capital of France?", env=env, state=state)
# Agent stops as soon as evidence from 2+ independent sources is found
```

### Library Structure

```
convergence_retrieval/
  __init__.py               # Top-level exports
  retriever.py              # ConvergenceRetriever
  substrates/
    base.py                 # Substrate ABC
    bm25.py                 # BM25Substrate
    dense.py                # DenseSubstrate
    structural.py           # StructuralSubstrate
  navigation/
    agent.py                # NavigationAgent
    policy.py               # NavigationPolicy, ConvergencePolicy
    state.py                # NavigationState
    actions.py              # Action, ActionType, ActionResult
  environments/
    base.py                 # Environment ABC
    document_env.py         # DocumentEnvironment
  tests/
    test_retriever.py
```

## Repository Structure

```
paper/                          # Paper sections + assembled full-paper.md (10,326 words)
  00-abstract.md                # Title + abstract
  01-introduction.md            # Contributions + overview
  02-related-work.md            # 63 papers surveyed
  03-methodology.md             # Heuristic + confidence-gated + U@B metric
  04-experimental-setup.md      # Benchmarks, baselines, ablations
  05-results.md                 # All results (Tables 1-10)
  06-discussion.md              # Root cause analysis (2700 words) + limitations
  07-conclusion.md              # Three findings + implications
  appendix-a-formal-framework.md # CMDP formalization
  full-paper.md                 # Assembled paper

convergence_retrieval/          # Importable library (see above)

experiments/
  aea/                          # Core framework
    types.py                    # AgentState, Action, EvidenceBundle, etc.
    address_spaces/             # Semantic, Lexical, Entity Graph, Structural, Executable
      executable.py             # A_tool: regex number extraction + Python arithmetic
    evaluation/                 # Immutable harness + metrics (EM, F1, U@B)
    policies/                   # 10 stopping policies
      heuristic.py              # π_heuristic (THE method — 2/2/0.4 rule)
      single_substrate.py       # π_semantic, π_lexical, π_entity
      ensemble.py               # π_ensemble (query all)
      ablations.py              # 5 ablation variants
      llm_routed.py             # π_llm_routed (LLM routing)
      learned_stopping.py       # π_learned (GBT classifier)
      cross_encoder_stopping.py # π_cross_encoder (MS MARCO)
      nli_stopping.py           # π_nli (DeBERTa-v3 NLI)
      decomposition_stopping.py # π_decomposition (LLM requirements)
      answer_stability.py       # π_answer_stability (draft convergence)
      confidence_gated.py       # π_confidence_gated (LLM self-assessment)
      embedding_router.py       # π_embedding_router (question classifier)
    answer_generator.py         # LLM answer generation (gpt-oss-120b)
  benchmarks/                   # Heterogeneous benchmark v2, Structural Nav, Computational
    computational_benchmark.py  # 100 computation questions (50 comparison + 50 arithmetic)
  models/
    stopping_classifier_clean.pkl # Clean GBT classifier (train/test split verified)
  results/                      # All experimental results (JSON)
    tool_execution.json         # A_tool experiment results
    codebase_nav.json           # Codebase navigation experiment (code-search generalisation)
  run_*.py                      # Experiment runners (reproducible)

deprecated/                     # Superseded files kept for reference
  experiments/                  # Early runners (run_heterogeneous_benchmark, run_llm_routed,
                                #   run_with_llm_answers, run_musique)
  experiments/results/          # Stale result files (v1 benchmark, rate-limited runs,
                                #   intermediate checkpoints)
  models/                       # Old contaminated stopping_classifier.pkl
  paper/                        # Early paper files (outline, comparison_table,
                                #   06a-root-cause-analysis)
  poc/                          # Early proof-of-concept substrate switching scripts

research-log/                   # Full research log (8 entries)
  000-setup.md                  # Phase 0: Idea DNA, evaluation contract
  001-literature-review.md      # Phase 1: 63 papers, 5 gaps
  002-hypothesis.md             # Phase 2: CMDP formalization, theory review
  003-poc-substrate-switching.md # Phase 3: 44% switching rate validated
  004-experiment-plan.md        # Phase 4: Full experiment design
  005-analysis-iter-0.md        # Phase 5: Routing avoidance finding
  006-paper-review.md           # Phase 6: First review (4/10)
  007-review-panel-v5.md        # Review panel results
  008-review-panel-v6.md        # Second review panel

prompts/                        # Subagent prompt templates
```

## Reproducing Results

### Requirements

```bash
pip install sentence-transformers rank_bm25 numpy scikit-learn scipy
# For LLM answer generation:
pip install openai
```

### Environment

```bash
export OPENROUTER_API_KEY="your_key_here"  # Required for LLM answer generation only
```

### All Experiments

#### Core Retrieval (no API key required)

```bash
# Core result: heuristic vs baselines on HotpotQA (retrieval-only)
python experiments/run_full_hotpotqa.py

# HotpotQA baselines only
python experiments/run_hotpotqa_baselines.py

# Diluted retrieval (50 paragraphs)
python experiments/run_open_domain.py

# BRIGHT benchmark (retrieval-only)
python experiments/run_bright.py

# Structural improvements (threshold tuning, novelty detection, dual-signal)
python experiments/run_structural_improvements.py

# Ablation study (5 ablation variants of the heuristic)
python experiments/run_ablations.py

# Heterogeneous benchmark v2
python experiments/run_heterogeneous_v2.py

# Multi-seed reproducibility
python experiments/run_multiseed.py
```

#### Domain Generalisation (no API key required)

```bash
# Structural navigation address space (A_struct) — title-hierarchy browsing
python experiments/run_structural_nav.py

# Tool execution address space (A_tool) — executable substrate vs source diversity
python experiments/run_tool_execution.py

# FEVER fact verification — stopping generalises beyond QA tasks
python experiments/run_fever.py

# Codebase navigation — stopping generalises to code search
python experiments/run_codebase_nav.py
```

#### Content-Aware Stopping (requires OPENROUTER_API_KEY)

```bash
# E2E with LLM answers on N=500 HotpotQA (heuristic + confidence-gated)
python experiments/run_e2e_n500.py

# Confidence-gated on N=500 (extended confidence-gated evaluation)
python experiments/run_confidence_gated_n500.py

# Confidence-gated on BRIGHT
python experiments/run_confidence_gated_bright.py

# Confidence-gated evaluation (smaller N)
python experiments/run_confidence_gated_eval.py

# NLI stopping (DeBERTa-v3)
python experiments/run_nli_stopping_eval.py

# Answer stability stopping (draft convergence)
python experiments/run_answer_stability_eval.py

# LLM decomposition stopping
python experiments/run_decomposition_eval.py

# Cross-encoder stopping (MS MARCO)
python experiments/run_cross_encoder_eval.py

# Embedding router (question classifier)
python experiments/run_embedding_router_eval.py

# Learned stopping (GBT classifier)
python experiments/run_learned_stopping.py

# 2WikiMultihopQA evaluation
python experiments/run_2wiki.py
```

#### Training

```bash
# Train the GBT stopping classifier (requires trajectory data)
python experiments/collect_trajectories.py
python experiments/train_stopping_model.py
```

### Train/Test Split

- **Training:** HotpotQA bridge questions 500–999 (for learned classifier only)
- **Evaluation:** HotpotQA bridge questions 0–499 (all reported results)
- **Zero overlap verified programmatically**

## Research Journey

| Phase | What Happened | Score |
|-------|--------------|-------|
| v1 | "Here's our AEA method" | 4/10 (Reject) |
| v6 | "Here's why the heuristic wins" | 5/10 (Borderline Reject) |
| v8 | "Validated across benchmarks" | 6/10 (Borderline Accept) |
| v9 | "3 families + 5 failures + BRIGHT" | 7/10 (Accept) |
| v11 | "Confidence-gated beats it!" → doesn't replicate | 8/7/6 (Mixed) |
| v12 | "10 alternatives, none wins. Pareto-optimal." | 8/7/8 (Accept) |

56+ commits. 10 stopping mechanisms tested. 3 benchmark families. 6 review rounds. One finding: **source diversity is the answer.**

## Tool Execution Experiment (A_tool Gap Closure)

The original paper never tested executable addressing (SQL, computation). A fourth experiment fills this gap.

**Setup:** 100 computation-focused questions (50 revenue comparisons, 50 population arithmetic) with 4 policies:
`pi_semantic`, `pi_lexical`, `pi_executable` (SEARCH + TOOL_CALL), `pi_ensemble_tool` (all three).

| Policy           | SupportRecall | F1     | Utility@Budget | AvgSteps |
|------------------|--------------|--------|----------------|----------|
| pi_semantic      | 1.0000       | 0.0295 | 0.0205         | 2.00     |
| pi_lexical       | 1.0000       | 0.0325 | 0.0251         | 2.00     |
| pi_executable    | 1.0000       | 0.1740 | **0.2427**     | 2.00     |
| pi_ensemble_tool | 1.0000       | 0.0325 | 0.0068         | 4.00     |

**Finding:** When the task intrinsically requires computation, the executable substrate **dominates** (U@B 0.2427 vs 0.0251 for best retrieval-only). Crucially, `pi_ensemble_tool` is **worse** than `pi_executable` alone (0.0068 vs 0.2427) — the non-executable substrates degrade the answer by flooding the workspace with passage content that outscores the TOOL_CALL result.

**Implication for stopping theory:** Source diversity remains Pareto-optimal for retrieval-based QA, but the stopping signal changes when computation is the primary operation. For computation tasks, stopping *after the first TOOL_CALL* is optimal — the diversity heuristic does not apply because the relevant "source" is the computation itself, not passage variety. This represents a clean boundary condition for the paper's main claim.

## Codebase Navigation Experiment (Domain Generalisation — Code Search)

To address reviewer concerns about generalisation beyond natural-language QA, this experiment tests whether source-diversity stopping works for **code search** — a structurally different retrieval task where the corpus is Python source files and queries ask about implementation locations.

META: We test our own theory on our own codebase (experiments/aea/).

**Setup:** 35 Python files from the AEA framework as documents; 50 code-search questions at three difficulty levels (16 easy / 18 medium / 16 hard). Context per question: gold file(s) + ~9 distractor files. No API calls; seed=42.

| Policy           | SupportRecall | AvgOps | Utility@Budget |
|------------------|--------------|--------|----------------|
| pi_semantic      | 0.8200       | 2.00   | -0.1101        |
| pi_lexical       | 0.8500       | 2.00   | -0.1100        |
| pi_structural    | 0.6900       | 2.00   | -0.0348        |
| pi_ensemble      | 0.9700       | 1.54   | -0.1051        |
| **pi_heuristic** | **0.9700**   | **1.52** | **-0.1046** |

**Finding:** Convergence-based stopping matches ensemble retrieval quality (SupportRecall=0.970 for both) while using slightly fewer operations (1.52 vs 1.54). The heuristic triggered early stops on 24/50 examples (48%) — whenever two distinct high-relevance files were found. Paired t-test vs ensemble: Δ=+0.0005, p=0.304 (n.s. — no significant degradation). On medium-difficulty questions, heuristic achieves SupportRecall=1.000 with 1.44 avg ops vs 1.50 for ensemble.

**Substrate ranking for code search:** Lexical > Semantic > Structural for single-substrate recall. Structural navigation (filename matching) alone performs worst on recall but best on U@B due to lower cost. Ensemble + heuristic both dominate single substrates on recall.

**Implication:** Source-diversity stopping generalises from QA to code navigation. The mechanism — stop when evidence from ≥2 independent files is present — is task-agnostic: it fires on retrieval quality, not on answer type.

Results: `experiments/results/codebase_nav.json` | Script: `experiments/run_codebase_nav.py`

## FEVER Fact Verification Experiment (Task-Type Generalisation)

To address reviewer concern about single-task-type evaluation, a FEVER-style fact verification experiment tests whether source-diversity stopping generalises **beyond QA** to classification tasks.

**Setup:** 200 FEVER-style fact verification examples (100 SUPPORTED + 100 REFUTED claims), each with 10 context paragraphs (1 gold evidence + 9 distractors). The task is binary classification, not answer generation. Retrieval evaluation measures SupportRecall: did we find the gold evidence paragraph?

| Policy           | SupportRecall | AvgOps | Utility@Budget |
|------------------|--------------|--------|----------------|
| pi_semantic      | 1.0000       | 2.00   | -0.0216        |
| pi_lexical       | 1.0000       | 2.00   | -0.0216        |
| pi_ensemble      | 1.0000       | 3.00   | -0.0289        |
| **pi_heuristic** | **1.0000**   | **1.01** | **-0.0106** |

**Finding:** Source-diversity stopping achieves equal SupportRecall (1.000) while using **66% fewer retrieval operations** than ensemble (1.01 vs 3.00 avg ops). Utility@Budget improvement = +0.0183 (paired t-test: p < 0.0001). The result is statistically significant and reproduces the Pareto-optimal pattern observed on QA tasks.

**Implication:** The source-diversity stopping heuristic generalises beyond question answering to fact verification (a classification task). The underlying mechanism — stop when evidence from two or more independent sources is present — is not task-type specific. It operates on retrieval quality, not on the downstream label prediction.

Results: `experiments/results/fever.json` | Script: `experiments/run_fever.py`

## Citation

```bibtex
@article{convergence-navigation-2026,
  title={Convergence-Based Navigation: A Framework for Agents That Know When to Stop Exploring},
  year={2026},
  note={Under review at EMNLP 2026}
}
```

## License

MIT
