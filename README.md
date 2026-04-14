# Source-Diversity Stopping is Pareto-Optimal for Multi-Substrate Retrieval

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

experiments/
  aea/                          # Core framework
    types.py                    # AgentState, Action, EvidenceBundle, etc.
    address_spaces/             # Semantic, Lexical, Entity Graph
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
  benchmarks/                   # Heterogeneous benchmark v2
  models/
    stopping_classifier_clean.pkl # Clean GBT classifier (train/test split verified)
  results/                      # All experimental results (JSON)
  run_*.py                      # Experiment runners (reproducible)
  collect_trajectories.py       # Trajectory data collection
  train_stopping_model.py       # Classifier training

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

### Key Experiments

```bash
# Core result: heuristic vs baselines on HotpotQA (retrieval-only, no API needed)
python experiments/run_full_hotpotqa.py

# Diluted retrieval (50 paragraphs, no API needed)
python experiments/run_open_domain.py

# BRIGHT benchmark (no API needed for retrieval-only)
python experiments/run_bright.py

# E2E with LLM answers (requires OPENROUTER_API_KEY)
python experiments/run_confidence_gated_n500.py

# Structural improvements (no API needed)
python experiments/run_structural_improvements.py
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

56 commits. 10 stopping mechanisms tested. 3 benchmark families. 6 review rounds. One finding: **source diversity is the answer.**

## Citation

```bibtex
@article{source-diversity-stopping-2026,
  title={Source-Diversity Stopping is Pareto-Optimal for Multi-Substrate Retrieval},
  year={2026},
  note={Under review at EMNLP 2026}
}
```

## License

MIT
