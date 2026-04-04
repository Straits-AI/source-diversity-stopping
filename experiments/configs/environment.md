# Experiment Environment

**Date:** 2026-04-04
**Status:** pending user confirmation

## Hardware

- **OS:** macOS Darwin 25.2.0
- **CPU:** [TO BE CONFIRMED]
- **RAM:** [TO BE CONFIRMED]
- **GPU:** [TO BE CONFIRMED — consumer GPU for router training, or none if API-only]
- **Disk:** [TO BE CONFIRMED]

## LLM Backend

- **Provider:** API-based (Anthropic / OpenAI)
- **Model:** [TO BE DETERMINED — likely Claude 3.5 Sonnet or GPT-4o]
- **Rationale:** Hold base LLM fixed; vary only the harness-level policy

## Software

- **Python:** [TO BE CONFIRMED]
- **Key libraries:**
  - Retrieval: FAISS, rank_bm25, sentence-transformers
  - Graph: networkx or neo4j
  - LLM API: anthropic / openai SDK
  - Evaluation: custom harness
  - Data: pandas, numpy
  - Visualization: matplotlib, seaborn

## Retrieval Infrastructure

- **Vector index:** FAISS (IVF-PQ or flat, depending on corpus size)
- **Lexical index:** BM25 via rank_bm25 or Elasticsearch
- **Graph store:** NetworkX (in-memory) for prototype, Neo4j for scale
- **Document parser:** [TO BE DETERMINED — likely unstructured or custom]

## Notes

User to confirm local machine specs. The harness and retrieval infrastructure run locally; LLM inference is API-based.
