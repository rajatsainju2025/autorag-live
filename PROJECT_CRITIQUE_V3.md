# Project Critique V3: Advanced Agentic RAG Enhancements

## Executive Summary

This is the third sprint of 10 commits focusing on cutting-edge agentic RAG capabilities.
Previous sprints covered: protocols, tool calling, ReAct, self-reflection, async execution,
structured output, memory, routing, compression, DSL, adaptive RAG, multi-hop, citations,
semantic cache, grounding, tracing, graph RAG, feedback loop, intent classification, and synthesis.

## Gap Analysis: 10 New Improvement Areas

### 1. **Query Planning & Decomposition** (Missing: Strategic Query Planning)
**Current State:** Basic query expansion exists, but no strategic query planning
**Gap:** No RAPTOR-style hierarchical planning or Self-Ask decomposition
**Solution:** Implement query planner with hierarchical decomposition and sub-query generation
**Reference:** RAPTOR (Sarthi et al., 2024), Self-Ask (Press et al., 2022)

### 2. **Speculative RAG** (Missing: Parallel Speculation)
**Current State:** Sequential retrieval and generation
**Gap:** No speculative execution for latency reduction
**Solution:** Implement speculative retrieval with parallel candidate generation
**Reference:** Speculative RAG (Wang et al., 2024)

### 3. **Document Reranking with Late Interaction** (Missing: ColBERT-style)
**Current State:** Basic MMR and cross-encoder reranking
**Gap:** No late interaction models or ColBERT-style reranking
**Solution:** Implement late interaction reranker with token-level matching
**Reference:** ColBERT (Khattab & Zaharia, 2020), ColBERTv2

### 4. **Contextual Compression** (Missing: LLMLingua-style)
**Current State:** Basic extractive/abstractive compression
**Gap:** No token-level pruning or LLMLingua-style compression
**Solution:** Implement selective context compression with perplexity-based pruning
**Reference:** LLMLingua (Jiang et al., 2023), LongLLMLingua

### 5. **Retrieval Augmented Fine-Tuning (RAFT)** (Missing: Training Integration)
**Current State:** No fine-tuning support
**Gap:** No mechanism for RAFT-style domain adaptation
**Solution:** Implement RAFT data generator and training pipeline hooks
**Reference:** RAFT (Zhang et al., 2024)

### 6. **Self-RAG with Reflection Tokens** (Missing: Inline Reflection)
**Current State:** Basic self-reflection exists
**Gap:** No special reflection tokens or inline critique markers
**Solution:** Implement Self-RAG with [Retrieve], [ISREL], [ISSUP], [ISUSE] tokens
**Reference:** Self-RAG (Asai et al., 2023)

### 7. **Hypothetical Document Embeddings (HyDE)** (Missing: Query2Doc)
**Current State:** Basic semantic search
**Gap:** No hypothetical document generation for improved retrieval
**Solution:** Implement HyDE with LLM-generated pseudo-documents
**Reference:** HyDE (Gao et al., 2023), Query2Doc

### 8. **Ensemble Retrieval with Learned Fusion** (Missing: Learned Weights)
**Current State:** Basic hybrid retrieval
**Gap:** No learned fusion weights or ensemble optimization
**Solution:** Implement learned retrieval ensemble with dynamic weight learning
**Reference:** Learned Sparse Retrieval, Contriever-MS MARCO

### 9. **Stepback Prompting** (Missing: Abstraction Layer)
**Current State:** Direct question answering
**Gap:** No abstraction-then-answer pattern
**Solution:** Implement stepback prompting with principle extraction
**Reference:** Stepback Prompting (Zheng et al., 2023)

### 10. **Chain-of-Note (CoN)** (Missing: Sequential Note-Taking)
**Current State:** Basic context aggregation
**Gap:** No sequential note-taking for complex documents
**Solution:** Implement Chain-of-Note with reading notes and evidence aggregation
**Reference:** Chain-of-Note (Yu et al., 2023)

---

## Implementation Plan: 10 Commits

| Commit | Feature | File | Description |
|--------|---------|------|-------------|
| 1 | Query Planner | `agent/query_planner_v2.py` | RAPTOR-style hierarchical planning |
| 2 | Speculative RAG | `agent/speculative_rag.py` | Parallel speculation for latency |
| 3 | Late Interaction | `rerank/late_interaction.py` | ColBERT-style token matching |
| 4 | LLMLingua Compression | `augment/llmlingua.py` | Perplexity-based context pruning |
| 5 | RAFT Generator | `evals/raft_generator.py` | Training data for domain adaptation |
| 6 | Self-RAG Tokens | `agent/self_rag.py` | Reflection tokens inline |
| 7 | HyDE Retrieval | `retrieval/hyde.py` | Hypothetical document embeddings |
| 8 | Ensemble Retriever | `retrieval/ensemble.py` | Learned fusion weights |
| 9 | Stepback Prompting | `prompts/stepback.py` | Abstraction prompting |
| 10 | Chain-of-Note | `augment/chain_of_note.py` | Sequential note aggregation |

---

*Generated: January 2026*
*Sprint: 3 of N*
