# Project Critique V4: Advanced Agentic RAG Improvements

## Sprint 4 Focus: Retrieval Quality & Generation Robustness

This sprint focuses on cutting-edge techniques for improving retrieval quality,
generation robustness, and attribution accuracy in RAG systems.

---

## Current State Analysis

### Strengths (from previous sprints)
- ✅ Protocol-based modular architecture
- ✅ ReAct agents with tool calling
- ✅ Self-RAG with reflection tokens
- ✅ HyDE and ensemble retrieval
- ✅ Stepback prompting and Chain-of-Note
- ✅ LLMLingua compression
- ✅ Multi-hop reasoning (IRCoT)
- ✅ Graph-RAG for entity awareness

### Gaps Identified for Sprint 4

1. **No Contextual Retrieval** - Documents lack surrounding context for better embedding
2. **No FLARE Pattern** - Missing forward-looking active retrieval during generation
3. **No Interleaved Generation** - Generation doesn't trigger retrieval mid-stream
4. **No Query2Doc** - Missing pseudo-document expansion for query enrichment
5. **Lost-in-the-Middle Problem** - No position-aware reranking
6. **No DPR Training Pipeline** - Can't train domain-specific bi-encoders
7. **No Attributed QA** - Missing inline citation generation
8. **No Active Learning** - Can't learn from retrieval feedback
9. **No Cross-Encoder Reranker** - Missing deep cross-attention reranking
10. **No RAG Fusion** - Missing multi-perspective query fusion

---

## Planned Improvements (10 Commits)

### Commit 1: Contextual Retrieval
**File**: `autorag_live/retrieval/contextual.py`

Implement Anthropic's Contextual Retrieval (2024) pattern:
- Prepend document-level context to each chunk before embedding
- Use LLM to generate situating context for chunks
- Improve retrieval by 49% on average (per Anthropic benchmarks)

Key components:
- `ContextualChunker`: Adds context to chunks
- `SituatingContextGenerator`: LLM-based context generation
- `ContextualEmbedder`: Embeds with context prefix

### Commit 2: FLARE (Forward-Looking Active Retrieval)
**File**: `autorag_live/agent/flare.py`

Implement FLARE (Jiang et al., 2023):
- Generate until low-confidence token detected
- Retrieve based on upcoming content prediction
- Continue generation with retrieved context

Key components:
- `FLAREGenerator`: Confidence-based retrieval trigger
- `ConfidenceMonitor`: Token probability tracking
- `LookaheadRetriever`: Predictive retrieval

### Commit 3: Interleaved Retrieval-Generation
**File**: `autorag_live/agent/interleaved.py`

Implement IRG pattern for fine-grained retrieval:
- Generate sentence by sentence
- Evaluate each sentence for factuality
- Retrieve supporting evidence as needed

Key components:
- `InterleavedGenerator`: Sentence-level generation
- `FactualityChecker`: Per-sentence verification
- `AdaptiveRetriever`: On-demand retrieval

### Commit 4: Query2Doc Expansion
**File**: `autorag_live/augment/query2doc.py`

Implement Query2Doc (Wang et al., 2023):
- Generate pseudo-document from query using LLM
- Combine query and pseudo-doc for retrieval
- Improve recall without additional training

Key components:
- `Query2DocExpander`: Pseudo-document generation
- `HybridQueryBuilder`: Query+pseudo-doc fusion
- `ExpansionStrategies`: Multiple expansion modes

### Commit 5: Position-Aware Reranking
**File**: `autorag_live/rerank/position_aware.py`

Address Lost-in-the-Middle (Liu et al., 2023):
- Reorder documents to place important ones at extremes
- Implement positional bias mitigation
- Improve long-context utilization

Key components:
- `PositionAwareReranker`: Optimal position assignment
- `ImportanceScorer`: Document importance estimation
- `ContextWindowOptimizer`: Position optimization

### Commit 6: DPR Training Pipeline
**File**: `autorag_live/training/dpr.py`

Implement Dense Passage Retrieval training:
- In-batch negative sampling
- Hard negative mining
- Contrastive learning for bi-encoders

Key components:
- `DPRTrainer`: Training loop with negatives
- `NegativeSampler`: Hard negative selection
- `BiEncoderModel`: Query/passage encoders

### Commit 7: Attributed Question Answering
**File**: `autorag_live/augment/attributed_qa.py`

Implement inline citation generation:
- Generate answers with [1], [2] style citations
- Map citations to source documents
- Verify citation accuracy

Key components:
- `AttributedGenerator`: Citation-aware generation
- `CitationMapper`: Source-to-citation mapping
- `AttributionVerifier`: Citation accuracy checking

### Commit 8: Active Learning for Retrieval
**File**: `autorag_live/evals/active_learning.py`

Implement active learning feedback loop:
- Identify uncertain retrievals
- Collect human/model feedback
- Update retrieval model

Key components:
- `UncertaintySampler`: Select uncertain examples
- `FeedbackCollector`: Gather relevance labels
- `ModelUpdater`: Online model improvement

### Commit 9: Cross-Encoder Reranker
**File**: `autorag_live/rerank/cross_encoder.py`

Implement deep cross-attention reranking:
- Full query-document cross-attention
- Fine-grained relevance scoring
- Model abstraction for various backends

Key components:
- `CrossEncoderReranker`: Deep relevance scoring
- `CrossAttentionScorer`: Attention-based scoring
- `ModelWrapper`: Backend abstraction

### Commit 10: RAG Fusion
**File**: `autorag_live/retrieval/rag_fusion.py`

Implement RAG Fusion (Raudaschl, 2023):
- Generate multiple query perspectives
- Retrieve for each perspective
- Reciprocal rank fusion of results

Key components:
- `RAGFusion`: Multi-perspective retrieval
- `QueryPerspectiveGenerator`: Query variations
- `FusionAggregator`: Result combination

---

## Research References

1. **Contextual Retrieval** - Anthropic (2024)
2. **FLARE** - Jiang et al., 2023: "Active Retrieval Augmented Generation"
3. **Query2Doc** - Wang et al., 2023: "Query2Doc: Query Expansion with Large Language Models"
4. **Lost-in-the-Middle** - Liu et al., 2023: "Lost in the Middle: How Language Models Use Long Contexts"
5. **DPR** - Karpukhin et al., 2020: "Dense Passage Retrieval for Open-Domain Question Answering"
6. **Attributed QA** - Bohnet et al., 2022: "Attributed Question Answering"
7. **RAG Fusion** - Raudaschl, 2023: "Forget RAG, the Future is RAG-Fusion"

---

## Implementation Priority

| Priority | Commit | Impact | Complexity |
|----------|--------|--------|------------|
| 1 | Contextual Retrieval | High | Medium |
| 2 | RAG Fusion | High | Low |
| 3 | Cross-Encoder Reranker | High | Medium |
| 4 | FLARE | High | High |
| 5 | Query2Doc | Medium | Low |
| 6 | Position-Aware Reranking | Medium | Low |
| 7 | Attributed QA | Medium | Medium |
| 8 | Interleaved Generation | Medium | High |
| 9 | Active Learning | Medium | Medium |
| 10 | DPR Training | Medium | High |

---

## Expected Outcomes

After Sprint 4:
- **49% improvement** in retrieval accuracy (Contextual Retrieval)
- **Better long-context handling** (Position-Aware)
- **Verifiable answers** (Attributed QA)
- **Multi-perspective coverage** (RAG Fusion)
- **Active improvement** (Active Learning + DPR)
