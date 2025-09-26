# AutoRAG-Live Project Critique and Optimization Roadmap

_Date: 2025-09-26_

## Overview

AutoRAG-Live has evolved into a feature-rich Retrieval-Augmented Generation experimentation platform. The codebase demonstrates solid modularity, configuration management via Hydra/OmegaConf, and an expanding automated test suite. Recent additions such as centralized logging, caching, and performance monitoring provide a strong foundation for production-grade experimentation.

However, several structural and performance issues remain that, if addressed, would improve the system's efficiency, maintainability, and developer experience.

## Strengths

- **Modular architecture** with separable retriever, augmentation, pipeline, and evaluation layers.
- **Strict typing** and Pydantic-based schema validation improve reliability.
- **Config-driven design** simplifies reproducibility across experiments.
- **Growing test coverage** for retrievers, reducing regression risk.
- **MkDocs documentation** offers newcomer-friendly onboarding.

## Areas for Improvement

### Performance Bottlenecks

1. **Redundant computation in dense retrievers**
   - Duplicate encoding work and repeated normalization inside search paths.
   - Deterministic fallback relies on `np.random.seed` per document, which is expensive.

2. **Caching inefficiencies**
   - TTL values are not enforced per entry, causing stale data to persist.
   - File-based cache performs full-index rewrites on each mutation and lacks size-aware eviction heuristics.

3. **Retriever adapters**
   - Qdrant and Elasticsearch adapters execute sequential network requests where batching is possible.
   - BM25 retriever re-tokenizes documents and queries without memoization.

4. **Performance monitoring overhead**
   - Frequent `psutil` calls add latency to short-lived operations.
   - Aggregation data structures allocate aggressively and compute quantiles on small sample sets.

### Maintainability Gaps

- Lack of benchmark scripts or profiling harnesses to validate improvements.
- Limited documentation on how to interpret new performance metrics.
- Incomplete tests for cache eviction and performance monitoring edge cases.

## Recommended Actions

1. Memoize dense embedding generations and persist normalized vectors to avoid repeated work.
2. Enforce TTL semantics in caches, reuse serialized payloads, and decouple index persistence from hot paths.
3. Batch Qdrant upserts and Elasticsearch queries; introduce reusable clients with connection pooling.
4. Reduce `psutil` sampling frequency and provide lightweight counters for low-latency sections.
5. Extend the test suite to simulate high-frequency retrieval scenarios and ensure caches behave under pressure.
6. Document tuning strategies for cache sizing, monitoring, and retriever-specific performance knobs.

## Next Steps

The subsequent commits in this series will implement a subset of the above recommendations, prioritizing high-impact optimizations with measurable gains while maintaining test coverage and documentation quality.
