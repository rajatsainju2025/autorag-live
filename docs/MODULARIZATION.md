# Modularization Guide for Agentic RAG

This project benefits from breaking responsibilities into small, testable modules.

Recommendations:
- `core/`: keep interfaces and lightweight utilities (`StateManager`, `AgentPolicy`).
- `pipeline/`: orchestrator and pipeline types only; avoid heavy logic at import time.
- `knowledge_graph/`: pure processing modules (extractors, graph builders) that accept dependencies by injection.
- `retrievers/` and `embeddings/`: pluggable implementations behind small adapters.
- `utils/`: small helpers (async utilities, batching, IO) with minimal external dependencies.

Design patterns:
- Dependency injection: pass `StateManager` / `AgentPolicy` into components rather than importing singletons.
- Async-friendly APIs: prefer async `run` methods and provide sync shims.
- Small public interfaces: export abstract interfaces for components to enable swapping implementations.

Testing and CI:
- Provide small unit tests for each module (see `tests/test_entity_extractor.py`).
- Keep CI focused: run fast unit tests and linting.

This document is a short starting point — expand with concrete examples as you refactor components.
