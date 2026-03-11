from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class SafetyCheckResult:
    is_safe: bool
    risk_level: Optional[str] = None
    issues: Optional[List[str]] = None


@runtime_checkable
class RetrieverProtocol(Protocol):
    async def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:  # pragma: no cover - protocol
        ...

    def add_documents(self, docs: List[Dict[str, Any]]) -> None:  # pragma: no cover - protocol
        ...


@runtime_checkable
class LLMProvider(Protocol):
    from __future__ import annotations

    from dataclasses import dataclass
    from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


    @dataclass
    class SafetyCheckResult:
        is_safe: bool
        risk_level: Optional[str] = None
        issues: Optional[List[str]] = None


    @runtime_checkable
    class RetrieverProtocol(Protocol):
        async def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:  # pragma: no cover - protocol
            ...

        def add_documents(self, docs: List[Dict[str, Any]]) -> None:  # pragma: no cover - protocol
            ...


    @runtime_checkable
    class LLMProvider(Protocol):
        async def agenerate(self, prompts: List[str]) -> Any:  # pragma: no cover - protocol
            ...

        async def ainvoke(self, prompt: str) -> Any:  # pragma: no cover - protocol
            ...

        def invoke(self, prompt: str) -> Any:  # pragma: no cover - protocol
            ...


    @runtime_checkable
    class RerankerProtocol(Protocol):
        async def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:  # pragma: no cover - protocol
            ...


    @runtime_checkable
    class SafetyCheckerProtocol(Protocol):
        def check_response(self, *, response: str, sources: List[str], query: str) -> SafetyCheckResult:  # pragma: no cover - protocol
            ...


    @runtime_checkable
    class AgentPolicyProtocol(Protocol):
        def decide(self, context: Dict[str, Any], state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:  # pragma: no cover - protocol
            ...


    __all__ = [
        "RetrieverProtocol",
        "LLMProvider",
        "RerankerProtocol",
        "SafetyCheckerProtocol",
        "AgentPolicyProtocol",
        "SafetyCheckResult",
    ]
    async def agenerate(self, prompts: List[str]) -> Any:  # pragma: no cover - protocol
