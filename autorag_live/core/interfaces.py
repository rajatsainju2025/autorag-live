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
    async def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        ...

    def add_documents(self, docs: List[Dict[str, Any]]) -> None:
        ...


@runtime_checkable
class LLMProvider(Protocol):
    async def agenerate(self, prompts: List[str]) -> Any:
        ...

    async def ainvoke(self, prompt: str) -> Any:
        ...

    def invoke(self, prompt: str) -> Any:
        ...


@runtime_checkable
class RerankerProtocol(Protocol):
    async def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ...


@runtime_checkable
class SafetyCheckerProtocol(Protocol):
    def check_response(self, *, response: str, sources: List[str], query: str) -> SafetyCheckResult:
        ...


@runtime_checkable
class AgentPolicyProtocol(Protocol):
    def decide(
        self, context: Dict[str, Any], state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        ...


__all__ = [
    "RetrieverProtocol",
    "LLMProvider",
    "RerankerProtocol",
    "SafetyCheckerProtocol",
    "AgentPolicyProtocol",
    "SafetyCheckResult",
]
