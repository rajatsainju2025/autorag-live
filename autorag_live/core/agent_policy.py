from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class AgentPolicy(ABC):
    """Abstract interface for agent decision policies.

    Implementations should be small, testable, and stateless where possible.
    This interface lets different agent policies be swapped into pipelines
    without changing the orchestration code.
    """

    @abstractmethod
    def decide(
        self, context: Dict[str, Any], state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Return an action dict given a context and optional state.

        The returned dict should contain a minimal, well-documented set of
        keys (e.g. 'action', 'params').
        """
        raise NotImplementedError()


__all__ = ["AgentPolicy"]
