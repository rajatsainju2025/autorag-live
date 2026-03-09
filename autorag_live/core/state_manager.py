from typing import Any, Dict, Optional


class StateManager:
    """Lightweight in-memory state manager for agentic RAG components.

    Intended as a small, modular utility that components can use to store
    and retrieve execution state. This keeps state handling decoupled from
    component logic and is suitable as a starting point for replacing with
    persistent or distributed stores later.
    """

    def __init__(self) -> None:
        self._state: Dict[str, Any] = {}

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return self._state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._state[key] = value

    def update(self, mapping: Dict[str, Any]) -> None:
        self._state.update(mapping)

    def snapshot(self) -> Dict[str, Any]:
        return dict(self._state)


__all__ = ["StateManager"]
