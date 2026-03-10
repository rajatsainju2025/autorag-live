from typing import Any, Callable, Dict


class DIRegistry:
    """Tiny dependency injection registry.

    Usage:
        registry = DIRegistry()
        registry.register("mock", lambda: MockImpl())
        inst = registry.resolve("mock")
    """

    def __init__(self) -> None:
        self._factories: Dict[str, Callable[[], Any]] = {}
        self._singletons: Dict[str, Any] = {}
        self._singleton_flags: Dict[str, bool] = {}

    def register(self, name: str, factory: Callable[[], Any], singleton: bool = False) -> None:
        self._factories[name] = factory
        self._singleton_flags[name] = singleton

    def resolve(self, name: str) -> Any:
        if name not in self._factories:
            raise KeyError(name)

        if self._singleton_flags.get(name):
            if name not in self._singletons:
                self._singletons[name] = self._factories[name]()
            return self._singletons[name]

        return self._factories[name]()


registry = DIRegistry()


__all__ = ["DIRegistry", "registry"]
