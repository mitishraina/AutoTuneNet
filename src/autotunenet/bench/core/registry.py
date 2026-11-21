"""
Name -> implementation registries for methods, tasks and scenarios.

Adding one is a new module plus a decorator; the runner never changes:

    from autotunenet.bench.core.registry import METHODS

    @METHODS.register("my_method")
    class MyMethod(SingleRunMethod):
        ...
"""
from typing import Dict, Generic, List, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, T] = {}

    def register(self, name: str):
        def decorator(obj: T) -> T:
            if name in self._items:
                raise ValueError(f"{self.kind} '{name}' is already registered")
            self._items[name] = obj
            return obj
        return decorator

    def add(self, name: str, obj: T) -> T:
        return self.register(name)(obj)

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(
                f"Unknown {self.kind} '{name}'. Available: {', '.join(self.names()) or 'none'}"
            ) from None

    def names(self) -> List[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items


METHODS: Registry = Registry("method")
TASKS: Registry = Registry("task")
SCENARIOS: Registry = Registry("scenario")


def load_plugins() -> None:
    """Import the built-in tasks, scenarios and methods so they register themselves."""
    import autotunenet.bench.methods  # noqa: F401
    import autotunenet.bench.scenarios  # noqa: F401
    import autotunenet.bench.tasks  # noqa: F401
