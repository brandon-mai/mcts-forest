from typing import Any, Tuple, Dict, List, TypeVar, Protocol

S = TypeVar("S")
A = TypeVar("A")

class EnvBase:
    """Base class for environment adapters."""
    def reset(self, seed: Any = None) -> S: ...
    def step(self, action: A) -> Tuple[S, float, bool, bool, Dict]: ...
    def get_state(self) -> S: ...
    def set_state(self, state: S) -> None: ...
    def get_legal_actions(self, state: S) -> List[A]: ...
    def close(self) -> None: ...

    def get_name(self) -> str:
        """Returns a sanitized name for the environment."""
        import re
        name = self.__class__.__name__
        # CamelCase to snake_case
        name = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
        # Remove 'adapter' or 'env' suffixes if present
        name = re.sub(r'(_adapter|_env)$', '', name)
        return name
