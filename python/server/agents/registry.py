"""
Agent loader registry.

Agent loaders register themselves with @register(game, agent_type).
Each loader is a callable (checkpoint_path: str, env) -> agent.
"""

from typing import Any, Callable

_REGISTRY: dict[tuple[str, str], Callable] = {}


def register(game: str, agent_type: str):
    """Decorator that registers an agent loader for a (game, agent_type) pair."""
    def decorator(fn: Callable) -> Callable:
        _REGISTRY[(game, agent_type)] = fn
        return fn
    return decorator


def load(
    game: str,
    agent_type: str,
    checkpoint_path: str,
    env: Any,
    agent_config: dict | None = None,
) -> Any:
    key = (game, agent_type)
    if key not in _REGISTRY:
        raise KeyError(
            f"No agent loader for {key!r}. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key](checkpoint_path, env, agent_config)


def registered() -> list[tuple[str, str]]:
    return sorted(_REGISTRY)
