"""
Generic GameAdapter framework for OpenSpiel games.

A GameAdapter wraps an OpenSpiel game and exposes two things:
  1. create_env()       — creates a fresh rl_environment.Environment
  2. serialize_state()  — converts an env + time_step into a JSON-friendly dict

Adapters are registered by name so the server can look them up dynamically.
"""

from abc import ABC, abstractmethod
from typing import Any

from open_spiel.python import rl_environment


class GameAdapter(ABC):
    @abstractmethod
    def create_env(self) -> rl_environment.Environment:
        """Return a fresh environment for this game configuration."""
        ...

    @abstractmethod
    def serialize_state(
        self,
        env: rl_environment.Environment,
        time_step: rl_environment.TimeStep,
    ) -> dict[str, Any]:
        """Convert env + time_step into a JSON-serializable dict."""
        ...

    def preview_action(
        self,
        env: rl_environment.Environment,
        time_step: rl_environment.TimeStep,
        player: int,
        action_id: int,
    ) -> dict[str, Any] | None:
        """Return a synthetic state showing the action applied without advancing
        the game, or None if no preview is needed.

        Used by the streaming layer to display a card in the trick before
        resolution — e.g. the 4th card in Oh Hell, which is normally cleared
        atomically when env.step() resolves the trick.
        """
        return None


_REGISTRY: dict[str, type[GameAdapter]] = {}


def register(name: str):
    """Class decorator that registers a GameAdapter under a given name."""
    def decorator(cls: type[GameAdapter]) -> type[GameAdapter]:
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_adapter(name: str) -> type[GameAdapter]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown game: {name!r}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]
