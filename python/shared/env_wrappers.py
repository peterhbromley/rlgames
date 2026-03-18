"""
Environment wrappers for OpenSpiel rl_environment.Environment.
"""


class CappedTricksEnv:
    """Wraps an Oh Hell environment and re-rolls reset() until the dealt hand
    has at most max_tricks tricks.

    This keeps the full deck intact (realistic suit distributions) while
    preventing excessively long hands during training and serving.

    All attributes and methods are proxied to the underlying environment so
    this wrapper is a drop-in replacement anywhere an Environment is expected.
    """

    def __init__(self, env, max_tricks: int) -> None:
        self._env = env
        self._max_tricks = max_tricks

    def reset(self):
        ts = self._env.reset()
        while self._num_tricks() > self._max_tricks:
            ts = self._env.reset()
        return ts

    def step(self, actions):
        return self._env.step(actions)

    def _num_tricks(self) -> int:
        for line in str(self._env.get_state).splitlines():
            if line.startswith("Num Total Tricks:"):
                return int(line.split(":", 1)[1].strip())
        return 0

    def __getattr__(self, name):
        return getattr(self._env, name)
