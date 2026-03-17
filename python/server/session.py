"""
Session manager for web gameplay.

One SessionManager is instantiated per game type at server startup.
It holds a single shared agent (eval-only) and a pool of active game sessions.

Design notes:
  - human_players is a frozenset of player-id ints.  Any mix of human and
    agent players is supported (e.g. {0} for solo play, {0,1} for two humans).
  - stream_action is the primary action interface: it validates the action
    synchronously, then returns a generator that yields SSE event dicts as the
    game advances step by step.  The route handler adds asyncio.sleep() between
    yields to control pacing.
  - apply_action is a thin wrapper around stream_action kept for test
    compatibility; it collects all events into a (state, transitions) tuple.
  - new_session still uses _advance_agents (list-based) because the bidding
    phase never resolves tricks mid-animation, so approach A is fine there.
  - The shared agent is safe for concurrent eval use: in is_evaluation=True mode
    the DQN never writes to its replay buffer or updates weights, and Python's
    GIL serialises any concurrent step() calls within a single process.
"""

import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from server.games.adapter import GameAdapter


@dataclass
class Session:
    id: str
    env: Any                        # rl_environment.Environment
    human_players: frozenset[int]
    time_step: Any                  # rl_environment.TimeStep
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)


class SessionError(Exception):
    pass


class SessionManager:
    SESSION_TTL = 3600  # seconds before an idle session is evicted

    def __init__(self, agent: Any, adapter: GameAdapter) -> None:
        """
        Args:
            agent:   Trained agent with a .step(time_step, is_evaluation) interface.
                     Shared read-only across all sessions.
            adapter: GameAdapter used to create envs and serialize state.
                     Shared across sessions (serialize_state is stateless).
        """
        self._agent = agent
        self._adapter = adapter
        self._sessions: dict[str, Session] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def new_session(
        self, human_players: list[int]
    ) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
        """
        Start a new game.

        Returns:
            (session_id, state, transitions) where state is the serialized
            GameState at the first human decision point and transitions captures
            any agent turns that occurred before the first human turn.
        """
        env = self._adapter.create_env()
        time_step = env.reset()
        time_step, transitions = self._advance_agents(env, time_step, frozenset[int](human_players))

        session_id = str(uuid.uuid4())
        self._sessions[session_id] = Session(
            id=session_id,
            env=env,
            human_players=frozenset(human_players),
            time_step=time_step,
        )
        return session_id, self._adapter.serialize_state(env, time_step), transitions

    def get_state(self, session_id: str) -> dict[str, Any]:
        """Return the current serialized state without advancing the game."""
        session = self._get(session_id)
        return self._adapter.serialize_state(session.env, session.time_step)

    def stream_action(self, session_id: str, action_id: int) -> Iterator[dict[str, Any]]:
        """
        Validate the action synchronously, then return a generator that yields
        SSE event dicts as the game advances.

        Each event is one of:
          {"type": "transition", "player": int, "state": dict}
          {"type": "final",      "player": None, "state": dict}

        Validation errors raise SessionError before the generator is returned,
        so the route handler can convert them to HTTP errors immediately.
        """
        session = self._get(session_id)
        current_player = session.time_step.observations["current_player"]

        if current_player not in session.human_players:
            raise SessionError(
                f"It is player {current_player}'s turn, which is not a human player."
            )

        legal = session.time_step.observations["legal_actions"][current_player]
        if action_id not in legal:
            raise SessionError(
                f"Action {action_id} is not legal for player {current_player}. "
                f"Legal actions: {legal}"
            )

        return self._stream_events(session, current_player, action_id)

    def apply_action(
        self, session_id: str, action_id: int
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Wrapper around stream_action for test compatibility."""
        transitions: list[dict[str, Any]] = []
        final_state: dict[str, Any] = {}
        for event in self.stream_action(session_id, action_id):
            if event["type"] == "final":
                final_state = event["state"]
            else:
                transitions.append({"player": event["player"], "state": event["state"]})
        return final_state, transitions

    def delete_session(self, session_id: str) -> None:
        """Remove a session. No-op if already gone."""
        self._sessions.pop(session_id, None)

    def cleanup_expired(self) -> int:
        """Evict sessions idle longer than SESSION_TTL. Returns count removed."""
        cutoff = time.time() - self.SESSION_TTL
        expired = [sid for sid, s in self._sessions.items() if s.last_active < cutoff]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _stream_events(
        self, session: Session, acting_player: int, action_id: int
    ) -> Iterator[dict[str, Any]]:
        """
        Generator that steps the game forward and yields one event per move.

        Emits a "transition" event for the human's own move and for each
        subsequent agent turn, then a "final" event when it reaches the next
        human decision point or terminal state.
        """
        def step_and_yield(player: int, action: int, ts: Any) -> Any:
            """Emit a preview if the action completes a trick, then step and emit the result."""
            preview = self._adapter.preview_action(session.env, ts, player, action)
            if preview:
                yield {"type": "transition", "player": player, "state": preview}
            ts = session.env.step([action])
            yield {"type": "transition", "player": player, "state": self._adapter.serialize_state(session.env, ts)}
            return ts

        # Human's move
        time_step = yield from step_and_yield(acting_player, action_id, session.time_step)

        # Consecutive agent turns
        while not time_step.last():
            agent_player = time_step.observations["current_player"]
            if agent_player in session.human_players:
                break
            agent_out = self._agent.step(time_step, is_evaluation=True)
            time_step = yield from step_and_yield(agent_player, agent_out.action, time_step)

        if time_step.last():
            self._agent.step(time_step, is_evaluation=True)

        session.time_step = time_step
        session.last_active = time.time()

        yield {
            "type": "final",
            "player": None,
            "state": self._adapter.serialize_state(session.env, time_step),
        }

    def _advance_agents(
        self,
        env: Any,
        time_step: Any,
        human_players: frozenset[int],
    ) -> tuple[Any, list[dict[str, Any]]]:
        """
        Used by new_session: drives the env through consecutive agent turns
        until the first human decision point or terminal.

        Returns (final_time_step, transitions) where each transition is
        {"player": int, "state": dict}.
        """
        transitions: list[dict[str, Any]] = []
        while not time_step.last():
            current_player = time_step.observations["current_player"]
            if current_player in human_players:
                break
            agent_out = self._agent.step(time_step, is_evaluation=True)
            time_step = env.step([agent_out.action])
            transitions.append({
                "player": current_player,
                "state": self._adapter.serialize_state(env, time_step),
            })

        if time_step.last():
            self._agent.step(time_step, is_evaluation=True)

        return time_step, transitions

    def _get(self, session_id: str) -> Session:
        session = self._sessions.get(session_id)
        if session is None:
            raise SessionError(f"Session not found: {session_id!r}")
        return session
