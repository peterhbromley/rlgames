"""Tests for SessionManager.

A lightweight mock agent always picks the first legal action, so these tests
run without requiring a trained checkpoint.
"""

import unittest
from unittest.mock import MagicMock

from server.games.oh_hell import OhHellAdapter
from server.session import SessionManager, SessionError


# ---------------------------------------------------------------------------
# Mock agent
# ---------------------------------------------------------------------------

class _FirstActionAgent:
    """Always picks the first legal action for the current player."""

    def step(self, time_step, is_evaluation: bool = False):
        if time_step.last():
            return None
        player = time_step.observations["current_player"]
        action = time_step.observations["legal_actions"][player][0]
        out = MagicMock()
        out.action = action
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager(human_players=None):
    adapter = OhHellAdapter()
    agent = _FirstActionAgent()
    manager = SessionManager(agent=agent, adapter=adapter)
    if human_players is None:
        human_players = [0]
    session_id, state, _ = manager.new_session(human_players)
    return manager, session_id, state


# ---------------------------------------------------------------------------
# SessionManager.new_session
# ---------------------------------------------------------------------------

class TestNewSession(unittest.TestCase):
    def test_returns_session_id_and_state(self):
        manager, sid, state = _make_manager()
        self.assertIsInstance(sid, str)
        self.assertIsInstance(state, dict)

    def test_session_ids_are_unique(self):
        adapter = OhHellAdapter()
        manager = SessionManager(agent=_FirstActionAgent(), adapter=adapter)
        ids = {manager.new_session([0])[0] for _ in range(10)}
        self.assertEqual(len(ids), 10)

    def test_initial_state_is_human_turn(self):
        # With human_players=[0], the first state should be player 0's turn.
        # Because Oh Hell deals randomly, the first bidder might not be player 0,
        # so agents should have been advanced until it is.
        manager, sid, state = _make_manager(human_players=[0])
        self.assertEqual(state["current_player"], 0)

    def test_initial_state_correct_phase(self):
        _, _, state = _make_manager()
        self.assertIn(state["phase"], ("bidding", "playing"))

    def test_all_human_players_no_agent_advance(self):
        # With all 4 players human, new_session should return immediately
        # after env.reset() — whoever the first player is.
        adapter = OhHellAdapter()
        manager = SessionManager(agent=_FirstActionAgent(), adapter=adapter)
        _, state, _ = manager.new_session([0, 1, 2, 3])
        self.assertIn(state["current_player"], [0, 1, 2, 3])


# ---------------------------------------------------------------------------
# SessionManager.get_state
# ---------------------------------------------------------------------------

class TestGetState(unittest.TestCase):
    def test_get_state_matches_new_session_state(self):
        manager, sid, initial_state = _make_manager()
        fetched_state = manager.get_state(sid)
        self.assertEqual(initial_state, fetched_state)

    def test_get_state_unknown_session_raises(self):
        manager, _, _ = _make_manager()
        with self.assertRaises(SessionError):
            manager.get_state("nonexistent")


# ---------------------------------------------------------------------------
# SessionManager.apply_action
# ---------------------------------------------------------------------------

class TestApplyAction(unittest.TestCase):
    def test_apply_legal_action_returns_state(self):
        manager, sid, state = _make_manager()
        action = state["legal_actions"][0]["id"]
        new_state, transitions = manager.apply_action(sid, action)
        self.assertIsInstance(new_state, dict)
        self.assertIsInstance(transitions, list)

    def test_apply_action_advances_to_next_human_turn(self):
        # After applying an action, it should be a human player's turn again
        # (or game over).
        manager, sid, state = _make_manager(human_players=[0])
        action = state["legal_actions"][0]["id"]
        new_state, _ = manager.apply_action(sid, action)
        self.assertIn(new_state["phase"], ("bidding", "playing", "terminal"))
        if new_state["phase"] != "terminal":
            self.assertEqual(new_state["current_player"], 0)

    def test_apply_illegal_action_raises(self):
        manager, sid, state = _make_manager()
        legal_ids = {a["id"] for a in state["legal_actions"]}
        # Find any action ID that's not legal.
        illegal = next(i for i in range(60) if i not in legal_ids)
        with self.assertRaises(SessionError):
            manager.apply_action(sid, illegal)

    def test_apply_action_wrong_player_raises(self):
        # human_players=[1], so player 0 is an agent; trying to submit an action
        # when it's not the human's turn should raise.
        adapter = OhHellAdapter()
        manager = SessionManager(agent=_FirstActionAgent(), adapter=adapter)
        # Peek at the env directly: if first bidder != player 1, agent was
        # advanced, so current_player should be 1.
        sid, state, _ = manager.new_session([1])
        self.assertEqual(state["current_player"], 1)
        # Now submit with wrong player — apply_action checks the current player,
        # not who you claim to be, so submitting a valid action is fine.
        # To trigger the error, we need the state to show it's an agent's turn.
        # That can't happen after new_session, so we test via a second session
        # where we observe an agent turn mid-game — instead just test the error
        # path directly by patching time_step.
        session = manager._sessions[sid]
        session.time_step = MagicMock()
        session.time_step.last.return_value = False
        session.time_step.observations = {
            "current_player": 0,  # agent's turn
            "legal_actions": {0: [52]},
        }
        with self.assertRaises(SessionError):
            manager.apply_action(sid, 52)

    def test_transitions_contain_agent_moves(self):
        # Transitions now always start with the acting player's post-move state,
        # followed by one entry per agent turn.
        manager, sid, state = _make_manager(human_players=[0])
        action = state["legal_actions"][0]["id"]
        _, transitions = manager.apply_action(sid, action)
        self.assertGreater(len(transitions), 0)
        # First transition is the human's own move (player 0).
        self.assertEqual(transitions[0]["player"], 0)
        # Remaining transitions are agent moves (players != 0).
        for t in transitions[1:]:
            self.assertNotEqual(t["player"], 0)
        for t in transitions:
            self.assertIsInstance(t["state"], dict)
            self.assertIn("phase", t["state"])

    def test_apply_action_unknown_session_raises(self):
        manager, _, _ = _make_manager()
        with self.assertRaises(SessionError):
            manager.apply_action("nonexistent", 0)

    def test_full_game_reaches_terminal(self):
        manager, sid, state = _make_manager(human_players=[0])
        while state["phase"] != "terminal":
            action = state["legal_actions"][0]["id"]
            state, _ = manager.apply_action(sid, action)
        self.assertIsNotNone(state["scores"])
        self.assertEqual(len(state["scores"]), 4)

    def test_state_updates_after_action(self):
        manager, sid, state = _make_manager()
        action = state["legal_actions"][0]["id"]
        new_state, _ = manager.apply_action(sid, action)
        # Something in the state must have changed.
        self.assertNotEqual(state, new_state)


# ---------------------------------------------------------------------------
# Multiple human players
# ---------------------------------------------------------------------------

class TestMultipleHumanPlayers(unittest.TestCase):
    def test_two_human_players(self):
        adapter = OhHellAdapter()
        manager = SessionManager(agent=_FirstActionAgent(), adapter=adapter)
        sid, state, _ = manager.new_session([0, 1])
        self.assertIn(state["current_player"], [0, 1])

    def test_two_humans_never_skip_to_agent_turn(self):
        adapter = OhHellAdapter()
        manager = SessionManager(agent=_FirstActionAgent(), adapter=adapter)
        sid, state, _ = manager.new_session([0, 1])

        # Play through the whole game; current_player must always be 0 or 1.
        while state["phase"] != "terminal":
            self.assertIn(state["current_player"], [0, 1])
            action = state["legal_actions"][0]["id"]
            state, _ = manager.apply_action(sid, action)

    def test_all_agent_players_game_completes_on_new_session(self):
        # With no human players, the entire game is played out during new_session.
        adapter = OhHellAdapter()
        manager = SessionManager(agent=_FirstActionAgent(), adapter=adapter)
        sid, state, _ = manager.new_session([])
        self.assertEqual(state["phase"], "terminal")


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

class TestSessionLifecycle(unittest.TestCase):
    def test_delete_session(self):
        manager, sid, _ = _make_manager()
        manager.delete_session(sid)
        with self.assertRaises(SessionError):
            manager.get_state(sid)

    def test_delete_nonexistent_is_noop(self):
        manager, _, _ = _make_manager()
        manager.delete_session("does-not-exist")  # should not raise

    def test_cleanup_expired_removes_old_sessions(self):
        manager, sid, _ = _make_manager()
        # Backdate last_active so the session looks stale.
        manager._sessions[sid].last_active = 0.0
        removed = manager.cleanup_expired()
        self.assertEqual(removed, 1)
        with self.assertRaises(SessionError):
            manager.get_state(sid)

    def test_cleanup_expired_keeps_fresh_sessions(self):
        manager, sid, _ = _make_manager()
        removed = manager.cleanup_expired()
        self.assertEqual(removed, 0)
        manager.get_state(sid)  # should not raise
