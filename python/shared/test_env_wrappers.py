"""Tests for shared environment wrappers."""

import unittest

from open_spiel.python import rl_environment

from shared.env_wrappers import CappedTricksEnv


def _make_env(max_tricks: int | None = None):
    env = rl_environment.Environment(
        "oh_hell",
        players=4,
        num_cards_per_suit=13,
        num_suits=4,
        off_bid_penalty=False,
        points_per_trick=1,
    )
    if max_tricks is not None:
        return CappedTricksEnv(env, max_tricks)
    return env


def _num_tricks(env) -> int:
    for line in str(env.get_state).splitlines():
        if line.startswith("Num Total Tricks:"):
            return int(line.split(":", 1)[1].strip())
    raise ValueError("Num Total Tricks not found in state string")


class TestCappedTricksEnv(unittest.TestCase):
    def test_reset_respects_cap(self):
        env = _make_env(max_tricks=3)
        for _ in range(20):
            env.reset()
            self.assertLessEqual(_num_tricks(env), 3)

    def test_cap_of_one_always_gives_one_trick(self):
        env = _make_env(max_tricks=1)
        for _ in range(10):
            env.reset()
            self.assertEqual(_num_tricks(env), 1)

    def test_step_proxied(self):
        env = _make_env(max_tricks=3)
        ts = env.reset()
        action = ts.observations["legal_actions"][ts.observations["current_player"]][0]
        ts2 = env.step([action])
        self.assertIsNotNone(ts2)

    def test_action_spec_proxied(self):
        env = _make_env(max_tricks=3)
        env.reset()
        spec = env.action_spec()
        self.assertIn("num_actions", spec)

    def test_observation_spec_proxied(self):
        env = _make_env(max_tricks=3)
        env.reset()
        spec = env.observation_spec()
        self.assertIn("info_state", spec)

    def test_get_state_proxied(self):
        env = _make_env(max_tricks=3)
        env.reset()
        self.assertIsNotNone(env.get_state)

    def test_num_players_proxied(self):
        env = _make_env(max_tricks=3)
        self.assertEqual(env.num_players, 4)

    def test_full_game_completes(self):
        env = _make_env(max_tricks=3)
        ts = env.reset()
        while not ts.last():
            player = ts.observations["current_player"]
            action = ts.observations["legal_actions"][player][0]
            ts = env.step([action])
        self.assertTrue(ts.last())


if __name__ == "__main__":
    unittest.main()
