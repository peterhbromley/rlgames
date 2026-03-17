"""Tests for the Oh Hell game adapter."""

import unittest
from dataclasses import asdict

from server.games.oh_hell import (
    OhHellAdapter,
    _bid_action,
    _card,
    _card_action,
    _parse_state_string,
    _parse_tricks_lines,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BIDDING_STATE = """\
Phase: Bid
Num Total Tricks: 2
Dealer: 0
Player: 0
    C: 3 2
    D:
    S: J T
    H:
Player: 1
    C:
    D: A K
    S:
    H: 5 4
Player: 2
    C: Q
    D: 6
    S:
    H: 9 8
Player: 3
    C: 7 6
    D:
    S: Q
    H:
Trump: C4
Bids:        -1 -1 -1 -1
Tricks Won:    0  0  0  0"""

PLAYING_STATE = """\
Phase: Play
Num Total Tricks: 2
Dealer: 0
Player: 0
    C: 3
    D:
    S: J T
    H:
Player: 1
    C:
    D: A K
    S:
    H: 5 4
Player: 2
    C: Q
    D: 6
    S:
    H: 9 8
Player: 3
    C: 7 6
    D:
    S: Q
    H:
Trump: C4
Bids:        1 0 2 0
Tricks Won:    0  0  0  0"""

# Player 1 led trick 1 (C3); player 2 won with H5 (trump H9).
# Trick 2 in progress: player 2 led DJ, player 3 played D6.
# Column layout (4 players): player = (col // 3) % 4
#   Trick 1 row "   C3 H5 D3 C4 ": cols 3,6,9,12 → players 1,2,3,0
#   Trick 2 row "      DJ D6    ":  cols 6,9     → players 2,3
PLAYING_WITH_TRICKS_STATE = """\
Phase: Play
Num Total Tricks: 2
Dealer: 0
Player: 0
    C: 4
    D:
    S:
    H:
Player: 1
    C: 9
    D:
    S:
    H:
Player: 2
    C:
    D: J
    S:
    H:
Player: 3
    C:
    D: 6
    S:
    H:
Trump: H9
Tricks:
0  1  2  3  0  1  2
   C3 H5 D3 C4
      DJ D6
Bids:        0 1 2 0
Tricks Won:    0 0 1 0"""

GAMEOVER_STATE = """\
Phase: GameOver
Num Total Tricks: 2
Dealer: 0
Player: 0
    C: 4
    D: 4
    S:
    H:
Player: 1
    C: 93
    D:
    S:
    H:
Player: 2
    C:
    D: J
    S:
    H: 5
Player: 3
    C:
    D: 63
    S:
    H:
Trump: H9
Tricks:
0  1  2  3  0  1  2
   C3 H5 D3 C4
      DJ D6 D4 C9
Bids:        0 0 0 0
Tricks Won:    0 0 2 0
Score:        10 10 2 10"""


# ---------------------------------------------------------------------------
# Unit tests: card / action helpers
# ---------------------------------------------------------------------------

class TestCardConstruction(unittest.TestCase):
    def test_action_id_formula(self):
        # rank_index * 4 + suit_index; suit: C=0 D=1 S=2 H=3
        self.assertEqual(_card("C", "2").id, 0)   # 0*4 + 0
        self.assertEqual(_card("D", "2").id, 1)   # 0*4 + 1
        self.assertEqual(_card("S", "2").id, 2)   # 0*4 + 2
        self.assertEqual(_card("H", "2").id, 3)   # 0*4 + 3
        self.assertEqual(_card("C", "A").id, 48)  # 12*4 + 0
        self.assertEqual(_card("D", "A").id, 49)  # 12*4 + 1
        self.assertEqual(_card("S", "A").id, 50)  # 12*4 + 2
        self.assertEqual(_card("H", "A").id, 51)  # 12*4 + 3

    def test_rank_display(self):
        self.assertEqual(_card("C", "T").rank, "10")
        self.assertEqual(_card("H", "J").rank, "J")
        self.assertEqual(_card("S", "2").rank, "2")

    def test_suit_fields(self):
        c = _card("H", "J")
        self.assertEqual(c.suit, "hearts")
        self.assertEqual(c.label, "HJ")

    def test_label_ten(self):
        self.assertEqual(_card("C", "T").label, "C10")


class TestActionParsing(unittest.TestCase):
    def test_card_action(self):
        a = _card_action(39, "HJ")
        self.assertEqual(a.id, 39)
        self.assertEqual(a.label, "HJ")
        self.assertEqual(a.type, "card")

    def test_card_action_ten(self):
        a = _card_action(34, "CT")
        self.assertEqual(a.label, "C10")

    def test_bid_action(self):
        a = _bid_action(52, "0")
        self.assertEqual(a.id, 52)
        self.assertEqual(a.label, "Bid 0")
        self.assertEqual(a.type, "bid")

    def test_bid_action_nonzero(self):
        a = _bid_action(54, "2")
        self.assertEqual(a.label, "Bid 2")


# ---------------------------------------------------------------------------
# Unit tests: state string parser
# ---------------------------------------------------------------------------

class TestParseStateString(unittest.TestCase):
    def setUp(self):
        self.bidding = _parse_state_string(BIDDING_STATE, 4)
        self.playing = _parse_state_string(PLAYING_STATE, 4)

    def test_phase_bidding(self):
        self.assertEqual(self.bidding["phase"], "bid")

    def test_phase_playing(self):
        self.assertEqual(self.playing["phase"], "play")

    def test_num_tricks(self):
        self.assertEqual(self.bidding["num_tricks"], 2)

    def test_trump_card(self):
        trump = self.bidding["trump_card"]
        self.assertIsNotNone(trump)
        self.assertEqual(trump.rank, "4")
        self.assertEqual(trump.suit, "clubs")
        self.assertEqual(trump.label, "C4")

    def test_bids_all_none_during_bidding(self):
        self.assertEqual(self.bidding["bids"], [None, None, None, None])

    def test_bids_set_during_playing(self):
        self.assertEqual(self.playing["bids"], [1, 0, 2, 0])

    def test_tricks_won_all_zero(self):
        self.assertEqual(self.bidding["tricks_won"], [0, 0, 0, 0])

    def test_player0_hand(self):
        labels = {c.label for c in self.bidding["hands"][0]}
        self.assertEqual(labels, {"C3", "C2", "SJ", "S10"})

    def test_player1_hand(self):
        labels = {c.label for c in self.bidding["hands"][1]}
        self.assertEqual(labels, {"DA", "DK", "H5", "H4"})

    def test_hand_card_ids_are_unique_per_player(self):
        for hand in self.bidding["hands"]:
            ids = [c.id for c in hand]
            self.assertEqual(len(ids), len(set(ids)), "Duplicate card IDs in hand")

    def test_playing_state_hand_reduced(self):
        # Player 0 played one card (C2) in the transition to PLAYING_STATE.
        labels = {c.label for c in self.playing["hands"][0]}
        self.assertNotIn("C2", labels)
        self.assertIn("C3", labels)

    def test_empty_suit_produces_no_cards(self):
        # Player 0 has no diamonds or hearts in the bidding state.
        diamonds = [c for c in self.bidding["hands"][0] if c.suit == "diamonds"]
        hearts = [c for c in self.bidding["hands"][0] if c.suit == "hearts"]
        self.assertEqual(diamonds, [])
        self.assertEqual(hearts, [])


# ---------------------------------------------------------------------------
# Unit tests: tricks section parser
# ---------------------------------------------------------------------------

class TestParseTricksLines(unittest.TestCase):
    def _lines(self, state_str: str) -> list[str]:
        """Extract raw lines from the Tricks: block of a state string."""
        lines = state_str.splitlines()
        result, in_tricks = [], False
        for line in lines:
            if line.strip() == "Tricks:":
                in_tricks = True
                continue
            if in_tricks:
                if line.strip().startswith("Bids:"):
                    break
                result.append(line)
        return result

    def test_complete_trick_one_player(self):
        # Trick 1: player 1 led C3, players 2,3,0 follow — all 4 cards present.
        lines = self._lines(PLAYING_WITH_TRICKS_STATE)
        tricks = _parse_tricks_lines(lines, 4)
        self.assertEqual(len(tricks[0]), 4)

    def test_trick1_player_assignment(self):
        lines = self._lines(PLAYING_WITH_TRICKS_STATE)
        tricks = _parse_tricks_lines(lines, 4)
        trick1 = {p: code for p, code in tricks[0]}
        self.assertEqual(trick1[1], "C3")  # player 1 led
        self.assertEqual(trick1[2], "H5")
        self.assertEqual(trick1[3], "D3")
        self.assertEqual(trick1[0], "C4")  # player 0 wrapped to col 12

    def test_incomplete_trick_detected(self):
        lines = self._lines(PLAYING_WITH_TRICKS_STATE)
        tricks = _parse_tricks_lines(lines, 4)
        # Trick 2 only has 2 cards (DJ by player 2, D6 by player 3).
        self.assertEqual(len(tricks[1]), 2)

    def test_trick2_player_assignment(self):
        lines = self._lines(PLAYING_WITH_TRICKS_STATE)
        tricks = _parse_tricks_lines(lines, 4)
        trick2 = {p: code for p, code in tricks[1]}
        self.assertEqual(trick2[2], "DJ")  # player 2 led trick 2
        self.assertEqual(trick2[3], "D6")

    def test_complete_game_two_full_tricks(self):
        lines = self._lines(GAMEOVER_STATE)
        tricks = _parse_tricks_lines(lines, 4)
        self.assertEqual(len(tricks), 2)
        self.assertTrue(all(len(t) == 4 for t in tricks))


class TestParseStateStringTricks(unittest.TestCase):
    def test_no_current_trick_during_bidding(self):
        result = _parse_state_string(BIDDING_STATE, 4)
        self.assertEqual(result["current_trick"], [])

    def test_current_trick_populated_during_play(self):
        result = _parse_state_string(PLAYING_WITH_TRICKS_STATE, 4)
        self.assertEqual(len(result["current_trick"]), 2)

    def test_current_trick_players(self):
        result = _parse_state_string(PLAYING_WITH_TRICKS_STATE, 4)
        players = [tc.player for tc in result["current_trick"]]
        self.assertEqual(players, [2, 3])

    def test_current_trick_card_labels(self):
        result = _parse_state_string(PLAYING_WITH_TRICKS_STATE, 4)
        labels = [tc.card.label for tc in result["current_trick"]]
        self.assertEqual(labels, ["DJ", "D6"])

    def test_no_current_trick_when_all_tricks_complete(self):
        result = _parse_state_string(GAMEOVER_STATE, 4)
        self.assertEqual(result["current_trick"], [])

    def test_gameover_phase_parsed(self):
        result = _parse_state_string(GAMEOVER_STATE, 4)
        self.assertEqual(result["phase"], "gameover")

    def test_bids_still_parsed_after_tricks_section(self):
        result = _parse_state_string(PLAYING_WITH_TRICKS_STATE, 4)
        self.assertEqual(result["bids"], [0, 1, 2, 0])

    def test_tricks_won_still_parsed_after_tricks_section(self):
        result = _parse_state_string(PLAYING_WITH_TRICKS_STATE, 4)
        self.assertEqual(result["tricks_won"], [0, 0, 1, 0])


# ---------------------------------------------------------------------------
# Integration tests: serialize_state against a live environment
# ---------------------------------------------------------------------------

class TestSerializeStateLive(unittest.TestCase):
    def setUp(self):
        self.adapter = OhHellAdapter()
        self.env = self.adapter.create_env()

    def _state(self, time_step):
        return self.adapter.serialize_state(self.env, time_step)

    def test_initial_state_is_bidding(self):
        ts = self.env.reset()
        state = self._state(ts)
        self.assertEqual(state["phase"], "bidding")

    def test_initial_legal_actions_are_bids(self):
        ts = self.env.reset()
        state = self._state(ts)
        self.assertTrue(len(state["legal_actions"]) > 0)
        self.assertTrue(all(a["type"] == "bid" for a in state["legal_actions"]))

    def test_initial_bid_labels(self):
        ts = self.env.reset()
        state = self._state(ts)
        labels = {a["label"] for a in state["legal_actions"]}
        # With num_tricks_fixed=2, bids are 0, 1, or 2.
        self.assertEqual(labels, {"Bid 0", "Bid 1", "Bid 2"})

    def test_initial_trump_card_present(self):
        ts = self.env.reset()
        state = self._state(ts)
        self.assertIsNotNone(state["trump_card"])
        self.assertIn("label", state["trump_card"])

    def test_initial_hands_dealt(self):
        ts = self.env.reset()
        state = self._state(ts)
        # Each player gets num_tricks_fixed (2) cards.
        for p in state["players"]:
            self.assertEqual(len(p["hand"]), 2)

    def test_initial_bids_all_null(self):
        ts = self.env.reset()
        state = self._state(ts)
        self.assertTrue(all(p["bid"] is None for p in state["players"]))

    def test_after_bid_phase_transitions_to_playing(self):
        ts = self.env.reset()
        state = self._state(ts)
        num_players = state["num_players"]

        # Submit bids for every player.
        for _ in range(num_players):
            bid_action = state["legal_actions"][0]["id"]
            ts = self.env.step([bid_action])
            state = self._state(ts)

        self.assertEqual(state["phase"], "playing")

    def test_playing_phase_legal_actions_are_cards(self):
        ts = self.env.reset()
        state = self._state(ts)
        num_players = state["num_players"]

        for _ in range(num_players):
            bid_action = state["legal_actions"][0]["id"]
            ts = self.env.step([bid_action])
            state = self._state(ts)

        self.assertTrue(all(a["type"] == "card" for a in state["legal_actions"]))

    def test_bids_recorded_after_bidding(self):
        ts = self.env.reset()
        state = self._state(ts)
        num_players = state["num_players"]

        for _ in range(num_players):
            bid_action = state["legal_actions"][0]["id"]
            ts = self.env.step([bid_action])
            state = self._state(ts)

        self.assertTrue(all(p["bid"] is not None for p in state["players"]))

    def test_full_game_reaches_terminal(self):
        ts = self.env.reset()
        state = self._state(ts)

        while not ts.last():
            action = state["legal_actions"][0]["id"]
            ts = self.env.step([action])
            state = self._state(ts)

        self.assertEqual(state["phase"], "terminal")

    def test_terminal_state_has_scores(self):
        ts = self.env.reset()
        state = self._state(ts)

        while not ts.last():
            action = state["legal_actions"][0]["id"]
            ts = self.env.step([action])
            state = self._state(ts)

        self.assertIsNotNone(state["scores"])
        self.assertEqual(len(state["scores"]), state["num_players"])

    def test_terminal_state_no_legal_actions(self):
        ts = self.env.reset()
        state = self._state(ts)

        while not ts.last():
            action = state["legal_actions"][0]["id"]
            ts = self.env.step([action])
            state = self._state(ts)

        self.assertEqual(state["legal_actions"], [])

    def test_hand_shrinks_as_cards_played(self):
        ts = self.env.reset()
        state = self._state(ts)
        num_players = state["num_players"]

        # Bid for all players.
        for _ in range(num_players):
            ts = self.env.step([state["legal_actions"][0]["id"]])
            state = self._state(ts)

        initial_hand_sizes = [len(p["hand"]) for p in state["players"]]

        # Play one card.
        ts = self.env.step([state["legal_actions"][0]["id"]])
        state = self._state(ts)

        new_hand_sizes = [len(p["hand"]) for p in state["players"]]
        total_before = sum(initial_hand_sizes)
        total_after = sum(new_hand_sizes)
        self.assertEqual(total_after, total_before - 1)

    def test_current_trick_empty_at_start_of_play(self):
        ts = self.env.reset()
        state = self._state(ts)
        num_players = state["num_players"]

        for _ in range(num_players):
            ts = self.env.step([state["legal_actions"][0]["id"]])
            state = self._state(ts)

        # No cards played yet in trick 1.
        self.assertEqual(state["current_trick"], [])

    def test_current_trick_grows_during_trick(self):
        ts = self.env.reset()
        state = self._state(ts)
        num_players = state["num_players"]

        for _ in range(num_players):
            ts = self.env.step([state["legal_actions"][0]["id"]])
            state = self._state(ts)

        sizes = []
        for _ in range(num_players):
            if ts.last():
                break
            ts = self.env.step([state["legal_actions"][0]["id"]])
            state = self._state(ts)
            sizes.append(len(state["current_trick"]))

        # Each successive state within a trick should have one more card,
        # then reset to 0 when the trick completes and a new one begins.
        self.assertIn(1, sizes)

    def test_current_trick_card_ids_are_valid(self):
        ts = self.env.reset()
        state = self._state(ts)
        num_players = state["num_players"]

        for _ in range(num_players):
            ts = self.env.step([state["legal_actions"][0]["id"]])
            state = self._state(ts)

        # Play one card to put something in the current trick.
        ts = self.env.step([state["legal_actions"][0]["id"]])
        state = self._state(ts)

        for tc in state["current_trick"]:
            self.assertIn("player", tc)
            self.assertIn("card", tc)
            self.assertGreaterEqual(tc["card"]["id"], 0)
            self.assertLess(tc["card"]["id"], 52)

    def test_state_is_json_serializable(self):
        import json
        ts = self.env.reset()
        state = self._state(ts)
        # Should not raise.
        json.dumps(state)

    def test_card_ids_in_hand_match_legal_action_ids_during_play(self):
        ts = self.env.reset()
        state = self._state(ts)
        num_players = state["num_players"]

        # Bid for all players.
        for _ in range(num_players):
            ts = self.env.step([state["legal_actions"][0]["id"]])
            state = self._state(ts)

        current = state["current_player"]
        hand_ids = {c["id"] for c in state["players"][current]["hand"]}
        legal_ids = {a["id"] for a in state["legal_actions"]}
        # Legal card actions must be a subset of the current player's hand.
        self.assertTrue(legal_ids.issubset(hand_ids))


if __name__ == "__main__":
    unittest.main()
