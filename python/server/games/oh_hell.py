"""
Oh Hell game adapter.

State string format emitted by OpenSpiel (oh_hell.cc ToString()):

    Phase: Bid
    Num Total Tricks: 2
    Dealer: 0
    Player: 0
        C: 3 2
        D:
        S: J T
        H:
    Player: 1
        ...
    Trump: C4
    Bids:        -1 -1 -1 -1
    Tricks Won:    0  0  0  0

action_to_string() format:
  Card plays → "<Suit><Rank>"  e.g. "D2", "HJ", "CA", "ST"
  Bids       → "<N>"           e.g. "0", "1", "2"

Card action IDs:  rank_index * 4 + suit_index
  suit_index: C=0, D=1, S=2, H=3
  rank_index: 2→0, 3→1, ..., 9→7, T→8, J→9, Q→10, K→11, A→12
Bid action IDs:  deck_size + bid_value
"""

from dataclasses import asdict, dataclass
from typing import Any, Optional

from open_spiel.python import rl_environment

from .adapter import GameAdapter, register


# ---------------------------------------------------------------------------
# Oh Hell state types
# ---------------------------------------------------------------------------

@dataclass
class Card:
    id: int           # OpenSpiel action ID
    rank: str         # "2"-"9", "10", "J", "Q", "K", "A"
    suit: str         # "clubs" | "diamonds" | "spades" | "hearts"
    label: str        # e.g. "C2", "HJ", "CA", "ST"


@dataclass
class TrickCard:
    player: int
    card: Card


@dataclass
class PlayerState:
    bid: Optional[int]   # None = not yet bid this round
    tricks_won: int
    hand: list[Card]     # empty list for hidden hands


@dataclass
class Action:
    id: int
    label: str           # e.g. "C2", "HJ", "CA", "ST", "Bid 2"
    type: str            # "play" | "bid"


@dataclass
class GameState:
    phase: str                      # "bidding" | "playing" | "terminal"
    current_player: int
    num_players: int
    num_tricks: int                 # tricks per player this round
    trump_card: Optional[Card]
    players: list[PlayerState]      # index = player id
    current_trick: list[TrickCard]
    scores: Optional[list[float]]   # None until terminal
    legal_actions: list[Action]

# ---------------------------------------------------------------------------
# Suit / rank lookup tables
# ---------------------------------------------------------------------------

_SUIT_INFO: dict[str, str] = {
    "C": "clubs",
    "D": "diamonds",
    "S": "spades",
    "H": "hearts",
}

_SUIT_INDEX: dict[str, int] = {"C": 0, "D": 1, "S": 2, "H": 3}
_SUIT_BY_INDEX: dict[int, str] = {v: k for k, v in _SUIT_INDEX.items()}

_RANK_ORDER = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
_RANK_INDEX: dict[str, int] = {r: i for i, r in enumerate(_RANK_ORDER)}

_RANK_DISPLAY: dict[str, str] = {r: r for r in _RANK_ORDER}
_RANK_DISPLAY["T"] = "10"

NUM_SUITS = 4


# ---------------------------------------------------------------------------
# Card construction helpers
# ---------------------------------------------------------------------------

def _card(suit_code: str, rank_code: str) -> Card:
    """Build a Card from OpenSpiel suit/rank codes."""
    suit_name = _SUIT_INFO[suit_code]
    rank_display = _RANK_DISPLAY[rank_code]
    action_id = _RANK_INDEX[rank_code] * NUM_SUITS + _SUIT_INDEX[suit_code]
    return Card(
        id=action_id,
        rank=rank_display,
        suit=suit_name,
        label=f"{suit_code}{rank_display}",
    )


def _card_action(action_id: int, label: str) -> Action:
    """Build an Action for a card play from action_to_string output (e.g. 'HJ')."""
    suit_code, rank_code = label[0], label[1]
    rank_display = _RANK_DISPLAY[rank_code]
    return Action(id=action_id, label=f"{suit_code}{rank_display}", type="play")


def _bid_action(action_id: int, label: str) -> Action:
    """Build an Action for a bid from action_to_string output (e.g. '2')."""
    return Action(id=action_id, label=f"Bid {label}", type="bid")


# ---------------------------------------------------------------------------
# State string parser
# ---------------------------------------------------------------------------

def _parse_tricks_lines(
    trick_lines: list[str], num_players: int
) -> list[list[tuple[int, str]]]:
    """
    Parse the body of the Tricks: section (excluding the "Tricks:" marker and
    the player-header line) into a list of tricks.

    Each trick is a list of (player_id, card_code) tuples in play order.
    Column layout: each slot is 3 chars wide; player = (col // 3) % num_players.
    When the lead player is not player 0 their card wraps past num_players-1,
    and the modulo resolves the correct player id.
    """
    # First line is the player-header (e.g. "0  1  2  3  0  1  2  ") — skip it.
    card_lines = trick_lines[1:]

    tricks = []
    for line in card_lines:
        trick: list[tuple[int, str]] = []
        col = 0
        while col + 1 < len(line):
            chunk = line[col : col + 2]
            if chunk[0] in _SUIT_INFO and chunk[1] in _RANK_INDEX:
                player_id = (col // 3) % num_players
                trick.append((player_id, chunk))
            col += 3
        if trick:
            tricks.append(trick)
    return tricks


def _parse_state_string(state_str: str, num_players: int) -> dict[str, Any]:
    """
    Parse the Oh Hell state string into a structured dict with keys:
      phase         : str  ("bid" | "play" | "gameover")
      num_tricks    : int | None
      trump_card    : Card | None
      hands         : list[list[Card]]  (indexed by player)
      bids          : list[int | None]  (None = not yet bid; -1 from state → None)
      tricks_won    : list[int]
      current_trick : list[TrickCard]   (empty during bidding phase)
    """
    result: dict[str, Any] = {
        "phase": None,
        "num_tricks": None,
        "trump_card": None,
        "hands": [[] for _ in range(num_players)],
        "bids": [None] * num_players,
        "tricks_won": [0] * num_players,
        "current_trick": [],
    }

    current_player: Optional[int] = None
    in_tricks_section = False
    tricks_lines: list[str] = []

    for raw_line in state_str.splitlines():
        line = raw_line.strip()

        if line.startswith("Phase:"):
            result["phase"] = line.split(":", 1)[1].strip().lower()
            in_tricks_section = False

        elif line.startswith("Num Total Tricks:"):
            result["num_tricks"] = int(line.split(":", 1)[1].strip())
            in_tricks_section = False

        elif line.startswith("Player:"):
            current_player = int(line.split(":", 1)[1].strip())
            in_tricks_section = False

        elif current_player is not None and len(line) >= 2 and line[0] in _SUIT_INFO and line[1] == ":":
            suit_code = line[0]
            rank_str = line[2:].strip()
            if not rank_str:
                continue
            # Ranks may be space-separated ("3 2 J") or concatenated ("32J").
            tokens = rank_str.split()
            rank_codes = tokens if all(len(t) == 1 for t in tokens) else list(rank_str.replace(" ", ""))
            for rc in rank_codes:
                if rc in _RANK_INDEX:
                    result["hands"][current_player].append(_card(suit_code, rc))

        elif line.startswith("Trump:"):
            trump_str = line.split(":", 1)[1].strip()
            if len(trump_str) >= 2:
                result["trump_card"] = _card(trump_str[0], trump_str[1])
            in_tricks_section = False

        elif line == "Tricks:":
            in_tricks_section = True
            current_player = None  # suit lines after this belong to Tricks, not a player

        elif in_tricks_section and line.startswith("Bids:"):
            # "Bids:" ends the Tricks section.
            in_tricks_section = False
            parts = line.split(":", 1)[1].split()
            for i, p in enumerate(parts[:num_players]):
                result["bids"][i] = None if p == "-1" else int(p)

        elif in_tricks_section:
            # Collect raw (unstripped) lines so column positions are preserved.
            tricks_lines.append(raw_line)

        elif line.startswith("Bids:"):
            parts = line.split(":", 1)[1].split()
            for i, p in enumerate(parts[:num_players]):
                result["bids"][i] = None if p == "-1" else int(p)

        elif line.startswith("Tricks Won:"):
            parts = line.split(":", 1)[1].split()
            for i, p in enumerate(parts[:num_players]):
                result["tricks_won"][i] = int(p)

    if tricks_lines:
        all_tricks = _parse_tricks_lines(tricks_lines, num_players)
        # The last trick is current if it has fewer cards than num_players.
        if all_tricks and len(all_tricks[-1]) < num_players:
            current = all_tricks[-1]
            result["current_trick"] = [
                TrickCard(player=p, card=_card(code[0], code[1]))
                for p, code in current
            ]

    return result


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

@register("oh_hell")
class OhHellAdapter(GameAdapter):
    DEFAULT_PARAMS = {
        "players": 4,
        "num_cards_per_suit": 13,
        "num_suits": 4,
        "num_tricks_fixed": 2,
        "off_bid_penalty": False,
        "points_per_trick": 1,
    }

    def __init__(self, params: dict | None = None, max_tricks: int | None = None) -> None:
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self._max_tricks = max_tricks

    def create_env(self):
        from shared.env_wrappers import CappedTricksEnv
        env = rl_environment.Environment("oh_hell", **self.params)
        if self._max_tricks is not None:
            return CappedTricksEnv(env, self._max_tricks)
        return env

    def preview_action(self, env, time_step, player: int, action_id: int) -> dict | None:
        """Synthesize the 4-card trick state when a card play will complete the trick.

        Returns None for bids, or when the trick won't complete (the post-step
        state will already show the card in current_trick).
        """
        deck_size = self.params["num_cards_per_suit"] * self.params["num_suits"]
        if action_id >= deck_size:
            return None  # bid action

        current = self.serialize_state(env, time_step)
        if len(current["current_trick"]) < self.params["players"] - 1:
            return None  # trick won't complete; post-step state will show the card normally

        suit_code = _SUIT_BY_INDEX[action_id % NUM_SUITS]
        rank_code = _RANK_ORDER[action_id // NUM_SUITS]
        card = _card(suit_code, rank_code)
        trick_card = {"player": player, "card": asdict(card)}
        return {**current, "current_trick": [*current["current_trick"], trick_card]}

    def serialize_state(
        self,
        env: rl_environment.Environment,
        time_step: rl_environment.TimeStep,
    ) -> dict[str, Any]:
        state = env.get_state
        num_players: int = self.params["players"]
        is_terminal: bool = time_step.last()
        current_player: int = time_step.observations["current_player"]

        parsed = _parse_state_string(str(state), num_players)

        # Map OpenSpiel phase names to frontend phases.
        if is_terminal or parsed["phase"] == "gameover":
            phase = "terminal"
        elif parsed["phase"] in ("bid", "bidding"):
            phase = "bidding"
        else:
            phase = "playing"

        # Legal actions with human-readable labels.
        legal_action_ids: list[int] = (
            time_step.observations["legal_actions"][current_player]
            if not is_terminal else []
        )
        legal_actions: list[Action] = []
        for action_id in legal_action_ids:
            label = state.action_to_string(current_player, action_id)
            if label.isdigit():
                legal_actions.append(_bid_action(action_id, label))
            else:
                legal_actions.append(_card_action(action_id, label))

        players = [
            PlayerState(
                bid=parsed["bids"][p],
                tricks_won=parsed["tricks_won"][p],
                hand=parsed["hands"][p],
            )
            for p in range(num_players)
        ]

        game_state = GameState(
            phase=phase,
            current_player=current_player,
            num_players=num_players,
            num_tricks=parsed["num_tricks"] or self.params["num_tricks_fixed"],
            trump_card=parsed["trump_card"],
            players=players,
            current_trick=parsed["current_trick"],
            scores=list[float](state.returns()) if is_terminal else None,
            legal_actions=legal_actions,
        )

        return asdict(game_state)
