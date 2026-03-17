# rlgames

Reinforcement learning experiments with [OpenSpiel](https://github.com/google-deepmind/open_spiel), plus a web server for playing trained agents interactively.

Currently implemented: **Oh Hell** with a shared DQN agent.

---

## Project structure

```
python/
  shared/
    dqn.py              SelfPlayDQN + make_shared_dqn_agent (used by training and server)
  training/
    oh_hell.py          DQN self-play training script for Oh Hell
    tictactoe_tabular_q.py  Tabular Q-learning experiment
  server/
    main.py             FastAPI server
    session.py          Per-game session management
    games/
      adapter.py        GameAdapter base class and registry
      oh_hell.py        Oh Hell state serialisation and adapter
    agents/
      registry.py       Agent loader registry
      dqn.py            DQN loader for Oh Hell
typescript/
  frontend/             Vite + React + TypeScript frontend
```

---

## Setup

### Python

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
cd python && uv sync
```

### Frontend

Requires Node.js.

```bash
cd typescript/frontend && npm install
```

---

## Training

Train a DQN agent for Oh Hell via self-play:

```bash
.venv/bin/python oh_hell.py
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--num_episodes` | 50 000 | Number of training episodes |
| `--checkpoint` | `oh_hell_dqn.pt` | Checkpoint save/load path |
| `--play_only` | `False` | Skip training; load checkpoint and play interactively |

The agent checkpoint is saved every 10 000 episodes and at the end of training. If a checkpoint already exists at `--checkpoint`, it is loaded before training begins (resuming from where training left off).

After training, a learning curves plot is saved to `learning_curves.png` and an interactive command-line game starts automatically.

### Game configuration

The training script uses a full 52-card deck (4 suits × 13 ranks) with `num_tricks_fixed = 2`. Relevant constants near the top of `oh_hell.py`:

```python
NUM_PLAYERS = 4
NUM_TRICKS = 2

DQN_PARAMS = {
    "players": NUM_PLAYERS,
    "num_cards_per_suit": 13,
    "num_suits": 4,
    "num_tricks_fixed": NUM_TRICKS,   # set to None to randomise tricks per hand
    ...
}
```

---

## Web server

The FastAPI server loads a trained checkpoint at startup and exposes a REST API for playing against the trained agent.

### Running

```bash
cd python
AGENTS_CONFIG='[{"game":"oh_hell","agent":"dqn","checkpoint":"training/oh_hell_dqn.pt"}]' \
  .venv/bin/uvicorn server.main:app --reload
```

Interactive API docs are available at `http://localhost:8000/docs`.

### Configuration

The server reads an `AGENTS_CONFIG` environment variable (JSON array) to determine which game/agent combinations to activate:

```bash
AGENTS_CONFIG='[{"game":"oh_hell","agent":"dqn","checkpoint":"oh_hell_dqn.pt"}]' \
  .venv/bin/uvicorn server.main:app
```

If `AGENTS_CONFIG` is not set, `oh_hell` + `dqn` with `oh_hell_dqn.pt` is used by default.

### API

All responses (except DELETE) return a `SessionResponse` object with `session_id`, `game`, `agent`, and `state` fields.

#### Start a game

```
POST /sessions
```

```json
{
  "game": "oh_hell",
  "agent": "dqn",
  "human_players": [0]
}
```

`human_players` is a list of player IDs that will be controlled by humans. All other player slots are controlled by the trained agent. For example, `[0, 1]` creates a two-human game.

#### Get current state

```
GET /sessions/{session_id}
```

#### Submit an action

```
POST /sessions/{session_id}/actions
```

```json
{ "action_id": 52 }
```

Action IDs come from the `legal_actions` array in the game state. The server advances through any consecutive agent turns after the human action and returns the state at the next human decision point (or the terminal state).

#### End a session

```
DELETE /sessions/{session_id}
```

### Response schema

```json
{
  "session_id": "uuid",
  "game": "oh_hell",
  "agent": "dqn",
  "state": {
    "phase": "bidding | playing | terminal",
    "current_player": 0,
    "num_players": 4,
    "num_tricks": 2,
    "trump_card": { "id": 23, "rank": "7", "suit": "hearts", "label": "H7" },
    "players": [
      {
        "bid": null,
        "tricks_won": 0,
        "hand": [
          { "id": 0, "rank": "2", "suit": "clubs", "label": "C2" }
        ]
      }
    ],
    "current_trick": [
      { "player": 1, "card": { "id": 5, "rank": "3", "suit": "diamonds", "label": "D3" } }
    ],
    "scores": null,
    "legal_actions": [
      { "id": 52, "label": "Bid 0", "type": "bid" },
      { "id": 5,  "label": "D3",    "type": "play" }
    ]
  }
}
```

`scores` is `null` until `phase == "terminal"`. `hand` for opponent players is always an empty list (hidden information). `current_trick` is empty at the start of each trick and outside the playing phase.

Card labels use OpenSpiel notation: `<Suit><Rank>` where suit is `C/D/S/H` and rank is `2–9`, `10`, `J`, `Q`, `K`, `A`. For example `H10`, `CA`, `SJ`.

---

## Adding a new game or agent

**New game:**

1. Create `server/games/<game_name>.py` implementing `GameAdapter` and decorate the class with `@register("<game_name>")`.
2. Import the module in `server/games/__init__.py`.
3. Add a `{"game": "<game_name>", "agent": "...", "checkpoint": "..."}` entry to `AGENTS_CONFIG`.

**New agent type:**

1. Create `server/agents/<agent_name>.py` with a loader function decorated with `@register("<game_name>", "<agent_type>")`.
2. Import the module in `server/agents/__init__.py`.

---

## Tests

All Python commands run from the `python/` directory.

```bash
cd python

# All tests
.venv/bin/python -m unittest discover -s . -p "test_*.py" -v

# Game adapter tests only
.venv/bin/python -m unittest server.games.test_oh_hell -v

# Session manager tests only
.venv/bin/python -m unittest server.test_session -v
```
