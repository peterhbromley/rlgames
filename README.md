# rlgames

Reinforcement learning experiments with [OpenSpiel](https://github.com/google-deepmind/open_spiel) and PyTorch, plus a web server and React UI for playing trained agents interactively.

Currently implemented: **Oh Hell** (trick-taking card game) with DQN and NFSP agents.

---

## Project structure

```
python/
  shared/
    dqn.py              SelfPlayDQN wrapper + make_shared_dqn_agent (shared by training and server)
    env_wrappers.py     CappedTricksEnv — limits max tricks per hand without changing the deck
  training/
    train_dqn.py        DQN training entry point
    train_nfsp.py       NFSP training entry point
    config.py           Pydantic config models (DQNRunConfig, NFSPRunConfig)
    configs/
      oh_hell_nfsp_3tricks.yaml    NFSP, 4 players, 3-trick fixed hands
      oh_hell_3tricks_stable.yaml  DQN, stability-focused hyperparameters
      oh_hell_3tricks.yaml         DQN, 3-trick fixed hands
      oh_hell_full.yaml            DQN, full variable-trick game
      oh_hell_small.yaml           DQN, small network for quick experiments
  server/
    main.py             FastAPI server (SSE streaming, session management)
    session.py          Per-game session state
    games/
      adapter.py        GameAdapter base class and registry
      oh_hell.py        Oh Hell state serialisation and adapter
    agents/
      registry.py       Agent loader registry
      dqn.py            DQN checkpoint loader for Oh Hell
      nfsp.py           NFSP checkpoint loader + NFSPEvalAgent for Oh Hell
typescript/
  frontend/             Vite + React + TypeScript UI
```

---

## Algorithms

### DQN (Deep Q-Network)

Shared DQN agent across all player positions (`SelfPlayDQN`). Each step, the agent observes the current player's info state and selects an action. All players share the same network weights, relying on positional symmetry in Oh Hell.

Training uses ε-greedy exploration with linear decay, experience replay, and a target network. Stability options: Huber loss (`loss_str: huber`) and gradient clipping.

### NFSP (Neural Fictitious Self-Play)

Four independent agents, one per player. Each agent maintains two networks:

- **Best-response network** (DQN): learns to exploit the current average strategy of opponents.
- **Average strategy network** (supervised): learns the time-average of the agent's own play.

At each step, each agent independently samples whether to act from its best-response or average strategy, controlled by the anticipatory parameter η (`anticipatory_param`, default 0.1). At evaluation time, all agents use their average strategy networks, which approximate a Nash equilibrium.

NFSP is better suited to Oh Hell than plain DQN because Oh Hell has imperfect information and requires reasoning about opponent strategy rather than just maximising expected reward.

---

## Setup

### Python

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

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

All training commands run from the `python/` directory. Both scripts accept a `--config` path pointing to a YAML config file.

### NFSP (recommended)

```bash
python -m training.train_nfsp --config training/configs/oh_hell_nfsp_3tricks.yaml
```

### DQN

```bash
python -m training.train_dqn --config training/configs/oh_hell_3tricks_stable.yaml
```

### Config structure

Each YAML config has three top-level sections:

```yaml
game:
  name: oh_hell
  players: 4
  num_cards_per_suit: 13
  num_suits: 4
  num_tricks_fixed: 3       # omit for variable tricks (full game)
  off_bid_penalty: false
  points_per_trick: 1

agent:
  # DQN or NFSP hyperparameters (see config.py for all fields)
  hidden_layers_sizes: [256, 256]
  gradient_clipping: 10.0
  ...

training:
  num_episodes: 2000000
  device: cpu               # or cuda
  checkpoint: checkpoints/oh_hell_nfsp_3tricks.pt
  log_interval: 10000
  wandb:
    enabled: false
    project: rlgames
    run_name: my_run
```

### Checkpointing and resuming

Checkpoints are saved every `log_interval` episodes and at the end of training. The episode counter, network weights, optimizer state, and iteration counters are all saved. If a checkpoint already exists at the configured path, training resumes from where it left off.

Note: replay buffers (experience replay and NFSP reservoir buffer) are **not** saved in checkpoints. They are rebuilt from scratch on resume. This means epsilon and the SL average policy start from a warm network but explore fresh experience.

### Experiment tracking

Set `wandb.enabled: true` in your config to log metrics to [Weights & Biases](https://wandb.ai). DQN logs `loss` and per-player rewards. NFSP logs `loss/rl`, `loss/sl`, and per-player rewards.

---

## Web server

The FastAPI server loads trained checkpoints at startup and exposes a REST + SSE API for playing against trained agents.

### Running

```bash
cd python
.venv/bin/uvicorn server.main:app --reload
```

By default the server loads both the DQN and NFSP checkpoints:

```python
_DEFAULT_CONFIG = [
    {"game": "oh_hell", "agent": "dqn",  "checkpoint": "checkpoints/oh_hell_dqn.pt"},
    {"game": "oh_hell", "agent": "nfsp", "checkpoint": "checkpoints/oh_hell_nfsp_3tricks.pt"},
]
```

To override, set `AGENTS_CONFIG` to a JSON array:

```bash
AGENTS_CONFIG='[{"game":"oh_hell","agent":"nfsp","checkpoint":"checkpoints/oh_hell_nfsp_3tricks.pt","params":{"num_tricks_fixed":3}}]' \
  .venv/bin/uvicorn server.main:app
```

Each entry accepts an optional `params` object whose keys override the adapter's `DEFAULT_PARAMS`. Use this to match the game configuration the checkpoint was trained with (e.g. `num_tricks_fixed`).

Interactive API docs: `http://localhost:8000/docs`.

### API

#### Start a game

```
POST /sessions
```

```json
{
  "game": "oh_hell",
  "agent": "nfsp",
  "human_players": [0]
}
```

`human_players` lists the player IDs controlled by a human. All others are agent-controlled.

#### Get current state

```
GET /sessions/{session_id}
```

#### Submit an action (streaming)

```
POST /sessions/{session_id}/actions
```

```json
{ "action_id": 52 }
```

Returns a Server-Sent Events stream. Each event is a JSON object:

```
data: {"type": "transition", "player": 1, "state": {...}}

data: {"type": "final", "player": null, "state": {...}}
```

`transition` events are emitted as each agent plays. The `final` event signals the next human decision point (or terminal state).

#### End a session

```
DELETE /sessions/{session_id}
```

### Response schema

```json
{
  "session_id": "uuid",
  "game": "oh_hell",
  "agent": "nfsp",
  "state": {
    "phase": "bidding | playing | terminal",
    "current_player": 0,
    "num_players": 4,
    "num_tricks": 3,
    "trump_card": { "id": 23, "rank": "7", "suit": "hearts", "label": "H7" },
    "players": [
      {
        "bid": null,
        "tricks_won": 0,
        "hand": [{ "id": 0, "rank": "2", "suit": "clubs", "label": "C2" }]
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
  },
  "transitions": []
}
```

`scores` is `null` until `phase == "terminal"`. Opponent `hand` arrays are always empty (hidden information). Card labels use OpenSpiel notation: `<Suit><Rank>` where suit is `C/D/S/H` and rank is `2–9`, `T`, `J`, `Q`, `K`, `A` (e.g. `H10`, `CA`, `SJ`).

---

## Frontend

```bash
cd typescript/frontend && npm run dev
```

Opens at `http://localhost:5173`. The UI connects to the server at `http://localhost:8000` by default (override with `VITE_API_BASE`).

Features:
- Choose opponent type (NFSP or DQN) before starting
- Bidding and card-play phases with legal action highlighting
- Animated agent moves streamed via SSE
- Scoreboard showing bids, tricks won, and final scores

---

## Tests

All commands run from the `python/` directory.

```bash
cd python

# All tests
.venv/bin/python -m unittest discover -s . -p "test_*.py" -v

# Game adapter tests
.venv/bin/python -m unittest server.games.test_oh_hell -v

# Session manager tests
.venv/bin/python -m unittest server.test_session -v

# Environment wrapper tests
.venv/bin/python -m unittest shared.test_env_wrappers -v
```

---

## Adding a new game or agent

**New game:**

1. Create `server/games/<game_name>.py` implementing `GameAdapter` and decorate with `@register("<game_name>")`.
2. Import the module in `server/games/__init__.py`.
3. Add a `GameConfig`-compatible section to your training YAML.
4. Add entries to `AGENTS_CONFIG` (or `_DEFAULT_CONFIG` in `main.py`).

**New agent type:**

1. Create `server/agents/<agent_name>.py` with a loader function decorated with `@register("<game_name>", "<agent_type>")`.
2. Import the module in `server/agents/__init__.py`.
3. Create a corresponding training script and config model in `training/`.
