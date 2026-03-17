"""
FastAPI server for OpenSpiel web gameplay.

Startup
-------
Reads AGENTS_CONFIG (a JSON array) to determine which (game, agent, checkpoint)
combinations to activate, then builds one SessionManager per pair.

Default config (used when AGENTS_CONFIG is not set):
  [{"game": "oh_hell", "agent": "dqn", "checkpoint": "oh_hell_dqn.pt"}]

To add a new game or agent type:
  1. Implement a GameAdapter and register it with @register in server/games/.
  2. Implement an agent loader and register it with @register in server/agents/.
  3. Import both modules in their respective __init__.py.
  4. Add an entry to AGENTS_CONFIG.

Routes
------
  POST   /sessions                  Start a new game session
  GET    /sessions/{id}             Get current game state
  POST   /sessions/{id}/actions     Submit a player action
  DELETE /sessions/{id}             End a session
"""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import server.agents  # noqa: F401 — triggers agent loader registration
import server.games   # noqa: F401 — triggers game adapter registration
from server.agents.registry import load as load_agent
from server.games.adapter import get_adapter
from server.session import SessionError, SessionManager


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRANSITION_DELAY_S = 0.8  # seconds between streamed events

_DEFAULT_CONFIG = [
    {"game": "oh_hell", "agent": "dqn", "checkpoint": "oh_hell_dqn.pt"},
]


def _load_config() -> list[dict]:
    raw = os.environ.get("AGENTS_CONFIG")
    if raw:
        return json.loads(raw)
    return _DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class NewSessionRequest(BaseModel):
    game: str = Field("oh_hell", description="Registered game name.")
    agent: str = Field("dqn", description="Registered agent type.")
    human_players: list[int] = Field(
        [0], description="Player IDs controlled by humans; all others are agents."
    )


class SessionResponse(BaseModel):
    session_id: str
    game: str
    agent: str
    state: dict[str, Any]
    transitions: list[dict[str, Any]] = []


class ActionRequest(BaseModel):
    action_id: int


# ---------------------------------------------------------------------------
# Application lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

_managers: dict[tuple[str, str], SessionManager] = {}
_session_index: dict[str, tuple[str, str]] = {}  # session_id → (game, agent)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = _load_config()
    for entry in config:
        game, agent_type, checkpoint = entry["game"], entry["agent"], entry["checkpoint"]
        adapter_cls = get_adapter(game)
        adapter = adapter_cls()
        env = adapter.create_env()
        agent = load_agent(game, agent_type, checkpoint, env)
        _managers[(game, agent_type)] = SessionManager(agent=agent, adapter=adapter)

    yield

    _managers.clear()
    _session_index.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="OpenSpiel Web", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_manager(game: str, agent: str) -> SessionManager:
    key = (game, agent)
    manager = _managers.get(key)
    if manager is None:
        available = [{"game": g, "agent": a} for g, a in sorted(_managers)]
        raise HTTPException(
            status_code=404,
            detail=f"No active manager for game={game!r} agent={agent!r}. "
                   f"Available: {available}",
        )
    return manager


def _get_manager_for_session(session_id: str) -> tuple[str, str, SessionManager]:
    key = _session_index.get(session_id)
    if key is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id!r}")
    game, agent = key
    return game, agent, _get_manager(game, agent)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/sessions", response_model=SessionResponse, status_code=201)
def new_session(body: NewSessionRequest):
    manager = _get_manager(body.game, body.agent)
    session_id, state, transitions = manager.new_session(body.human_players)
    _session_index[session_id] = (body.game, body.agent)
    return SessionResponse(session_id=session_id, game=body.game, agent=body.agent, state=state, transitions=transitions)


@app.get("/sessions/{session_id}", response_model=SessionResponse)
def get_state(session_id: str):
    game, agent, manager = _get_manager_for_session(session_id)
    try:
        state = manager.get_state(session_id)
    except SessionError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return SessionResponse(session_id=session_id, game=game, agent=agent, state=state)


@app.post("/sessions/{session_id}/actions")
async def apply_action(session_id: str, body: ActionRequest):
    game, agent, manager = _get_manager_for_session(session_id)
    try:
        event_iter = manager.stream_action(session_id, body.action_id)
    except SessionError as e:
        status = 404 if "not found" in str(e).lower() else 400
        raise HTTPException(status_code=status, detail=str(e))

    async def generate():
        for event in event_iter:
            yield f"data: {json.dumps(event)}\n\n"
            if event["type"] != "final":
                await asyncio.sleep(TRANSITION_DELAY_S)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str):
    game, agent, manager = _get_manager_for_session(session_id)
    manager.delete_session(session_id)
    _session_index.pop(session_id, None)
