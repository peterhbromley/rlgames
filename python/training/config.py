"""
Pydantic config models for the training platform.

Configs are loaded from YAML files. Example usage:

    cfg = RunConfig.from_yaml("configs/oh_hell_full.yaml")
    env = cfg.game.make_env()
    agent = cfg.agent.make_agent(state_size, num_actions, cfg.training.num_episodes, num_players)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class GameConfig(BaseModel):
    name: str
    # Oh Hell params — extend with new fields when adding other games.
    players: int = 4
    num_cards_per_suit: int = 13
    num_suits: int = 4
    num_tricks_fixed: Optional[int] = None   # None → variable tricks (full game)
    off_bid_penalty: bool = False
    points_per_trick: int = 1

    def make_env(self):
        from open_spiel.python import rl_environment
        params = self.model_dump(exclude={"name"}, exclude_none=True)
        return rl_environment.Environment(self.name, **params)

    def to_openspiel_params(self) -> dict[str, Any]:
        return self.model_dump(exclude={"name"}, exclude_none=True)


class AgentConfig(BaseModel):
    hidden_layers_sizes: list[int] = [256, 256]
    replay_buffer_capacity: int = 50_000
    batch_size: int = 128
    learning_rate: float = 0.001
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.8   # fraction of total steps over which ε decays
    update_target_network_every: int = 500
    learn_every: int = 10

    def make_agent(
        self,
        state_size: int,
        num_actions: int,
        num_episodes: int,
        num_players: int,
        device: str = "cpu",
    ):
        from shared.dqn import make_shared_dqn_agent
        return make_shared_dqn_agent(
            state_size=state_size,
            num_actions=num_actions,
            num_episodes=num_episodes,
            num_players=num_players,
            hidden_layers_sizes=self.hidden_layers_sizes,
            replay_buffer_capacity=self.replay_buffer_capacity,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            epsilon_start=self.epsilon_start,
            epsilon_end=self.epsilon_end,
            epsilon_decay_fraction=self.epsilon_decay_fraction,
            update_target_network_every=self.update_target_network_every,
            learn_every=self.learn_every,
            device=device,
        )


class WandbConfig(BaseModel):
    enabled: bool = False
    project: str = "rlgames"
    run_name: Optional[str] = None
    tags: list[str] = Field(default_factory=list)


class TrainingConfig(BaseModel):
    num_episodes: int = 50_000
    device: str = "cpu"
    checkpoint: str = "checkpoints/agent.pt"
    log_interval: int = 10_000
    wandb: WandbConfig = Field(default_factory=WandbConfig)


class RunConfig(BaseModel):
    game: GameConfig
    agent: AgentConfig = Field(default_factory=AgentConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
