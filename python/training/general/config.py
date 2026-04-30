"""
Config models for the training pipelines.

Shared GameConfig works with any OpenSpiel game. Each algorithm type
(PPO, MCCFR, Deep CFR) has its own RunConfig.

Usage:
    cfg = RunConfig.from_yaml("training/configs/oh_hell_ppo.yaml")
    cfg = MCCFRRunConfig.from_yaml("training/configs/liars_dice_mccfr.yaml")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class GameConfig(BaseModel):
    """Game-agnostic config — passes arbitrary params to OpenSpiel."""
    name: str
    params: dict[str, Any] = Field(default_factory=dict)

    def make_env(self):
        from open_spiel.python import rl_environment
        return rl_environment.Environment(self.name, **self.params)

    def make_game(self):
        import pyspiel
        return pyspiel.load_game(self.name, self.params)


class AgentConfig(BaseModel):
    hidden_layers_sizes: list[int] = [256, 256]
    lr: float = 3e-4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    batch_size: int = 256
    episodes_per_iter: int = 1024
    pool_size: int = 20
    pool_save_interval: int = 10
    eval_episodes: int = 200


class WandbConfig(BaseModel):
    enabled: bool = False
    project: str = "rlgames"
    group: Optional[str] = None
    run_name: Optional[str] = None
    tags: list[str] = Field(default_factory=list)


class TrainingConfig(BaseModel):
    num_iterations: int = 5000
    device: str = "cpu"
    checkpoint: str = "checkpoints/agent.pt"
    log_interval: int = 10
    wandb: WandbConfig = Field(default_factory=WandbConfig)


class RunConfig(BaseModel):
    """PPO self-play training config."""
    game: GameConfig
    agent: AgentConfig = Field(default_factory=AgentConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# MCCFR config
# ---------------------------------------------------------------------------

class MCCFRConfig(BaseModel):
    """Tabular MCCFR training config."""
    num_iterations: int = 100_000
    log_interval: int = 1000
    checkpoint: str = "checkpoints/mccfr.pkl"

class MCCFRRunConfig(BaseModel):
    """Top-level config for tabular MCCFR training."""
    game: GameConfig
    mccfr: MCCFRConfig = Field(default_factory=MCCFRConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> MCCFRRunConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# Deep CFR config
# ---------------------------------------------------------------------------

class DeepCFRConfig(BaseModel):
    """Deep CFR algorithm config."""
    num_iterations: int = 100
    num_traversals: int = 100
    lr: float = 1e-3
    advantage_network_layers: list[int] = [128, 128]
    policy_network_layers: list[int] = [256, 256]
    batch_size_advantage: int = 2048
    batch_size_strategy: int = 2048
    memory_capacity: int = 1_000_000
    advantage_network_train_steps: int = 750
    policy_network_train_steps: int = 5000
    reinitialize_advantage_networks: bool = True
    device: str = "cpu"
    log_interval: int = 10
    checkpoint: str = "checkpoints/deep_cfr.pkl"

class DeepCFRRunConfig(BaseModel):
    """Top-level config for Deep CFR training."""
    game: GameConfig
    deep_cfr: DeepCFRConfig = Field(default_factory=DeepCFRConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> DeepCFRRunConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
