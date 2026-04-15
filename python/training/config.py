"""
Pydantic config models for the training platform.

Configs are loaded from YAML files. Example usage:

    cfg = DQNRunConfig.from_yaml("configs/oh_hell_full.yaml")
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
    max_tricks: Optional[int] = None         # cap hand size without changing the deck

    def make_env(self):
        from open_spiel.python import rl_environment
        from shared.env_wrappers import CappedTricksEnv
        params = self.model_dump(exclude={"name", "max_tricks"}, exclude_none=True)
        env = rl_environment.Environment(self.name, **params)
        if self.max_tricks is not None:
            return CappedTricksEnv(env, self.max_tricks)
        return env

    def to_openspiel_params(self) -> dict[str, Any]:
        return self.model_dump(exclude={"name", "max_tricks"}, exclude_none=True)


class DQNAgentConfig(BaseModel):
    hidden_layers_sizes: list[int] = [256, 256]
    replay_buffer_capacity: int = 50_000
    batch_size: int = 128
    learning_rate: float = 0.001
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.8   # fraction of total steps over which ε decays
    update_target_network_every: int = 500
    learn_every: int = 10
    gradient_clipping: Optional[float] = None
    loss_str: str = "mse"

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
            gradient_clipping=self.gradient_clipping,
            loss_str=self.loss_str,
            device=device,
        )


class NFSPAgentConfig(BaseModel):
    hidden_layers_sizes: list[int] = [256, 256]
    reservoir_buffer_capacity: int = 2_000_000
    anticipatory_param: float = 0.1
    batch_size: int = 256
    rl_learning_rate: float = 0.0001
    sl_learning_rate: float = 0.001
    learn_every: int = 64
    gradient_clipping: Optional[float] = None
    # DQN (best-response) kwargs passed through to the inner RL agent.
    replay_buffer_capacity: int = 200_000
    epsilon_start: float = 0.06
    epsilon_end: float = 0.001
    epsilon_decay_duration: int = 10_000_000
    update_target_network_every: int = 1000
    min_buffer_size_to_learn: int = 1000

    def make_agents(
        self,
        state_size: int,
        num_actions: int,
        num_players: int,
        device: str = "cpu",
    ) -> list:
        from open_spiel.python.pytorch import nfsp
        return [
            nfsp.NFSP(
                player_id=i,
                state_representation_size=state_size,
                num_actions=num_actions,
                hidden_layers_sizes=self.hidden_layers_sizes,
                reservoir_buffer_capacity=self.reservoir_buffer_capacity,
                anticipatory_param=self.anticipatory_param,
                batch_size=self.batch_size,
                rl_learning_rate=self.rl_learning_rate,
                sl_learning_rate=self.sl_learning_rate,
                learn_every=self.learn_every,
                min_buffer_size_to_learn=self.min_buffer_size_to_learn,
                gradient_clipping=self.gradient_clipping,
                optimizer_str="adam",
                # DQN kwargs
                replay_buffer_capacity=self.replay_buffer_capacity,
                epsilon_start=self.epsilon_start,
                epsilon_end=self.epsilon_end,
                epsilon_decay_duration=self.epsilon_decay_duration,
                update_target_network_every=self.update_target_network_every,
                device=device,
            )
            for i in range(num_players)
        ]


class WandbConfig(BaseModel):
    enabled: bool = False
    project: str = "rlgames"
    run_name: Optional[str] = None
    tags: list[str] = Field(default_factory=list)


class TrainingConfig(BaseModel):
    num_episodes: int = 50_000
    num_iterations: Optional[int] = None  # PPO uses iterations instead of episodes
    device: str = "cpu"
    checkpoint: str = "checkpoints/agent.pt"
    log_interval: int = 10_000
    num_workers: int = 1  # parallel rollout workers (1 = sequential)
    wandb: WandbConfig = Field(default_factory=WandbConfig)


class DQNRunConfig(BaseModel):
    game: GameConfig
    agent: DQNAgentConfig = Field(default_factory=DQNAgentConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DQNRunConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)


class PPOAgentConfig(BaseModel):
    hidden_layers_sizes: list[int] = [256, 256]
    lr: float = 3e-4              # play network learning rate
    bid_lr: Optional[float] = None  # bid network learning rate (defaults to lr if unset)
    gamma: float = 1.0           # no discounting within a hand
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01       # play network entropy coefficient
    bid_entropy_coef: float = 0.05   # bid network entropy coefficient (higher = more bid exploration)
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    bid_ppo_epochs: int = 8          # PPO epochs for bid network (more to compensate for smaller dataset)
    play_ppo_epochs: int = 4         # PPO epochs for play network
    batch_size: int = 256
    episodes_per_iter: int = 1024  # games played between each PPO update
    pool_size: int = 20            # max past-policy snapshots kept
    pool_save_interval: int = 10   # snapshot current policy every N iterations
    eval_episodes: int = 200       # greedy eval episodes run at each log interval
    lr_schedule: Optional[str] = None  # None = constant, "cosine" = cosine annealing to 0


class NFSPRunConfig(BaseModel):
    game: GameConfig
    agent: NFSPAgentConfig = Field(default_factory=NFSPAgentConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "NFSPRunConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)


class CurriculumStage(BaseModel):
    until_iter: int
    max_tricks: int


class PPORunConfig(BaseModel):
    game: GameConfig
    agent: PPOAgentConfig = Field(default_factory=PPOAgentConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    curriculum: list[CurriculumStage] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PPORunConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)
