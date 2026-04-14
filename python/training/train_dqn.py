"""
DQN training entry point.

Usage:
    python -m training.train_dqn --config training/configs/oh_hell_small.yaml
    python -m training.train_dqn --config training/configs/oh_hell_full.yaml
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np

from training.config import DQNRunConfig


def train(cfg: DQNRunConfig) -> None:
    """Run a full training loop according to cfg."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # ── Environment ──────────────────────────────────────────────────────────
    env = cfg.game.make_env()
    num_actions: int = env.action_spec()["num_actions"]
    state_size: int = env.observation_spec()["info_state"][0]
    num_players: int = cfg.game.players
    logging.info(
        "Game: %s | players=%d state_size=%d num_actions=%d",
        cfg.game.name,
        num_players,
        state_size,
        num_actions,
    )

    # ── Config summary ────────────────────────────────────────────────────────
    logging.info("Game config:\n%s", cfg.game.model_dump_json(indent=2))
    logging.info("Agent config:\n%s", cfg.agent.model_dump_json(indent=2))

    # ── Agent ─────────────────────────────────────────────────────────────────
    agent = cfg.agent.make_agent(
        state_size=state_size,
        num_actions=num_actions,
        num_episodes=cfg.training.num_episodes,
        num_players=num_players,
        device=cfg.training.device,
    )

    checkpoint_path = cfg.training.checkpoint
    start_episode = 0
    if os.path.exists(checkpoint_path):
        start_episode = agent.load(checkpoint_path, device=cfg.training.device)
        logging.info("Resuming from episode %d", start_episode)
    else:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    # ── WandB ─────────────────────────────────────────────────────────────────
    # WandB is initialized lazily so that runs without wandb.enabled=true don't
    # require the package to be installed (though it is listed as a dependency).
    wb_run = None
    if cfg.training.wandb.enabled:
        import wandb
        wb_run = wandb.init(
            project=cfg.training.wandb.project,
            name=cfg.training.wandb.run_name,
            tags=cfg.training.wandb.tags,
            config={
                "game": cfg.game.model_dump(),
                "agent": cfg.agent.model_dump(),
                "training": cfg.training.model_dump(exclude={"wandb"}),
            },
        )
        logging.info("WandB run: %s", wb_run.url)

    # ── Training loop ─────────────────────────────────────────────────────────
    num_episodes = cfg.training.num_episodes
    log_interval = cfg.training.log_interval

    reward_window = np.zeros(num_players)
    loss_window: list[float] = []

    for ep in range(start_episode, num_episodes):
        time_step = env.reset()
        while not time_step.last():
            agent_out = agent.step(time_step)
            time_step = env.step([agent_out.action])
        agent.step(time_step)

        rewards = np.asarray(time_step.rewards, dtype=float)
        reward_window += (rewards - reward_window) / (ep % log_interval + 1)

        loss = agent.loss
        if loss is not None:
            loss_window.append(float(loss))

        if (ep + 1) % log_interval == 0:
            mean_loss = float(np.mean(loss_window)) if loss_window else float("nan")
            logging.info(
                "Episode %d/%d | avg reward: %s | loss: %.4f",
                ep + 1,
                num_episodes,
                np.round(reward_window, 3),
                mean_loss,
            )

            if wb_run is not None:
                import wandb
                log_dict: dict = {
                    "episode": ep + 1,
                    "loss": mean_loss,
                }
                for p in range(num_players):
                    log_dict[f"reward/player_{p}"] = reward_window[p]
                log_dict["reward/mean"] = float(reward_window.mean())
                wandb.log(log_dict, step=ep + 1)

            reward_window = np.zeros(num_players)
            loss_window = []
            agent.save(checkpoint_path, episode=ep + 1)

    agent.save(checkpoint_path, episode=num_episodes)
    logging.info("Training complete. Checkpoint: %s", checkpoint_path)

    if wb_run is not None:
        wb_run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DQN agent.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a YAML DQNRunConfig (e.g. training/configs/oh_hell_small.yaml)",
    )
    args = parser.parse_args()

    cfg = DQNRunConfig.from_yaml(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
