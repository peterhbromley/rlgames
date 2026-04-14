"""
NFSP training entry point.

Usage:
    python -m training.train_nfsp --config training/configs/oh_hell_nfsp_3tricks.yaml
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch

from open_spiel.python.pytorch import nfsp

from training.config import NFSPRunConfig


def save_nfsp(agents: list[nfsp.NFSP], path: str, episode: int = 0) -> None:
    checkpoint = {"episode": episode, "agents": []}
    for agent in agents:
        checkpoint["agents"].append({
            "q_network": agent._rl_agent._q_network.state_dict(),
            "target_q_network": agent._rl_agent._target_q_network.state_dict(),
            "rl_optimizer": agent._rl_agent._optimizer.state_dict(),
            "rl_iteration": agent._rl_agent._iteration,
            "avg_network": agent._avg_network.state_dict(),
            "sl_optimizer": agent._optimizer.state_dict(),
            "nfsp_iteration": agent._iteration,
        })
    torch.save(checkpoint, path)
    logging.info("NFSP agents saved to %s (episode %d)", path, episode)


def load_nfsp(agents: list[nfsp.NFSP], path: str, device: str = "cpu") -> int:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    for agent, state in zip(agents, checkpoint["agents"]):
        agent._rl_agent._q_network.load_state_dict(state["q_network"])
        agent._rl_agent._target_q_network.load_state_dict(state["target_q_network"])
        agent._rl_agent._optimizer.load_state_dict(state["rl_optimizer"])
        agent._rl_agent._iteration = state["rl_iteration"]
        agent._avg_network.load_state_dict(state["avg_network"])
        agent._optimizer.load_state_dict(state["sl_optimizer"])
        agent._iteration = state["nfsp_iteration"]
    episode = checkpoint.get("episode", 0)
    logging.info(
        "NFSP agents loaded from %s (episode %d, iterations %s)",
        path,
        episode,
        [a._iteration for a in agents],
    )
    return episode


def train(cfg: NFSPRunConfig) -> None:
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

    # ── Agents ────────────────────────────────────────────────────────────────
    agents = cfg.agent.make_agents(
        state_size=state_size,
        num_actions=num_actions,
        num_players=num_players,
        device=cfg.training.device,
    )

    checkpoint_path = cfg.training.checkpoint
    start_episode = 0
    if os.path.exists(checkpoint_path):
        start_episode = load_nfsp(agents, checkpoint_path, device=cfg.training.device)
        logging.info("Resuming from episode %d", start_episode)
    else:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    # ── WandB ─────────────────────────────────────────────────────────────────
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
    sl_loss_window: list[float] = []
    rl_loss_window: list[float] = []

    for ep in range(start_episode, num_episodes):
        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent_output = agents[player_id].step(time_step)
            time_step = env.step([agent_output.action])

        # Terminal step for all agents.
        for agent in agents:
            agent.step(time_step)

        rewards = np.asarray(time_step.rewards, dtype=float)
        reward_window += (rewards - reward_window) / (ep % log_interval + 1)

        # Collect losses from agent 0 as representative (all are symmetric).
        sl_loss, rl_loss = agents[0].loss
        if sl_loss is not None:
            sl_loss_window.append(float(sl_loss))
        if rl_loss is not None:
            rl_loss_window.append(float(rl_loss))

        if (ep + 1) % log_interval == 0:
            mean_sl_loss = float(np.mean(sl_loss_window)) if sl_loss_window else float("nan")
            mean_rl_loss = float(np.mean(rl_loss_window)) if rl_loss_window else float("nan")
            logging.info(
                "Episode %d/%d | avg reward: %s | rl_loss: %.4f | sl_loss: %.4f",
                ep + 1,
                num_episodes,
                np.round(reward_window, 3),
                mean_rl_loss,
                mean_sl_loss,
            )

            if wb_run is not None:
                import wandb
                log_dict: dict = {
                    "episode": ep + 1,
                    "loss/rl": mean_rl_loss,
                    "loss/sl": mean_sl_loss,
                }
                for p in range(num_players):
                    log_dict[f"reward/player_{p}"] = reward_window[p]
                log_dict["reward/mean"] = float(reward_window.mean())
                wandb.log(log_dict, step=ep + 1)

            reward_window = np.zeros(num_players)
            sl_loss_window = []
            rl_loss_window = []
            save_nfsp(agents, checkpoint_path, episode=ep + 1)

    save_nfsp(agents, checkpoint_path, episode=num_episodes)
    logging.info("Training complete. Checkpoint: %s", checkpoint_path)

    if wb_run is not None:
        wb_run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NFSP agents.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a YAML NFSPRunConfig (e.g. training/configs/oh_hell_nfsp_3tricks.yaml)",
    )
    args = parser.parse_args()

    cfg = NFSPRunConfig.from_yaml(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
