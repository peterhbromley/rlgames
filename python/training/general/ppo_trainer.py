"""
PPO self-play trainer for OpenSpiel games.

Works with any sequential imperfect-information game. Uses a single
policy-value network (no game-specific head splitting).

Usage:
    python -m training.general.ppo_trainer --config training/configs/oh_hell_ppo.yaml
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch

from training.ppo import ActorCritic, RolloutBuffer, PolicyPool, ppo_update
from training.general.config import RunConfig


# ---------------------------------------------------------------------------
# Episode collection
# ---------------------------------------------------------------------------

def _act_network(net, obs_np, legal, num_actions, device, greedy=False):
    """Get action from a network with legal-action masking."""
    mask = np.zeros(num_actions, dtype=np.bool_)
    mask[legal] = True
    obs_t = torch.as_tensor(obs_np, device=device)
    mask_t = torch.as_tensor(mask, device=device)
    action, lp, val = net.act(obs_t, mask_t, greedy=greedy)
    return action, lp, val, mask


def collect_episodes(
    env,
    net: ActorCritic,
    opp_net: ActorCritic,
    pool: PolicyPool,
    *,
    num_episodes: int,
    num_players: int,
    num_actions: int,
    device: str = "cpu",
) -> tuple[RolloutBuffer, np.ndarray]:
    """Self-play episodes. Learner seat rotates. Returns buffer + rewards."""
    buffer = RolloutBuffer()
    rewards = np.zeros(num_episodes)

    for ep in range(num_episodes):
        learner = ep % num_players

        if len(pool) > 0:
            opp_net.load_state_dict(pool.sample())

        time_step = env.reset()
        while not time_step.last():
            player = time_step.observations["current_player"]
            legal = time_step.observations["legal_actions"][player]

            if player == learner:
                obs_np = np.asarray(
                    time_step.observations["info_state"][player], dtype=np.float32,
                )
                action, lp, val, mask = _act_network(
                    net, obs_np, legal, num_actions, device,
                )
                buffer.add(obs_np, action, lp, val, mask)
            else:
                obs_np = np.asarray(
                    time_step.observations["info_state"][player], dtype=np.float32,
                )
                action, _, _, _ = _act_network(
                    opp_net, obs_np, legal, num_actions, device,
                )

            time_step = env.step([action])

        reward = time_step.rewards[learner]
        buffer.finish_episode(reward)
        rewards[ep] = reward

    return buffer, rewards


def eval_episodes(
    env,
    net: ActorCritic,
    opp_net: ActorCritic,
    pool: PolicyPool,
    *,
    num_episodes: int,
    num_players: int,
    num_actions: int,
    device: str = "cpu",
) -> float:
    """Greedy eval episodes, return mean reward."""
    rewards = np.zeros(num_episodes)

    for ep in range(num_episodes):
        learner = ep % num_players
        if len(pool) > 0:
            opp_net.load_state_dict(pool.sample())

        time_step = env.reset()
        while not time_step.last():
            player = time_step.observations["current_player"]
            legal = time_step.observations["legal_actions"][player]

            if player == learner:
                obs_np = np.asarray(
                    time_step.observations["info_state"][player], dtype=np.float32,
                )
                action, _, _, _ = _act_network(
                    net, obs_np, legal, num_actions, device, greedy=True,
                )
            else:
                obs_np = np.asarray(
                    time_step.observations["info_state"][player], dtype=np.float32,
                )
                action, _, _, _ = _act_network(
                    opp_net, obs_np, legal, num_actions, device,
                )

            time_step = env.step([action])

        rewards[ep] = time_step.rewards[learner]

    return float(rewards.mean())


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(net, optimizer, pool, path, iteration, scheduler=None):
    data = {
        "net": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "pool": pool.state_dict(),
        "iteration": iteration,
    }
    if scheduler is not None:
        data["scheduler"] = scheduler.state_dict()
    torch.save(data, path)
    logging.info("Saved checkpoint to %s (iteration %d)", path, iteration)


def load_checkpoint(net, optimizer, pool, path, device="cpu", scheduler=None):
    cp = torch.load(path, map_location=device, weights_only=False)
    net.load_state_dict(cp["net"])
    optimizer.load_state_dict(cp["optimizer"])
    pool.load_state_dict(cp.get("pool", []))
    if scheduler is not None and "scheduler" in cp:
        scheduler.load_state_dict(cp["scheduler"])
    iteration = cp.get("iteration", 0)
    logging.info("Loaded checkpoint from %s (iteration %d)", path, iteration)
    return iteration


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: RunConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.getLogger("absl").setLevel(logging.WARNING)

    game = cfg.game.make_game()
    env = cfg.game.make_env()
    state_size: int = env.observation_spec()["info_state"][0]
    num_actions: int = env.action_spec()["num_actions"]
    num_players: int = game.num_players()
    device = cfg.training.device
    ac = cfg.agent

    logging.info(
        "Game: %s | players=%d state_size=%d num_actions=%d",
        cfg.game.name, num_players, state_size, num_actions,
    )

    net = ActorCritic(state_size, num_actions, ac.hidden_layers_sizes).to(device)
    opp_net = ActorCritic(state_size, num_actions, ac.hidden_layers_sizes).to(device)
    opp_net.eval()

    optimizer = torch.optim.Adam(net.parameters(), lr=ac.lr)
    pool = PolicyPool(max_size=ac.pool_size)

    checkpoint_path = cfg.training.checkpoint
    start_iter = 0
    if os.path.exists(checkpoint_path):
        start_iter = load_checkpoint(net, optimizer, pool, checkpoint_path, device)
    else:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    if len(pool) == 0:
        pool.add(net)

    wb_run = None
    if cfg.training.wandb.enabled:
        import wandb
        wb_run = wandb.init(
            project=cfg.training.wandb.project,
            group=cfg.training.wandb.group,
            name=cfg.training.wandb.run_name,
            tags=cfg.training.wandb.tags,
            config=cfg.model_dump(),
        )

    for it in range(start_iter, cfg.training.num_iterations):
        net.eval()
        buf, rewards = collect_episodes(
            env, net, opp_net, pool,
            num_episodes=ac.episodes_per_iter,
            num_players=num_players,
            num_actions=num_actions,
            device=device,
        )

        net.train()
        dataset = buf.build_dataset(
            gamma=ac.gamma, gae_lambda=ac.gae_lambda, device=device,
        )
        info = (
            ppo_update(
                net, optimizer, dataset,
                epochs=ac.ppo_epochs,
                batch_size=ac.batch_size,
                clip_epsilon=ac.clip_epsilon,
                entropy_coef=ac.entropy_coef,
                value_coef=ac.value_coef,
                max_grad_norm=ac.max_grad_norm,
            )
            if dataset
            else {"policy_loss": 0, "value_loss": 0, "entropy": 0}
        )

        if (it + 1) % ac.pool_save_interval == 0:
            pool.add(net)

        if (it + 1) % cfg.training.log_interval == 0:
            net.eval()
            eval_reward = eval_episodes(
                env, net, opp_net, pool,
                num_episodes=ac.eval_episodes,
                num_players=num_players,
                num_actions=num_actions,
                device=device,
            )

            logging.info(
                "Iter %d | train: reward %.3f +/-%.3f | eval: %.3f | "
                "pg=%.4f vf=%.4f ent=%.3f | pool=%d",
                it + 1, rewards.mean(), rewards.std(), eval_reward,
                info["policy_loss"], info["value_loss"], info["entropy"],
                len(pool),
            )

            if wb_run:
                import wandb
                wandb.log({
                    "iteration": it + 1,
                    "train/reward_mean": float(rewards.mean()),
                    "train/reward_std": float(rewards.std()),
                    "eval/reward_mean": eval_reward,
                    "policy_loss": info["policy_loss"],
                    "value_loss": info["value_loss"],
                    "entropy": info["entropy"],
                    "pool_size": len(pool),
                })

            save_checkpoint(net, optimizer, pool, checkpoint_path, it + 1)

    save_checkpoint(net, optimizer, pool, checkpoint_path, cfg.training.num_iterations)
    logging.info("Training complete.")
    if wb_run:
        wb_run.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a PPO agent for any OpenSpiel game.",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()
    train(RunConfig.from_yaml(args.config))


if __name__ == "__main__":
    main()
