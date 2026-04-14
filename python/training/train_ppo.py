"""
PPO self-play training for Oh Hell with separate bid and play networks.

The learner seat rotates across episodes so training data covers all positions.
Opponents are sampled from a pool of past policy snapshots to prevent cyclic
strategies.

Rollout collection is parallelised across CPU cores via ProcessPoolExecutor.
Each worker runs a self-contained episode loop and returns raw episode data;
the main process merges buffers and runs the PPO update.

Usage:
    python -m training.train_ppo --config training/configs/oh_hell_ppo_3tricks.yaml
"""

import argparse
import logging
import os
import resource
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from training.config import PPORunConfig
from training.ppo import ActorCritic, PolicyPool, RolloutBuffer, ppo_update


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_ppo(
    bid_net: ActorCritic,
    play_net: ActorCritic,
    bid_opt: torch.optim.Optimizer,
    play_opt: torch.optim.Optimizer,
    pool: PolicyPool,
    path: str,
    iteration: int,
    bid_sched=None,
    play_sched=None,
) -> None:
    data = {
        "bid_net": bid_net.state_dict(),
        "play_net": play_net.state_dict(),
        "bid_optimizer": bid_opt.state_dict(),
        "play_optimizer": play_opt.state_dict(),
        "pool": pool.state_dict(),
        "iteration": iteration,
    }
    if bid_sched is not None:
        data["bid_scheduler"] = bid_sched.state_dict()
    if play_sched is not None:
        data["play_scheduler"] = play_sched.state_dict()
    torch.save(data, path)
    logging.info("Saved checkpoint to %s (iteration %d)", path, iteration)


def load_ppo(
    bid_net: ActorCritic,
    play_net: ActorCritic,
    bid_opt: torch.optim.Optimizer,
    play_opt: torch.optim.Optimizer,
    pool: PolicyPool,
    path: str,
    device: str = "cpu",
    bid_sched=None,
    play_sched=None,
) -> int:
    cp = torch.load(path, map_location=device, weights_only=False)
    bid_net.load_state_dict(cp["bid_net"])
    play_net.load_state_dict(cp["play_net"])
    bid_opt.load_state_dict(cp["bid_optimizer"])
    play_opt.load_state_dict(cp["play_optimizer"])
    pool.load_state_dict(cp.get("pool", []))
    if bid_sched is not None and "bid_scheduler" in cp:
        bid_sched.load_state_dict(cp["bid_scheduler"])
    if play_sched is not None and "play_scheduler" in cp:
        play_sched.load_state_dict(cp["play_scheduler"])
    iteration = cp.get("iteration", 0)
    logging.info(
        "Loaded checkpoint from %s (iteration %d, pool=%d)",
        path, iteration, len(pool),
    )
    return iteration


# ---------------------------------------------------------------------------
# Parallel rollout workers
#
# These must be module-level functions (not closures) so that ProcessPoolExecutor
# can pickle them for the subprocess on macOS (spawn start method).
# ---------------------------------------------------------------------------

def _rollout_worker(args: tuple) -> tuple:
    """Collect training episodes in a subprocess.

    Returns (bid_episodes, play_episodes, rewards, bid_correct_count, bid_total).
    """
    (
        bid_sd, play_sd, pool_entries,
        num_episodes, seed,
        game_cfg,
        state_size, num_bids, deck_size, num_players,
        hidden_sizes, points_per_trick,
    ) = args

    import logging as _logging
    import random as _random
    import numpy as _np
    import torch as _torch
    from training.ppo import ActorCritic as _AC, RolloutBuffer as _RB, PolicyPool as _PP

    _logging.disable(_logging.CRITICAL)
    # Prevent intra-op thread parallelism from fighting across workers.
    _torch.set_num_threads(1)
    _random.seed(seed)
    _np.random.seed(seed)
    _torch.manual_seed(seed)

    env = game_cfg.make_env()

    bid_net = _AC(state_size, num_bids, hidden_sizes)
    play_net = _AC(state_size, deck_size, hidden_sizes)
    bid_net.load_state_dict(bid_sd)
    play_net.load_state_dict(play_sd)
    bid_net.eval()
    play_net.eval()

    opp_bid_net = _AC(state_size, num_bids, hidden_sizes)
    opp_play_net = _AC(state_size, deck_size, hidden_sizes)
    # Default opponent = current policy; overridden per episode if pool is non-empty.
    opp_bid_net.load_state_dict(bid_sd)
    opp_play_net.load_state_dict(play_sd)
    opp_bid_net.eval()
    opp_play_net.eval()

    pool = _PP(max_size=len(pool_entries) + 1)
    pool._entries = list(pool_entries)

    buffer = _RB()
    rewards = _np.zeros(num_episodes)
    bid_correct_count = 0
    bid_total = 0

    for ep_idx in range(num_episodes):
        learner = ep_idx % num_players
        learner_bid_idx = None

        if len(pool) > 0:
            snapshot = pool.sample()
            opp_bid_net.load_state_dict(snapshot["bid"])
            opp_play_net.load_state_dict(snapshot["play"])

        time_step = env.reset()

        while not time_step.last():
            player = time_step.observations["current_player"]
            obs_np = _np.asarray(
                time_step.observations["info_state"][player], dtype=_np.float32,
            )
            legal = time_step.observations["legal_actions"][player]
            is_bid = legal[0] >= deck_size
            is_learner = player == learner
            obs_t = _torch.as_tensor(obs_np)

            if is_bid:
                bid_indices = [a - deck_size for a in legal]
                mask = _np.zeros(num_bids, dtype=_np.bool_)
                mask[bid_indices] = True
                mask_t = _torch.as_tensor(mask)
                net = bid_net if is_learner else opp_bid_net
                action_idx, lp, val = net.act(obs_t, mask_t)
                if is_learner:
                    learner_bid_idx = action_idx
                    buffer.add_bid(obs_np, action_idx, lp, val, mask)
                action = action_idx + deck_size
            else:
                mask = _np.zeros(deck_size, dtype=_np.bool_)
                mask[legal] = True
                mask_t = _torch.as_tensor(mask)
                net = play_net if is_learner else opp_play_net
                action_idx, lp, val = net.act(obs_t, mask_t)
                if is_learner:
                    buffer.add_play(obs_np, action_idx, lp, val, mask)
                action = action_idx

            time_step = env.step([action])

        reward = time_step.rewards[learner]
        buffer.finish_episode(reward)
        rewards[ep_idx] = reward

        if learner_bid_idx is not None:
            bid_total += 1
            if abs(reward - (learner_bid_idx * points_per_trick + 10)) < 1e-6:
                bid_correct_count += 1

    return buffer._bid_eps, buffer._play_eps, rewards, bid_correct_count, bid_total


def _eval_worker(args: tuple) -> tuple:
    """Greedy eval episodes in a subprocess.

    Returns (rewards, bid_correct_count, bid_total).
    """
    (
        bid_sd, play_sd, pool_entries,
        num_episodes, seed,
        game_cfg,
        state_size, num_bids, deck_size, num_players,
        hidden_sizes, points_per_trick,
    ) = args

    import logging as _logging
    import numpy as _np
    import torch as _torch
    from training.ppo import ActorCritic as _AC, PolicyPool as _PP

    _logging.disable(_logging.CRITICAL)
    _torch.set_num_threads(1)
    _np.random.seed(seed)
    _torch.manual_seed(seed)

    env = game_cfg.make_env()

    bid_net = _AC(state_size, num_bids, hidden_sizes)
    play_net = _AC(state_size, deck_size, hidden_sizes)
    bid_net.load_state_dict(bid_sd)
    play_net.load_state_dict(play_sd)
    bid_net.eval()
    play_net.eval()

    opp_bid_net = _AC(state_size, num_bids, hidden_sizes)
    opp_play_net = _AC(state_size, deck_size, hidden_sizes)
    opp_bid_net.load_state_dict(bid_sd)
    opp_play_net.load_state_dict(play_sd)
    opp_bid_net.eval()
    opp_play_net.eval()

    pool = _PP(max_size=len(pool_entries) + 1)
    pool._entries = list(pool_entries)

    rewards = _np.zeros(num_episodes)
    bid_correct_count = 0
    bid_total = 0

    for ep_idx in range(num_episodes):
        learner = ep_idx % num_players
        learner_bid_idx = None

        if len(pool) > 0:
            snapshot = pool.sample()
            opp_bid_net.load_state_dict(snapshot["bid"])
            opp_play_net.load_state_dict(snapshot["play"])

        time_step = env.reset()

        while not time_step.last():
            player = time_step.observations["current_player"]
            obs_np = _np.asarray(
                time_step.observations["info_state"][player], dtype=_np.float32,
            )
            legal = time_step.observations["legal_actions"][player]
            is_bid = legal[0] >= deck_size
            is_learner = player == learner
            obs_t = _torch.as_tensor(obs_np)

            if is_bid:
                bid_indices = [a - deck_size for a in legal]
                mask = _np.zeros(num_bids, dtype=_np.bool_)
                mask[bid_indices] = True
                mask_t = _torch.as_tensor(mask)
                net = bid_net if is_learner else opp_bid_net
                action_idx, _, _ = net.act(obs_t, mask_t, greedy=is_learner)
                if is_learner:
                    learner_bid_idx = action_idx
                action = action_idx + deck_size
            else:
                mask = _np.zeros(deck_size, dtype=_np.bool_)
                mask[legal] = True
                mask_t = _torch.as_tensor(mask)
                net = play_net if is_learner else opp_play_net
                action_idx, _, _ = net.act(obs_t, mask_t, greedy=is_learner)
                action = action_idx

            time_step = env.step([action])

        reward = time_step.rewards[learner]
        rewards[ep_idx] = reward

        if learner_bid_idx is not None:
            bid_total += 1
            if abs(reward - (learner_bid_idx * points_per_trick + 10)) < 1e-6:
                bid_correct_count += 1

    return rewards, bid_correct_count, bid_total


# ---------------------------------------------------------------------------
# Rollout collection — sequential (num_workers=1) and parallel
# ---------------------------------------------------------------------------

def _bid_correct(reward: float, bid_idx: int, points_per_trick: int) -> bool:
    """Return True if the agent hit their bid exactly."""
    return abs(reward - (bid_idx * points_per_trick + 10)) < 1e-6


def _make_worker_args(
    bid_net, play_net, pool,
    num_episodes, num_workers, iteration,
    game_cfg, state_size, num_bids, deck_size, num_players,
    hidden_sizes, points_per_trick,
):
    """Build the args list for _rollout_worker or _eval_worker."""
    bid_sd = {k: v.cpu() for k, v in bid_net.state_dict().items()}
    play_sd = {k: v.cpu() for k, v in play_net.state_dict().items()}
    pool_entries = pool.state_dict()

    base, rem = divmod(num_episodes, num_workers)
    counts = [base + (1 if i < rem else 0) for i in range(num_workers)]

    return [
        (
            bid_sd, play_sd, pool_entries,
            counts[i],
            iteration * num_workers + i,   # unique seed per worker per iteration
            game_cfg,
            state_size, num_bids, deck_size, num_players,
            hidden_sizes, points_per_trick,
        )
        for i in range(num_workers)
    ]


def collect_rollouts(
    env,
    bid_net: ActorCritic,
    play_net: ActorCritic,
    opp_bid_net: ActorCritic,
    opp_play_net: ActorCritic,
    pool: PolicyPool,
    *,
    num_episodes: int,
    num_players: int,
    deck_size: int,
    num_bids: int,
    points_per_trick: int = 1,
    device: str = "cpu",
) -> tuple[RolloutBuffer, np.ndarray, float]:
    """Sequential rollout collection (num_workers=1 path)."""
    buffer = RolloutBuffer()
    rewards = np.zeros(num_episodes)
    bid_correct_count = 0
    bid_total = 0

    for ep_idx in range(num_episodes):
        learner = ep_idx % num_players
        learner_bid_idx: int | None = None

        if len(pool) > 0:
            snapshot = pool.sample()
            opp_bid_net.load_state_dict(snapshot["bid"])
            opp_play_net.load_state_dict(snapshot["play"])

        time_step = env.reset()

        while not time_step.last():
            player = time_step.observations["current_player"]
            obs_np = np.asarray(
                time_step.observations["info_state"][player], dtype=np.float32,
            )
            legal = time_step.observations["legal_actions"][player]
            is_bid = legal[0] >= deck_size
            is_learner = player == learner
            obs_t = torch.as_tensor(obs_np, device=device)

            if is_bid:
                bid_indices = [a - deck_size for a in legal]
                mask = np.zeros(num_bids, dtype=np.bool_)
                mask[bid_indices] = True
                mask_t = torch.as_tensor(mask, device=device)
                net = bid_net if is_learner else opp_bid_net
                action_idx, lp, val = net.act(obs_t, mask_t)
                if is_learner:
                    learner_bid_idx = action_idx
                    buffer.add_bid(obs_np, action_idx, lp, val, mask)
                action = action_idx + deck_size
            else:
                mask = np.zeros(deck_size, dtype=np.bool_)
                mask[legal] = True
                mask_t = torch.as_tensor(mask, device=device)
                net = play_net if is_learner else opp_play_net
                action_idx, lp, val = net.act(obs_t, mask_t)
                if is_learner:
                    buffer.add_play(obs_np, action_idx, lp, val, mask)
                action = action_idx

            time_step = env.step([action])

        reward = time_step.rewards[learner]
        buffer.finish_episode(reward)
        rewards[ep_idx] = reward

        if learner_bid_idx is not None:
            bid_total += 1
            if _bid_correct(reward, learner_bid_idx, points_per_trick):
                bid_correct_count += 1

    bid_acc = bid_correct_count / bid_total if bid_total > 0 else 0.0
    return buffer, rewards, bid_acc


def collect_rollouts_parallel(
    executor: ProcessPoolExecutor,
    bid_net: ActorCritic,
    play_net: ActorCritic,
    pool: PolicyPool,
    *,
    num_episodes: int,
    num_workers: int,
    iteration: int,
    game_cfg,
    state_size: int,
    num_bids: int,
    deck_size: int,
    num_players: int,
    hidden_sizes: list,
    points_per_trick: int = 1,
) -> tuple[RolloutBuffer, np.ndarray, float]:
    """Parallel rollout collection across num_workers subprocesses."""
    args_list = _make_worker_args(
        bid_net, play_net, pool,
        num_episodes, num_workers, iteration,
        game_cfg, state_size, num_bids, deck_size, num_players,
        hidden_sizes, points_per_trick,
    )

    all_bid_eps: list = []
    all_play_eps: list = []
    all_rewards: list = []
    total_correct = 0
    total_bids = 0

    for bid_eps, play_eps, rewards, bid_correct, bid_total in executor.map(_rollout_worker, args_list):
        all_bid_eps.extend(bid_eps)
        all_play_eps.extend(play_eps)
        all_rewards.append(rewards)
        total_correct += bid_correct
        total_bids += bid_total

    buffer = RolloutBuffer()
    buffer._bid_eps = all_bid_eps
    buffer._play_eps = all_play_eps

    bid_acc = total_correct / total_bids if total_bids > 0 else 0.0
    return buffer, np.concatenate(all_rewards), bid_acc


def eval_rollouts(
    env,
    bid_net: ActorCritic,
    play_net: ActorCritic,
    opp_bid_net: ActorCritic,
    opp_play_net: ActorCritic,
    pool: PolicyPool,
    *,
    num_episodes: int,
    num_players: int,
    deck_size: int,
    num_bids: int,
    points_per_trick: int = 1,
    device: str = "cpu",
) -> tuple[float, float]:
    """Sequential greedy eval (num_workers=1 path)."""
    rewards = np.zeros(num_episodes)
    bid_correct_count = 0
    bid_total = 0

    bid_net.eval()
    play_net.eval()

    for ep_idx in range(num_episodes):
        learner = ep_idx % num_players
        learner_bid_idx: int | None = None

        if len(pool) > 0:
            snapshot = pool.sample()
            opp_bid_net.load_state_dict(snapshot["bid"])
            opp_play_net.load_state_dict(snapshot["play"])

        time_step = env.reset()

        while not time_step.last():
            player = time_step.observations["current_player"]
            obs_np = np.asarray(
                time_step.observations["info_state"][player], dtype=np.float32,
            )
            legal = time_step.observations["legal_actions"][player]
            is_bid = legal[0] >= deck_size
            is_learner = player == learner
            obs_t = torch.as_tensor(obs_np, device=device)

            if is_bid:
                bid_indices = [a - deck_size for a in legal]
                mask = np.zeros(num_bids, dtype=np.bool_)
                mask[bid_indices] = True
                mask_t = torch.as_tensor(mask, device=device)
                net = bid_net if is_learner else opp_bid_net
                action_idx, _, _ = net.act(obs_t, mask_t, greedy=is_learner)
                if is_learner:
                    learner_bid_idx = action_idx
                action = action_idx + deck_size
            else:
                mask = np.zeros(deck_size, dtype=np.bool_)
                mask[legal] = True
                mask_t = torch.as_tensor(mask, device=device)
                net = play_net if is_learner else opp_play_net
                action_idx, _, _ = net.act(obs_t, mask_t, greedy=is_learner)
                action = action_idx

            time_step = env.step([action])

        reward = time_step.rewards[learner]
        rewards[ep_idx] = reward

        if learner_bid_idx is not None:
            bid_total += 1
            if _bid_correct(reward, learner_bid_idx, points_per_trick):
                bid_correct_count += 1

    bid_acc = bid_correct_count / bid_total if bid_total > 0 else 0.0
    return float(rewards.mean()), bid_acc


def eval_rollouts_parallel(
    executor: ProcessPoolExecutor,
    bid_net: ActorCritic,
    play_net: ActorCritic,
    pool: PolicyPool,
    *,
    num_episodes: int,
    num_workers: int,
    iteration: int,
    game_cfg,
    state_size: int,
    num_bids: int,
    deck_size: int,
    num_players: int,
    hidden_sizes: list,
    points_per_trick: int = 1,
) -> tuple[float, float]:
    """Parallel greedy eval across num_workers subprocesses."""
    args_list = _make_worker_args(
        bid_net, play_net, pool,
        num_episodes, num_workers, iteration,
        game_cfg, state_size, num_bids, deck_size, num_players,
        hidden_sizes, points_per_trick,
    )

    all_rewards: list = []
    total_correct = 0
    total_bids = 0

    for rewards, bid_correct, bid_total in executor.map(_eval_worker, args_list):
        all_rewards.append(rewards)
        total_correct += bid_correct
        total_bids += bid_total

    bid_acc = total_correct / total_bids if total_bids > 0 else 0.0
    return float(np.concatenate(all_rewards).mean()), bid_acc


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: PPORunConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    # Suppress noisy OpenSpiel env logs (e.g. "Using game settings: ...").
    logging.getLogger("absl").setLevel(logging.WARNING)

    # Raise the open file descriptor limit — parallel workers use many pipes.
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))

    # Create one env in the main process to query observation/action sizes.
    env = cfg.game.make_env()
    state_size: int = env.observation_spec()["info_state"][0]
    num_actions: int = env.action_spec()["num_actions"]
    num_players: int = cfg.game.players
    deck_size: int = cfg.game.num_cards_per_suit * cfg.game.num_suits
    num_bids: int = num_actions - deck_size

    num_workers: int = cfg.training.num_workers

    logging.info(
        "Game: %s | players=%d state_size=%d deck_size=%d num_bids=%d workers=%d",
        cfg.game.name, num_players, state_size, deck_size, num_bids, num_workers,
    )
    logging.info("Game config:\n%s", cfg.game.model_dump_json(indent=2))
    logging.info("Agent config:\n%s", cfg.agent.model_dump_json(indent=2))

    device = cfg.training.device
    ac = cfg.agent

    bid_net = ActorCritic(state_size, num_bids, ac.hidden_layers_sizes).to(device)
    play_net = ActorCritic(state_size, deck_size, ac.hidden_layers_sizes).to(device)

    # Sequential-path opponent copies (unused when num_workers > 1).
    opp_bid_net = ActorCritic(state_size, num_bids, ac.hidden_layers_sizes).to(device)
    opp_play_net = ActorCritic(state_size, deck_size, ac.hidden_layers_sizes).to(device)
    opp_bid_net.eval()
    opp_play_net.eval()

    bid_lr = ac.bid_lr if ac.bid_lr is not None else ac.lr
    bid_opt = torch.optim.Adam(bid_net.parameters(), lr=bid_lr)
    play_opt = torch.optim.Adam(play_net.parameters(), lr=ac.lr)

    num_iterations = cfg.training.num_iterations or (
        cfg.training.num_episodes // ac.episodes_per_iter
    )

    bid_sched = play_sched = None
    if ac.lr_schedule == "cosine":
        bid_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            bid_opt, T_max=num_iterations, eta_min=0,
        )
        play_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            play_opt, T_max=num_iterations, eta_min=0,
        )

    pool = PolicyPool(max_size=ac.pool_size)

    checkpoint_path = cfg.training.checkpoint
    start_iter = 0
    if os.path.exists(checkpoint_path):
        start_iter = load_ppo(
            bid_net, play_net, bid_opt, play_opt, pool, checkpoint_path, device,
            bid_sched=bid_sched, play_sched=play_sched,
        )
    else:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    if len(pool) == 0:
        pool.add(bid_net, play_net)

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

    log_interval = cfg.training.log_interval

    # Shared kwargs for parallel collection helpers.
    _parallel_kw = dict(
        game_cfg=cfg.game,
        state_size=state_size,
        num_bids=num_bids,
        deck_size=deck_size,
        num_players=num_players,
        hidden_sizes=ac.hidden_layers_sizes,
        points_per_trick=cfg.game.points_per_trick,
    )

    executor = ProcessPoolExecutor(max_workers=num_workers) if num_workers > 1 else None

    try:
        for it in range(start_iter, num_iterations):
            bid_net.eval()
            play_net.eval()

            if executor is not None:
                buf, rewards, train_bid_acc = collect_rollouts_parallel(
                    executor, bid_net, play_net, pool,
                    num_episodes=ac.episodes_per_iter,
                    num_workers=num_workers,
                    iteration=it,
                    **_parallel_kw,
                )
            else:
                opp_bid_net.load_state_dict(bid_net.state_dict())
                opp_play_net.load_state_dict(play_net.state_dict())
                buf, rewards, train_bid_acc = collect_rollouts(
                    env, bid_net, play_net, opp_bid_net, opp_play_net, pool,
                    num_episodes=ac.episodes_per_iter,
                    num_players=num_players,
                    deck_size=deck_size,
                    num_bids=num_bids,
                    points_per_trick=cfg.game.points_per_trick,
                    device=device,
                )

            # PPO update
            bid_net.train()
            play_net.train()

            bid_data = buf.build_bid_dataset(
                gamma=ac.gamma, gae_lambda=ac.gae_lambda, device=device,
            )
            play_data = buf.build_play_dataset(
                gamma=ac.gamma, gae_lambda=ac.gae_lambda, device=device,
            )

            _common_kw = dict(
                batch_size=ac.batch_size,
                clip_epsilon=ac.clip_epsilon,
                value_coef=ac.value_coef,
                max_grad_norm=ac.max_grad_norm,
            )
            bid_info = (
                ppo_update(bid_net, bid_opt, bid_data,
                           epochs=ac.bid_ppo_epochs, entropy_coef=ac.bid_entropy_coef, **_common_kw)
                if bid_data
                else {"policy_loss": 0, "value_loss": 0, "entropy": 0}
            )
            play_info = (
                ppo_update(play_net, play_opt, play_data,
                           epochs=ac.play_ppo_epochs, entropy_coef=ac.entropy_coef, **_common_kw)
                if play_data
                else {"policy_loss": 0, "value_loss": 0, "entropy": 0}
            )

            if bid_sched is not None:
                bid_sched.step()
            if play_sched is not None:
                play_sched.step()

            if (it + 1) % ac.pool_save_interval == 0:
                pool.add(bid_net, play_net)

            if (it + 1) % log_interval == 0:
                ep_total = (it + 1) * ac.episodes_per_iter

                bid_net.eval()
                play_net.eval()

                if executor is not None:
                    eval_reward, eval_bid_acc = eval_rollouts_parallel(
                        executor, bid_net, play_net, pool,
                        num_episodes=ac.eval_episodes,
                        num_workers=num_workers,
                        iteration=it,
                        **_parallel_kw,
                    )
                else:
                    eval_reward, eval_bid_acc = eval_rollouts(
                        env, bid_net, play_net, opp_bid_net, opp_play_net, pool,
                        num_episodes=ac.eval_episodes,
                        num_players=num_players,
                        deck_size=deck_size,
                        num_bids=num_bids,
                        points_per_trick=cfg.game.points_per_trick,
                        device=device,
                    )

                logging.info(
                    "Iter %d (%d eps) | "
                    "train: reward %.2f +/-%.2f bid_acc %.3f | "
                    "eval: reward %.2f bid_acc %.3f | "
                    "bid: pg=%.4f vf=%.4f ent=%.3f | "
                    "play: pg=%.4f vf=%.4f ent=%.3f | pool=%d",
                    it + 1,
                    ep_total,
                    rewards.mean(),
                    rewards.std(),
                    train_bid_acc,
                    eval_reward,
                    eval_bid_acc,
                    bid_info["policy_loss"],
                    bid_info["value_loss"],
                    bid_info["entropy"],
                    play_info["policy_loss"],
                    play_info["value_loss"],
                    play_info["entropy"],
                    len(pool),
                )

                if wb_run:
                    import wandb
                    wandb.log(
                        {
                            "iteration": it + 1,
                            "episodes": ep_total,
                            "train/reward_mean": float(rewards.mean()),
                            "train/reward_std": float(rewards.std()),
                            "train/bid_acc": train_bid_acc,
                            "eval/reward_mean": eval_reward,
                            "eval/bid_acc": eval_bid_acc,
                            "bid/policy_loss": bid_info["policy_loss"],
                            "bid/value_loss": bid_info["value_loss"],
                            "bid/entropy": bid_info["entropy"],
                            "play/policy_loss": play_info["policy_loss"],
                            "play/value_loss": play_info["value_loss"],
                            "play/entropy": play_info["entropy"],
                            "pool_size": len(pool),
                            "lr/bid": bid_opt.param_groups[0]["lr"],
                            "lr/play": play_opt.param_groups[0]["lr"],
                        },
                        step=ep_total,
                    )

                save_ppo(
                    bid_net, play_net, bid_opt, play_opt, pool,
                    checkpoint_path, it + 1,
                    bid_sched=bid_sched, play_sched=play_sched,
                )

    finally:
        if executor is not None:
            executor.shutdown(wait=False)

    # Final save
    save_ppo(
        bid_net, play_net, bid_opt, play_opt, pool,
        checkpoint_path, num_iterations,
        bid_sched=bid_sched, play_sched=play_sched,
    )
    logging.info("Training complete.")

    if wb_run:
        wb_run.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO agents for Oh Hell.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML PPORunConfig.",
    )
    args = parser.parse_args()
    cfg = PPORunConfig.from_yaml(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
