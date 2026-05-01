"""
Deep CFR trainer for 2-player zero-sum OpenSpiel games.

Wraps OpenSpiel's PyTorch DeepCFRSolver. Drives the solver's inner loop
manually rather than calling solve() so we can log and checkpoint at intervals.

We do NOT compute full exploitability during training — for non-trivial games
it requires materializing a tabular policy over every info state and traversing
the entire game tree, which is memory-prohibitive in OpenSpiel's Python
implementation. Instead we log advantage-network loss and memory sizes as
training health signals. The strategy (average policy) network is trained
once at the end so the saved checkpoint produces a usable policy. Evaluate
quality via head-to-head play against a reference solver (e.g. tabular MCCFR).

Usage:
    python -m training.general.dcfr_trainer --config training/configs/liars_dice_dcfr.yaml
"""

import argparse
import gc
import logging
import os
import pickle
import time
from pathlib import Path

from training.general.config import DeepCFRRunConfig


def train(cfg: DeepCFRRunConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.getLogger("absl").setLevel(logging.WARNING)

    from open_spiel.python.pytorch.deep_cfr import DeepCFRSolver

    game = cfg.game.make_game()
    dc = cfg.deep_cfr

    logging.info(
        "Game: %s | players=%d",
        cfg.game.name, game.num_players(),
    )

    checkpoint_path = dc.checkpoint
    start_iter = 0

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            cp = pickle.load(f)
        solver = cp["solver"]
        start_iter = cp["iteration"]
        logging.info("Loaded checkpoint from %s (iteration %d)", checkpoint_path, start_iter)
    else:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        solver = DeepCFRSolver(
            game,
            policy_network_layers=tuple(dc.policy_network_layers),
            advantage_network_layers=tuple(dc.advantage_network_layers),
            num_iterations=dc.num_iterations,
            num_traversals=dc.num_traversals,
            learning_rate=dc.lr,
            batch_size_advantage=dc.batch_size_advantage,
            batch_size_strategy=dc.batch_size_strategy,
            memory_capacity=dc.memory_capacity,
            policy_network_train_steps=dc.policy_network_train_steps,
            advantage_network_train_steps=dc.advantage_network_train_steps,
            reinitialize_advantage_networks=dc.reinitialize_advantage_networks,
            device=dc.device,
        )

    wb_run = None
    if cfg.wandb.enabled:
        import wandb
        wb_run = wandb.init(
            project=cfg.wandb.project,
            group=cfg.wandb.group,
            name=cfg.wandb.run_name,
            tags=cfg.wandb.tags,
            config=cfg.model_dump(),
        )

    num_players = game.num_players()
    root_state = game.new_initial_state()

    for it in range(start_iter, dc.num_iterations):
        iter_start = time.time()
        adv_losses = []

        for p in range(num_players):
            for _ in range(dc.num_traversals):
                solver._traverse_game_tree(root_state, p)
            if dc.reinitialize_advantage_networks:
                solver._reinitialize_advantage_network(p)
            adv_losses.append(solver._learn_advantage_network(p))
        solver._iteration += 1

        iter_time = time.time() - iter_start

        if (it + 1) % dc.log_interval == 0:
            adv_mem_size = sum(len(m) for m in solver._advantage_memories)
            strat_mem_size = len(solver._strategy_memories)
            mean_adv_loss = sum(adv_losses) / len(adv_losses) if adv_losses else 0.0

            logging.info(
                "Iter %d / %d | adv_loss: %.4f | adv_mem: %d | strat_mem: %d | iter_time: %.2fs",
                it + 1, dc.num_iterations, mean_adv_loss,
                adv_mem_size, strat_mem_size, iter_time,
            )

            if wb_run:
                import wandb
                metrics = {
                    "iteration": it + 1,
                    "advantage_loss": mean_adv_loss,
                    "advantage_memory_size": adv_mem_size,
                    "strategy_memory_size": strat_mem_size,
                    "iter_time_seconds": iter_time,
                }
                for p, loss in enumerate(adv_losses):
                    metrics[f"advantage_loss_p{p}"] = loss
                wandb.log(metrics)

            cp = {"solver": solver, "iteration": it + 1}
            with open(checkpoint_path, "wb") as f:
                pickle.dump(cp, f)

            gc.collect()

    # Train the strategy (average policy) network once at the end so the
    # saved checkpoint produces a usable policy for inference / evaluation.
    logging.info("Training final strategy network...")
    solver._reinitialize_policy_network()
    policy_loss = solver._learn_strategy_network()
    logging.info(
        "Final policy_loss: %s",
        f"{policy_loss:.6f}" if policy_loss is not None else "N/A",
    )

    if wb_run and policy_loss is not None:
        import wandb
        wandb.log({"policy_loss": policy_loss})

    cp = {"solver": solver, "iteration": dc.num_iterations}
    with open(checkpoint_path, "wb") as f:
        pickle.dump(cp, f)
    logging.info("Saved final checkpoint to %s", checkpoint_path)

    if wb_run:
        wb_run.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Deep CFR for a 2-player zero-sum OpenSpiel game.",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()
    train(DeepCFRRunConfig.from_yaml(args.config))


if __name__ == "__main__":
    main()
