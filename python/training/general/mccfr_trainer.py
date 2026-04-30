"""
Tabular MCCFR trainer for small 2-player zero-sum OpenSpiel games.

Runs external sampling MCCFR and periodically logs exploitability.
The solver and average policy are saved as pickle checkpoints.

Usage:
    python -m training.general.mccfr_trainer --config training/configs/liars_dice_mccfr.yaml
"""

import argparse
import logging
import os
import pickle
from pathlib import Path

from training.general.config import MCCFRRunConfig


def train(cfg: MCCFRRunConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.getLogger("absl").setLevel(logging.WARNING)

    import pyspiel
    from open_spiel.python.algorithms.external_sampling_mccfr import (
        ExternalSamplingSolver,
    )
    from open_spiel.python.algorithms import exploitability

    game = cfg.game.make_game()
    mc = cfg.mccfr

    logging.info(
        "Game: %s | players=%d",
        cfg.game.name, game.num_players(),
    )

    checkpoint_path = mc.checkpoint
    start_iter = 0

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            cp = pickle.load(f)
        solver = cp["solver"]
        start_iter = cp["iteration"]
        logging.info("Loaded checkpoint from %s (iteration %d)", checkpoint_path, start_iter)
    else:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        solver = ExternalSamplingSolver(game)

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

    for it in range(start_iter, mc.num_iterations):
        solver.iteration()

        if (it + 1) % mc.log_interval == 0:
            avg_policy = solver.average_policy()
            expl = exploitability.exploitability(game, avg_policy)

            logging.info(
                "Iter %d / %d | exploitability: %.6f",
                it + 1, mc.num_iterations, expl,
            )

            if wb_run:
                import wandb
                wandb.log({
                    "iteration": it + 1,
                    "exploitability": expl,
                })

            cp = {"solver": solver, "iteration": it + 1}
            with open(checkpoint_path, "wb") as f:
                pickle.dump(cp, f)

    # Final save
    avg_policy = solver.average_policy()
    expl = exploitability.exploitability(game, avg_policy)
    logging.info("Final exploitability: %.6f", expl)

    cp = {"solver": solver, "iteration": mc.num_iterations}
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
        description="Train tabular MCCFR for a small 2p zero-sum OpenSpiel game.",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()
    train(MCCFRRunConfig.from_yaml(args.config))


if __name__ == "__main__":
    main()
