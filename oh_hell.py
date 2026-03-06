import logging
import sys
from typing import Dict, List, Sequence

from absl import app
from absl import flags
import numpy as np

import matplotlib.pyplot as plt
from open_spiel.python import rl_environment
from open_spiel.python import rl_tools
from open_spiel.python.algorithms import random_agent
from open_spiel.python.algorithms import tabular_qlearner
from open_spiel.python.pytorch import dqn

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_episodes", int(5e4), "Number of training episodes.")
flags.DEFINE_enum(
    "mode",
    "tabular",
    ["tabular", "dqn"],
    "Agent type: 'tabular' (small deck) or 'dqn' (full deck).",
)

NUM_PLAYERS = 4

# Small deck for tabular Q-learning: 4 suits * 3 ranks = 12 cards.
# With 3 players and 2 tricks each needs 6 cards + 1 trump reveal = 7 cards minimum.
TABULAR_PARAMS = {
    "players": NUM_PLAYERS,
    "num_cards_per_suit": 4,
    "num_suits": 3,
    "off_bid_penalty": False,
    "points_per_trick": 1,
}

# Full deck for DQN: standard 52-card deck.
DQN_PARAMS = {
    "players": NUM_PLAYERS,
    "num_cards_per_suit": 13,
    "num_suits": 4,
    "off_bid_penalty": False,
    "points_per_trick": 1,
}


def make_tabular_agents(
    num_actions: int, num_episodes: int
) -> List[tabular_qlearner.QLearner]:
    # Estimate total training steps: ~10 steps/episode across 3 players.
    total_steps = num_episodes * 10
    return [
        tabular_qlearner.QLearner(
            player_id=idx,
            num_actions=num_actions,
            step_size=0.5,
            epsilon_schedule=rl_tools.LinearSchedule(
                init_val=0.8, final_val=0.05, num_steps=total_steps
            ),
        )
        for idx in range(NUM_PLAYERS)
    ]


def make_dqn_agents(
    state_size: int, num_actions: int, num_episodes: int
) -> List[dqn.DQN]:
    # Estimate total training steps: ~10 steps/episode across 3 players.
    total_steps = num_episodes * 10
    return [
        dqn.DQN(
            player_id=idx,
            state_representation_size=state_size,
            num_actions=num_actions,
            hidden_layers_sizes=[256, 256],
            replay_buffer_capacity=50000,
            batch_size=128,
            learning_rate=0.001,
            optimizer_str="adam",
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_duration=int(total_steps * 0.8),
            update_target_network_every=500,
            learn_every=10,
        )
        for idx in range(NUM_PLAYERS)
    ]


def train(
    env: rl_environment.Environment, agents: list, num_episodes: int
) -> Dict[str, np.ndarray]:
    """Train agents for num_episodes, printing progress every 10k episodes.

    Returns a dict with:
      "rewards": float array of shape (num_episodes, NUM_PLAYERS)
      "losses":  float array of shape (num_episodes, NUM_PLAYERS), NaN where unavailable
    """
    episode_rewards = np.zeros((num_episodes, NUM_PLAYERS))
    episode_losses = np.full((num_episodes, NUM_PLAYERS), np.nan)
    reward_window = np.zeros(NUM_PLAYERS)
    window_size = int(1e4)

    for ep in range(num_episodes):
        if ep % window_size == 0:
            logging.info(
                "Episode %d/%d | avg reward (last %d eps): %s",
                ep,
                num_episodes,
                window_size,
                reward_window,
            )
            reward_window = np.zeros(NUM_PLAYERS)

        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent_out = agents[player_id].step(time_step)
            time_step = env.step([agent_out.action])

        for agent in agents:
            agent.step(time_step)

        episode_rewards[ep] = time_step.rewards
        for i, agent in enumerate(agents):
            loss = agent.loss
            if loss is not None:
                # DQN returns a tensor; tabular returns a float.
                episode_losses[ep, i] = float(loss)

        n = (ep % window_size) + 1
        reward_window += (time_step.rewards - reward_window) / n

    logging.info("Training complete. Final window avg reward: %s", reward_window)
    return {"rewards": episode_rewards, "losses": episode_losses}


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean ignoring NaN."""
    out = np.full_like(values, np.nan)
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1) : i + 1]
        valid = chunk[~np.isnan(chunk)]
        if valid.size > 0:
            out[i] = valid.mean()
    return out


def plot_curves(
    data: Dict[str, np.ndarray],
    mode: str,
    smooth_window: int = 500,
    filename: str = "learning_curves.png",
) -> None:
    """Plot reward (all modes) and loss (DQN only) learning curves."""
    rewards = data["rewards"]   # (num_episodes, NUM_PLAYERS)
    losses = data["losses"]     # (num_episodes, NUM_PLAYERS)
    episodes = np.arange(len(rewards))

    has_loss = not np.all(np.isnan(losses))
    num_plots = 2 if has_loss else 1
    fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 5))
    if num_plots == 1:
        axes = [axes]

    # Reward plot
    ax = axes[0]
    for i in range(NUM_PLAYERS):
        smoothed = _smooth(rewards[:, i], smooth_window)
        ax.plot(episodes, smoothed, label=f"Player {i}")
    ax.set_title(f"Average Reward per Episode ({mode})")
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Reward (smoothed, window={smooth_window})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Loss plot (DQN only)
    if has_loss:
        ax = axes[1]
        for i in range(NUM_PLAYERS):
            smoothed = _smooth(losses[:, i], smooth_window)
            valid = ~np.isnan(smoothed)
            ax.plot(episodes[valid], smoothed[valid], label=f"Player {i}")
        ax.set_title("DQN Loss per Episode")
        ax.set_xlabel("Episode")
        ax.set_ylabel(f"Loss (smoothed, window={smooth_window})")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    logging.info("Learning curves saved to %s", filename)


def print_state(
    env: rl_environment.Environment, time_step: rl_environment.TimeStep
) -> None:
    state = env.get_state
    print("\n" + str(state))
    player_id = time_step.observations["current_player"]
    legal_actions = time_step.observations["legal_actions"][player_id]
    action_names = {a: state.action_to_string(player_id, a) for a in legal_actions}
    print(f"Player {player_id}'s turn. Legal actions: {action_names}")


def command_line_action(
    env: rl_environment.Environment, time_step: rl_environment.TimeStep
) -> int:
    player_id = time_step.observations["current_player"]
    legal_actions = time_step.observations["legal_actions"][player_id]
    action = -1
    while action not in legal_actions:
        print(f"Choose action id {legal_actions}: ", end="")
        sys.stdout.flush()
        try:
            action = int(input())
        except ValueError:
            continue
    return action


def play_interactive(env: rl_environment.Environment, agents: list) -> None:
    human_player = 0
    logging.info("You are playing as Player %d.", human_player)

    while True:
        time_step = env.reset()
        while not time_step.last():
            print_state(env, time_step)
            player_id = time_step.observations["current_player"]
            if player_id == human_player:
                action = command_line_action(env, time_step)
            else:
                agent_out = agents[player_id].step(time_step, is_evaluation=True)
                action = agent_out.action
                state = env.get_state
                logging.info(
                    "Player %d plays: %s",
                    player_id,
                    state.action_to_string(player_id, action),
                )
            time_step = env.step([action])

        print("\n" + str(env.get_state))
        print(f"Game over! Scores: {time_step.rewards}")
        print("Play again? [y/n] ", end="")
        if input().strip().lower() != "y":
            break


def main(_: Sequence[str]) -> None:
    logging.basicConfig(level=logging.INFO)

    if FLAGS.mode == "tabular":
        logging.info("Mode: tabular Q-learning (small deck: 4 cards/suit, 3 suits)")
        params = TABULAR_PARAMS
    else:
        logging.info("Mode: DQN (full deck: 13 cards/suit, 4 suits)")
        params = DQN_PARAMS

    env = rl_environment.Environment("oh_hell", **params)
    num_actions = env.action_spec()["num_actions"]

    if FLAGS.mode == "tabular":
        agents = make_tabular_agents(num_actions, FLAGS.num_episodes)
    else:
        state_size = env.observation_spec()["info_state"][0]
        agents = make_dqn_agents(state_size, num_actions, FLAGS.num_episodes)

    data = train(env, agents, FLAGS.num_episodes)
    plot_curves(data, FLAGS.mode)
    play_interactive(env, agents)


if __name__ == "__main__":
    app.run(main)
