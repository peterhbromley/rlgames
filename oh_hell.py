import logging
import os
import sys
from typing import Dict, Sequence

from absl import app
from absl import flags
import numpy as np
import torch

import matplotlib.pyplot as plt
from open_spiel.python import rl_environment
from open_spiel.python.pytorch import dqn

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_episodes", int(5e4), "Number of training episodes.")
flags.DEFINE_string("checkpoint", "oh_hell_dqn.pt", "Path for saving/loading the agent checkpoint.")
flags.DEFINE_boolean("play_only", False, "Skip training and load checkpoint for interactive play.")

NUM_PLAYERS = 4
NUM_TRICKS = 2

DQN_PARAMS = {
    "players": NUM_PLAYERS,
    "num_cards_per_suit": 13,
    "num_suits": 4,
    "num_tricks_fixed": NUM_TRICKS,
    "off_bid_penalty": False,
    "points_per_trick": 1,
}


class SelfPlayDQN:
    """Single DQN shared across all players for symmetric self-play.

    One network and one replay buffer are updated from every player's
    perspective each episode, instead of training N independent networks.
    Per-player _prev_timestep/_prev_action slots are swapped in/out around
    each DQN.step() call so transitions are formed correctly.
    """

    def __init__(self, dqn_agent: dqn.DQN, num_players: int) -> None:
        self._agent = dqn_agent
        self._num_players = num_players
        self._prev_timesteps = [None] * num_players
        self._prev_actions = [None] * num_players

    def step(self, time_step, is_evaluation: bool = False):
        if time_step.last():
            # Commit final transitions for every player then reset slots.
            for p in range(self._num_players):
                self._agent.player_id = p
                self._agent._prev_timestep = self._prev_timesteps[p]
                self._agent._prev_action = self._prev_actions[p]
                self._agent.step(time_step, is_evaluation=is_evaluation)
            self._prev_timesteps = [None] * self._num_players
            self._prev_actions = [None] * self._num_players
            return

        player_id = time_step.observations["current_player"]
        self._agent.player_id = player_id
        self._agent._prev_timestep = self._prev_timesteps[player_id]
        self._agent._prev_action = self._prev_actions[player_id]

        agent_out = self._agent.step(time_step, is_evaluation=is_evaluation)

        self._prev_timesteps[player_id] = self._agent._prev_timestep
        self._prev_actions[player_id] = self._agent._prev_action
        return agent_out

    @property
    def loss(self):
        return self._agent.loss

    def save(self, path: str) -> None:
        torch.save(
            {
                "q_network": self._agent._q_network.state_dict(),
                "target_q_network": self._agent._target_q_network.state_dict(),
                "optimizer": self._agent._optimizer.state_dict(),
                "step_counter": self._agent._step_counter,
            },
            path,
        )
        logging.info("Agent saved to %s", path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, weights_only=False)
        self._agent._q_network.load_state_dict(checkpoint["q_network"])
        self._agent._target_q_network.load_state_dict(checkpoint["target_q_network"])
        self._agent._optimizer.load_state_dict(checkpoint["optimizer"])
        self._agent._step_counter = checkpoint["step_counter"]
        logging.info("Agent loaded from %s", path)


def make_shared_dqn_agent(
    state_size: int, num_actions: int, num_episodes: int
) -> SelfPlayDQN:
    # Estimate total training steps: ~10 steps/episode across 3 players.
    total_steps = num_episodes * 10
    agent = dqn.DQN(
        player_id=0,
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
    return SelfPlayDQN(agent, NUM_PLAYERS)


def train(
    env: rl_environment.Environment,
    agents: SelfPlayDQN,
    num_episodes: int,
    checkpoint_path: str,
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
            if ep > 0:
                agents.save(checkpoint_path)

        time_step = env.reset()
        while not time_step.last():
            agent_out = agents.step(time_step)
            time_step = env.step([agent_out.action])

        agents.step(time_step)

        episode_rewards[ep] = time_step.rewards
        loss = agents.loss
        if loss is not None:
            episode_losses[ep, 0] = float(loss)

        n = (ep % window_size) + 1
        reward_window += (time_step.rewards - reward_window) / n

    logging.info("Training complete. Final window avg reward: %s", reward_window)
    agents.save(checkpoint_path)
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
    smooth_window: int = 500,
    filename: str = "learning_curves.png",
) -> None:
    """Plot reward and loss learning curves."""
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
    ax.set_title("Average Reward per Episode (DQN)")
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Reward (smoothed, window={smooth_window})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Loss plot
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


def play_interactive(env: rl_environment.Environment, agents: SelfPlayDQN) -> None:
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
                agent_out = agents.step(time_step, is_evaluation=True)
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
    logging.info("Mode: DQN (full deck: 13 cards/suit, 4 suits)")

    env = rl_environment.Environment("oh_hell", **DQN_PARAMS)
    num_actions = env.action_spec()["num_actions"]
    state_size = env.observation_spec()["info_state"][0]
    agents = make_shared_dqn_agent(state_size, num_actions, FLAGS.num_episodes)

    if os.path.exists(FLAGS.checkpoint):
        agents.load(FLAGS.checkpoint)

    if FLAGS.play_only:
        logging.info("Skipping training, going straight to interactive play.")
    else:
        data = train(env, agents, FLAGS.num_episodes, FLAGS.checkpoint)
        plot_curves(data)

    play_interactive(env, agents)


if __name__ == "__main__":
    app.run(main)
