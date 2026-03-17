"""
Shared DQN agent code used by both training and server inference.
"""

import logging

import torch
from open_spiel.python.pytorch import dqn


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
    state_size: int,
    num_actions: int,
    num_episodes: int,
    num_players: int,
    hidden_layers_sizes: list[int] | None = None,
    replay_buffer_capacity: int = 50000,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_fraction: float = 0.8,
    update_target_network_every: int = 500,
    learn_every: int = 10,
    device: str = "cpu",
) -> SelfPlayDQN:
    if hidden_layers_sizes is None:
        hidden_layers_sizes = [256, 256]
    total_steps = num_episodes * 10
    agent = dqn.DQN(
        player_id=0,
        state_representation_size=state_size,
        num_actions=num_actions,
        hidden_layers_sizes=hidden_layers_sizes,
        replay_buffer_capacity=replay_buffer_capacity,
        batch_size=batch_size,
        learning_rate=learning_rate,
        optimizer_str="adam",
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_duration=int(total_steps * epsilon_decay_fraction),
        update_target_network_every=update_target_network_every,
        learn_every=learn_every,
        device=device,
    )
    return SelfPlayDQN(agent, num_players)
