"""PPO agent loader for Oh Hell."""

import numpy as np
import torch
from open_spiel.python import rl_agent, rl_environment

from training.ppo import ActorCritic
from .registry import register


class PPOEvalAgent:
    """Wraps the PPO bid + play ActorCritic pair for greedy evaluation.

    Uses argmax action selection (no sampling) to get the policy's best move.
    Exposes the .step(time_step, is_evaluation) interface expected by SessionManager.
    """

    def __init__(
        self,
        bid_net: ActorCritic,
        play_net: ActorCritic,
        deck_size: int,
        num_bids: int,
    ) -> None:
        self._bid_net = bid_net
        self._play_net = play_net
        self._deck_size = deck_size
        self._num_bids = num_bids

    @torch.no_grad()
    def step(self, time_step, is_evaluation: bool = True) -> rl_agent.StepOutput | None:
        if time_step.last():
            return None

        player = time_step.observations["current_player"]
        obs_np = np.asarray(
            time_step.observations["info_state"][player], dtype=np.float32
        )
        legal = time_step.observations["legal_actions"][player]
        is_bid = legal[0] >= self._deck_size

        obs_t = torch.as_tensor(obs_np)

        if is_bid:
            bid_indices = [a - self._deck_size for a in legal]
            mask = np.zeros(self._num_bids, dtype=np.bool_)
            mask[bid_indices] = True
            mask_t = torch.as_tensor(mask)
            action_idx, _, _ = self._bid_net.act(obs_t, mask_t, greedy=True)
            action = action_idx + self._deck_size
        else:
            mask = np.zeros(self._deck_size, dtype=np.bool_)
            mask[legal] = True
            mask_t = torch.as_tensor(mask)
            action_idx, _, _ = self._play_net.act(obs_t, mask_t, greedy=True)
            action = action_idx

        return rl_agent.StepOutput(action=action, probs=None)


@register("oh_hell", "ppo")
def load(
    checkpoint_path: str,
    env: rl_environment.Environment,
    agent_config: dict | None = None,
) -> PPOEvalAgent:
    num_actions = env.action_spec()["num_actions"]
    state_size = env.observation_spec()["info_state"][0]
    cfg = agent_config or {}

    deck_size = cfg.get("deck_size", 52)
    num_bids = num_actions - deck_size
    hidden = cfg.get("hidden_layers_sizes", [256, 256])

    bid_net = ActorCritic(state_size, num_bids, hidden)
    play_net = ActorCritic(state_size, deck_size, hidden)

    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    bid_net.load_state_dict(cp["bid_net"])
    play_net.load_state_dict(cp["play_net"])
    bid_net.eval()
    play_net.eval()

    return PPOEvalAgent(bid_net, play_net, deck_size, num_bids)
