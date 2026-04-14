"""
PPO components for Oh Hell training with separate bid and play networks.

Architecture:
  - BidNet:  info_state during bidding → bid action (0..num_tricks)
  - PlayNet: info_state during play   → card action (0..deck_size-1)

Both are ActorCritic models with illegal-action masking.
A PolicyPool stores past snapshots for diverse self-play opponents.
"""

import copy
import random as stdlib_random

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """Actor-critic with illegal-action masking.

    The actor and critic share a backbone but have separate output heads.
    Orthogonal initialization is used (standard PPO practice).
    """

    def __init__(self, obs_size: int, num_actions: int, hidden_sizes: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        prev = obs_size
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.actor_head = nn.Linear(prev, num_actions)
        self.critic_head = nn.Linear(prev, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.backbone:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def forward(self, obs: torch.Tensor, legal_mask: torch.Tensor):
        features = self.backbone(obs)
        logits = self.actor_head(features)
        logits = logits.masked_fill(~legal_mask, float("-inf"))
        value = self.critic_head(features).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor, legal_mask: torch.Tensor, *, greedy: bool = False) -> tuple[int, float, float]:
        """Sample action from policy. Returns (action_index, log_prob, value).

        greedy=True: argmax (for evaluation); greedy=False: sample (for training).
        """
        logits, value = self.forward(obs.unsqueeze(0), legal_mask.unsqueeze(0))
        dist = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if greedy else dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor, legal_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batch evaluation for PPO update. Returns (log_probs, entropy, values)."""
        logits, values = self.forward(obs, legal_mask)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Collects separate bid and play trajectories, computes GAE advantages."""

    def __init__(self):
        self._bid_eps: list[list[dict]] = []
        self._play_eps: list[list[dict]] = []
        self._cur_bid: list[dict] = []
        self._cur_play: list[dict] = []

    @property
    def num_episodes(self) -> int:
        return len(self._bid_eps)

    def add_bid(self, obs, action, log_prob, value, legal_mask):
        self._cur_bid.append(dict(
            obs=obs, action=action, log_prob=log_prob,
            value=value, legal_mask=legal_mask,
        ))

    def add_play(self, obs, action, log_prob, value, legal_mask):
        self._cur_play.append(dict(
            obs=obs, action=action, log_prob=log_prob,
            value=value, legal_mask=legal_mask,
        ))

    def finish_episode(self, reward: float):
        """Assign terminal reward and flush the current episode's transitions."""
        # Bid: single-step trajectory — reward is the full hand outcome.
        for t in self._cur_bid:
            t["reward"] = reward
        # Play: multi-step trajectory — reward only at the final step.
        for t in self._cur_play:
            t["reward"] = 0.0
        if self._cur_play:
            self._cur_play[-1]["reward"] = reward
        if self._cur_bid:
            self._bid_eps.append(self._cur_bid)
        if self._cur_play:
            self._play_eps.append(self._cur_play)
        self._cur_bid = []
        self._cur_play = []

    @staticmethod
    def _build_dataset(episodes, gamma, gae_lambda, device):
        all_obs, all_act, all_lp, all_mask = [], [], [], []
        all_ret, all_adv = [], []
        for ep in episodes:
            T = len(ep)
            values = [t["value"] for t in ep]
            rewards = [t["reward"] for t in ep]
            # GAE
            advantages = [0.0] * T
            gae = 0.0
            for t in reversed(range(T)):
                next_val = values[t + 1] if t + 1 < T else 0.0
                delta = rewards[t] + gamma * next_val - values[t]
                gae = delta + gamma * gae_lambda * gae
                advantages[t] = gae
            returns = [a + v for a, v in zip(advantages, values)]
            for t, tr in enumerate(ep):
                all_obs.append(tr["obs"])
                all_act.append(tr["action"])
                all_lp.append(tr["log_prob"])
                all_mask.append(tr["legal_mask"])
                all_ret.append(returns[t])
                all_adv.append(advantages[t])
        if not all_obs:
            return None
        adv = torch.tensor(all_adv, dtype=torch.float32, device=device)
        return dict(
            obs=torch.tensor(np.array(all_obs), dtype=torch.float32, device=device),
            actions=torch.tensor(all_act, dtype=torch.long, device=device),
            old_log_probs=torch.tensor(all_lp, dtype=torch.float32, device=device),
            legal_masks=torch.tensor(np.array(all_mask), dtype=torch.bool, device=device),
            returns=torch.tensor(all_ret, dtype=torch.float32, device=device),
            advantages=(adv - adv.mean()) / (adv.std() + 1e-8),
        )

    def build_bid_dataset(self, gamma=1.0, gae_lambda=0.95, device="cpu"):
        return self._build_dataset(self._bid_eps, gamma, gae_lambda, device)

    def build_play_dataset(self, gamma=1.0, gae_lambda=0.95, device="cpu"):
        return self._build_dataset(self._play_eps, gamma, gae_lambda, device)


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(
    net: ActorCritic,
    optimizer: torch.optim.Optimizer,
    dataset: dict,
    *,
    epochs: int = 4,
    batch_size: int = 256,
    clip_epsilon: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.5,
) -> dict[str, float]:
    """Run PPO clipped-objective update epochs. Returns mean losses."""
    n = dataset["obs"].shape[0]
    total_pg, total_vf, total_ent, num_steps = 0.0, 0.0, 0.0, 0

    for _ in range(epochs):
        idx = torch.randperm(n, device=dataset["obs"].device)
        for start in range(0, n, batch_size):
            batch = idx[start : start + batch_size]
            obs = dataset["obs"][batch]
            actions = dataset["actions"][batch]
            old_lp = dataset["old_log_probs"][batch]
            masks = dataset["legal_masks"][batch]
            returns = dataset["returns"][batch]
            advantages = dataset["advantages"][batch]

            log_probs, entropy, values = net.evaluate(obs, actions, masks)
            ratio = (log_probs - old_lp).exp()

            pg_loss = -torch.min(
                ratio * advantages,
                ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantages,
            ).mean()
            vf_loss = (returns - values).pow(2).mean()
            loss = pg_loss + value_coef * vf_loss - entropy_coef * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()

            total_pg += pg_loss.item()
            total_vf += vf_loss.item()
            total_ent += entropy.mean().item()
            num_steps += 1

    d = max(num_steps, 1)
    return {"policy_loss": total_pg / d, "value_loss": total_vf / d, "entropy": total_ent / d}


# ---------------------------------------------------------------------------
# Policy pool
# ---------------------------------------------------------------------------

class PolicyPool:
    """Fixed-size FIFO pool of past bid/play network snapshots."""

    def __init__(self, max_size: int = 20):
        self.max_size = max_size
        self._entries: list[dict[str, dict]] = []

    def __len__(self):
        return len(self._entries)

    def add(self, bid_net: ActorCritic, play_net: ActorCritic):
        self._entries.append({
            "bid": copy.deepcopy(bid_net.state_dict()),
            "play": copy.deepcopy(play_net.state_dict()),
        })
        if len(self._entries) > self.max_size:
            self._entries.pop(0)

    def sample(self) -> dict[str, dict]:
        return stdlib_random.choice(self._entries)

    def state_dict(self) -> list[dict]:
        return list(self._entries)

    def load_state_dict(self, entries: list[dict]):
        self._entries = list(entries)
