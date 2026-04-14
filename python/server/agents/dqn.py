"""DQN agent loader for Oh Hell."""

from open_spiel.python import rl_environment

from shared.dqn import SelfPlayDQN, make_shared_dqn_agent
from .registry import register


@register("oh_hell", "dqn")
def load(
    checkpoint_path: str,
    env: rl_environment.Environment,
    agent_config: dict | None = None,
) -> SelfPlayDQN:
    num_actions = env.action_spec()["num_actions"]
    state_size = env.observation_spec()["info_state"][0]
    num_players = env.num_players
    cfg = agent_config or {}
    # num_episodes only affects epsilon decay, irrelevant for inference.
    agent = make_shared_dqn_agent(
        state_size,
        num_actions,
        num_episodes=1,
        num_players=num_players,
        hidden_layers_sizes=cfg.get("hidden_layers_sizes", [64, 64]),
    )
    agent.load(checkpoint_path)
    return agent
