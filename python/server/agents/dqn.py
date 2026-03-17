"""DQN agent loader for Oh Hell."""

from open_spiel.python import rl_environment

from shared.dqn import SelfPlayDQN, make_shared_dqn_agent
from .registry import register


@register("oh_hell", "dqn")
def load(checkpoint_path: str, env: rl_environment.Environment) -> SelfPlayDQN:
    """
    Instantiate a SelfPlayDQN matching the training architecture, then restore
    weights from a checkpoint saved by SelfPlayDQN.save().

    Args:
        checkpoint_path: Path to the .pt file.
        env: Live environment for the same game config — used to read the
             state-representation size and number of actions.
    """
    num_actions = env.action_spec()["num_actions"]
    state_size = env.observation_spec()["info_state"][0]
    num_players = env.num_players
    # num_episodes only affects epsilon decay, which is irrelevant for inference.
    agent = make_shared_dqn_agent(state_size, num_actions, num_episodes=1, num_players=num_players)
    agent.load(checkpoint_path)
    return agent
