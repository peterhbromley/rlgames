"""NFSP agent loader for Oh Hell."""

import torch
from open_spiel.python import rl_agent, rl_environment
from open_spiel.python.pytorch import nfsp

from .registry import register


class NFSPEvalAgent:
    """Wraps N independent NFSP agents for evaluation.

    Uses the average policy network for action selection (the converged strategy).
    Exposes a single .step() interface matching what SessionManager expects.
    """

    def __init__(self, agents: list[nfsp.NFSP]) -> None:
        self._agents = agents

    def step(self, time_step, is_evaluation: bool = True) -> rl_agent.StepOutput | None:
        if time_step.last():
            return None
        player_id = time_step.observations["current_player"]
        agent = self._agents[player_id]
        with agent.temp_mode_as(nfsp.MODE.AVERAGE_POLICY):
            return agent.step(time_step, is_evaluation=True)


@register("oh_hell", "nfsp")
def load(
    checkpoint_path: str,
    env: rl_environment.Environment,
    agent_config: dict | None = None,
) -> NFSPEvalAgent:
    num_actions = env.action_spec()["num_actions"]
    state_size = env.observation_spec()["info_state"][0]
    num_players = env.num_players
    cfg = agent_config or {}

    agents = [
        nfsp.NFSP(
            player_id=i,
            state_representation_size=state_size,
            num_actions=num_actions,
            hidden_layers_sizes=cfg.get("hidden_layers_sizes", [256, 256]),
            reservoir_buffer_capacity=10,  # minimal — not training
            anticipatory_param=cfg.get("anticipatory_param", 0.1),
            batch_size=1,
            rl_learning_rate=cfg.get("rl_learning_rate", 0.0001),
            sl_learning_rate=cfg.get("sl_learning_rate", 0.001),
            learn_every=64,
            optimizer_str="adam",
            device="cpu",
        )
        for i in range(num_players)
    ]

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for agent, state in zip(agents, checkpoint["agents"]):
        agent._rl_agent._q_network.load_state_dict(state["q_network"])
        agent._avg_network.load_state_dict(state["avg_network"])
        agent._iteration = state["nfsp_iteration"]

    return NFSPEvalAgent(agents)
