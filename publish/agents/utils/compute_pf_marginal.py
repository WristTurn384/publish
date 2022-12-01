from publish.envs import BaseEnvironment
from publish.utils import get_device
from publish.utils.constants import BACKWARD
from torchtyping import TensorType
import torch


def to_action_tensor(action_id: int) -> TensorType[int]:
    return torch.LongTensor([[action_id]]).to(device=get_device())

class AgentMarginalPFEvaluator:
    def __init__(
        self,
        env: BaseEnvironment
    ):
        self.env = env
        self.terminal_obs = env.terminal_state

    def compute_pf_marginal(
        self,
        agent: 'BaseAgent',
        log_space: bool = True
    ) -> TensorType['num_states']:
        assert self.env.num_states_tractable
        terminating_probs = torch.zeros(
            self.env.num_states,
            device=get_device()
        )

        state_flows = torch.ones_like(terminating_probs)
        done_computing_index = torch.zeros_like(terminating_probs).bool()

        self._compute_pf_marginal_recurse(
            agent,
            terminating_probs,
            state_flows,
            done_computing_index,
            self.terminal_obs
        )

        return terminating_probs.log() if log_space else terminating_probs

    def _compute_pf_marginal_recurse(
        self,
        agent: 'BaseAgent',
        terminating_probs: TensorType['num_states'],
        state_flows: TensorType['num_states'],
        done_computing_index: TensorType['num_states'],
        state: TensorType['obs_dim']
    ) -> None:
        state_id = self.env.get_state_ids(state).item()
        if done_computing_index[state_id]:
            return state_flows[state_id]

        if self.env.is_origin_state(state):
            terminating_probs[state_id] = agent.get_log_pf(
                state,
                to_action_tensor(self.env.done_action)
            ).sum(dim=1).exp()

            done_computing_index[state_id] = True
            return state_flows[state_id]

        state_flows[state_id] = self._get_curr_state_flow(
            agent,
            terminating_probs,
            state_flows,
            done_computing_index,
            state
        )

        with torch.no_grad():
            terminating_probs[state_id] = (
                state_flows[state_id].log() + agent.get_log_pf(
                    state,
                    to_action_tensor(self.env.done_action)
                )
            ).exp()

        done_computing_index[state_id] = True
        return state_flows[state_id]

    def _get_curr_state_flow(
        self,
        agent: 'BaseAgent',
        terminating_probs: TensorType['num_states'],
        state_flows: TensorType['num_states'],
        done_computing_index: TensorType['num_states'],
        state: TensorType['obs_dim']
    ) -> float:
        curr_state_flow = 0
        valid_back_actions = ~self.env.get_invalid_action_mask(state, mode=BACKWARD)
        for i, is_valid in enumerate(valid_back_actions):
            if not is_valid:
                continue

            action = to_action_tensor(i)
            parent_state = self.env.step_backward(action, state).squeeze(0)
            parent_flow = self._compute_pf_marginal_recurse(
                agent,
                terminating_probs,
                state_flows,
                done_computing_index,
                parent_state
            )

            action_id = to_action_tensor(
                self.env.get_action_ids(parent_state, state)
            )

            transition_log_pf = agent.get_log_pf(parent_state, action_id)
            curr_state_flow += (parent_flow.log() + transition_log_pf).exp()

        return curr_state_flow
