from publish.agents.flow_functions import BaseFlowFunction
from publish.data import HypergridTrajectoryDataset
from publish.utils import get_device, get_grid_one_hot_int_encoder_mask
from torchtyping import TensorType
from typing import Dict
import torch

FORWARD  = 'frwd'
BACKWARD = 'back'

class ExactFlowFunction(BaseFlowFunction):
    def __init__(self, config: Dict[str, object]):
        self.mode = config['mode']
        assert self.mode in [FORWARD, BACKWARD]

        env = config['env']
        num_states = env.side_length ** env.num_dims
        self.state_flows = torch.zeros(
            num_states,
            device=get_device(),
            requires_grad=False
        ).exp()

        traj_dataset = HypergridTrajectoryDataset.create(
            env.side_length,
            env.num_dims,
            env.R_0,
            env.R_1,
            env.R_2,
            use_depth_first_search=False
        )

        self.states, self.actions, self.back_actions, self.rewards, self.dones = (
            traj_dataset.one_hot_states,
            traj_dataset.actions,
            traj_dataset.back_actions,
            traj_dataset.rewards,
            traj_dataset.dones
        )

        self.num_terminal_states = num_states
        int_encoder_mask = get_grid_one_hot_int_encoder_mask(
            env.num_dims,
            env.side_length
        )

        self.int_encoded_states = (
            self.states * int_encoder_mask
        ).sum(dim=-1).long()

        self.int_encoded_states[self.int_encoded_states <= -1] = num_states
        self.update(config['agent'])

    def __str__(self) -> str:
        return 'ExactFlowFunction'

    def update(self, agent) -> None:
        probs, multiplier = None, None
        if self.mode == FORWARD:
            probs = agent.get_log_pf(
                self.states,
                self.actions
            ).sum(dim=-1).exp()
            multiplier = agent.true_Z
        else:
            probs = agent.get_log_pb(
                self.states,
                self.back_actions
            ).sum(dim=-1).exp()

            multiplier = self.rewards

        traj_flows = probs * multiplier
        state_flows_pre = torch.zeros(
            self.num_terminal_states + 1,
            device=get_device()
        )

        state_flows_pre.index_add_(
            dim=0,
            index=self.int_encoded_states.flatten(),
            source=traj_flows.repeat_interleave(self.actions.shape[-1])
        )

        self.state_flows = state_flows_pre[:-1].detach()

    def get_flows(
        self,
        int_encoded_states: TensorType['num_states', int],
        log_Z: TensorType[float]
    ) -> TensorType['num_states', float]:
        is_origin_mask = int_encoded_states == 0
        if self.mode == BACKWARD:
            is_origin_mask[:] = False

        int_encoded_states[int_encoded_states <= -1] = 0

        return (
            (~is_origin_mask * self.state_flows[int_encoded_states]) +
            (is_origin_mask * log_Z.exp())
        )
