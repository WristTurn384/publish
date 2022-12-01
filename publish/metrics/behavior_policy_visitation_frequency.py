from publish.metrics import BaseMetric
from publish.metrics.utils import heatmap
from publish.workers import RolloutWorker
from publish.utils import get_grid_one_hot_int_encoder_mask, get_first_done_idx
from hive.replays import BaseReplayBuffer
from functools import partial
from torchtyping import TensorType
from typing import Dict, Tuple
from torch.distributions.categorical import Categorical
import wandb
import torch
import torch.nn.functional as F
import numpy as np


class BehaviorPolicyVisitationFrequency(BaseMetric):
    def __init__(self, config: Dict[str, object]):
        self.num_states_to_track = config['num_states_to_track']
        if not isinstance(self.num_states_to_track, list):
            self.num_states_to_track = list(self.num_states_to_track)

        self.env = config['env']
        self.side_length = self.env.side_length
        self.num_dim = self.env.num_dims

        self.behavior_agent_batch_size = config['behavior_agent_batch_size']
        self.target_agent_batch_size = config.get('target_agent_pf_batch_size', 0)

        self.batch_size = \
            self.behavior_agent_batch_size + self.target_agent_batch_size

        self.all_emp_state_visit_arrs = [
            np.full((size,), fill_value=self.side_length ** self.num_dim)
            for size in self.num_states_to_track
        ]

        self.behavior_agent_emp_state_visit_arrs = [
            np.full((size,), fill_value=self.side_length ** self.num_dim)
            for size in self.num_states_to_track
        ]

        self.encoder_mask = get_grid_one_hot_int_encoder_mask(
            self.num_dim,
            self.side_length
        )

        self.heatmap_compute_period = config.get('heatmap_compute_period', 250)
        self.num_times_called = 0

    def update(self, update_infos: Dict[str, object]) -> None:
        terminal_int_states = self._get_terminal_int_states(
            update_infos['target_agent_replay_buffer']
        )

        np_int_encoded_states = terminal_int_states.cpu().numpy()
        for i in range(len(self.all_emp_state_visit_arrs)):
            self.all_emp_state_visit_arrs[i] = np.roll(
                self.all_emp_state_visit_arrs[i],
                len(np_int_encoded_states)
            )

            self.all_emp_state_visit_arrs[i][:len(np_int_encoded_states)] = \
                np_int_encoded_states

        behavior_terminal_states = \
            np_int_encoded_states[self.target_agent_batch_size:]
        for i in range(len(self.behavior_agent_emp_state_visit_arrs)):
            self.behavior_agent_emp_state_visit_arrs[i] = np.roll(
                self.behavior_agent_emp_state_visit_arrs[i],
                len(behavior_terminal_states)
            )

            self.behavior_agent_emp_state_visit_arrs[i][
                :len(behavior_terminal_states)
            ] = behavior_terminal_states

        self.num_times_called += 1

    def _get_terminal_int_states(
        self,
        replay_buffer: BaseReplayBuffer
    ) -> TensorType['batch_size']:
        buffer_out = replay_buffer.get_last_n_inserted(self.batch_size)
        first_dones_idx = get_first_done_idx(buffer_out['dones'])

        recent_states = buffer_out['states']
        recent_terminal_states = recent_states[
            torch.arange(len(recent_states), device=recent_states.device),
            first_dones_idx
        ]

        return (recent_terminal_states * self.encoder_mask).sum(dim=1)

    def compute_for_step_result_dict(self) -> Dict[str, object]:
        return {}

    def compute_for_wandb_dict(self) -> Dict[str, object]:
        ret_dict = {}

        if self.num_times_called % self.heatmap_compute_period == 0:
            for visit_arr in self.all_emp_state_visit_arrs:
                state_visits = np.bincount(
                    visit_arr,
                    minlength=(self.side_length ** self.num_dim) + 1
                )[:-1]

                ret_dict['last_%d_mixed_behavior_visits' % len(visit_arr)] = \
                    heatmap(
                        state_visits.reshape(self.side_length, self.side_length)
                    )

            for behavior_visit_arr in self.behavior_agent_emp_state_visit_arrs:
                state_visits = np.bincount(
                    behavior_visit_arr,
                    minlength=(self.side_length ** self.num_dim) + 1
                )[:-1]

                ret_dict['last_%d_behavior_visits' % len(behavior_visit_arr)] = \
                    heatmap(
                        state_visits.reshape(self.side_length, self.side_length)
                    )

        return ret_dict
