from publish.metrics import BaseMetric
from publish.metrics.utils import heatmap
from publish.workers import RolloutWorker
from publish.utils import get_grid_one_hot_int_encoder_mask, get_first_done_idx
from functools import partial
from torchtyping import TensorType
from typing import Dict, Tuple
from torch.distributions.categorical import Categorical
import wandb
import torch
import torch.nn.functional as F
import numpy as np

STATES_IDX, DONES_IDX = 0, 3

class DistributionMetrics(BaseMetric):
    def __init__(self, config: Dict[str, object]):
        self.num_states_to_track = config['num_states_to_track']
        self.batch_size = config.get('batch_size', 64)

        self.env = config['env']
        self.side_length = self.env.side_length
        self.num_dim = self.env.num_dims
        self.true_density, _, _ = self.env.true_density()
        self.true_density = self.true_density.numpy()
        self.true_density_bins = np.arange(len(self.true_density) + 1)

        self.emp_state_visits = np.full(
            (self.num_states_to_track,),
            fill_value=self.side_length ** self.num_dim
        )

        self.empirical_distribution = None

        self.encoder_mask = get_grid_one_hot_int_encoder_mask(
            self.num_dim,
            self.side_length
        )

        self.num_sampled_so_far = 0
        self._worker = None
        self._min_l1_distance = 1e8
        self._min_kl_divergence = 1e8

        self.heatmap_compute_period = config.get('heatmap_compute_period', 250)
        self.num_times_called = 0

        self.density_diff_vmin, self.density_diff_vmax = 1e7, 0

    def update(self, update_infos: Dict[str, object]) -> None:
        new_terminal_states = self._get_new_terminal_states(update_infos)
        int_encoded_states = (new_terminal_states * self.encoder_mask).sum(dim=1)

        self.emp_state_visits = np.roll(
            self.emp_state_visits,
            len(int_encoded_states)
        )

        self.emp_state_visits[:len(int_encoded_states)] = \
            int_encoded_states.cpu().numpy()

        self.num_sampled_so_far += len(new_terminal_states)

        state_visits = np.bincount(
            self.emp_state_visits,
            minlength=len(self.true_density) + 1
        )[:-1]

        self.empirical_density = state_visits / state_visits.sum()

    def _get_new_terminal_states(
        self,
        update_infos: Dict[str, object]
    ) -> TensorType['batch_size', 'obs_dim']:
        states, dones = self._get_states_dones(update_infos)
        first_done_idx = get_first_done_idx(dones)

        return states[torch.arange(len(states)), first_done_idx]

    def _get_states_dones(
        self,
        update_infos: Dict[str, object]
    ) -> Tuple[
        TensorType['batch_size', 'horizon', 'obs_dim'],
        TensorType['batch_size', 'horizon'],
    ]:
        rollout = self._get_worker(update_infos).sample()
        return rollout[STATES_IDX], rollout[DONES_IDX]

    def _get_worker(self, update_infos: Dict[str, object]) -> RolloutWorker:
        if self._worker is None:
            self._worker = RolloutWorker(
                self.env.clone(self.batch_size),
                update_infos['target_agent'],
                self.batch_size
            )

        return self._worker

    def compute_for_step_result_dict(self) -> Dict[str, object]:
        l1_distance = np.abs(
            self.true_density - self.empirical_density
        ).mean().item()

        kl_divergence = F.kl_div(
            torch.nan_to_num(torch.tensor(self.empirical_density).log()),
            torch.tensor(self.true_density).log(),
            reduction='batchmean',
            log_target=True
        ).item()

        self._min_l1_distance = np.minimum(self._min_l1_distance, l1_distance)
        self._min_kl_divergence = np.minimum(self._min_kl_divergence, kl_divergence)

        return {
            'l1_distance': l1_distance,
            'min_l1_distance': self._min_l1_distance,
            'num_trajs_sampled': self.num_sampled_so_far,
            'kl_divergence': kl_divergence,
            'min_kl_divergence': self._min_kl_divergence,
            'empirical_entropy': Categorical(
                probs=torch.tensor(self.empirical_density)
            ).entropy().item(),
        }

    def compute_for_wandb_dict(self) -> Dict[str, object]:
        ret_dict = {}
        if self.num_times_called % self.heatmap_compute_period == 0:
            density_diff = np.abs(self.empirical_density - self.true_density)
            self.density_diff_vmin = min(self.density_diff_vmin, density_diff.min())
            self.density_diff_vmax = max(self.density_diff_vmax, density_diff.max())

            ret_dict.update({
                'true_distribution_heatmap': heatmap(
                    self.true_density.reshape(self.side_length, self.side_length)
                ),
                'empirical_distribution_heatmap': heatmap(
                    self.empirical_density.reshape(
                        self.side_length,
                        self.side_length
                    )
                ),
                'density_diff_heatmap': heatmap(
                    np.abs(self.empirical_density - self.true_density).reshape(
                        self.side_length,
                        self.side_length
                    )
                )
            })

        self.num_times_called += 1
        return ret_dict
