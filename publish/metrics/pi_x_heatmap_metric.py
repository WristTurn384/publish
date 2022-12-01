from publish.metrics.utils import heatmap
from publish.metrics import (
    PeriodicComputableMetric,
    BaseExactTrajectoryDistributionMetric
)
from publish.samplers.expected_pb_loss_sampler import get_pi_terminal_state
from publish.utils import get_losses_and_pb, get_p_given_terminal_state
from typing import Dict
import torch
import numpy as np

BATCH_SIZE = 100_000

class PiXHeatmapMetric(
    PeriodicComputableMetric,
    BaseExactTrajectoryDistributionMetric
):
    def __init__(self, config: Dict[str, object]):
        super().__init__(config)
        self.normalize_pi_x = config.get('normalize_pi_x', False)
        self.pi_x = None
        self.traj_per_state = torch.bincount(self.int_terminal_states)

        self.min_max_heatmap_vals = {}

        self.target_agent = None

    def update(self, update_infos: Dict[str, object]) -> None:
        self.target_agent = update_infos['target_agent']
        self.losses, pb = get_losses_and_pb(
            update_infos['target_agent'],
            self.states,
            self.actions,
            self.back_actions,
            self.dones,
            self.rewards,
            BATCH_SIZE
        )

        pb_given_terminal_state = get_p_given_terminal_state(
            pb,
            self.num_terminal_states,
            self.int_terminal_states
        )

        self.pi_x = get_pi_terminal_state(
            self.losses,
            pb_given_terminal_state,
            self.num_terminal_states,
            self.int_terminal_states,
            self.normalize_pi_x
        )

    def compute_for_wandb_dict(self) -> Dict[str, object]:
        loss_per_state = torch.zeros(
            self.num_terminal_states,
            device=self.states.device
        )

        loss_per_state.index_add_(
            dim=0,
            index=self.int_terminal_states,
            source=self.losses
        )

        avg_state_loss = loss_per_state / self.traj_per_state
        #self._debug_losses(avg_state_loss)

        normalized_str = 'normalized' if self.normalize_pi_x else 'unnormalized'
        pi_x_name = '%s_pi_x' % normalized_str
        return {
            pi_x_name: self._get_heatmap(self.pi_x, pi_x_name),
            'avg_state_loss': self._get_heatmap(avg_state_loss, 'avg_state_loss')
        }

    def _debug_losses(self, avg_state_loss):
        five_worst_states = avg_state_loss.argsort()[-5:]
        for state_idx in five_worst_states:
            idx = self.int_terminal_states == state_idx
            state_losses = get_losses_and_pb(
                self.target_agent,
                self.states[idx],
                self.actions[idx],
                self.back_actions[idx],
                self.dones[idx],
                self.rewards[idx],
                BATCH_SIZE
            )

        idx = self.int_terminal_states == ((6 * 8) + 6)
        state_losses = get_losses_and_pb(
            self.target_agent,
            self.states[idx],
            self.actions[idx],
            self.back_actions[idx],
            self.dones[idx],
            self.rewards[idx],
            BATCH_SIZE
        )

    def _get_heatmap(self, vals: np.ndarray, name: str):
        if name not in self.min_max_heatmap_vals:
            self.min_max_heatmap_vals[name] = {'min': 1e7, 'max': -1e7}

        self.min_max_heatmap_vals[name]['min'] = min(
            self.min_max_heatmap_vals[name]['min'],
            vals.min()
        )

        self.min_max_heatmap_vals[name]['max'] = max(
            self.min_max_heatmap_vals[name]['max'],
            vals.max()
        )

        return heatmap(
            vals.cpu().numpy().reshape([
                self.env.side_length for _ in range(self.env.num_dims)
            ]),
            #vmin=self.min_max_heatmap_vals[name]['min'],
            #vmax=self.min_max_heatmap_vals[name]['max'],
            cmap='plasma'
        )
