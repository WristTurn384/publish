from publish.metrics import BaseExactTrajectoryDistributionMetric
from publish.metrics.utils import heatmap
from publish.utils.constants import FORWARD
from typing import Dict
import math
import torch.nn.functional as F
import torch

class ExactKLDivergenceMetric(BaseExactTrajectoryDistributionMetric):
    def __init__(self, config: Dict[str, object]):
        super().__init__(config)
        true_density, _, _ = self.env.true_density()
        self.log_true_density = true_density.to(device=self.states.device).log()
        self.compute_period = config.get('compute_period', 250)

        self.kl_div = None
        self.min_kl_div = 1e7

        self.l1_dist = None
        self.min_l1_dist = 1e7

    @property
    def agent_names_to_compute_for(self) -> str:
        return ['target']

    @property
    def modes_to_compute_for(self) -> str:
        return [FORWARD]

    def update(self, update_infos: Dict[str, object]) -> None:
        super().update(update_infos)
        self.probs = torch.zeros(self.num_terminal_states, device=self.states.device)
        self.probs.index_add_(
            dim=0,
            index=self.int_terminal_states,
            source=self.target_agent_fwd_distrib.probs
        )

        self.kl_div = F.kl_div(
            torch.nan_to_num(self.probs.log()),
            self.log_true_density,
            reduction='batchmean',
            log_target=True
        ).item()

        self.l1_dist = (self.probs - self.log_true_density.exp()).abs().mean().item()

        self.min_l1_dist = min(self.l1_dist, self.min_l1_dist)
        self.min_kl_div = min(self.kl_div, self.min_kl_div)

        # True density is in log space while probs is not
        self.kl_div_per_state = self.log_true_density.exp() * (
            self.log_true_density - self.probs.log()
        )

    def compute_for_step_result_dict(self) -> Dict[str, object]:
        return {
            'exact_kl_divergence': self.kl_div,
            'min_exact_kl_divergence': self.min_kl_div,
            'exact_l1_distance': self.l1_dist,
            'min_exact_l1_distance': self.min_l1_dist
        }

    def compute_for_wandb_dict(self) -> Dict[str, object]:
        if self.iter_num % self.compute_period != 0:
            return {}

        reshape_shape = [self.env.side_length for _ in range(self.env.num_dims)]
        return {
            'exact_terminal_distribution': heatmap(
                self.probs.cpu().numpy().reshape(reshape_shape),
                cmap='plasma'
            ),
            'kl_div_contribution_per_state': heatmap(
                self.kl_div_per_state.cpu().numpy().reshape(reshape_shape),
                cmap='plasma'
            )
        }
