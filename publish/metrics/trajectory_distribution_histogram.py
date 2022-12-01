from publish.metrics import PeriodicComputableMetric, BaseExactTrajectoryDistributionMetric
from publish.metrics.utils import heatmap
from torchtyping import TensorType
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

CONDITIONAL_PB_ARG_IDX = -1

def plot_hist(data: TensorType):
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(data)), data)
    plt.yscale('log')
    plt.close(fig)
    return wandb.Image(fig)

def plot_expected_num_sampled_past_threshold(
    probs: TensorType,
    num_to_sample: int = 100_000,
    num_points_linspace: int = 100
):
    expected_num_samples = (probs * num_to_sample)
    thresholds = torch.linspace(
        1,
        num_to_sample,
        num_points_linspace,
        device=probs.device
    ).reshape(-1, 1)

    expected_repeated = expected_num_samples.repeat(num_points_linspace, 1)

    expected_past_threshold = (expected_repeated >= thresholds).sum(dim=1)

    x = thresholds.squeeze().cpu().numpy()
    y = expected_past_threshold.squeeze().cpu().numpy()

    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.xscale('log')
    plt.close(fig)
    return wandb.Image(fig)

class TrajectoryDistributionHistogram(
    PeriodicComputableMetric,
    BaseExactTrajectoryDistributionMetric
):
    def __init__(self, config: Dict[str, object]):
        super().__init__(config)
        self.target_agent_back_entropy_vmin = 1e7
        self.target_agent_back_entropy_vmax = 0

    def compute_for_wandb_dict(self) -> Dict[str, object]:
        result_dict = {}
        for arg_name in self.distrib_arg_names:
            distrib = getattr(self, arg_name)
            if distrib is None:
                continue

            if arg_name != self.distrib_arg_names[-1]:
                probs = distrib.probs
                result_dict['%s_histogram' % arg_name] = plot_hist(
                    np.sort(probs.cpu().numpy())
                )

                result_dict['%s_expected_num_sampled_past_threshold' % arg_name] = \
                    plot_expected_num_sampled_past_threshold(probs)
            else:
                entropies = torch.zeros(
                    self.num_terminal_states,
                    device=self.states.device
                )

                entropies.index_add_(
                    dim=0,
                    index=self.int_terminal_states,
                    source=-(distrib * distrib.log())
                )

                self.target_agent_back_entropy_vmin = min(
                    self.target_agent_back_entropy_vmin,
                    entropies.min().item()
                )

                self.target_agent_back_entropy_vmax = max(
                    self.target_agent_back_entropy_vmax,
                    entropies.max().item()
                )

                result_dict['target_agent_back_entropies'] = heatmap(
                    entropies.cpu().numpy().reshape(
                        self.env.side_length,
                        self.env.side_length
                    ),
                    vmin=self.target_agent_back_entropy_vmin,
                    vmax=self.target_agent_back_entropy_vmax,
                    cmap='plasma'
                )

        return result_dict
