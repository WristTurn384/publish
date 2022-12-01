from publish.metrics import BaseExactTrajectoryDistributionMetric
from torch.distributions.categorical import Categorical
from torchtyping import TensorType
from typing import Dict
import torch

CONDITIONAL_PB_ARG_IDX = -1

class TrajectoryDistributionEntropy(BaseExactTrajectoryDistributionMetric):
    def __init__(self, config: Dict[str, object]) -> None:
        super().__init__(config)

        self.entropy_arg_names = (
            'sampler_entropy',
            'behavior_agent_fwd_entropy',
            'behavior_agent_back_entropy',
            'target_agent_fwd_entropy',
            'target_agent_back_entropy',
            'target_agent_back_given_term_entropies',
        )

        list(map(lambda x: setattr(self, x, None), self.entropy_arg_names))

    def update(self, update_infos: Dict[str, object]) -> None:
        super().update(update_infos)
        for arg_name in self.entropy_arg_names:
            name_split = arg_name.split('_')
            name_split[-1] = 'distrib'
            distrib_arg_name = '_'.join(name_split)

            distrib = getattr(self, distrib_arg_name)
            if distrib is None:
                continue

            if arg_name != self.entropy_arg_names[CONDITIONAL_PB_ARG_IDX]:
                setattr(self, arg_name, distrib.entropy().item())
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

                self.target_agent_back_given_term_entropies = \
                    entropies.mean().item()

    def compute_for_step_result_dict(self) -> Dict[str, object]:
        return {
            arg_name: getattr(self, arg_name)
            for arg_name in self.entropy_arg_names
            if getattr(self, arg_name) is not None
        }
