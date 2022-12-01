from publish.agents import TabularGFlowNet
from publish.utils import get_grid_one_hot_int_encoder_mask
from torchtyping import TensorType
from typing import Dict, Tuple, Iterable
from torch.distributions.categorical import Categorical
import torch

class OnPolicyTabularGFlowNet(TabularGFlowNet):
    def _setup(self, config: Dict[str, object]):
        self.fwd_transition_logits = config['target_agent'].fwd_transition_logits

        self.env = config['env']
        self.encoder_mask = get_grid_one_hot_int_encoder_mask(
            self.env.num_dims,
            self.env.side_length
        )

    def update(self, update_infos: Dict[str, object]) -> Dict[str, object]:
        self.fwd_transition_logits = update_infos['target_agent'].fwd_transition_logits
        return torch.tensor(0.0)

    def get_metrics(self) -> Dict[str, object]:
        return {}
