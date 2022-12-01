from publish.agents import BasicGFlowNet
from publish.utils import get_grid_one_hot_int_encoder_mask
from torchtyping import TensorType
from typing import Dict, Tuple, Iterable
from torch.distributions.categorical import Categorical
import torch


class OnPolicyBasicGFlowNet(BasicGFlowNet):
    def _setup(self, config: Dict[str, object]):
        self._init_base_agent(
            config['target_agent'].model,
            loss_fxn=None,
            env=config['env'],
            optimizer_config={},
            build_optimizer=False
        )

    def update(self, update_infos: Dict[str, object]) -> Dict[str, object]:
        self.model = update_infos['target_agent'].model
        return torch.tensor(0.0)

    def get_metrics(self) -> Dict[str, object]:
        return {}
