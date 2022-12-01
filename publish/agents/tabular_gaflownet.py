from publish.agents import TabularGFlowNet, BasicGAFlowNet
from publish.agents.flow_functions import TabularFlowFunction
from publish.agents.losses import BaseLoss, GAFlowNetTrajectoryBalanceLoss
from publish.agents.utils import RandomNetworkDistillation
from publish.utils import get_grid_one_hot_int_encoder_mask, get_first_done_idx
from torchtyping import TensorType
from typing import Dict, Tuple, Iterable
from torch.distributions.categorical import Categorical
import torch


class TabularGAFlowNet(BasicGAFlowNet, TabularGFlowNet):
    def _get_config_overrides(
        self,
        config: Dict[str, object]
    ) -> Dict[str, object]:
        overrides = super()._get_config_overrides(config)
        overrides['flow_function_config'] = {'type': TabularFlowFunction}

        return overrides
