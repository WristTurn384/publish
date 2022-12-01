from publish.agents.flow_functions import ExactFlowFunction
from publish.utils import get_device
from torch.distributions import LogNormal
from torchtyping import TensorType
from typing import Dict
import torch

class GaussianNoiseExactFlowFunction(ExactFlowFunction):
    def __init__(self, config: Dict[str, object]):
        self.gaussian_std = config['gaussian_std']
        self.noise_distrib = LogNormal(loc=0.0, scale=self.gaussian_std)
        self.device = get_device()

        super().__init__(config)

    def __str__(self) -> str:
        return 'NoisyExactFlowFunction Std=%f' % self.gaussian_std

    def update(self, agent) -> None:
        super().update(agent)
        self.state_flows = self.state_flows + self.noise_distrib.sample(
            sample_shape=(len(self.state_flows),)
        ).to(device=self.device)
