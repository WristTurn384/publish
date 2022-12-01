from publish.agents.flow_functions import BaseFlowFunction
from publish.utils import get_device
from torchtyping import TensorType
from typing import Dict, Iterable
import torch

class TabularFlowFunction(BaseFlowFunction):
    def __init__(self, config: Dict[str, object]):
        env = config['env']
        if env.num_states == -1:
            raise ValueError(
                'Cannot use a tabular flow function with an environment ' +
                'where the number of states is not computable'
            )

        self.state_flows = torch.ones(
            env.num_states,
            device=get_device(),
            requires_grad=True
        )

    def __str__(self) -> str:
        return 'TabularFlowFunction'

    # This flow function is updated via gradients in part of the outer loop.
    # For this reason we just pass in the update method.
    def update(self, agent) -> None:
        pass

    def parameters(self) -> Iterable[TensorType]:
        return [self.state_flows]

    def get_flows(
        self,
        int_encoded_states: TensorType['num_states', int],
        log_Z: TensorType[float]
    ) -> TensorType['num_states', float]:
        is_origin_mask = int_encoded_states == 0

        int_encoded_states[int_encoded_states <= -1] = 0

        return (
            (~is_origin_mask * self.state_flows[int_encoded_states]) +
            (is_origin_mask * log_Z.exp())
        )
