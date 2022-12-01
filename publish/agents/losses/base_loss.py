from abc import ABC, abstractmethod
from torchtyping import TensorType
from typing import Dict

class BaseLoss(ABC):
    def __init__(self, config: Dict[str, object] = {}):
        pass

    @abstractmethod
    def __call__(self, loss_infos: Dict[str, object]) -> TensorType['batch_size']:
        pass

    @property
    @abstractmethod
    def requires_state_flows(self) -> bool:
        pass
