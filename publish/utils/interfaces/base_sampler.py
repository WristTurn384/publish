from torchtyping import TensorType
from typing import Dict, Tuple
from abc import ABC, abstractmethod

class BaseSampler(ABC):
    @abstractmethod
    def sample(self, num_trajs: int) -> Tuple[
        TensorType['batch_size', 'horizon', 'obs_dim'],
        TensorType['batch_size', 'horizon'],
        TensorType['batch_size', 'horizon'],
        TensorType['batch_size', 'horizon'],
        TensorType['batch_size']
    ]:
        pass

    def get_metrics(self) -> Dict[str, object]:
        return {}
