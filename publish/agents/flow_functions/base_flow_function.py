from abc import ABC, abstractmethod
from typing import Iterable
from torchtyping import TensorType

class BaseFlowFunction(ABC):
    @abstractmethod
    def update(self, agent) -> None:
        pass

    @abstractmethod
    def get_flows(
        self,
        states: TensorType['num_states']
    ) -> TensorType['num_states']:
        pass

    def parameters(self) -> Iterable[TensorType]:
        return []
