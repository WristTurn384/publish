from abc import ABC, abstractmethod
from torchtyping import TensorType
from typing import Dict

class BaseParameterScheduler(ABC):
    def __init__(self, config: Dict[str, object]):
        self._parameter = config['parameter']

    @property
    def current_value(self):
        return self._parameter

    @abstractmethod
    def step(self, update_infos: Dict[str, object] = {}) -> None:
        pass
