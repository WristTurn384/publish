from abc import ABC, abstractmethod
from typing import Dict

class BaseMetric(ABC):
    def __init__(self, config: Dict[str, object]):
        self.iter_num = 0

    @abstractmethod
    def update(self, update_infos: Dict[str, object]) -> None:
        self.iter_num += 1

    def should_compute(self, iter_num: int) -> bool:
        return True

    def compute_for_step_result_dict(self) -> Dict[str, object]:
        return {}

    def compute_for_wandb_dict(self) -> Dict[str, object]:
        return {}
