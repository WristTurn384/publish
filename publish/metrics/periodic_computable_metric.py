from publish.metrics import BaseMetric
from typing import Dict

class PeriodicComputableMetric(BaseMetric):
    def __init__(self, config: Dict[str, object]):
        super().__init__(config)
        self.compute_period = config['compute_period']

    def should_compute(self, iter_num: int) -> bool:
        super_cond = super().should_compute(iter_num)
        self_cond = iter_num % self.compute_period == 0

        return super_cond and self_cond
