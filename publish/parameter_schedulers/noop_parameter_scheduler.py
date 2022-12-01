from publish.parameter_schedulers import BaseParameterScheduler
from torchtyping import TensorType
from typing import Dict

class NoopParameterScheduler(BaseParameterScheduler):
    def step(self, update_infos: Dict[str, object] = {}) -> None:
        pass
