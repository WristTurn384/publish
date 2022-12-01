from publish.parameter_schedulers import BaseParameterScheduler
from torchtyping import TensorType
from typing import Dict
import torch

class LinearParameterScheduler(BaseParameterScheduler):
    def __init__(self, config: Dict[str, object]):
        super().__init__(config)
        self._change_per_step = config['change_per_step']

        str_mode = config.get('mode', 'min')
        assert str_mode in ['min', 'max']
        self._is_min_mode = str_mode == 'min'

        self._boundary_value = config.get('boundary_value', None)
        if self._boundary_value is not None:
            self._boundary_value = torch.tensor(
                self._boundary_value,
                device=self._parameter.device
            )

    def step(self, update_infos: Dict[str, object] = {}) -> None:
        new_value_pre = self._parameter + self._change_per_step
        if self._boundary_value is not None:
            check_fxn = torch.minimum if self._is_min_mode else torch.maximum
            new_value_pre = check_fxn(self._parameter, new_value_pre)

        self._parameter = new_value_pre
        return
