from publish.agents import BasicGFlowNet
from publish.parameter_schedulers import NoopParameterScheduler
from torch.distributions.categorical import Categorical
from torchtyping import TensorType
from typing import Dict
import numpy as np
import torch

class TemperedGFlowNet(BasicGFlowNet):
    def _setup(self, config: Dict[str, object]):
        super()._setup(config)
        self.model = config['target_agent'].model

        scheduler_conf = {
            'parameter': config['temperature'],
            **config.get(
                'temperature_scheduler_config',
                {'type': NoopParameterScheduler}
            )
        }

        self.temperature_scheduler = scheduler_conf['type'](scheduler_conf)
        temperature = self.temperature_scheduler.current_value
        assert 0.0 <= temperature

    @property
    def does_grad_update(self) -> bool:
        return False

    def act(
        self,
        observations: TensorType['batch_size', 'obs_dim']
    ) -> TensorType['batch_size']:
        untempered_logits = self.model(observations)[:, : self._act_dim]

        curr_temp = self.temperature_scheduler.current_value
        tempered_logits = untempered_logits * curr_temp

        logits = super()._get_masked_action_logits(observations, tempered_logits)
        return Categorical(logits=logits).sample()

    def update(self, update_infos: Dict[str, object]) -> Dict[str, object]:
        self.model = update_infos['target_agent'].model

        # The return value of the update method is tracked as the loss
        # of the behavior agent. Since the loss doesn't really make sense
        # for an epsilon-noisy GFN, we just return a value of 0.
        return torch.tensor(0.0)

    def end_epoch(self) -> None:
        self.temperature_scheduler.step()
