from publish.agents import TabularGFlowNet
from publish.parameter_schedulers import NoopParameterScheduler
from torchtyping import TensorType
from torch.distributions.categorical import Categorical
from typing import Dict
import torch
import numpy as np

class TemperedTabularGFlowNet(TabularGFlowNet):
    def _setup(self, config: Dict[str, object]):
        super()._setup(config)

        assert isinstance(config['target_agent'], TabularGFlowNet)
        self.fwd_transition_logits = config['target_agent'].fwd_transition_logits
        self.back_transition_logits = config['target_agent'].back_transition_logits

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
        logits = self._get_tempered_fwd_logits(observations)
        return Categorical(logits=logits).sample()

    def get_log_pf(
        self,
        states: TensorType['batch_size', 'horizon', 'obs_dim'],
        actions: TensorType['batch_size', 'horizon']
    ):
        distrib = Categorical(logits=self._get_tempered_fwd_logits(states))
        return distrib.log_prob(actions)

    def _get_tempered_fwd_logits(
        self,
        observations: TensorType['batch_size', 'obs_dim']
    ) -> TensorType['batch_size']:
        int_encoded_states = (observations * self.encoder_mask).sum(dim=1).long()
        untempered_logits = self.fwd_transition_logits[int_encoded_states]

        curr_temp = self.temperature_scheduler.current_value
        tempered_logits = untempered_logits - curr_temp

        return super()._get_masked_action_logits(observations, tempered_logits)

    def update(self, update_infos: Dict[str, object]) -> Dict[str, object]:
        self.fwd_transition_logits, self.back_transition_logits = (
            update_infos['target_agent'].fwd_transition_logits,
            update_infos['target_agent'].back_transition_logits
        )

        # The return value of the update method is tracked as the loss
        # of the behavior agent. Since the loss doesn't really make sense
        # for an tempered GFN, we just return a value of 0.
        return torch.tensor(0.0)

    def end_epoch(self) -> None:
        self.temperature_scheduler.step()
