from publish.agents import BasicGFlowNet
from publish.parameter_schedulers import NoopParameterScheduler
from torchtyping import TensorType
from typing import Dict
import torch

class EpsilonNoisyGFlowNet(BasicGFlowNet):
    def _setup(self, config: Dict[str, object]):
        super()._setup(config)
        self.model = config['target_agent'].model

        scheduler_conf = {
            'parameter': config['epsilon'],
            **config.get(
                'epsilon_scheduler_config',
                {'type': NoopParameterScheduler}
            )
        }

        self.epsilon_scheduler = scheduler_conf['type'](scheduler_conf)
        epsilon = self.epsilon_scheduler.current_value
        assert 0.0 <= epsilon and epsilon <= 1.0

    @property
    def does_grad_update(self) -> bool:
        return False

    def act(
        self,
        observations: TensorType['batch_size', 'obs_dim']
    ) -> TensorType['batch_size']:
        should_sample_pre = torch.rand(
            (len(observations),),
            device=observations.device
        )

        should_sample = should_sample_pre <= self.epsilon_scheduler.current_value
        rand_actions = torch.randint(
            self._act_dim,
            size=(len(observations),),
            device=observations.device
        )

        gfn_actions = BasicGFlowNet.act(self, observations)
        return (rand_actions * should_sample) + (gfn_actions * ~should_sample)

    def update(self, update_infos: Dict[str, object]) -> Dict[str, object]:
        self.model = update_infos['target_agent'].model

        # The return value of the update method is tracked as the loss
        # of the behavior agent. Since the loss doesn't really make sense
        # for an epsilon-noisy GFN, we just return a value of 0.
        return torch.tensor(0.0)

    def end_epoch(self) -> None:
        self.epsilon_scheduler.step()
