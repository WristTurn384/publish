from publish.agents import BasicGFlowNet
from publish.agents.flow_functions import TabularFlowFunction
from publish.agents.losses import BaseLoss, GAFlowNetTrajectoryBalanceLoss
from publish.agents.utils import RandomNetworkDistillation
from torchtyping import TensorType
from typing import Dict, Iterable, Callable
import torch


class BasicGAFlowNet(BasicGFlowNet):
    def _setup(self, config: Dict[str, object]):
        config['flow_function_config'] = {'type': TabularFlowFunction}

        self.random_network_distillation = RandomNetworkDistillation({
            'env': config['env'],
            **config['random_network_distillation_config']
        })

        self.intrinsic_reward_weight = config.get('intrinsic_reward_weight', 1.0)

        super()._setup(config)

    def _get_config_overrides(
        self,
        config: Dict[str, object]
    ) -> Dict[str, object]:
        overrides = super()._get_config_overrides(config)
        overrides['loss_fxn_config'] = {'type': GAFlowNetTrajectoryBalanceLoss}

        return overrides

    def parameters(self) -> Iterable[TensorType]:
        params = list(super().parameters())
        params.extend(self.random_network_distillation.parameters())

        return tuple(params)

    def loss(
        self,
        update_infos: Dict[str, object],
        reduction: str = 'mean',
        loss_fxn: BaseLoss = None
    ) -> TensorType:
        policy_loss = super().loss(update_infos, reduction, loss_fxn)
        rnd_loss = self._do_rnd_call(
            self.random_network_distillation.loss,
            update_infos
        )

        return {'policy': policy_loss, 'rnd': rnd_loss}

    def _get_loss_infos(
        self,
        update_infos: Dict[str, object],
    ) -> Dict[str, object]:
        loss_infos = super()._get_loss_infos(update_infos)
        loss_infos['intrinsic_rewards'] = \
            self.intrinsic_reward_weight * self._do_rnd_call(
                self.random_network_distillation,
                loss_infos
            )

        return loss_infos

    def _do_rnd_call(
        self,
        rnd_callable: Callable,
        update_infos: Dict[str, object]
    ) -> TensorType:
        return rnd_callable(update_infos)
