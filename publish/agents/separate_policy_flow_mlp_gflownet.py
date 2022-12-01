from publish.agents import BasicGFlowNet
from publish.agents.losses import BaseLoss
from publish.utils import build_mlp, get_device
from collections.abc import Iterable
from torchtyping import TensorType
from typing import Dict
import copy
import itertools
import torch

class SeparatePolicyFlowMLPGFlowNet(BasicGFlowNet):
    def _setup(self, config: Dict[str, object]):
        device = get_device()
        config['param_backward_policy'] = False

        self.back_policy_model = build_mlp(
            config['obs_dim'],
            config['action_dim'] - 1,
            config['hidden_layer_dim'],
            config['num_hidden_layers'],
            should_model_state_flow=False,
            should_param_backward_policy=False,
            activation=config.get('activation', torch.nn.LeakyReLU())
        ).to(device=device)

        self.flow_model = build_mlp(
            config['obs_dim'],
            1,
            config['hidden_layer_dim'],
            config['num_hidden_layers'],
            activation=config.get('activation', torch.nn.LeakyReLU())
        ).to(device=get_device())

        super()._setup(config)

    def parameters(self) -> Iterable:
        return itertools.chain(
            self.model.parameters(),
            self.back_policy_model.parameters(),
            self.flow_model.parameters()
        )

    def _get_should_model_state_flow(
        self,
        config: Dict[str, object],
        loss_fxn: BaseLoss
    ) -> bool:
        return False

    def _get_loss_infos(
        self,
        update_infos: Dict[str, object]
    ) -> Dict[str, object]:
        return {
            'log_state_flows': self.flow_model(update_infos['states']),
            **super()._get_loss_infos(update_infos)
        }

    def _gather_log_probs(
        self,
        model_outs: TensorType['batch_size', 'horizon', 'model_out_dim'],
        observations: TensorType['batch_size', 'horizon', 'obs_dim'],
        actions: TensorType['batch_size', 'horizon'],
        mode: str
    ) -> TensorType['batch_size', 'horizon']:
        assert mode in ['fwd', 'back']
        if mode == 'fwd':
            return super()._gather_log_probs(
                model_outs,
                observations,
                actions,
                mode
            )

        # At this point we're just concerned with the backward policy
        if self.use_tree_pb:
            return torch.zeros_like(actions)

        pb_unmasked_logits = self.back_policy_model(observations)
        return self._gather_masked_log_probs(
            pb_unmasked_logits,
            observations,
            actions,
            mode
        )
