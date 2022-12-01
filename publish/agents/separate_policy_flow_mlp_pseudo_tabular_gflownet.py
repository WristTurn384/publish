from publish.agents import SeparatePolicyFlowMLPGFlowNet
from publish.agents.losses import BaseLoss, NaiveSubtrajectoryBalanceLoss
from publish.utils import get_grid_one_hot_int_encoder_mask
from torch.distributions.categorical import Categorical
from torchtyping import TensorType
from typing import Dict
import copy
import torch

class SeparatePolicyFlowMLPPseudoTabularGFlowNet(SeparatePolicyFlowMLPGFlowNet):
    def _setup(self, config: Dict[str, object]):
        super()._setup(config)

        env = config['env']
        int_states = torch.arange(
            env.side_length ** env.num_dims,
            device=env.device
        )

        self.all_states = torch.zeros(
            (len(int_states), env.obs_dim),
            device=env.device
        )

        idx1 = int_states.div(env.side_length, rounding_mode='floor').long()
        idx2 = int_states.remainder(env.side_length).long() + env.side_length

        self.all_states[torch.arange(len(int_states)), idx1] = 1.
        self.all_states[torch.arange(len(int_states)), idx2] = 1.

        self.encoder_mask = get_grid_one_hot_int_encoder_mask(
            env.num_dims,
            env.side_length
        )

    def _get_loss_infos(
        self,
        update_infos: Dict[str, object]
    ) -> Dict[str, object]:
        loss_infos = update_infos

        table_logits = self.model(self.all_states)
        loss_infos['fwd_table_logits'] = self.model(self.all_states)
        loss_infos['back_table_logits'] = self.back_policy_model(self.all_states)

        int_encoded_states = (
            update_infos['states'] * self.encoder_mask
        ).sum(dim=-1).long()

        for mode in ['fwd', 'back']:
            loss_infos['%s_log_probs' % mode] = self._gather_log_probs(
                update_infos['%s_table_logits' % mode],
                update_infos['states'],
                int_encoded_states.clone(),
                update_infos['%s_actions' % mode],
                mode
            )

        loss_infos['log_state_flows'] = self.flow_model(update_infos['states'])

        if isinstance(self._loss_fxn, NaiveSubtrajectoryBalanceLoss):
            loss_infos['log_Z'] = self.flow_model(self.all_states[0])
        else:
            loss_infos['log_Z'] = self.log_Z

        return loss_infos

    def _gather_log_probs(
        self,
        logits_table: TensorType['num_states', 'action_dim'],
        one_hot_states: TensorType['batch_size', 'horizon', 'obs_dim'],
        int_encoded_states: TensorType['batch_size', 'horizon'],
        actions: TensorType['batch_size', 'horizon'],
        mode: str
    ) -> TensorType['batch_size', 'horizon']:
        assert mode in ['fwd', 'back']
        if mode == 'back' and self.use_tree_pb:
            return torch.zeros_like(actions)

        scoped_actions = actions.clone()
        invalid_action_idx = torch.nonzero(scoped_actions == -1)

        int_encoded_states[invalid_action_idx[:, 0], invalid_action_idx[:, 1]] = 0

        logits = super()._get_masked_action_logits(
            one_hot_states,
            logits_table[int_encoded_states],
            mode
        )

        cat_distrib = Categorical(logits=logits)
        scoped_actions[invalid_action_idx[:, 0], invalid_action_idx[:, 1]] = 0

        out_log_probs = cat_distrib.log_prob(scoped_actions)
        out_log_probs[invalid_action_idx[:, 0], invalid_action_idx[:, 1]] = 0.

        return out_log_probs

    def get_log_pf(
        self,
        states: TensorType['batch_size', 'horizon', 'obs_dim'],
        actions: TensorType['batch_size', 'horizon']
    ):
        table_logits = self.model(self.all_states)
        int_encoded_states = (
            states * self.encoder_mask
        ).sum(dim=-1).long()

        return self._gather_log_probs(
            table_logits,
            states,
            int_encoded_states,
            actions,
            'fwd'
        )

    def get_log_pb(
        self,
        states: TensorType['batch_size', 'horizon', 'obs_dim'],
        actions: TensorType['batch_size', 'horizon']
    ):
        table_logits = self.back_policy_model(self.all_states)
        int_encoded_states = (
            states * self.encoder_mask
        ).sum(dim=-1).long()

        return self._gather_log_probs(
            table_logits,
            states,
            int_encoded_states,
            actions,
            'back'
        )
