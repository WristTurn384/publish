from publish.agents import BasicGFlowNet
from publish.agents.losses import NaiveSubtrajectoryBalanceLoss
from publish.agents.flow_functions import TabularFlowFunction
from publish.utils import get_grid_one_hot_int_encoder_mask, get_first_done_idx
from torchtyping import TensorType
from typing import Dict, Tuple, Iterable
from torch.distributions.categorical import Categorical
import torch

class TabularGFlowNet(BasicGFlowNet):
    def _setup(self, config: Dict[str, object]):
        env = config['env']
        num_states = env.side_length ** env.num_dims

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fwd_transition_logits = torch.zeros(
            (num_states, env.action_dim),
            device=device,
            requires_grad=True
        )

        self.back_transition_logits = torch.zeros(
            (num_states, env.action_dim - 1),
            device=device,
            requires_grad=True
        )

        flow_fxn_config = config.get(
            'flow_function_config',
            {'type': TabularFlowFunction}
        )

        self.state_flow_function = flow_fxn_config['type']({
            'env': env,
            'agent': self,
            **flow_fxn_config
        })

        self.state_flow_function.update(self)

        # We aren't going to use a model here, so assign it to None
        # for superclass initialization
        super()._setup(config)

        encoder_mask_part1 = torch.arange(
            env.side_length,
            device=device
        ).repeat(env.num_dims)

        encoder_mask_part2 = (
            env.side_length ** torch.arange(
                env.num_dims,
                device=device
            ).flip(0).repeat_interleave(env.side_length)
        )

        self.encoder_mask = get_grid_one_hot_int_encoder_mask(
            env.num_dims,
            env.side_length
        )

        self.use_tree_pb = config.get('use_tree_pb', False)

    def parameters(self) -> Iterable[TensorType]:
        params = [
            self.fwd_transition_logits,
            self.back_transition_logits,
        ]

        if self._loss_fxn.requires_state_flows:
            params.extend(self.state_flow_function.parameters())

        return tuple(params)

    def update(self, update_infos: Dict[str, object]) -> Dict[str, object]:
        loss = super().update(update_infos)
        if self.state_flow_function is not None:
            self.state_flow_function.update(self)

        return loss

    def act(
        self,
        observations: TensorType['batch_size', 'obs_dim']
    ) -> TensorType['batch_size']:
        int_encoded_states = (observations * self.encoder_mask).sum(dim=1).long()
        logits = super()._get_masked_action_logits(
            observations,
            self.fwd_transition_logits[int_encoded_states]
        )

        return Categorical(logits=logits).sample()

    def get_log_pf(
        self,
        states: TensorType['batch_size', 'horizon', 'obs_dim'],
        actions: TensorType['batch_size', 'horizon']
    ):
        int_encoded_states = (states * self.encoder_mask).sum(dim=-1).long()
        return self._gather_log_probs(states, int_encoded_states, actions, 'fwd')

    def get_log_pb(
        self,
        states: TensorType['batch_size', 'horizon', 'obs_dim'],
        actions: TensorType['batch_size', 'horizon']
    ):
        int_encoded_states = (states * self.encoder_mask).sum(dim=-1).long()
        return self._gather_log_probs(states, int_encoded_states, actions, 'back')

    def _get_loss_infos(
        self,
        update_infos: Dict[str, object],
        flow_fxn=None
    ) -> Dict[str, object]:
        loss_infos = update_infos
        loss_infos['log_Z'] = self.log_Z

        int_encoded_states = (
            update_infos['states'] * self.encoder_mask
        ).sum(dim=-1).long()

        for mode in ['fwd', 'back']:
            loss_infos['%s_log_probs' % mode] = self._gather_log_probs(
                update_infos['states'],
                int_encoded_states.clone(),
                update_infos['%s_actions' % mode],
                mode
            )

        log_flows = self.state_flow_function.get_flows(
            int_encoded_states,
            loss_infos['log_Z']
        ).log()

        loss_infos['log_state_flows'] = log_flows
        loss_infos['log_Z'] = log_flows[0, 0]
        return loss_infos

    def _gather_log_probs(
        self,
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
        all_logits = (
            self.fwd_transition_logits
            if mode == 'fwd'
            else self.back_transition_logits
        )

        int_encoded_states[invalid_action_idx[:, 0], invalid_action_idx[:, 1]] = 0

        logits = super()._get_masked_action_logits(
            one_hot_states,
            all_logits[int_encoded_states],
            mode
        )

        cat_distrib = Categorical(logits=logits)
        scoped_actions[invalid_action_idx[:, 0], invalid_action_idx[:, 1]] = 0

        out_log_probs = cat_distrib.log_prob(scoped_actions)
        out_log_probs[invalid_action_idx[:, 0], invalid_action_idx[:, 1]] = 0.

        return out_log_probs
