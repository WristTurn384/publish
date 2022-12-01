from collections.abc import Iterable
from publish.agents import BaseAgent
from publish.agents.losses import BaseLoss, TrajectoryBalanceLoss
from publish.utils import build_mlp
from publish.utils.constants import LOG_ZERO, FORWARD, BACKWARD
from torchtyping import TensorType
from typing import Dict, Tuple
from torch.distributions.categorical import Categorical
from pathlib import Path
import numpy as np
import torch
import copy

def should_model_state_flow(
    config: Dict[str, object],
    loss_fxn: BaseLoss
) -> bool:
    return config.get('model_state_flow') or loss_fxn.requires_state_flows


def get_periodic_log_Z_settings(config: Dict[str, object]) -> Tuple[bool, int, int]:
    change_config = config.get('periodic_log_Z_change_config', None)
    if change_config is None:
        return False, None, None

    return (
        True,
        change_config['change_value'],
        change_config['num_updates_per_change']
    )

class BasicGFlowNet(BaseAgent):
    def _setup(self, config: Dict[str, object]):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        loss_fxn_config = {
            **config,
            **config.get('loss_fxn_config', {'type': TrajectoryBalanceLoss})
        }

        loss_fxn = loss_fxn_config['type'](loss_fxn_config)
        self.model_state_flow = self._get_should_model_state_flow(config, loss_fxn)

        model = build_mlp(
            config['obs_dim'],
            config['action_dim'],
            config['hidden_layer_dim'],
            config['num_hidden_layers'],
            self.model_state_flow,
            config.get('param_backward_policy', True),
            config.get('activation', torch.nn.LeakyReLU())
        ).to(device=device)

        self._init_base_agent(
            model,
            loss_fxn,
            config['env'],
            config.get('optim_config')
        )

        self.true_Z = config['true_Z']
        if config.get('use_true_log_Z', False):
            self.init_log_Z_val = np.log(config['true_Z'])
        else:
            self.init_log_Z_val = config.get('init_log_Z_val', 0.0)

        self.log_Z = torch.nn.Parameter(
            torch.tensor(self.init_log_Z_val, requires_grad=True, device=device)
        )

        self.log_Z_optim_config = config['log_Z_optim_config']
        self.log_Z_optim, self.log_Z_lr_scheduler = self._build_optim(
            copy.deepcopy(self.log_Z_optim_config),
            [self.log_Z]
        )

        (
            self.do_periodic_log_Z_bump,
            self.periodic_log_Z_change,
            self.log_Z_change_period
        ) = get_periodic_log_Z_settings(config)

        self.num_updates_done = 0

        self.invalid_action_idx = torch.zeros(
            config['obs_dim'],
            dtype=torch.long,
            device=device
        )

        self.use_tree_pb = config.get('use_tree_pb', False)

    def _get_should_model_state_flow(
        self,
        config: Dict[str, object],
        loss_fxn: BaseLoss
    ) -> bool:
        return should_model_state_flow(config, loss_fxn)

    def act(
        self,
        observations: TensorType['batch_size', 'obs_dim'],
        mode: str = FORWARD
    ) -> TensorType['batch_size']:
        logits_pre = self._get_unmasked_logits_from_model_outs(
            self.model(observations),
            mode
        )

        logits = self._get_masked_action_logits(observations, logits_pre, mode)
        return Categorical(logits=logits).sample()

    def _get_masked_action_logits(
        self,
        observations: TensorType['batch_size', 'obs_dim'],
        model_out_logits: TensorType['batch_size', 'num_actions'],
        mode: str = FORWARD
    ) -> TensorType['batch_size', 'num_actions']:
        invalid_action_mask = self.env.get_invalid_action_mask(observations, mode)
        return (
            (~invalid_action_mask * model_out_logits) +
            (invalid_action_mask * LOG_ZERO)
        )

    def update(self, update_infos: Dict[str, object]) -> Dict[str, object]:
        loss = self.loss(update_infos)

        if isinstance(loss, dict):
            scalar_loss = sum(loss.values())
        else:
            scalar_loss = loss

        self.optim.zero_grad()
        self.log_Z_optim.zero_grad()
        scalar_loss.backward()
        self.optim.step()
        self.log_Z_optim.step()

        if self.lr_scheduler is not None:
            self._lr_scheduler_step(self.lr_scheduler, scalar_loss)

        if self.log_Z_lr_scheduler is not None:
            self._lr_scheduler_step(self.log_Z_lr_scheduler, scalar_loss)

        self.num_updates_done += 1
        self._maybe_do_log_Z_bump()

        return loss

    def _lr_scheduler_step(
        self,
        scheduler: torch.optim.lr_scheduler,
        loss: TensorType[1]
    ) -> None:
        call_kwargs = {}
        reduce_lr_plat_type = torch.optim.lr_scheduler.ReduceLROnPlateau
        if isinstance(scheduler, reduce_lr_plat_type):
            call_kwargs['metrics'] = loss

        scheduler.step(**call_kwargs)

    def _maybe_do_log_Z_bump(self) -> None:
        cond1 = self.do_periodic_log_Z_bump
        cond2 = self.num_updates_done != 1
        cond3 = False
        if cond1:
            cond3 = self.num_updates_done % self.log_Z_change_period == 0

        if cond1 and cond2 and cond3:
            self.log_Z = torch.nn.Parameter(
                self.log_Z + self.periodic_log_Z_change
            )

            self.log_Z_optim, self.log_Z_lr_scheduler = self._build_optim(
                copy.deepcopy(self.log_Z_optim_config),
                [self.log_Z]
            )

    def get_log_pf(
        self,
        states: TensorType['batch_size', 'horizon', 'obs_dim'],
        actions: TensorType['batch_size', 'horizon']
    ):
        model_outs = self.model(states)
        return self._gather_log_probs(model_outs, states, actions, 'fwd')

    def get_log_pb(
        self,
        states: TensorType['batch_size', 'horizon', 'obs_dim'],
        actions: TensorType['batch_size', 'horizon']
    ):
        model_outs = self.model(states)
        return self._gather_log_probs(model_outs, states, actions, 'back')

    def _get_loss_infos(
        self,
        update_infos: Dict[str, object]
    ) -> Dict[str, object]:
        loss_infos = update_infos
        loss_infos['log_Z'] = self.log_Z

        model_outs = self.model(update_infos['states'])
        if self.model_state_flow:
            loss_infos['log_state_flows'] = model_outs[..., -1]

        loss_infos['back_log_probs'] = self._gather_log_probs(
            model_outs,
            self._get_env_observations(update_infos),
            update_infos['back_actions'],
            BACKWARD
        )

        loss_infos['fwd_log_probs'] = self._gather_log_probs(
            model_outs,
            self._get_env_observations(update_infos),
            update_infos['fwd_actions'],
            FORWARD
        )

        return loss_infos

    def _get_env_observations(
        self,
        update_infos: Dict[str, object]
    ) -> TensorType['batch_size', 'horizon', 'obs_dim']:
        return update_infos['states']

    def _gather_log_probs(
        self,
        model_outs: TensorType['batch_size', 'horizon', 'model_out_dim'],
        observations: TensorType['batch_size', 'horizon', 'obs_dim'],
        actions: TensorType['batch_size', 'horizon'],
        mode: str
    ) -> TensorType['batch_size', 'horizon']:
        assert mode in [FORWARD, BACKWARD]
        if mode == BACKWARD and self.use_tree_pb:
            return torch.zeros_like(actions)

        unmasked_logits = self._get_unmasked_logits_from_model_outs(
            model_outs,
            mode
        )

        return self._gather_masked_log_probs(
            unmasked_logits,
            observations,
            actions,
            mode
        )

    def _get_unmasked_logits_from_model_outs(
        self,
        model_outs: TensorType['batch_size', 'horizon', 'model_out_dim'],
        mode: str
    ) -> TensorType:
        if mode == FORWARD:
            return model_outs[..., :self._act_dim]
        else:
            return model_outs[..., self._act_dim:(2 * self._act_dim) - 1]

    def _gather_masked_log_probs(
        self,
        unmasked_logits: TensorType['batch_size', 'horizon', 'act_dim'],
        observations: TensorType['batch_size', 'horizon', 'obs_dim'],
        actions: TensorType['batch_size', 'horizon'],
        mode: str
    ) -> TensorType['batch_size', 'horizon']:
        logits = self._get_masked_action_logits(
            observations,
            unmasked_logits,
            mode
        )

        cat_distrib = Categorical(logits=logits)

        # Clone the actions so we don't change them overall in-place
        scoped_actions = actions.clone()
        invalid_action_idx = torch.nonzero(scoped_actions == -1)
        scoped_actions[invalid_action_idx[:, 0], invalid_action_idx[:, 1]] = 0

        out_log_probs = cat_distrib.log_prob(scoped_actions)
        out_log_probs[invalid_action_idx[:, 0], invalid_action_idx[:, 1]] = 0.

        return out_log_probs

    def save(self, filename) -> None:
        torch.save(self.model.state_dict(), filename)

    def load(self, filename) -> None:
        self.model.load_state_dict(torch.load(filename))

    def get_metrics(self) -> Dict[str, object]:
        metric_dict = {
            'log_Z': self.log_Z.item()
        }

        if self.log_Z_lr_scheduler:
            metric_dict['log_Z_lr'] = self.log_Z_lr_scheduler._last_lr[0]

        if self.lr_scheduler:
            metric_dict['model_lr'] = self.lr_scheduler._last_lr[0]

        return metric_dict
