from publish.agents.losses import BaseLoss
from publish.envs import BaseEnvironment
from hive.agents.agent import Agent
from collections.abc import Iterable
from torchtyping import TensorType
from typing import Callable, Dict, Iterable
import torch
import copy

REDUCTIONS = {
    'mean': torch.mean,
    'sum': torch.sum,
    'identity': lambda x: x,
    None: lambda x: x,
}

class BaseAgent(Agent):
    def __init__(self, config: Dict[str, object]):
        config_overrides = self._get_config_overrides(config)

        # This is a really basic way to do overrides.  But I don't need
        # anything more fancy for now, so this will suffice.
        for key, val in config_overrides.items():
            config[key] = val

        self._setup(config)

    def _get_config_overrides(self, config: Dict[str, object]):
        return {}

    def _setup(self, config: Dict[str, object]) -> None:
        '''
        Method to implement for setting up your agent.  New agents
        should NOT implement __init__ as this has logic that should
        be run for setting up the config before setup is called.
        '''
        pass

    def _init_base_agent(
        self,
        model: torch.nn.Module,
        loss_fxn: BaseLoss,
        env: BaseEnvironment,
        optimizer_config: Dict[str, object],
        build_optimizer: bool = True
    ):
        self._loss_fxn = loss_fxn
        self.model = model
        self.env = env

        # RLHive is designed for multi-agent RL and so takes an agent ID
        # as input.  We're not doing anything multi-agent here, so we just
        # supply an id of 0.
        Agent.__init__(self, self.env.obs_dim, self.env.action_dim, id=0)

        if build_optimizer:
            self.optimizer_config = optimizer_config
            self.optim, self.lr_scheduler = self._build_optim(
                copy.deepcopy(self.optimizer_config)
            )

    def loss(
        self,
        update_infos: Dict[str, object],
        reduction: str = 'mean',
        loss_fxn: BaseLoss = None
    ) -> TensorType:
        augmented_infos = self._get_loss_infos(update_infos)
        augmented_infos['agent'] = self

        loss_fxn = loss_fxn if loss_fxn is not None else self._loss_fxn
        unreduced_loss : TensorType['batch_size'] = loss_fxn(augmented_infos)
        return REDUCTIONS[reduction](unreduced_loss)

    def _get_loss_infos(
        self,
        update_infos: Dict[str, object]
    ) -> Dict[str, object]:
        return update_infos

    def parameters(self) -> Iterable:
        return self.model.parameters()

    def get_agent_state(
        self,
        all_states: TensorType['batch_size', 'horizon', 'ndim_times_side_len'],
        all_actions: TensorType['batch_size', 'horizon'],
        all_dones: TensorType['batch_size', 'horizon'],
        all_rewards: TensorType['batch_size'],
        iter_num: int
    ) -> TensorType:
        return all_states[:, iter_num]

    @property
    def does_grad_update(self) -> bool:
        return True

    def _build_optim(
        self,
        optimizer_config: Dict[str, object],
        parameters: Iterable = None
    ) -> torch.optim.Optimizer:
        optim_type = optimizer_config.pop('type', torch.optim.Adam)
        lr_scheduler_config = optimizer_config.pop('lr_scheduler_config', None)
        lr = optimizer_config.pop('lr')

        optim = optim_type(
            parameters or self.parameters(),
            lr=lr,
            **optimizer_config
        )

        lr_scheduler = None
        if lr_scheduler_config is not None:
            scheduler_type = lr_scheduler_config.pop('type')
            lr_scheduler = scheduler_type(optim, **lr_scheduler_config)

        return optim, lr_scheduler

    def get_log_pf(
        self,
        states: TensorType['batch_size', 'horizon', 'obs_dim'],
        actions: TensorType['batch_size', 'horizon']
    ):
        return torch.zeros_like(actions)

    def get_log_pb(
        self,
        states: TensorType['batch_size', 'horizon', 'obs_dim'],
        actions: TensorType['batch_size', 'horizon']
    ):
        return torch.zeros_like(actions)

    def get_metrics(self) -> Dict[str, object]:
        return {}

    # A method to allow subclasses to perform logic at the end of an
    # epoch if they would like.
    def end_epoch(self) -> None:
        pass
