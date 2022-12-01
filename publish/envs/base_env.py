from abc import abstractmethod
from typing import Tuple
from torchtyping import TensorType
from hive.envs.base import BaseEnv
import numpy as np

class BaseEnvironment(BaseEnv):
    def __init__(self, env_spec):
        super().__init__(env_spec=env_spec, num_players=1)

    @abstractmethod
    def true_density(self) -> Tuple[np.array, np.array, np.array]:
        """Compute the true reward density for the environment

        Returns:
            Tuple of three np.arrays.  The first is the true density for the
            environment.  The second is the set of all interior states (i.e.,
            states in the environment which can be moved from).  The third
            is the set of unnormalized rewards (for each interior state).
        """
        pass

    def true_Z(self) -> Tuple[bool, float]:
        return False, 0.0

    @property
    @abstractmethod
    def done_action(self) -> int:
        pass

    @property
    @abstractmethod
    def obs_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def action_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def horizon(self) -> int:
        pass

    @property
    def num_states(self) -> int:
        return -1

    @property
    def num_states_tractable(self) -> bool:
        return self.num_states != -1

    @property
    def terminal_state(self) -> TensorType['obs_dim']:
        return None

    def get_state_ids(
        self,
        state: TensorType['batch_size', 'obs_dim']
    ) -> TensorType['batch_size', int]:
        return None

    @abstractmethod
    def is_origin_state(
        self,
        state: TensorType['obs_dim']
    ) -> TensorType['batch_size', bool]:
        pass

    @abstractmethod
    def get_backward_actions(
        self,
        actions: TensorType['horizon', 'batch_size']
    ) -> TensorType['horizon', 'batch_size']:
        pass

    @abstractmethod
    def clone(self, new_batch_size: int = None) -> 'BaseEnvironment':
        pass

    @abstractmethod
    def get_rewards(
        self,
        states: TensorType['batch_size', 'obs_dim']
    ) -> TensorType['batch_size']:
        pass

    @abstractmethod
    def get_invalid_action_mask(
        self,
        observations: TensorType['batch_size', 'obs_dim'],
        mode: str
    ) -> TensorType['batch_size', 'num_actions']:
        pass
