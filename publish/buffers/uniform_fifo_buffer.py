from publish.utils.interfaces import BaseSampler
from hive.replays import BaseReplayBuffer
from torchtyping import TensorType
from typing import Dict, Tuple
import torch

TOKEN = -1

BUFF_NAMES = (
    'traj_obs',
    'traj_actions',
    'traj_back_actions',
    'traj_dones',
    'traj_rewards'
)

CAPACITY_FILLED_ATTR_NAME = '_capacity_filled'

ALL_ATTR_NAMES = BUFF_NAMES + (CAPACITY_FILLED_ATTR_NAME,)

class UniformFifoBuffer(BaseSampler, BaseReplayBuffer):
    def __init__(self, config: Dict[str, object]):
        obs_dim, horizon, capacity = \
            config['obs_dim'], config['horizon'], config['capacity']

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.traj_obs = torch.full(
            (capacity, horizon, obs_dim),
            fill_value=TOKEN,
            dtype=torch.float,
            device=device
        )

        self.traj_actions = torch.full(
            (capacity, horizon),
            fill_value=TOKEN,
            device=device
        )

        self.traj_back_actions = torch.full(
            (capacity, horizon),
            fill_value=TOKEN,
            device=device
        )

        self.traj_dones = torch.full(
            (capacity, horizon),
            fill_value=TOKEN,
            device=device
        )

        self.traj_rewards = torch.full(
            (capacity,),
            fill_value=TOKEN,
            dtype=torch.float,
            device=device
        )

        self._capacity = capacity
        self._capacity_filled = torch.zeros(capacity, dtype=torch.int)
        self._last_size = 0
        self._have_warned = False

    def add(
        self,
        observations: TensorType['batch_size', 'horizon', 'obs_dim'],
        actions: TensorType['batch_size', 'horizon'],
        back_actions: TensorType['batch_size', 'horizon'],
        dones: TensorType['batch_size', 'horizon'],
        rewards: TensorType['batch_size']
    ) -> None:
        def cycle_buffer(buff_and_to_add):
            buff_name, to_add = buff_and_to_add
            buff = getattr(self, buff_name)

            buff = buff.roll(len(to_add), dims=0)
            buff[:len(to_add)] = to_add

            setattr(self, buff_name, buff)

        to_adds = (observations, actions, back_actions, dones, rewards)
        tuple(map(cycle_buffer, zip(BUFF_NAMES, to_adds)))
        if self._last_size != self._capacity:
            cycle_buffer((
                CAPACITY_FILLED_ATTR_NAME,
                torch.ones(len(observations), dtype=torch.int)
            ))

        return

    def size(self) -> int:
        if self._last_size != self._capacity:
            self._last_size = self._capacity_filled.sum()

        return self._last_size

    def get_last_n_inserted(self, batch_size: int = 32) -> Tuple[
        TensorType['batch_size', 'horizon', 'obs_dim'],
        TensorType['batch_size', 'horizon'],
        TensorType['batch_size', 'horizon'],
        TensorType['batch_size', 'horizon'],
        TensorType['batch_size']
    ]:
        curr_size = self.size()
        if batch_size > curr_size:
            print('Batch size: %d, current size: %d' % (batch_size, curr_size))
            assert batch_size <= curr_size

        return {
            'states': self.traj_obs[:batch_size],
            'fwd_actions': self.traj_actions[:batch_size],
            'back_actions': self.traj_back_actions[:batch_size],
            'dones': self.traj_dones[:batch_size],
            'rewards': self.traj_rewards[:batch_size],
        }

    def sample(self, batch_size: int = 32) -> Tuple[
        TensorType['batch_size', 'horizon', 'obs_dim'],
        TensorType['batch_size', 'horizon'],
        TensorType['batch_size', 'horizon'],
        TensorType['batch_size', 'horizon'],
        TensorType['batch_size']
    ]:
        curr_size = self.size()
        if batch_size >= curr_size:
            if not self._have_warned:
                print(
                    ('WARNING: Wanted to sample %d trajectories from the ' % batch_size) +
                    ('buffer but current size is only %d.  Sampling only ' % curr_size) +
                    ('%d trajectories.' % curr_size)
                )

                self._have_warned = True

            batch_size = curr_size

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sample_idxs = torch.randperm(
            curr_size.item(),
            dtype=torch.long,
            device=device
        )[:batch_size]

        return {
            'states': self.traj_obs[sample_idxs],
            'fwd_actions': self.traj_actions[sample_idxs],
            'back_actions': self.traj_back_actions[sample_idxs],
            'dones': self.traj_dones[sample_idxs],
            'rewards': self.traj_rewards[sample_idxs],
        }

    def save(self, dir_name: str) -> None:
        def _save(attr_name):
            torch.save(getattr(self, attr_name), dir_name + '/' + attr_name)

        tuple(map(_save, ALL_ATTR_NAMES))

    def load(self, dir_name: str) -> None:
        def _load(attr_name):
            setattr(self, attr_name, torch.load(dir_name + '/' + attr_name))

        tuple(map(_load, ALL_ATTR_NAMES))
        self._last_size = 0
