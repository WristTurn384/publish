from publish.utils import Trajectory, get_grid_one_hot_int_encoder_mask
from publish.utils.constants import FORWARD
from publish.envs import BaseEnvironment
from typing import Dict, Tuple
from torchtyping import TensorType
import torch
import itertools
import numpy as np
import random
import queue

TOKEN_VAL = -1

class HypergridEnv(BaseEnvironment):
    def __init__(self, env_spec: Dict[str, int]):
        super().__init__(env_spec)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.side_length = self.env_spec['side_length']
        self.num_dims = self.env_spec['num_dims']
        self.sample_batch_size = self.env_spec['sample_batch_size']
        self.R_0 = self.env_spec['R_0']
        self.R_1 = self.env_spec.get('R_1', 0.5)
        self.R_2 = self.env_spec.get('R_2', 2.0)

        self._one_hot_idx1_cached = torch.arange(
            self.sample_batch_size,
            device=self.device
        ).repeat_interleave(self.num_dims)

        self._one_hot_idx2_cached = \
            self.side_length * torch.arange(self.num_dims, device=self.device)

        self._int_encoder_mask = get_grid_one_hot_int_encoder_mask(
            self.num_dims,
            self.side_length
        )

        self._true_density = None
        self._paths_to_state = {}
        self._stop_action_paths_to_state = {}
        if self.num_dims == 2:
            states = torch.tensor([
                [i, j]
                for i in range(self.side_length)
                for j in range(self.side_length)
            ])

            rewards = self._reward_fxn(states)
            self._state_to_reward = {
                tuple(states[i].tolist()): rewards[i] for i in range(len(states))
            }

        self.fwd_border_state_idxs = (
            self.side_length * torch.arange(1, self.num_dims + 1)
        ) - 1

        self.back_border_state_idxs = \
            self.side_length * torch.arange(self.num_dims)

        self.reset()

    @property
    def obs_dim(self) -> int:
        return self.num_dims * self.side_length

    @property
    def action_dim(self) -> int:
        return self.num_dims + 1

    @property
    def horizon(self) -> int:
        return self.num_dims * (self.side_length - 1) + 1

    @property
    def stop_action(self) -> int:
        return self.num_dims

    @property
    def num_states(self) -> int:
        return self.side_length ** self.num_dims

    def reset(self) -> TensorType['batch_size', 'ndim_times_side_len', int]:
        """Resets the environment.

        Returns:
            np.array -- The current position on the grid in format
                        {0, ..., side_length}^num_dims
        """
        self.states = torch.zeros(
            (self.sample_batch_size, self.num_dims),
            dtype=torch.long,
            device=self.device
        )

        self.dones = torch.zeros(
            self.sample_batch_size,
            dtype=torch.bool,
            device=self.device
        )

        self.rewards = torch.zeros(
            self.sample_batch_size,
            dtype=torch.float,
            device=self.device
        )

        return self._one_hot_state(self.states)

    def step(
        self,
        actions: TensorType['batch_size', int],
        states: TensorType['batch_size', 'ndim_times_side_len', int] = None
    ) -> Tuple[
        TensorType['batch_size', 'ndim_times_side_len', int],
        TensorType['batch_size', float],
        TensorType['batch_size', bool],
    ]:
        """Takes a step according to the action.

        Args:
            action: int -- Index of action to take

        Returns:
            object: The current position on the grid in format
                    {0, ..., side_length}^num_dims

            float: The reward for this step
            bool: A boolean representing whether this step concludes the episode
            dict: A dictionary of extra information for the episode.  In our
                  case, this is always empty.
        """
        use_self_tnsrs = states is None

        dones = self.dones if use_self_tnsrs else torch.zeros_like(actions)
        rewards = self.rewards if use_self_tnsrs else torch.zeros_like(actions)
        states = self.states if use_self_tnsrs else self._one_hot_to_internal(
            states
        )

        if not dones.all():
            old_dones = dones.clone()
            dones_pre = self._get_new_dones(dones, actions)

            not_done_nnz = torch.nonzero(~dones_pre).flatten()
            states[not_done_nnz, actions[not_done_nnz]] += 1

            if use_self_tnsrs:
                self.dones = dones_pre

            dones = dones_pre
            rewards = self._update_rewards(
                old_dones,
                actions,
                states,
                rewards,
                use_self_tnsrs
            )

        return (
            self._one_hot_state(states, use_self_tnsrs),
            rewards.clone(),
            dones.clone()
        )

    def step_backward(
        self,
        actions: TensorType["batch_size", int],
        states: TensorType["batch_size", "ndim_times_side_len"] = None
    ) -> TensorType["batch_size", "ndim_times_side_len"]:
        use_self_tnsrs = states is None

        states = self.states if use_self_tnsrs else self._one_hot_to_internal(states)
        not_is_origin_mask = (states != 0).any(dim=-1)
        not_origin_nnz = torch.nonzero(not_is_origin_mask).flatten()

        states[not_origin_nnz, actions[not_origin_nnz]] -= 1
        return self._one_hot_state(states, use_self_tnsrs)

    @property
    def terminal_state(self) -> TensorType["ndim_times_side_len"]:
        pre_state = torch.full(
            size=(1, self.num_dims),
            fill_value=self.side_length - 1,
            device=self.states.device
        )

        return self._one_hot_state(pre_state, use_self_tnsrs=False).squeeze()

    def is_origin_state(
        self,
        states: TensorType["batch_size", "ndim_times_side_len"]
    ) -> TensorType["batch_size", bool]:
        return self.get_state_ids(states) == 0

    def get_action_ids(
        self,
        parent_states: TensorType["ndim_times_side_len"],
        child_states: TensorType["ndim_times_side_len"],
        mode: str = FORWARD
    ) -> int:
        parent_states = self._one_hot_to_internal(parent_states)
        child_states = self._one_hot_to_internal(child_states)

        diff = child_states - parent_states
        assert (diff.sum(dim=-1) == 1).all()

        nnz = diff.nonzero()
        if len(nnz.shape) == 2:
            nnz = nnz[:, 1]

        return nnz.flatten()

    def get_state_ids(
        self,
        states: TensorType["batch_size", "ndim_times_side_len"]
    ) -> int:
        if len(states.shape) == 1:
            states = states.unsqueeze(0)

        return self._one_hot_to_int_encoding(states)

    def _one_hot_to_internal(
        self,
        states: TensorType["batch_size", "ndim_times_side_len"]
    ) -> TensorType["batch_size", "ndims"]:
        if len(states.shape) == 1:
            states = states.unsqueeze(0)

        batch_size = states.shape[0]
        nnz = states.nonzero()

        internal_rep = torch.zeros(
            (batch_size, self.num_dims),
            dtype=torch.long,
            device=states.device
        )

        idx2 = torch.arange(self.num_dims).repeat(batch_size)

        internal_rep[nnz[:, 0], idx2] = nnz[:, 1]
        return internal_rep % self.side_length

    def _update_rewards(
        self,
        old_dones: TensorType["batch_size", bool],
        actions: TensorType["batch_size", int],
        states: TensorType["batch_size", "ndim"],
        rewards: TensorType["batch_size", float],
        use_self_tnsrs: bool
    ) -> None:
        # use xor operation
        new_dones_idx = \
            self.dones ^ old_dones if use_self_tnsrs else torch.ones_like(actions)

        new_rewards = self._reward_fxn(
            states[new_dones_idx],
            actions[new_dones_idx]
        )

        rewards[new_dones_idx] = new_rewards
        return rewards

    def get_rewards(
        self,
        states: TensorType["batch_size", "ndim_times_side_len"]
    ) -> TensorType["batch_size", float]:
        return self._reward_fxn(self._one_hot_to_internal(states))

    def _reward_fxn(
        self,
        states: TensorType[int],
        actions: TensorType[int] = None
    ) -> TensorType[float]:
        abs_val = torch.abs((states / float(self.side_length - 1)) - 0.5)

        mid_reward_ind = (abs_val > 0.25) & (abs_val <= 0.5)
        top_reward_ind = (abs_val > 0.3) & (abs_val < 0.4)

        reward = (
            self.R_0 +
            (self.R_1 * mid_reward_ind.prod(dim=-1)) +
            (self.R_2 * top_reward_ind.prod(dim=-1))
        )

        if actions is not None:
            reward[actions != self.num_dims] = 0.0

        return reward

    def _get_new_dones(
        self,
        dones: TensorType['batch_size', bool],
        actions: TensorType['batch_size', int]
    ) -> TensorType[int]:
        move_actions_idx = ~(dones | (actions == self.stop_action))
        move_actions_int_idx = torch.nonzero(move_actions_idx).flatten()
        move_actions_pre = actions[move_actions_int_idx]

        idx1, idx2 = torch.arange(len(move_actions_pre)), move_actions_pre
        to_move_states = self.states[move_actions_idx][idx1, idx2]

        move_actions_idx[move_actions_int_idx] = \
            to_move_states != self.side_length - 1

        return ~move_actions_idx

    def _one_hot_state(
        self,
        states: TensorType["batch_size", "ndim", int],
        use_self_tnsrs: bool = True
    ) -> TensorType["batch_size", "ndim_times_sidelen", int]:
        """Converts the current state to a one_hot grid
           of shape (num_dims, side_length)
        """
        one_hot_grid = torch.zeros(
            (states.shape[0], self.side_length * self.num_dims),
            dtype=torch.float,
            device=self.device
        )

        idx2 = (states + self._one_hot_idx2_cached).flatten()

        if use_self_tnsrs:
            one_hot_idx1 = self._one_hot_idx1_cached
        else:
            one_hot_idx1 = torch.arange(
                states.shape[0],
                device=self.device
            ).repeat_interleave(self.num_dims)

        one_hot_grid[one_hot_idx1, idx2] = 1
        return one_hot_grid

    @property
    def done_action(self) -> int:
        return self.num_dims

    def true_density(self) -> Tuple[np.array, np.array, np.array]:
        """Compute the true reward density for the environment

        Returns:
            Tuple of three np.arrays.  The first is the true density for the
            environment.  The second is the set of all interior states (i.e.,
            states in the environment which can be moved from).  The third
            is the set of unnormalized rewards (for each interior state).
        """
        if self._true_density is not None:
            return self._true_density

        all_states = torch.tensor(list(
            itertools.product(*[list(range(self.side_length))] * self.num_dims)
        ), dtype=torch.float)

        trajectory_rewards = self._reward_fxn(all_states)

        density = trajectory_rewards / trajectory_rewards.sum()
        all_states = list(map(tuple, all_states))

        self._true_density = (density, all_states, trajectory_rewards)
        return self._true_density

    def get_invalid_action_mask(
        self,
        observations: TensorType['batch_size', 'obs_dim'],
        mode: str
    ) -> TensorType['batch_size', 'num_actions']:
        if mode == 'fwd':
            border_state_idxs = self.fwd_border_state_idxs
        else:
            border_state_idxs = self.back_border_state_idxs

        invalid_action_mask = observations[..., border_state_idxs] == 1

        if mode == 'fwd':
            invalid_action_mask = torch.cat([
                invalid_action_mask,
                torch.zeros(
                    [*observations.shape[:-1]] + [1],
                    dtype=torch.bool,
                    device=observations.device
                )
            ], dim=-1)

        return invalid_action_mask

    def true_Z(self) -> Tuple[bool, float]:
        _, _, traj_rewards = self.true_density()
        return True, traj_rewards.sum()

    def seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _one_hot_to_int_encoding(
        self,
        states: TensorType[..., 'obs_dim']
    ) -> TensorType[..., 'num_dims']:
        return (states * self._int_encoder_mask).sum(dim=-1).long()

    def get_backward_actions(
        self,
        states: TensorType['batch_size', 'horizon', 'obs_dim'],
        actions: TensorType['batch_size', 'horizon']
    ) -> TensorType['batch_size', 'horizon']:
        cloned_actions = actions.clone()
        idx = torch.arange(
            start=cloned_actions.shape[1],
            end=0,
            step=-1,
            device=cloned_actions.device
        )

        act_eq_token = cloned_actions == -1
        tmp = act_eq_token * idx
        term_action_idx_pre = tmp.argmax(dim=1) - 1

        mask = act_eq_token.any(dim=1)
        term_action_idx = (
            (mask * term_action_idx_pre) + (~mask * (cloned_actions.shape[1] - 1))
        )

        idx1 = torch.arange(len(cloned_actions), device=cloned_actions.device)
        cloned_actions[idx1, term_action_idx] = -1

        return cloned_actions.roll(1)

    def clone(self, new_batch_size: int = None) -> 'HypergridEnv':
        new_conf = {
            'side_length': self.side_length,
            'num_dims': self.num_dims,
            'sample_batch_size': new_batch_size or self.sample_batch_size,
            'R_0': self.R_0,
        }

        return HypergridEnv(new_conf)

    def get_depth_first_search_trajectories(self):
        start_state = tuple((0 for _ in range(self.num_dims)))
        stop_action = self.num_dims

        init_trajectory = Trajectory(
            [start_state],
            [stop_action],
            self._state_to_reward[start_state],
            stop_action,
            [1]
        )

        return [init_trajectory] + self._get_child_trajectories_depth_first_search(
            init_trajectory,
            start_state,
            seen_so_far={start_state}
        )

    def _depth_first_search_recurse(
        self,
        curr_traj: Trajectory,
        curr_state: Tuple[int],
        action: int,
        seen_so_far: set
    ):
        if curr_state in seen_so_far:
            return []

        seen_so_far.add(curr_state)

        new_traj = curr_traj.copy()
        new_traj.append(
            curr_state,
            action,
            reward=self._state_to_reward[curr_state],
            stopped=True,
            stop_action=self.num_dims
        )

        return [new_traj] + self._get_child_trajectories_depth_first_search(
            new_traj,
            curr_state,
            seen_so_far
        )

    def _get_child_trajectories_depth_first_search(self, curr_traj, curr_state, seen_so_far):
        result = []
        for i in np.random.permutation(self.num_dims):
            if curr_state[i] == self.side_length - 1:
                continue

            new_state = tuple((
                curr_state[j] if j != i else curr_state[j] + 1
                for j in range(self.num_dims)
            ))

            result.extend(
                self._depth_first_search_recurse(
                    curr_traj,
                    new_state,
                    i,
                    seen_so_far
                )
            )

        return result


    def get_all_trajectories(self):
        '''
        This is only implemented for environments with 2 dimensions.  Note that
        if the side length is too long this method will just hang.  The method's
        time complexity is exponential in the side length, so limit using this to
        environments with a small side length.
        '''
        assert self.num_dims == 2
        if self.side_length >= 10:
            print(
                'Be aware that your invocation of this method may hang.  The ' +
                'method\'s time complexity is exponential in the side length ' +
                'and should be limited to use on environments with small side ' +
                'lengths.  Given that your side length is %d, ' % self.side_length +
                'expect your invocation to take a long time'
            )

        if not self._paths_to_state:
            self._build_paths_to_state()

        all_trajs = []
        for paths_to_state in self._paths_to_state.values():
            all_trajs.extend(paths_to_state)

        return all_trajs

    def _build_paths_to_state(self):
        start_state, stop_action = (0, 0), self.num_dims
        init_trajectory = Trajectory(
            [start_state],
            [stop_action],
            self._state_to_reward[start_state],
            stop_action,
            [1]
        )

        self._paths_to_state[start_state] = [init_trajectory]
        self._stop_action_paths_to_state[start_state] = [init_trajectory]

        children_queue = queue.Queue()
        children_queue.put(start_state)
        while not children_queue.empty():
            curr_state = children_queue.get()
            if curr_state in self._paths_to_state and curr_state != start_state:
                continue

            parents = [list(curr_state) for _ in range(2)]
            parents[0][0] -= 1
            parents[1][1] -= 1

            all_term_paths_to_curr_state, stop_action_term_paths_to_curr_state = \
                self._extend_trajectory(parents, curr_state)

            if all_term_paths_to_curr_state:
                self._paths_to_state[curr_state] = all_term_paths_to_curr_state
                self._stop_action_paths_to_state[curr_state] = \
                    stop_action_term_paths_to_curr_state

            self._add_children_to_queue(children_queue, curr_state)

        return

    def _extend_trajectory(self, parents, curr_state):
        all_terminating_paths_to_curr_state = []
        stop_action_terminating_paths_to_curr_state = []

        for i, parent in enumerate(parents):
            if parent[0] == -1 or parent[1] == -1:
                continue

            stop_actions = [self.num_dims]
            #stop_actions = [
            #    i for i in range(self.num_dims)
            #    if curr_state[i] == self.side_length - 1
            #] + [self.num_dims]

            for stop_action in stop_actions:
                new_paths = [
                    traj.copy()
                    for traj in self._stop_action_paths_to_state[tuple(parent)]
                ]

                reward = 0.0
                if stop_action == self.num_dims:
                    reward=self._state_to_reward[curr_state]

                func = lambda x: x.append(
                    curr_state,
                    i,
                    reward=reward,
                    stopped=True,
                    stop_action=stop_action
                )

                list(map(func, new_paths))
                all_terminating_paths_to_curr_state.extend(new_paths)
                if stop_action == self.num_dims:
                    stop_action_terminating_paths_to_curr_state.extend(new_paths)

        return (
            all_terminating_paths_to_curr_state,
            stop_action_terminating_paths_to_curr_state
        )

    def _add_children_to_queue(self, children_queue, curr_state):
        for i in range(2):
            if curr_state[i] == self.side_length - 1:
                continue

            children_queue.put((curr_state[0] + (1 - i), curr_state[1] + i))

        return

    def _build_trajectory(self, down_move_order):
        states = [[0, 0]]
        down_move_order = sorted(down_move_order, reverse=True)

        x_pos, y_pos = 0, 0
        for i in range(2 * len(down_move_order)):
            if down_move_order and i == down_move_order[-1]:
                x_pos += 1
                down_move_order.pop()

            else:
                y_pos += 1

            states.append([x_pos, y_pos])

        return torch.tensor(states)
