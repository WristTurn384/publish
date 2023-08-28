from gfn_exploration.envs import AutoregressiveSequenceGenerationEnvironment
from gfn_exploration.utils import get_device
from gfn_exploration.utils.constants import FORWARD, BACKWARD
from torch.utils.data import TensorDataset
from torchtyping import TensorType
from typing import Dict, Tuple
import torch

class MutationBasedSequenceGenerationEnvironment(AutoregressiveSequenceGenerationEnvironment):
    def __init__(self, config: Dict[str, int]):
        self.device = get_device()

        self.sequence_length = config.get('sequence_length', config['horizon'])
        self.starting_states = config.get('starting_states', None)
        if self.starting_states is not None:
            self.starting_states = self.starting_states.to(device=self.device)

        super().__init__(config)

    def _get_test_set(self) -> TensorDataset:
        return None

    @property
    def obs_dim(self) -> int:
        return self.horizon + 1

    @property
    def forward_action_dim(self) -> int:
        return (len(self.vocab) * self.sequence_length) + int(self.allow_stop_action)

    @property
    def backward_action_dim(self) -> int:
        return self.forward_action_dim

    @property
    def padding_token(self) -> int:
        return self.action_dim

    @property
    def eos_token(self) -> int:
        return self.padding_token + 1

    @property
    def has_done_action(self) -> bool:
        return self.allow_stop_action

    @property
    def done_action(self) -> int:
        return self.action_dim - 1 if self.allow_stop_action else -1

    @property
    def is_tree_state_space(self) -> bool:
        return False

    def clone(
        self,
        new_batch_size: int = None
    ) -> 'MutationBasedSequenceGenerationEnvironment':
        return MutationBasedSequenceGenerationEnvironment(
            self._get_clone_config(new_batch_size)
        )

    def _get_clone_config(self, new_batch_size: int = None) -> Dict[str, object]:
        return {
            'sequence_length': self.sequence_length,
            **super()._get_clone_config(new_batch_size)
        }

    def is_origin_state(
        self,
        states: TensorType['batch_size', 'obs_dim']
    ) -> TensorType['batch_size', bool]:
        return states[:, -1] == 0

    def get_backward_actions(
        self,
        states: TensorType['batch_size', 'horizon', 'obs_dim'],
        actions: TensorType['batch_size', 'horizon']
    ) -> TensorType['batch_size', 'horizon']:
        action_seq_idxs = actions // len(self.vocab)
        back_actions_char_idxs = states[
            torch.arange(len(states)).repeat_interleave(self.horizon),
            torch.arange(self.horizon).repeat(len(states)),
            action_seq_idxs.flatten().long()
        ].reshape(actions.shape)

        return (action_seq_idxs * len(self.vocab)) + back_actions_char_idxs

    def reset(self, batch_size: int = None) -> TensorType['batch_size', 'horizon', int]:
        batch_size = batch_size or self.sample_batch_size
        if self.starting_states is None:
            self.states = torch.randint(
                len(self.vocab),
                (batch_size, self.sequence_length),
                dtype=torch.long,
                device=self.device
            )
        else:
            self.states = self.starting_states[
                torch.multinomial(
                    torch.ones(len(self.starting_states)),
                    batch_size,
                    replacement=True
                )
            ]

        self.dones = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=self.device
        )

        self.log_rewards = torch.zeros(
            batch_size,
            dtype=torch.float,
            device=self.device
        )

        self.step_idx = 0

        return self._get_step_encoded_states()

    def _get_step_encoded_states(
        self,
        states: TensorType['batch_size', 'horizon', int] = None,
        step_idx: int = None
    ) -> TensorType['batch_size', 'horizon_plus_one', int]:
        states = states if states is not None else self.states
        step_tnsr = torch.full(
            (self.sample_batch_size, 1),
            step_idx or self.step_idx,
            device=states.device
        )

        return torch.cat([states, step_tnsr], dim=1)

    def step_backward(
        self,
        actions: TensorType["batch_size", int],
        states: TensorType["batch_size", "horizon_plus_one"] = None
    ) -> TensorType["batch_size", "horizon_plus_one"]:
        use_self_tnsrs = states is None

        states = self.states if use_self_tnsrs else states

        last_step_idx = states[0, -1]
        assert last_step_idx > 0
        assert (states[:, -1] == last_step_idx).all()

        action_seq_idxs = actions // len(self.vocab)
        action_char_idxs = actions % len(self.vocab)

        states[torch.arange(len(states)), action_seq_idxs] = action_char_idxs
        states[:, -1] = last_step_idx - 1

        return states

    def get_action_ids(
        self,
        parent_states: TensorType["batch_size", "horizon"],
        child_states: TensorType["batch_size", "horizon"],
        mode: str = FORWARD,
        other_dir_actions: TensorType["batch_size"] = None
    ) -> int:
        parent_states = self._unstep_encode_states(parent_states)
        child_states = self._unstep_encode_states(child_states)

        diff_idxs = (parent_states != child_states).nonzero()

        vocab_idx = None
        if mode == FORWARD:
            vocab_idx = child_states[diff_idxs[:, 0], diff_idxs[:, 1]]
        elif mode == BACKWARD:
            vocab_idx = parent_states[diff_idxs[:, 0], diff_idxs[:, 1]]
        else:
            raise ValueError(
                'Mode must be in (%s, %s) but was %s' % (FORWARD, BACKWARD, mode)
            )

        # str_diff_locs is the indices in the strings which were different
        str_diff_locs = diff_idxs[:, 1]
        pre_action_ids = (str_diff_locs * len(self.vocab)) + vocab_idx

        if other_dir_actions is not None:
            action_ids = other_dir_actions.clone()
            action_ids[diff_idxs[:, 0]] = pre_action_ids
        else:
            action_ids = pre_action_ids

        return action_ids

    def get_state_ids(
        self,
        states: TensorType["batch_size", "horizon"]
    ) -> int:
        '''
        This method is used in computing the exact marginal

        P_F^T(x) = \sum_{tau : T(tau) = x} P_F(tau)

        with T(tau) a function which returns the terminal state of trajectory
        tau.  Since there's no hope of tractably doing this in this environment,
        just return None.
        '''
        return None

    def get_log_rewards(
        self,
        states: TensorType["batch_size", "horizon"]
    ) -> TensorType["batch_size", float]:
        return self._reward_fxn.get_log_rewards(self._unstep_encode_states(states))

    def _unstep_encode_states(
        self,
        states: TensorType["batch_size", "horizon"]
    ):
        last_dim_diff = states.shape[-1] - self.sequence_length

        # Clause is true if states are step encoded
        if last_dim_diff != 0:
            states = states[..., :-last_dim_diff]

        return states

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

            float: The log_reward for this step
            bool: A boolean representing whether this step concludes the episode
            dict: A dictionary of extra information for the episode.  In our
                  case, this is always empty.
        """
        use_self_tnsrs = states is None

        states = self.states if use_self_tnsrs else states
        dones = self.dones if use_self_tnsrs else torch.zeros_like(actions)

        states = self._unstep_encode_states(states)

        if not dones.all():
            old_dones = dones.clone()
            new_dones_pre, stop_actions = self._get_new_dones(dones, actions)

            should_not_change_idx = old_dones | stop_actions
            not_done_nnz = torch.nonzero(~should_not_change_idx).flatten()

            valid_actions = actions[not_done_nnz]
            seq_idxs = valid_actions // len(self.vocab)
            char_idxs = valid_actions % len(self.vocab)
            assert (char_idxs < len(self.vocab)).all()

            states[not_done_nnz, seq_idxs] = char_idxs

            if use_self_tnsrs:
                self.dones = new_dones_pre

            dones = new_dones_pre

            self.step_idx += 1

        return (
            self._get_step_encoded_states(states),
            dones.clone()
        )
