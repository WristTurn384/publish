import torch

class Trajectory:
    def __init__(
        self,
        states,
        actions,
        reward,
        stop_action,
        dones_idx=[],
        pad_value=-1
    ):
        self.states = states
        self.actions = actions
        self.reward = reward
        self.stop_action = stop_action
        self.dones_idx = dones_idx
        self.pad_value = pad_value

    @staticmethod
    def combine_trajectories(trajectories):
        states = torch.stack(tuple(map(lambda x: x.states, trajectories)))
        actions = torch.stack(tuple(map(lambda x: x.actions, trajectories)))
        rewards = torch.stack(tuple(map(lambda x: x.reward, trajectories)))
        dones = torch.stack(tuple(map(lambda x: x.dones_idx, trajectories)))

        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'dones': dones
        }

    def copy(self, pop_last_action=True):
        cp_states = self.states.copy()
        cp_actions = self.actions.copy()
        cp_dones = self.dones_idx.copy()
        if len(cp_actions) > 0 and pop_last_action:
            cp_actions.pop()
            cp_dones.pop()
            cp_dones.append(0)

        return Trajectory(
            cp_states,
            cp_actions,
            self.reward,
            self.stop_action,
            cp_dones
        )

    def append(
        self,
        new_state,
        new_action,
        reward=None,
        stopped=True,
        stop_action=None
    ):
        self.states.append(new_state)
        self.actions.append(new_action)

        if reward is not None:
            self.reward = reward

        if stopped:
            sa = stop_action if stop_action is not None else self.stop_action
            self.actions.append(sa)
            self.dones_idx.append(1)

    def to_torch(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.states = torch.tensor(self.states, device=device)
        self.actions = torch.tensor(self.actions, device=device)
        self.reward = torch.tensor(self.reward, device=device)
        self.dones_idx = torch.tensor(self.dones_idx, device=device)

    def pad_trajectory(self, desired_length):
        state_len = len(self.states)
        assert desired_length >= state_len

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_vals_to_pad = desired_length - state_len

        state_pad = torch.full(
            (num_vals_to_pad, self.states.shape[-1]),
            self.pad_value,
            device=device
        )

        action_pad = torch.full((num_vals_to_pad,), self.pad_value, device=device)
        done_pad = torch.full((num_vals_to_pad,), self.pad_value, device=device)
        reward_pad = torch.full((num_vals_to_pad,), self.pad_value, device=device, dtype=torch.float)

        self.states = torch.cat((self.states, state_pad), dim=0)
        self.actions = torch.cat((self.actions, action_pad), dim=0)
        self.dones_idx = torch.cat((self.dones_idx, done_pad), dim=0)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (self.states[idx], self.actions[idx])
