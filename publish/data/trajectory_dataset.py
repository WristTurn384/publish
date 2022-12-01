from torch.utils.data import Dataset
from publish.envs import HypergridEnv
from publish.utils import Trajectory
from publish.utils import observation_to_multi_one_hot, get_int_encoded_terminal_states
from pathlib import Path
import torch
import ray

TRY_LOAD = True
REL_SAVE_DIR_PATH = 'saved_datasets'

def get_dataset_path(side_length, num_dims, R_0, R_1, R_2):
    save_dir = Path(__file__).parent.absolute() / REL_SAVE_DIR_PATH
    save_dir.mkdir(exist_ok=True)

    rel_fname = 'sidelen_%d_numdims_%d' % (side_length, num_dims)
    suffix = 'r0_{:.4f}_r1_{:.4f}_r2_{:.4f}.pt'.format(R_0, R_1, R_2)

    return save_dir / (rel_fname + suffix)

class HypergridTrajectoryDataset(Dataset):
    def __init__(
        self,
        side_length,
        num_dims,
        R_0,
        R_1,
        R_2,
        states,
        actions,
        back_actions,
        rewards,
        dones,
        one_hot_states
    ):
        self.side_length = side_length
        self.num_dims = num_dims
        self.R_0 = R_0
        self.R_1 = R_1
        self.R_2 = R_2

        self.states = states
        self.actions = actions
        self.back_actions = back_actions
        self.rewards = rewards
        self.dones = dones
        self.one_hot_states = one_hot_states

        self.log_density = self._compute_log_density()
        self.save_if_needed()

    def save_if_needed(self):
        dataset_path = get_dataset_path(
            self.side_length,
            self.num_dims,
            self.R_0,
            self.R_1,
            self.R_2
        )
        if dataset_path.exists():
            return

        save_dict = {
            'states': self.states,
            'actions': self.actions,
            'back_actions': self.back_actions,
            'rewards': self.rewards,
            'dones': self.dones,
            'one_hot_states': self.one_hot_states,
        }

        torch.save(save_dict, dataset_path)

    def to(self, device):
        self.states = self.states.to(device=device)
        self.actions = self.actions.to(device=device)
        self.back_actions = self.back_actions.to(device=device)
        self.rewards = self.rewards.to(device=device)
        self.dones = self.dones.to(device=device)
        self.one_hot_states = self.one_hot_states.to(device=device)
        self.log_density = self.log_density.to(device=device)

    def truncate(self, num_to_keep):
        self.states = self.states[:num_to_keep]
        self.actions = self.actions[:num_to_keep]
        self.back_actions = self.back_actions[:num_to_keep]
        self.rewards = self.rewards[:num_to_keep]
        self.dones = self.dones[:num_to_keep]
        self.one_hot_states = self.one_hot_states[:num_to_keep]

    def get_reward_distribution(self):
        dones_idx = (self.dones == 1).nonzero()
        rewards = self.rewards[dones_idx[:, 0], dones_idx[:, 1]]

        return rewards / rewards.sum()

    @staticmethod
    def create(
        side_length,
        num_dims,
        R_0,
        R_1=0.5,
        R_2=2.0,
        use_depth_first_search=False
    ):
        if not use_depth_first_search:
            dataset = HypergridTrajectoryDataset.try_load_dataset(
                side_length,
                num_dims,
                R_0,
                R_1,
                R_2
            )

            if dataset:
                return dataset

        env = HypergridEnv({
            'side_length': side_length,
            'num_dims': num_dims,
            'sample_batch_size': 1,
            'R_0': R_0,
            'R_1': R_1,
            'R_2': R_2
        })

        desired_len = (num_dims * side_length) - 1

        if use_depth_first_search:
            all_trajectories = env.get_depth_first_search_trajectories()
        else:
            all_trajectories = env.get_all_trajectories()

        list(map(lambda x: x.to_torch(), all_trajectories))
        list(map(lambda x: x.pad_trajectory(desired_len), all_trajectories))

        combined_traj_dict = Trajectory.combine_trajectories(all_trajectories)
        states = combined_traj_dict['states']
        actions = combined_traj_dict['actions']
        rewards = combined_traj_dict['rewards']
        dones = combined_traj_dict['dones']

        pad_value = all_trajectories[0].pad_value if all_trajectories else -1

        one_hot_states = observation_to_multi_one_hot(
            states,
            side_length,
            num_dims,
            pad_value
        )

        back_actions = env.get_backward_actions(one_hot_states, actions)
        return HypergridTrajectoryDataset(
            side_length,
            num_dims,
            R_0,
            R_1,
            R_2,
            states,
            actions,
            back_actions,
            rewards,
            dones,
            one_hot_states
        )

    def _compute_log_density(self):
        rewards = self.rewards.flatten()
        return rewards.log() - rewards.sum().log()

    def prune_to_max_num_trajs_per_terminal_state(
        self,
        max_num_trajs_per_terminal_state: int
    ) -> None:
        int_terminal_states = get_int_encoded_terminal_states(
            self.one_hot_states,
            self.dones,
            self.side_length,
            self.num_dims
        )

        idx_arrs = []
        for val in int_terminal_states.unique():
            val_idxs = (int_terminal_states == val).nonzero().squeeze(1)
            val_idxs = val_idxs[torch.randperm(len(val_idxs))]

            idx_arrs.append(val_idxs[:max_num_trajs_per_terminal_state])

        filter_idxs = torch.cat(idx_arrs)

        self.states = self.states[filter_idxs]
        self.actions = self.actions[filter_idxs]
        self.back_actions = self.back_actions[filter_idxs]
        self.rewards = self.rewards[filter_idxs]
        self.dones = self.dones[filter_idxs]
        self.one_hot_states = self.one_hot_states[filter_idxs]

    @staticmethod
    def try_load_dataset(side_length, num_dims, R_0, R_1, R_2):
        new_path = get_dataset_path(side_length, num_dims, R_0, R_1, R_2)
        if not (TRY_LOAD and new_path.exists()):
            return None

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tensors = torch.load(new_path, map_location=device)
        return HypergridTrajectoryDataset(
            side_length,
            num_dims,
            R_0,
            R_1,
            R_2,
            tensors['states'],
            tensors['actions'],
            tensors['back_actions'],
            tensors['rewards'],
            tensors['dones'],
            tensors['one_hot_states'],
        )

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            'states': self.one_hot_states[idx],
            'actions': self.actions[idx],
            'back_actions': self.back_actions[idx],
            'rewards': self.rewards[idx],
            'dones': self.dones[idx],
        }

if __name__ == '__main__':
    dataset = HypergridTrajectoryDataset.create(8, 2, 1e-3)
    print(dataset[4])
    print(len(dataset))
