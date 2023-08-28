from gfn_exploration.workers import RolloutWorkerRemote
from gfn_exploration.utils import get_first_eq_idx
from gfn_exploration.utils.type_defs import TrajectoryContainer
from collections import OrderedDict
from torchtyping import TensorType
from tensordict import TensorDict
from typing import Dict, List
import torch
import ray

def traj_container_to_tensor_dict(
    trajectory_containers: List[TrajectoryContainer]
) -> TensorDict:
    tensor_dicts = [
        TensorDict(
            {
                'states': container[0],
                'frwd_actions': container[1],
                'back_actions': container[2],
                'dones': container[3],
                'log_rewards': container[4]
            },
            batch_size=[len(container[0])]
        )
        for container in trajectory_containers
    ]

    return torch.cat(tensor_dicts)

def tensor_dict_to_traj_container(tensor_dict: TensorDict) -> TrajectoryContainer:
    return (
        tensor_dict['states'],
        tensor_dict['frwd_actions'],
        tensor_dict['back_actions'],
        tensor_dict['dones'],
        tensor_dict['log_rewards'],
    )

def _get_terminal_states(
    samples: TrajectoryContainer,
    bool_indexer: TensorType['batch_size', bool] = None
) -> TensorType['batch_size', 'state_dim']:
    states, dones = samples[0], samples[3]
    if bool_indexer is not None:
        states, dones = states[bool_indexer], dones[bool_indexer]

    first_dones_idx = get_first_eq_idx(dones)

    return states[torch.arange(len(states)), first_dones_idx]

def _get_mh_back_forth_terminal_states(
    temperature: float,
    starting_states: TensorType['batch_size', 'state_dim'],
    worker: RolloutWorkerRemote,
    num_steps: int
):
    have_accepted = torch.zeros(
        len(starting_states),
        dtype=torch.bool,
        device=starting_states.device
    )

    new_states = torch.empty_like(starting_states)

    gfn = ray.get(worker.get_agent())
    while not have_accepted.all():
        backward_trajs, frwd_trajs, _ = ray.get(
            worker.k_step_state_perturbation.remote(
                starting_states[~have_accepted],
                num_steps
            )
        )

        # Temper the energies
        backward_trajs[-1] = (1 / temperature) * backward_trajs[-1]
        frwd_trajs[-1] = (1 / temperature) * frwd_trajs[-1]

        now_accepted = have_accepted.clone()
        now_accepted[~have_accepted] = gfn_mh_cond_should_accept(
            gfn,
            backward_trajs,
            frwd_trajs
        )

        new_accepted = now_accepted ^ have_accepted
        new_states[new_accepted] = _get_terminal_states(
            frwd_trajs,
            new_accepted
        )

        have_accepted = now_accepted

    return new_states

class TemperatureAnnealedSampler:
    def __init__(
        self,
        temperature_to_workers: Dict[float, RolloutWorkerRemote]
    ):
        self.temperature_to_workers = OrderedDict()

        # TODO: Figure out a schedule or something here
        self.constant_num_backwards_steps = 5

        for temp in sorted(temperature_to_workers.keys()):
            self.temperature_to_workers[temp] = temperature_to_workers[temp]

    def sample(self, batch_size: int, num_backwards_steps: int = None):
        temperatures = list(self.temperature_to_workers.keys())

        highest_temp_samples = ray.get(
            self.temperature_to_workers[temperatures[0]].sample.remote(batch_size)
        )

        prev_temp_terminal_states = _get_terminal_states(highest_temp_samples)
        for temperature in temperatures[1:]:
            prev_temp_terminal_states = _get_mh_back_forth_terminal_states(
                temperature,
                prev_temp_terminal_states,
                self.temperature_to_workers[temperature],
                self._get_num_perturbation_steps(temperature, num_backwards_steps)
            )

        return prev_temp_terminal_states

    def _get_num_perturbation_steps(
        self,
        temperature: float,
        num_backwards_steps: int = None
    ) -> int:
        if num_backwards_steps is not None:
            return num_backwards_steps
        else:
            return self.constant_num_backwards_steps
