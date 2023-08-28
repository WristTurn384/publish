from gfn_exploration.replica_exchange.state_assigners import BaseStateAssigner
from gfn_exploration.replica_exchange.utils import traj_container_to_tensor_dict, tensor_dict_to_traj_container
from gfn_exploration.utils import get_device
from gfn_exploration.utils.type_defs import TrajectoryContainer
from abc import abstractmethod
from torchrl.data import TensorDictPrioritizedReplayBuffer
from torchrl.data.replay_buffers import LazyTensorStorage
from torchtyping import TensorType
from typing import Dict, List

class PrioritizedReplayStateAssigner(BaseStateAssigner):
    def __init__(
        self,
        config: Dict[str, object],
        temperature_to_updater: Dict[float, 'GFNUpdater']
    ):
        super().__init__(config, temperature_to_updater)

        replay_config = config['buffer_config']
        self.updater_to_replay = {
            updater: TensorDictPrioritizedReplayBuffer(
                alpha=replay_config.get('alpha', 0.7),
                beta=replay_config.get('beta', 1.1),
                storage=LazyTensorStorage(
                    max_size=replay_config['max_size'],
                    device=get_device()
                ),
                batch_size=replay_config.get('batch_size', None),
                priority_key=self.priority_key
            )
            for temperature, updater in temperature_to_updater.items()
        }

        self.updater_to_temperature = {
            updater: temperature
            for temperature, updater in self.temperature_to_updater.items()
        }

    @property
    @abstractmethod
    def priority_key(self) -> str:
        raise NotImplementedError

    def assign_loss_states(
        self,
        trajectories: List[TrajectoryContainer]
    ) -> Dict['GFNUpdater', List[TrajectoryContainer]]:
        '''
        This method computes priorities then adds them to a prioritized replay
        and gets the loss states by sampling from the replay buffer.

        Arguments:
            trajectories: A list of trajectory containers that the method can
                          use to create its trajectory assignment.
        Returns:
            A dictionary mapping each GFNUpdater to a list of the trajectories
            that will be used to update the GFN this iteration
        '''
        result = {}
        for updater, replay_buffer in self.updater_to_replay.items():
            traj_tensor_dict = traj_container_to_tensor_dict(trajectories)
            traj_tensor_dict[self.priority_key] = self._get_priorities(
                updater,
                traj_tensor_dict
            )

            replay_buffer.extend(traj_tensor_dict)

            result[updater] = tensor_dict_to_traj_container(replay_buffer.sample())

        return result

    @abstractmethod
    def _get_priorities(
        self,
        updater: 'GFNUpdater',
        trajectories: List[TrajectoryContainer]
    ) -> TensorType[float]:
        '''
        This method takes a GFNUpdater and a list of trajectories
        and returns the priority of each trajectory.  The priorities
        are returned in a 1D tensor which has the same length as the
        list of trajectories.
        '''
        pass
