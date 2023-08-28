from gfn_exploration.replica_exchange.state_assigners import PrioritizedReplayStateAssigner
from gfn_exploration.utils.type_defs import TrajectoryContainer
from torchtyping import TensorType
from typing import Dict, List
import torch

class RewardPrioritizedReplayStateAssigner(PrioritizedReplayStateAssigner):
    def __init__(
        self,
        config: Dict[str, object],
        temperature_to_updater: Dict[float, 'GFNUpdater']
    ):
        super().__init__(config, temperature_to_updater)

        self.temper_reward_for_priorities = config.get(
            'temper_reward_for_priorities',
            False
        )

    @property
    def priority_key(self) -> str:
        return 'reward'

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
        log_rewards = torch.tensor([
            trajectory['log_rewards']
            for trajectory in trajectories
        ])

        if self.temper_reward_for_priorities:
            log_rewards = 10 * self.updater_to_temperature[updater] * log_rewards

        return log_rewards
