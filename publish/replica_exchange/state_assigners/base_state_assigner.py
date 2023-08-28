from gfn_exploration.utils.type_defs import TrajectoryContainer
from abc import ABC, abstractmethod
from typing import Dict, List

class BaseStateAssigner(ABC):
    def __init__(
        self,
        config: Dict[str, object],
        temperature_to_updater: Dict[float, 'GFNUpdater']
    ):
        self.temperature_to_updater = temperature_to_updater

    @abstractmethod
    def assign_loss_states(
        self,
        trajectories: List[TrajectoryContainer]
    ) -> Dict['GFNUpdater', List[TrajectoryContainer]]:
        '''
        Given a list of many trajectories, this method selects the trajectories
        to actually update each different temperature GFlowNet with. This can
        take many forms, from an entropic OT assignment to a prioritized replay
        based on reward, or whatever else.  The method returns a dictionary mapping
        each GFNUpdater to a list of the trajectories that will be used to update
        the GFN this iteration.

        Arguments:
            trajectories: A list of trajectory containers that the method can
                          use to create its trajectory assignment.
        Returns:
            A dictionary mapping each GFNUpdater to a list of the trajectories
            that will be used to update the GFN this iteration
        '''
        pass
