from publish.agents import BaseAgent
from publish.data import HypergridTrajectoryDataset
from publish.samplers import DirectTrajectorySampler
from publish.metrics import BaseMetric
from publish.utils import get_grid_one_hot_int_encoder_mask, get_int_encoded_terminal_states, get_p_given_terminal_state
from publish.utils.constants import FORWARD, BACKWARD
from publish.utils.interfaces import BaseSampler
from torchtyping import TensorType
from torch.distributions.categorical import Categorical
from typing import Dict
import torch

class BaseExactTrajectoryDistributionMetric(BaseMetric):
    def __init__(self, config: Dict[str, object]):
        super().__init__(config)

        self.env = config['env']

        traj_dataset = HypergridTrajectoryDataset.create(
            self.env.side_length,
            self.env.num_dims,
            self.env.R_0
        )

        self.states, self.actions, self.back_actions, self.dones, self.rewards = (
            traj_dataset.one_hot_states,
            traj_dataset.actions,
            traj_dataset.back_actions,
            traj_dataset.dones,
            traj_dataset.rewards,
        )

        encoder_mask = get_grid_one_hot_int_encoder_mask(
            self.env.num_dims,
            self.env.side_length
        )

        self.int_terminal_states = get_int_encoded_terminal_states(
            self.states,
            traj_dataset.dones,
            self.env.side_length,
            self.env.num_dims
        )

        self.num_terminal_states = len(self.int_terminal_states.unique())

        self.distrib_arg_names = (
            'sampler_distrib',
            'behavior_agent_fwd_distrib',
            'behavior_agent_back_distrib',
            'target_agent_fwd_distrib',
            'target_agent_back_distrib',
            'target_agent_back_given_term_distrib',
        )

        list(map(lambda x: setattr(self, x, None), self.distrib_arg_names))

    def update(self, update_infos: Dict[str, object]) -> None:
        super().update(update_infos)

        for agent_type in self.agent_names_to_compute_for:
            agent_key = '%s_agent' % agent_type
            if not agent_key in update_infos:
                continue

            curr_state_sampling_probs = torch.zeros(
                self.num_terminal_states,
                device=self.states.device
            )

            for mode in self.modes_to_compute_for:
                probs = self._compute_trajectory_distribution(
                    update_infos[agent_key],
                    mode
                )

                if mode == FORWARD:
                    curr_state_sampling_probs.index_add_(
                        dim=0,
                        index=self.int_terminal_states,
                        source=probs
                    )
                else:
                    self.target_agent_back_given_term_distrib = \
                        get_p_given_terminal_state(
                            probs,
                            self.num_terminal_states,
                            self.int_terminal_states
                        )

                    probs = probs * curr_state_sampling_probs[
                        self.int_terminal_states
                    ]

                self_key = '%s_%s_distrib' % (agent_key, mode)
                setattr(self, self_key, Categorical(probs=probs))

        if 'sampler' in update_infos:
            probs = self._compute_trajectory_distribution(
                update_infos['sampler']
            )

            self.sampler_distrib = Categorical(probs=probs)

    @property
    def agent_names_to_compute_for(self) -> str:
        return ['target']

    @property
    def modes_to_compute_for(self) -> str:
        return [FORWARD, BACKWARD]

    def _compute_trajectory_distribution(
        self,
        generator: BaseSampler,
        agent_fwd_bwd_mode: str = None
    ) -> TensorType['num_possible_trajectories']:
        if isinstance(generator, DirectTrajectorySampler):
            return self._compute_trajectory_distribution_sampler(generator)
        elif isinstance(generator, BaseAgent):
            return self._compute_trajectory_distribution_agent(
                generator,
                agent_fwd_bwd_mode
            )
        else:
            raise TypeError(
                'Generator must be a subclas of one of types '    +
                'DirectTrajectorySampler or BaseAgent but was of' +
                'type %s instead' % str(type(generator))
            )

    def _compute_trajectory_distribution_sampler(
        self,
        generator: DirectTrajectorySampler
    ) -> TensorType['num_possible_trajectories']:
        priorities = generator.priorities
        return priorities / priorities.sum()

    def _compute_trajectory_distribution_agent(
        self,
        generator: BaseAgent,
        compute_fwd_bwd_mode: str
    ) -> TensorType['num_possible_trajectories']:
        assert compute_fwd_bwd_mode in [FORWARD, BACKWARD]

        log_p_traj = None
        with torch.no_grad():
            if compute_fwd_bwd_mode == FORWARD:
                log_p_traj = generator.get_log_pf(self.states, self.actions)
            else:
                log_p_traj = generator.get_log_pb(self.states, self.back_actions)

        return log_p_traj.sum(dim=1).exp()
