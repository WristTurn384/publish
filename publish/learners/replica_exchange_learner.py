from gfn_exploration.learners import BasicLearner
from gfn_exploration.learners.basic_learner import merge_traj_infos
from gfn_exploration.replica_exchange import GFNUpdater
from gfn_exploration.replica_exchange.utils import traj_container_to_tensor_dict
from typing import Dict
from torchtyping import TensorType
import numpy as np
import torch
import ray

def get_temperature_schedule(
    config: Dict[str, object]
) -> TensorType['num_temps', float]:
    temperature_config = config['temperature_schedule_config']

    schedule = None
    schedule_type = temperature_config['schedule_type']
    if schedule_type.lower() == 'linear':
        schedule = torch.linspace(
            temperature_config['min_val'],
            temperature_config['max_val'],
            temperature_config['num_temperatures']
        )

    elif schedule_type.lower() == 'log':
        schedule = torch.logspace(
            np.log10(temperature_config['min_val']),
            np.log10(temperature_config['max_val']),
            temperature_config['num_temperatures'],
            base=temperature_config.get('log_base', 10.0)
        )

    else:
        raise ValueError(f'Unsupported temperature schedule type {schedule_type}')

    return schedule


class ReplicaExchangeLearner(BasicLearner):
    def setup(self, config: Dict[str, object]) -> None:
        super().setup(config)

        temperature_schedule = get_temperature_schedule(config)
        self.temperature_to_updaters = {
            temperature: GFNUpdater.remote(self.env, temperature, config)
            for temperature in temperature_schedule
        }

        self.updater_to_temperature = {
            updater: temperature
            for temperature, updater in self.temperature_to_updaters.items()
        }

        self.gfn_updaters = list(self.temperature_to_updaters.values())

        self.workers = ray.get([
            updater.get_rollout_worker.remote()
            for updater in self.gfn_updaters
        ])

        state_assigner_config = config['state_assigner_config']
        self.state_assigner = state_assigner_config['type'](
            state_assigner_config,
            self.temperature_to_updaters
        )

    def step(self) -> Dict[str, object]:
        all_rollouts = ray.get([worker.sample.remote() for worker in self.workers])

        state_assignments = self.state_assigner.assign_loss_states(all_rollouts)

        update_tasks = []
        for updater, state_assignment in state_assignments.items():
            update_tasks.append(updater.update.remote(state_assignment))

        gfn_losses = ray.get(update_tasks)

        self.step_num += 1
        return self._process_metrics(
            rollout_infos=all_rollouts,
            target_update_losses=[
                (self.updater_to_temperature[updater], loss.item())
                for updater, loss in zip(state_assignments.keys(), gfn_losses)
            ]
        )

    def _get_merged_metric_update_infos(self, **kwargs) -> Dict[str, object]:
        return {
            'losses': {
                temperature: loss
                for temperature, loss in kwargs['target_update_losses']
            },
            'behavior_rollout': merge_traj_infos(kwargs['rollout_infos'])
        }
