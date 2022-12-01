from publish.workers import RolloutWorker, RolloutWorkerRemote
from publish.metrics import BaseMetric
from publish.learners.utils import get_seed
from ray.tune import Trainable
from ray.tune.integration.wandb import WandbTrainableMixin
from hive.replays import BaseReplayBuffer
from typing import Dict, List, Tuple, Union
from torchtyping import TensorType
import numpy as np
import torch
import random
import ray

from torch.profiler import profile, ProfilerActivity, record_function

def get_target_agent_buff_conf_and_should_reuse(
    config: Dict[str, object]
) -> Tuple[Dict[str, object], bool]:
    should_reuse_target_buff = not (
        config.get('reuse_target_agent_buffer_for_behavior_agent', False) or
        'behavior_agent_buffer_config' in config
    )

    buff_conf_in_conf = 'buffer_config' in config
    targ_agent_buff_conf_in_conf = 'target_agent_buffer_config' in config

    # Validate that XOR is true
    assert buff_conf_in_conf ^ targ_agent_buff_conf_in_conf

    target_conf_key = \
        'buffer_config' if buff_conf_in_conf else 'target_agent_buffer_config'

    return config[target_conf_key], should_reuse_target_buff


def build_buffer(
    buffer_config_pre: Dict[str, object],
    obs_dim: int,
    horizon: int
) -> BaseReplayBuffer:
    buffer_type = buffer_config_pre.pop('type')
    buffer_config = {
        'obs_dim': obs_dim,
        'horizon': horizon,
        **buffer_config_pre
    }

    return buffer_type(buffer_config)


def get_buffers(
    config: Dict[str, object],
    obs_dim: int,
    horizon: int
) -> Tuple[BaseReplayBuffer, BaseReplayBuffer, bool]:
    target_agent_buff_config, reuse_target_agent_buffer = \
        get_target_agent_buff_conf_and_should_reuse(config)

    target_agent_buffer = build_buffer(target_agent_buff_config, obs_dim, horizon)
    behavior_agent_buffer = target_agent_buffer
    if not reuse_target_agent_buffer:
        behavior_agent_buffer = build_buffer(
            config['behavior_agent_buffer_config'],
            obs_dim,
            horizon
        )

    return target_agent_buffer, behavior_agent_buffer, reuse_target_agent_buffer


def unpack_losses(iter_losses):
    if not isinstance(iter_losses[0], dict):
        return torch.tensor(iter_losses)

    flattened_losses = {}
    for losses in iter_losses:
        for key, val in losses.items():
            if key not in flattened_losses:
                flattened_losses[key] = []

            flattened_losses[key].append(val)

    return {k: torch.tensor(v) for k, v in flattened_losses.items()}


class BasicLearner(WandbTrainableMixin, Trainable):
    def setup(self, config: Dict[str, object]) -> None:
        seed = get_seed(config)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        self.train_batch_size = config['train_batch_size']

        env_config = {
            'sample_batch_size': self.train_batch_size,
            **config['env_config']
        }
        self.env = env_config['type'](env_config)

        (
            self.target_agent_buffer,
            self.behavior_agent_buffer,
            self.reuse_target_agent_buffer
        ) = get_buffers(config, self.env.obs_dim, self.env.horizon)

        _, self.true_Z = self.env.true_Z()
        true_Z_computable = config.get('use_true_log_Z', False)

        agent_conf_base = {
            'env': self.env,
            'obs_dim': self.env.obs_dim,
            'action_dim': self.env.action_dim,
            'horizon': self.env.horizon,
            'true_Z_computable': true_Z_computable,
            'true_Z': self.true_Z,
            'batch_size': config['train_batch_size'],
        }

        self.target_agent = config['target_agent_config']['type']({
            **agent_conf_base,
            **config['target_agent_config']
        })
        self.behavior_agent = config['behavior_agent_config']['type']({
            'target_agent': self.target_agent,
             **agent_conf_base,
             **config['behavior_agent_config']
        })

        self.use_remote_rollout = config.get('use_remote_rollout', False)
        worker_type = \
            RolloutWorkerRemote if self.use_remote_rollout else RolloutWorker

        self.rollout_workers = [
            worker_type(self.env, self.behavior_agent, self.train_batch_size)
            for _ in range(config.get('num_rollout_workers', 1))
        ]

        self.num_target_train_batches_per_step = \
            config['num_target_train_batches_per_step']
        self.num_behavior_train_batches_per_step = \
            config['num_behavior_train_batches_per_step']

        self.step_num = 0

        self.metrics = self._get_metrics(config)
        self._last_result_dict = {}

    def _get_metrics(self, config: Dict[str, object]) -> List[BaseMetric]:
        return [
            metric_conf_dict['type'](self._get_metric_conf(metric_conf_dict))
            for metric_conf_dict in config['metrics_config']
        ]

    def _get_metric_conf(
        self,
        metric_conf_dict: Dict[str, object]
    ) -> Dict[str, object]:
        return_dict = {
            'env': self.env,
            'target_agent': self.target_agent,
            'target_agent_sampler': self.target_agent_buffer,
            'trainer_batch_size': self.train_batch_size,
            'behavior_agent_batch_size': self.train_batch_size,
            **metric_conf_dict
        }

        behavior_agent_cond = (
            self.behavior_agent is not None and
            self.behavior_agent.does_grad_update
        )

        if behavior_agent_cond:
            return_dict.update({
                'behavior_agent': self.behavior_agent,
                'behavior_agent_sampler': self.behavior_agent_buffer,
            })

        return return_dict

    def step(self) -> Dict[str, object]:
        rollout_infos = {'behavior_rollout': self._do_rollout()}
        self._add_to_buffers(rollout_infos['behavior_rollout'])
        trajs_to_add, custom_trajs_infos = self._get_custom_trajs()
        if trajs_to_add:
            self._add_to_buffers(custom_trajs_infos['rollout'])

        target_update_losses = [
            self.target_agent.update(
                self.target_agent_buffer.sample(self.train_batch_size)
            )
            for _ in range(self.num_target_train_batches_per_step)
        ]

        behavior_update_losses = [
            self.behavior_agent.update({
                'target_agent': self.target_agent,
                **self.behavior_agent_buffer.sample(self.train_batch_size)
            })
            for _ in range(self.num_behavior_train_batches_per_step)
        ]

        self.step_num += 1
        return self._process_metrics(
            rollout_infos=rollout_infos,
            custom_trajs_infos=custom_trajs_infos,
            target_update_losses=target_update_losses,
            behavior_update_losses=behavior_update_losses
        )

    def _get_custom_trajs(self) -> Tuple[
        bool,
        Tuple[
            TensorType, # ['horizon', 'batch_size', 'ndim_times_side_len', int],
            TensorType, # ['horizon', 'batch_size', int],
            TensorType, # ['horizon', 'batch_size', float],
            TensorType, # ['horizon','batch_size', bool],
        ]
    ]:
        return False, {}

    def _do_rollout(self) -> Tuple[
        TensorType, # ['horizon', 'batch_size', 'ndim_times_side_len', int],
        TensorType, # ['horizon', 'batch_size', int],
        TensorType, # ['horizon', 'batch_size', int]
        TensorType, # ['horizon', 'batch_size', float],
        TensorType, # ['horizon','batch_size', bool],
    ]:
        def sample(worker):
            if isinstance(worker, type(RolloutWorkerRemote)):
                return ray.get(worker.sample())
            else:
                return worker.sample()

        all_states, all_actions, all_back_actions, all_dones, all_rewards = \
            [], [], [], [], []

        all_lists = \
            (all_states, all_actions, all_back_actions, all_dones, all_rewards)

        for worker in self.rollout_workers:
            tuple(map(lambda x: x[0].append(x[1]), zip(all_lists, sample(worker))))

        return tuple(map(torch.cat, all_lists))

    def _add_to_buffers(
        self,
        rollout: Tuple[
            TensorType, # ['horizon', 'batch_size', 'ndim_times_side_len', int],
            TensorType, # ['horizon', 'batch_size', int],
            TensorType, # ['horizon', 'batch_size', float],
            TensorType, # ['horizon','batch_size', bool],
        ]
    ) -> None:
        self.target_agent_buffer.add(*rollout)
        if not self.reuse_target_agent_buffer:
            self.behavior_agent_buffer.add(*rollout)

    def _process_metrics(
        self,
        **kwargs: Dict[str, object]
    ) -> Dict[str, object]:
        self._update_metrics(**kwargs)

        wandb_log_dict = self._get_wandb_log_dict()
        wandb_log_dict.update(self._get_learner_step_result_dict())
        if wandb_log_dict:
            self.wandb.log(wandb_log_dict)

        result_dict = self._get_step_result_dict()
        result_dict.update(self._get_learner_wandb_log_dict())
        self._last_result_dict = result_dict

        return result_dict

    def _get_learner_step_result_dict(self) -> Dict[str, object]:
        return {'true_log_Z': np.log(self.true_Z)}

    def _get_learner_wandb_log_dict(self) -> Dict[str, object]:
        return {}

    def _update_metrics(
        self,
        rollout_infos: Dict[str, object],
        custom_trajs_infos: Dict[str, object],
        target_update_losses: Union[Dict[str, TensorType], List[TensorType]],
        behavior_update_losses: Union[Dict[str, TensorType], List[TensorType]]
    ) -> None:
        merged_infos = {
            'target_policy_losses': unpack_losses(target_update_losses),
            'behavior_policy_losses': unpack_losses(behavior_update_losses),
            'target_agent': self.target_agent,
            'behavior_agent': self.behavior_agent,
            'target_agent_replay_buffer': self.target_agent_buffer,
            'behavior_agent_replay_buffer': self.behavior_agent_buffer,
            **rollout_infos,
            **custom_trajs_infos,
        }

        tuple(map(lambda x: x.update(merged_infos), self.metrics))

    def _get_wandb_log_dict(self) -> Dict[str, object]:
        wandb_dict = {}
        for metric in self.metrics:
            if metric.should_compute(self.step_num):
                wandb_dict.update(metric.compute_for_wandb_dict())

        return wandb_dict

    def _get_step_result_dict(self) -> Dict[str, object]:
        metrics_dict = {}
        for metric in self.metrics:
            if metric.should_compute(self.step_num):
                metrics_dict.update(metric.compute_for_step_result_dict())

        metrics_dict.update(self._get_agent_metrics())
        return metrics_dict

    def _get_agent_metrics(self) -> Dict[str, object]:
        agent_metrics_dict = {}
        for key, val in self.target_agent.get_metrics().items():
            agent_metrics_dict['target_agent_%s' % key] = val

        for key, val in self.behavior_agent.get_metrics().items():
            agent_metrics_dict['behavior_agent_%s' % key] = val

        return agent_metrics_dict

    def stop(self) -> None:
        self.wandb.summary.update(self._last_result_dict)
        super().stop()

    def cleanup(self) -> None:
        if self.use_remote_rollout:
            for worker in self.rollout_workers:
                worker.__ray_terminate__.remote()
