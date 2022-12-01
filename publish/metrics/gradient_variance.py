from publish.agents import BasicGFlowNet
from publish.metrics import BaseMetric
from publish.utils.interfaces import BaseSampler
from torchtyping import TensorType
from typing import Dict, Tuple
import torch

LOG_Z_GRAD_TUPLE_IDX = 0
MODEL_PARAM_GRAD_TUPLE_IDX = 1

def get_batch_sizes(config: Dict[str, object]) -> Tuple[int]:
    if 'batch_sizes' in config:
        return config['batch_sizes']
    else:
        return [config['trainer_batch_size']]


def compute_grad_variance(
    agent: BasicGFlowNet,
    sampler: BaseSampler,
    batch_size: int,
    num_grads_to_sample: int
) -> Dict[str, float]:
    log_Z_grads, model_grads = [], []
    for _ in range(num_grads_to_sample):
        grads = sample_single_grad(agent, sampler, batch_size)

        log_Z_grads.append(grads[LOG_Z_GRAD_TUPLE_IDX])
        model_grads.append(grads[MODEL_PARAM_GRAD_TUPLE_IDX])

    log_Z_grads, model_grads = torch.stack(log_Z_grads), torch.stack(model_grads)
    return {
        'log_Z_grad_var': log_Z_grads.var().item(),
        'unnormalized_model_grad_var': model_grads.var(
            dim=0,
            unbiased=True
        ).mean().item(),
        'normalized_model_grad_var': compute_normalized_var(model_grads)
    }


def sample_single_grad(
    agent: BasicGFlowNet,
    sampler: BaseSampler,
    batch_size: int
) -> Tuple[TensorType[1], TensorType['num_params']]:
    trajectories_dict = sampler.sample(batch_size)
    loss = agent.loss(trajectories_dict)

    agent.optim.zero_grad()
    agent.log_Z_optim.zero_grad()
    loss.backward()

    model_grads = torch.cat([
        param.grad.view(-1)
        for param in agent.parameters()
        if param.grad is not None
    ])

    return agent.log_Z.grad.clone(), model_grads


def compute_normalized_var(
    grads: TensorType['num_grads_sampled', 'num_params']
) -> float:
    mean_grad = grads.mean(dim=0).abs() + 1e-8
    denom = (len(grads) - 1) * mean_grad

    return ((grads - mean_grad).pow(2) / denom).sum(dim=0).mean().item()


class GradientVarianceMetric(BaseMetric):
    def __init__(self, config: Dict[str, object]):
        self.target_agent, self.target_agent_sampler = \
            config['target_agent'], config['target_agent_sampler']

        self.behavior_agent, self.behavior_agent_sampler = None, None
        if 'behavior_agent' in config:
            self.behavior_agent = config['behavior_agent']
            self.behavior_agent_sampler = (
                config.get('behavior_agent_sampler', None) or
                self.target_agent_sampler
            )

        self.batch_sizes = get_batch_sizes(config)
        self.num_grads_to_sample = config['num_grads_to_sample']

    # Don't need to actually do anything to update since we've
    # got a reference to the agents and samplers, so the updates.
    # in the learner will automatically propagate here
    def update(self, update_infos: Dict[str, object]) -> None:
        pass

    def compute_for_step_result_dict(self) -> Dict[str, object]:
        agents_to_compute_for = ['target']
        if self.behavior_agent is not None:
            agents_to_compute_for.append('behavior')

        result_dict = {}
        for agent_name in agents_to_compute_for:
            agent = getattr(self, '%s_agent' % agent_name)
            sampler = getattr(self, '%s_agent_sampler' % agent_name)

            for batch_size in self.batch_sizes:
                inner_result_dict = compute_grad_variance(
                    agent,
                    sampler,
                    batch_size,
                    self.num_grads_to_sample
                )

                for key, val in inner_result_dict.items():
                    new_key = '%s_agent_%s_batch_size_%d' % (
                        agent_name,
                        key,
                        batch_size
                    )

                    result_dict[new_key] = val

        return result_dict
