from publish.agents import BasicGFlowNet, TabularGFlowNet
from publish.agents.flow_functions import GaussianNoiseExactFlowFunction
from publish.agents.losses import BaseLoss, NaiveSubtrajectoryBalanceLoss, TrajectoryBalanceLoss, DetailedBalanceLoss
from publish.metrics import BaseMetric, PeriodicComputableMetric
from publish.workers import RolloutWorker
from publish.utils.interfaces import BaseSampler
from torchtyping import TensorType
from typing import Dict, Tuple, List
import numpy as np
import torch
import wandb

WORKER_ARG_NAMES = ('states', 'fwd_actions', 'back_actions', 'dones', 'rewards')

LOSS_NAME_IDX     = 0
COMP_NAME_IDX     = 0
FLOW_FXN_NAME_IDX = 1
GRAD_NAME_IDX     = 2

def batch_cosine_similarity(
    mtx: TensorType['batch_size', 'num_policy_model_params'],
    n_traj_to_sample_exp: int,
    full_batch_grad: TensorType['num_policy_model_params'] = None,
    eps: float = 1e-8,
    shuffle_batches_cosine_sim: bool = False
):
    if full_batch_grad is None:
        full_batch_grad = mtx.mean(dim=0)

    full_grad_norm = torch.linalg.norm(full_batch_grad, ord=2).reshape(1, -1)
    full_batch_grad = full_batch_grad.reshape(1, -1)

    cosine_sims = []
    for i in range(n_traj_to_sample_exp + 1):
        batch_size = 2 ** i
        batch_grads = _get_batch_grads(mtx, batch_size)
        batch_grad_norms = torch.linalg.norm(batch_grads, dim=1, ord=2)

        dot_prod = (full_batch_grad * batch_grads).sum(dim=1)
        normalization = torch.max(
            full_grad_norm * batch_grad_norms,
            torch.tensor(eps)
        )

        cosine_sims.append((dot_prod / normalization).mean())

    return torch.stack(cosine_sims)

def _get_batch_grads(mtx, batch_size, shuffle_batches_cosine_sim=False):
    num_grads = len(mtx)
    num_batches = int(num_grads / batch_size)

    batch_grads = torch.zeros(
        (num_batches, mtx.shape[1]),
        device=mtx.device
    )

    batch_idx = torch.arange(
        num_batches,
        device=mtx.device
    ).repeat_interleave(batch_size)

    if shuffle_batches_cosine_sim:
        shuffle_idx = torch.randperm(len(batch_idx), device=mtx.device)
        batch_idx = batch_idx[shuffle_idx]

    batch_grads.index_add_(0, batch_idx, mtx)
    return batch_grads / batch_size


def get_grads(
    agent: BasicGFlowNet,
    trajectories_dict: Dict[str, object],
    loss_fxn: BaseLoss,
    compute_log_prob_grad: bool = False,
    flow_fxn: object = None
) -> TensorType['batch_size', 'num_policy_model_params']:
    grads_dict = {}
    for i in range(len(trajectories_dict['states'])):
        # Make it a 2D tensor so that, once the dictionary tensors are
        # indexed, they keep their original dimensionlity
        item_dict = {
            key: val[i].unsqueeze(0)
            for key, val in trajectories_dict.items()
        }

        loss_infos = agent._get_loss_infos(item_dict, flow_fxn)
        loss_infos['agent'] = agent

        if 'fwd_table_logits' in loss_infos:
            loss_infos['fwd_table_logits'].retain_grad()

        loss = loss_fxn(loss_infos).mean()

        agent.optim.zero_grad()
        loss.backward()

        _update_grad_dict(grads_dict, agent, loss_infos)

    return {
        key: torch.stack(grads)
        for key, grads in grads_dict.items()
    }


def _update_grad_dict(
    grads_dict: Dict[str, List[TensorType]],
    agent: BasicGFlowNet,
    loss_infos: Dict[str, object]
) -> None:
    is_tabular = isinstance(agent, TabularGFlowNet)

    if is_tabular:
        inner_dict = {
            'policy_logits_grad': agent.fwd_transition_logits.grad.flatten().clone()
        }

    else:
        inner_dict = {
            'nn_param_grad': torch.cat([
                param.grad.view(-1)
                for param in agent.model.parameters()
                if param.grad is not None
            ]),
            'policy_logits_grad': loss_infos['fwd_table_logits'].grad.flatten().clone()
        }

    for key, grad in inner_dict.items():
        if key not in grads_dict:
            grads_dict[key] = []

        grads_dict[key].append(grad)

    return


class GradientCosineSimilarity(PeriodicComputableMetric, BaseMetric):
    def __init__(self, config: Dict[str, object]):
        super().__init__(config)

        self.target_agent = config['target_agent']
        self.batch_size = config['batch_size']
        self.n_traj_to_sample_exponent = int(np.log2(self.batch_size))

        self.target_agent_sampler = RolloutWorker(
            config['env'].clone(self.batch_size),
            self.target_agent,
            self.batch_size
        )

        self.plot_xs = list(range(1, self.n_traj_to_sample_exponent + 1))

        self.training_via_subtb = isinstance(
            self.target_agent._loss_fxn,
            NaiveSubtrajectoryBalanceLoss
        )

        self.training_loss_name = str(type(self.target_agent._loss_fxn))

        self.losses = {
            'DB': DetailedBalanceLoss(),
            'SubTB': NaiveSubtrajectoryBalanceLoss({
                'lambda': config.get('subtb_lambda', 0.8)
            }),
            'TB': TrajectoryBalanceLoss(),
        }

        self.flow_functions = {'learned': None}
        for flow_fxn_config in config.get('flow_fxn_configs', []):
            flow_fxn = flow_fxn_config['type']({
                'env': config['env'],
                'agent': self.target_agent,
                **flow_fxn_config
            })

            self.flow_functions[str(flow_fxn)] = flow_fxn

    # Don't need to actually do anything to update since we've
    # got a reference to the agents and samplers, so the updates.
    # in the learner will automatically propagate here
    def update(self, update_infos: Dict[str, object]) -> None:
        super().update(update_infos)

    def compute_for_wandb_dict(self) -> Dict[str, object]:
        worker_out = self.target_agent_sampler.sample()
        trajectories_dict = {
            key: val
            for key, val in zip(WORKER_ARG_NAMES, worker_out)
        }

        for flow_fxn in self.flow_functions.values():
            if flow_fxn is not None:
                flow_fxn.update(self.target_agent)

        grad_matrices = {}
        for loss_name, loss in self.losses.items():
            for flow_fxn_name, flow_fxn in self.flow_functions.items():
                if isinstance(flow_fxn, GaussianNoiseExactFlowFunction) and loss_name != 'SubTB':
                    continue

                grad_matrices.update({
                    (loss_name, flow_fxn_name, grad_name): grad
                    for grad_name, grad in get_grads(
                        self.target_agent,
                        trajectories_dict,
                        loss,
                        flow_fxn=flow_fxn
                    ).items()
                })

        cos_sim_results = {}
        for grad_ident_1, grad_mtx_1 in grad_matrices.items():
            for grad_ident_2, grad_mtx_2 in grad_matrices.items():
                grad_name = grad_ident_1[GRAD_NAME_IDX]
                if grad_name != grad_ident_2[GRAD_NAME_IDX]:
                    continue

                grad_cmp_name = '%s vs %s' % (
                    grad_ident_1[LOSS_NAME_IDX],
                    grad_ident_2[LOSS_NAME_IDX]
                )

                flow_cmp_name = '%s vs %s' % (
                    grad_ident_1[FLOW_FXN_NAME_IDX],
                    grad_ident_2[FLOW_FXN_NAME_IDX],
                )

                cos_sim_results[(grad_cmp_name, flow_cmp_name, grad_name)] = \
                    batch_cosine_similarity(
                        grad_mtx_1,
                        self.n_traj_to_sample_exponent,
                        full_batch_grad=grad_mtx_2.mean(dim=0)
                    )

        cols = [
            'Training Loss',
            'Loss Comp Name',
            'Flow Fxn Comp Name',
            'Grad Name',
            'K',
            'Cosine Similarity',
            'Training Step'
        ]

        table = wandb.Table(columns=cols)
        for cmp_ident, cos_sims in cos_sim_results.items():
            for i in range(len(cos_sims)):
                table.add_data(
                    self.training_loss_name,
                    cmp_ident[COMP_NAME_IDX],
                    cmp_ident[FLOW_FXN_NAME_IDX],
                    cmp_ident[GRAD_NAME_IDX],
                    i,
                    cos_sims[i],
                    self.iter_num
                )

        return {'Gradient Cosine Similarity Table': table}
