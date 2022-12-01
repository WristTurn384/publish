from publish.agents.losses import BaseLoss
from publish.utils import get_first_done_idx, get_log_rewards
from typing import Dict
from torchtyping import TensorType
import torch

def _compute_loss_single_traj(
    fwd_logits: TensorType['horizon'],
    back_logits: TensorType['horizon'],
    log_state_flows: TensorType['horizon'],
    dones: TensorType['horizon'],
    log_reward: float,
    lambda_val: float
) -> TensorType[float]:
    # dones is currently a 1D tensor, but
    # get_first_done_idx expects a 2D tensor
    traj_len = get_first_done_idx(dones.unsqueeze(0)).item()
    traj_loss = 0.0
    idx = torch.arange(1, traj_len + 1)
    lambda_sum = ((lambda_val ** idx) * (traj_len + 1 - idx)).sum()

    # Subtrajectories are of length 1 until traj_len
    for i, subtraj_len in enumerate(range(1, traj_len + 1)):
        frwd_subtraj_idxs = torch.arange(subtraj_len).tile(
            traj_len - subtraj_len + 1,
            1
        )

        to_add = torch.arange(frwd_subtraj_idxs.shape[0]).reshape(-1, 1)

        frwd_subtraj_idxs = frwd_subtraj_idxs + to_add
        back_subtraj_idxs = frwd_subtraj_idxs + 1

        frwd_side = (
            fwd_logits[frwd_subtraj_idxs].sum(dim=1) +
            log_state_flows[frwd_subtraj_idxs[:, 0]]
        )

        end_log_flow_mask = dones[back_subtraj_idxs[:, -1]]
        end_log_flow = (
            (end_log_flow_mask * log_reward) +
            ((1 - end_log_flow_mask) * log_state_flows[back_subtraj_idxs[:, -1]])
        )

        back_side = back_logits[back_subtraj_idxs].sum(dim=1) + end_log_flow

        traj_length_weight = (lambda_val ** subtraj_len) / lambda_sum
        subtraj_loss = traj_length_weight * ((frwd_side - back_side)).pow(2).sum()
        traj_loss = traj_loss + subtraj_loss

    return traj_loss

class NaiveSubtrajectoryBalanceLoss(BaseLoss):
    def __init__(self, config: Dict[str, object]):
        super().__init__(config)
        self.lambda_val = config['lambda']

    def __call__(self, loss_infos: Dict[str, object]) -> TensorType['batch_size']:
        losses = []
        log_rewards = get_log_rewards(loss_infos)
        for i in range(len(loss_infos['fwd_log_probs'])):
            this_loss = _compute_loss_single_traj(
                loss_infos['fwd_log_probs'][i],
                loss_infos['back_log_probs'][i],
                loss_infos['log_state_flows'][i],
                loss_infos['dones'][i],
                log_rewards[i],
                self.lambda_val
            )

            losses.append(this_loss)

        return torch.stack(losses)

    @property
    def requires_state_flows(self) -> bool:
        return True
