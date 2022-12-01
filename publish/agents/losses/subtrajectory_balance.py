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
    lambda_sum = 0.0

    # Subtrajectories are of length 1 until traj_len
    for subtraj_len in range(1, traj_len + 1):
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

        # Er, for the back side I need a mask for if the traj is done
        # to use the log reward instead of the state flow
        back_side = back_logits[back_subtraj_idxs].sum(dim=1) + end_log_flow

        traj_length_weight = lambda_val ** subtraj_len
        subtraj_loss = traj_length_weight * (frwd_side - back_side).pow(2).sum()
        traj_loss = traj_loss + subtraj_loss

        lambda_sum += len(frwd_subtraj_idxs) * traj_length_weight

    return traj_loss / lambda_sum

class SubtrajectoryBalanceLoss(BaseLoss):
    def __init__(self, config: Dict[str, object]):
        super().__init__(config)
        self.lambda_val = config['lambda']

        horizon = config['horizon']
        frwd_idxs, back_idxs = [], []
        for subtraj_len in range(1, horizon):
            num_subtrajs_this_len = traj_len - subtraj_len + 1
            frwd_subtraj_idxs = torch.arange(subtraj_len).tile(
                num_subtrajs_this_len,
                1
            )

            to_add = torch.arange(frwd_subtraj_idxs.shape[0]).reshape(-1, 1)

            frwd_subtraj_idxs = frwd_subtraj_idxs + to_add
            back_subtraj_idxs = frwd_subtraj_idxs + 1

            to_cat = torch.full(
                (num_subtrajs_this_len, horizon - subtraj_len),
                fill_value=horizon
            )

            frwd_idxs.append(torch.cat((frwd_subtraj_idxs, to_cat), dim=1))
            back_idxs.append(torch.cat((back_subtraj_idxs, to_cat), dim=1))

        frwd_idxs.append(torch.full((1, horizon), fill_value=horizon))
        back_idxs.append(torch.full((1, horizon), fill_value=horizon))

        self.frwd_subtraj_idxs = torch.cat(frwd_idxs, dim=0)
        self.back_subtraj_idxs = torch.cat(back_idxs, dim=0)

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
