from publish.agents.losses import BaseLoss
from publish.utils import get_log_rewards
from torchtyping import TensorType
from typing import Dict, Tuple
import torch

def unpack_dict(loss_infos: Dict[str, object]) -> Tuple[
    TensorType['batch_size', 'horizon'],
    TensorType['batch_size', 'horizon'],
    TensorType['batch_size', 'horizon'],
    TensorType['batch_size', 'horizon'],
    TensorType['batch_size'],
]:
    return (
        loss_infos['fwd_log_probs'],
        loss_infos['back_log_probs'],
        loss_infos['log_state_flows'],
        loss_infos['dones'],
        get_log_rewards(loss_infos)
    )

class DetailedBalanceLoss(BaseLoss):
    def __call__(
        self,
        loss_infos: Dict[str, object]
    ) -> TensorType['batch_size']:
        frwd_log_probs, back_log_probs, log_state_flows, dones, log_rewards = \
            unpack_dict(loss_infos)

        log_rewards = log_rewards.reshape(-1, 1)
        end_log_flow_pre = (dones * log_rewards) + ((1 - dones) * log_state_flows)
        rolled_back_log_probs = back_log_probs.roll(-1, dims=1)
        rolled_end_log_flow = end_log_flow_pre.roll(-1, dims=1)

        frwd_side = frwd_log_probs + log_state_flows
        back_side = rolled_back_log_probs + rolled_end_log_flow

        num_transitions_in_traj = (1 - dones).sum(dim=1)
        unmasked_loss = (frwd_side - back_side).pow(2)
        return ((1 - dones) * unmasked_loss).sum(dim=1) / num_transitions_in_traj

    @property
    def requires_state_flows(self) -> bool:
        return True
