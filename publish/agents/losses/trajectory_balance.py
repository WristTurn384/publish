from publish.agents.losses import BaseLoss
from publish.utils import get_log_rewards
from torchtyping import TensorType
from typing import Dict, Tuple
import torch

def unpack_dict(update_infos: Dict[str, object]) -> Tuple[
    TensorType['batch_size', 'horizon', 'num_dim'],
    TensorType['batch_size', 'horizon', 'num_dim'],
    TensorType['batch_size', 'horizon'],
    TensorType['batch_size', 'horizon'],
    TensorType
]:
    return (
        update_infos['fwd_log_probs'],
        update_infos['back_log_probs'],
        get_log_rewards(update_infos),
        update_infos['log_Z']
    )


class TrajectoryBalanceLoss(BaseLoss):
    def __call__(self, loss_infos: Dict[str, object]) -> TensorType['batch_size']:
        (
            fwd_log_probs,
            back_log_probs,
            log_rewards,
            Z
        ) = unpack_dict(loss_infos)

        frwd_side = Z + fwd_log_probs.sum(dim=1)
        back_side = back_log_probs.sum(dim=1) + log_rewards

        return (frwd_side - back_side).pow(2)

    @property
    def requires_state_flows(self) -> bool:
        return False
