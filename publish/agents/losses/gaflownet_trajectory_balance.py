from publish.agents.losses import BaseLoss
from publish.utils import get_log_rewards, get_first_done_idx
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
        update_infos['back_log_probs'].exp(),
        update_infos['intrinsic_rewards'],
        get_log_rewards(update_infos).exp(),
        update_infos['log_state_flows'].exp(),
        update_infos['log_Z'],
        update_infos['dones']
    )


class GAFlowNetTrajectoryBalanceLoss(BaseLoss):
    def __call__(self, loss_infos: Dict[str, object]) -> TensorType['batch_size']:
        (
            fwd_log_probs,
            back_probs,
            intrinsic_rewards,
            rewards,
            state_flows,
            log_Z,
            dones
        ) = unpack_dict(loss_infos)

        frwd_side = log_Z + fwd_log_probs.sum(dim=1)

        idx1 = torch.arange(len(intrinsic_rewards))
        idx2 = get_first_done_idx(dones) - 1

        terminal_intrinsic_rewards = intrinsic_rewards[idx1, idx2].clone()
        intrinsic_rewards[idx1, idx2] = 0.0

        back_side_pre = back_probs + (intrinsic_rewards / state_flows.roll(-1))
        back_side = (
            (rewards + terminal_intrinsic_rewards) * back_side_pre.prod(dim=1)
        ).log()

        return (frwd_side - back_side).pow(2)

    @property
    def requires_state_flows(self) -> bool:
        return True
