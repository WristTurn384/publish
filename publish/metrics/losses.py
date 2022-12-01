from publish.metrics import BaseMetric
from torchtyping import TensorType
from typing import Dict

class Losses(BaseMetric):
    def __init__(self, config: Dict[str, object]):
        self.target_policy_losses = None
        self.behavior_policy_losses = None

        self.first_k_losses_mean_k_vals = config.get(
            'first_k_losses_mean_k_vals',
            []
        )

        self.last_k_losses_mean_k_vals = config.get(
            'last_k_losses_mean_k_vals',
            []
        )

    def update(self, update_infos: Dict[str, object]) -> None:
        if 'target_policy_losses' in update_infos:
            self.target_policy_losses = update_infos['target_policy_losses']
        if 'behavior_policy_losses' in update_infos:
            self.behavior_policy_losses = update_infos['behavior_policy_losses']

    def compute_for_step_result_dict(self) -> Dict[str, object]:
        assert (
            self.target_policy_losses is not None or
            self.behavior_policy_losses is not None
        )

        to_add = []
        if self.target_policy_losses is not None:
            to_add.append('target')
        if self.behavior_policy_losses is not None:
            to_add.append('behavior')

        result_dict = {}
        for policy_name in to_add:
            policy_losses = getattr(self, '%s_policy_losses' % policy_name)
            if isinstance(policy_losses, dict):
                for loss_name, losses in policy_losses.items():
                    self._add_loss_to_result_dict(
                        result_dict,
                        losses,
                        policy_name,
                        loss_name
                    )

            else:
                self._add_loss_to_result_dict(
                    result_dict,
                    policy_losses,
                    policy_name
                )

        return result_dict

    def _add_loss_to_result_dict(
        self,
        result_dict: Dict[str, object],
        policy_losses: TensorType,
        policy_name: str,
        loss_name: str = 'policy'
    ) -> None:
        key_prefix = '%s_%s' % (policy_name, loss_name)

        result_dict['%s_loss_all_mean' % key_prefix] = \
            policy_losses.mean().item()

        for k in self.first_k_losses_mean_k_vals:
            key = '%s_loss_first_%d_mean' % (key_prefix, k)
            result_dict[key] = policy_losses[:k].mean().item()

        for k in self.last_k_losses_mean_k_vals:
            key = '%s_loss_last_%d_mean' % (key_prefix, k)
            result_dict[key] = policy_losses[-k:].mean().item()

        return
