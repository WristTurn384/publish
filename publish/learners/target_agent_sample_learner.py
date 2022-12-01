from publish.learners import BasicLearner
from publish.workers import RolloutWorker
from typing import Dict, Tuple
from torchtyping import TensorType

class TargetAgentSamplesIncorporationLearner(BasicLearner):
    def setup(self, config: Dict[str, object]):
        self.target_agent_batch_size = \
            config['target_agent_pf_samples_batch_size']

        super().setup(config)

        self.target_agent_worker = RolloutWorker(
            self.env.clone(self.target_agent_batch_size),
            self.target_agent,
            self.target_agent_batch_size
        )

    def _get_metric_conf(
        self,
        metric_conf_dict: Dict[str, object]
    ) -> Dict[str, object]:
        ret_dict = super()._get_metric_conf(metric_conf_dict)
        ret_dict['target_agent_pf_batch_size'] = self.target_agent_batch_size

        return ret_dict

    def _get_custom_trajs(self) -> Tuple[
        bool,
        Tuple[
            TensorType, # ['horizon', 'batch_size', 'ndim_times_side_len', int],
            TensorType, # ['horizon', 'batch_size', int],
            TensorType, # ['horizon', 'batch_size', float],
            TensorType  # ['horizon','batch_size', bool],
        ]
    ]:
        ret_dict = {'target_policy_rollout': self.target_agent_worker.sample()}
        ret_dict['rollout'] = ret_dict['target_policy_rollout']

        return True, ret_dict
