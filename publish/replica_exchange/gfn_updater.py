from gfn_exploration.workers import RolloutWorkerRemote
from gfn_exploration.utils import traj_container_to_dict, get_device
from torch.utils._pytree import tree_map
import inspect
import ray
import torch

STATES_IDX = 0

@ray.remote(num_cpus=1e-4, num_gpus=1e-4)
class GFNUpdater:
    def __init__(self, env, temperature, config):
        agent_conf_base = {
            'env': env,
            'obs_dim': env.obs_dim,
            'action_dim': env.forward_action_dim,
            'horizon': env.horizon,
            'true_Z_computable': False,
            'true_Z': 0.0,
            'use_tree_pb': env.is_tree_state_space,
            'batch_size': config['train_batch_size'],
        }

        agent_conf = {
            **agent_conf_base,
            **config['target_agent_config']
        }

        agent_conf['reward_exponent'] = temperature
        self.target_agent = config['target_agent_config']['type'](agent_conf)

        self.temperature = temperature

        self.env = env
        self.env.device = torch.device('cpu')
        self.rollout_worker = RolloutWorkerRemote.remote(
            self.env.to(device=torch.device('cpu')),#tree_map(lambda x: x.to(device='cpu'), env),
            self.target_agent.to(device=torch.device('cpu')),#tree_map(lambda x: x.to(device='cpu'), self.target_agent),#self.target_agent.to(device=cpu_device),
            config['train_batch_size'],
            torch.device('cpu')
        )
        self.env.device = get_device()

    def get_rollout_worker(self):
        return self.rollout_worker

    def update(self, trajectories):#: TrajectoryContainer):
        traj_states = trajectories[STATES_IDX]
        if self.env.is_tree_state_space:
            trajs = trajectories
        else:
            trajs = ray.get(
                self.rollout_worker.sample_backward.remote(
                    traj_states[:, -1].cpu()
                )
            )

        trajs = tuple(x.to(device=get_device()) for x in trajs)
        losses = self.target_agent.update(traj_container_to_dict(*trajs))
        self.rollout_worker.set_agent_weights.remote(
            self.target_agent.get_state_dict(device=torch.device('cpu'))
        )

        return losses

if __name__ == '__main__':
    dumped = ray.cloudpickle.dumps(GFNUpdater)
    loaded = ray.cloudpickle.loads(dumped)
