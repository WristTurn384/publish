from publish.agents import BasicGFlowNet, EpsilonNoisyGFlowNet, BasicExplorationGFlowNet
from publish.agents.debug_uniform_traj_policy import DebugUniformTrajPolicy
from publish.agents.losses import trajectory_balance_loss
from publish.learners import BasicLearner
from publish.envs import HypergridEnv
from publish.metrics import Losses, DistributionMetrics
from publish.buffers import UniformFifoBuffer
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from ray import tune
import torch

TRAINER = BasicLearner

SEARCH_ALGORITHM = OptunaSearch(metric='l1_distance', mode='min')

SCHEDULER_ALGORITHM = ASHAScheduler(
    time_attr='training_iteration',
    metric='l1_distance',
    mode='min',
    grace_period=3800,
    max_t=25000
)

CONFIG = {
    'target_agent_pf_samples_batch_size': 64,
    'train_batch_size': 64,
    'num_target_train_batches_per_step': tune.randint(1, 8),
    'num_behavior_train_batches_per_step': 1,
    'env_config': {
        'type': HypergridEnv,
        'side_length': 64,
        'num_dims': 2,
        'R_0': 1e-2,
    },
    'buffer_config': {
        'type': UniformFifoBuffer,
        'capacity': 100,
    },
    'target_agent_config': {
        'type': BasicGFlowNet,
        'hidden_layer_dim': 256,
        'num_hidden_layers': 2,
        'loss_fxn': trajectory_balance_loss,
        'param_backward_policy': True,
        'init_log_Z_val': tune.uniform(0.0, 40.0),
        'log_Z_optim_config': {
            'type': torch.optim.Adam,
            'lr': tune.loguniform(1e-5, 1e-1),
            'lr_scheduler_config': {
                'type': torch.optim.lr_scheduler.ReduceLROnPlateau,
                'patience': tune.randint(5000, 25000),
                'verbose': True
            }
        },
        'optim_config': {
            'type': torch.optim.Adam,
            'lr': tune.loguniform(1e-5, 1e-1),
            'lr_scheduler_config': {
                'type': torch.optim.lr_scheduler.ReduceLROnPlateau,
                'patience': tune.randint(2500, 25000),
                'verbose': True
            }
        }
    },
    'behavior_agent_config': { # Just do on-policy training
        'type': EpsilonNoisyGFlowNet,
        'epsilon': tune.uniform(0.01, 0.2),
        'hidden_layer_dim': 256,
        'num_hidden_layers': 2,
        'loss_fxn': trajectory_balance_loss,
        'param_backward_policy': True,
        'log_Z_optim_config': {
            'type': torch.optim.Adam,
            'lr': 1e-1
        },
        'optim_config': {
            'type': torch.optim.Adam,
            'lr': 1e-3
        }
    },
    'metrics_config': [
        {'type': Losses},
        {'type': DistributionMetrics, 'num_states_to_track': 200_000}
    ]
}
