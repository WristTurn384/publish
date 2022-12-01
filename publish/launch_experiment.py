import os
from ray.tune.integration.wandb import WandbLoggerCallback
from ray import tune
from publish.utils import get_wandb_config
from typing import Dict
from pathlib import Path
import argparse
import importlib
import os
import torch
import ray


SLURM_JOB_CPUS_FILENAME = '/sys/fs/cgroup/cpuset/slurm/uid_%s/job_%s/cpuset.cpus'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_name", default=None)
    parser.add_argument("-t", "--entity_name", default=None)
    parser.add_argument("-c", "--config_name", default=None)
    parser.add_argument("-w", "--wandb_log_dir", default=None)
    parser.add_argument("-n", "--num_training_iterations", default=2500, type=int)

    return parser.parse_args(), parser


def get_config_module(args):
    if args.config_name is None:
        return None

    git_tracked_conf_path, git_untracked_conf_path = (
        "publish.configs",
        "publish.configs.untracked_configs",
    )

    try:
        module_name = ".".join([git_tracked_conf_path, args.config_name])
        return importlib.import_module(module_name)
    except ImportError:
        module_name = ".".join([git_untracked_conf_path, args.config_name])
        return importlib.import_module(module_name)


def get_config(config_module):
    try:
        return config_module.CONFIG
    except AttributeError:
        raise AttributeError(
            "The config in a config file should be stored in the "
            + "variable CONFIG, otherwise it cannot be found."
        )


def get_scheduler(config_module):
    return config_module.SCHEDULER_ALGORITHM


def get_search_algorithm(config_module):
    return config_module.SEARCH_ALGORITHM


def get_trainer(config_module):
    return config_module.TRAINER


def get_num_cpus():
    if 'SLURM_JOB_ID' not in os.environ:
        return 1

    uid, slurm_job_id = os.getuid(), os.environ['SLURM_JOB_ID']
    fname = Path(SLURM_JOB_CPUS_FILENAME % (uid, slurm_job_id))

    num_cpus = 0
    with open(fname, 'r') as f:
        line = f.read().replace('\n', '')
        for substr in line.split(','):
            if '-' not in substr:
                num_cpus += 1
                continue

            cpu_nums = list(map(int, substr.split('-')))
            num_cpus += cpu_nums[1] - cpu_nums[0] + 1

    return num_cpus


def main():
    cli_args, parser = parse_args()

    config_module = get_config_module(cli_args)
    if not config_module:
        print('Need config to be specified to run this script!')
        parser.print_help()

        return

    config = get_config(config_module)
    scheduler = get_scheduler(config_module)
    search_algorithm = get_search_algorithm(config_module)
    trainer = get_trainer(config_module)

    ray.init(
        num_gpus=torch.cuda.device_count(),
        num_cpus=get_num_cpus()
    )

    wandb_mixin_config, wandb_callback_config = get_wandb_config(
        cli_args.experiment_name or cli_args.config_name,
        cli_args.entity_name,
        cli_args.wandb_log_dir
    )

    config['wandb'] = wandb_mixin_config
    callbacks = (
        [WandbLoggerCallback(**wandb_callback_config)]
        if wandb_callback_config else []
    )

    tune.run(
        trainer,
        name=cli_args.experiment_name,
        stop={"training_iteration": cli_args.num_training_iterations},
        search_alg=search_algorithm,
        scheduler=scheduler,
        max_failures=0,
        resources_per_trial={
            'gpu': config.get('num_gpus', 0.5),
            'cpu': config.get('num_cpus', 1.0)
        },
        config=config,
        max_concurrent_trials=12,
        num_samples=config.get('num_tune_samples', 1),
        local_dir='~/scratch/ray_results',
        callbacks=callbacks,
    )


if __name__ == "__main__":
    main()
