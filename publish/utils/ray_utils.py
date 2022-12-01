from ray.tune.integration.wandb import WandbLoggerCallback
from pathlib import Path
from typing import Dict, Tuple
import copy
import ray
import os

REPO_NAME = 'publish'
WANDB_PROJECT = "WANDB_PROJECT"
WANDB_API_KEY_FILE = "WANDB_API_KEY_FILE"

def get_wandb_config(
    group_name: str,
    entity_name: str,
    log_dir: str = None
) -> Tuple[Dict[str, object], Dict[str, object]]:
    if WANDB_API_KEY_FILE not in os.environ:
        print(
            (
                "\n\n\nWARNING: Did not find environment variable %s -- "
                + "will not log runs to wandb\n\n\n"
            )
            % WANDB_API_KEY_FILE
        )

        return {}, {}

    mixin_config = {
        'project': os.environ.get(WANDB_PROJECT, "publish"),
        'group': group_name,
        'entity': entity_name,
        'api_key_file': os.environ[WANDB_API_KEY_FILE],
        'dir': _get_wandb_log_dir(log_dir),
    }

    callback_config = copy.deepcopy(mixin_config)
    callback_config['log_config'] = True

    return mixin_config, callback_config

def _get_wandb_log_dir(log_dir: str = None) -> str:
    if log_dir:
        return log_dir

    wandb_scratch_dir = Path.home() / 'scratch/wandb'
    if not wandb_scratch_dir.exists():
        wandb_scratch_dir.mkdir()

    log_dir = wandb_scratch_dir / REPO_NAME
    if not log_dir.exists():
        log_dir.mkdir()

    return log_dir.as_posix()
