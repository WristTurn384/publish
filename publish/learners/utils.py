#from ray.tune.suggest.repeater import TRIAL_INDEX
from typing import Dict

def get_seed(config: Dict[str, object]) -> int:
    seed = None

    if 'seed' in config:
        seed = config['seed']
#    elif TRIAL_INDEX in config:
#        seed = config[TRIAL_INDEX]

    return seed
