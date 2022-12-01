from publish.buffers import UniformFifoBuffer
from typing import Dict

class NoBuffer(UniformFifoBuffer):
    def __init__(self, config: Dict[str, object]):
        config['capacity'] = (
            config['target_agent_pf_samples_batch_size'] +
            config['direct_sampler_batch_size']
        )

        super().__init__(config)
