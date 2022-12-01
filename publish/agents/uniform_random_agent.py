from publish.agents import BaseAgent
from torchtyping import TensorType
from typing import Dict
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import torch

class UniformRandomActionAgent(BaseAgent):
    def _setup(self, config: Dict[str, object]):
        self.action_dim = config['action_dim']
        self.env = config['env']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def act(
        self,
        observations: TensorType['batch_size', 'obs_dim']
    ) -> TensorType['batch_size']:
        unnormalized_probs = torch.ones(
            (len(observations), self.action_dim),
            device=self.device
        )

        invalid_action_mask = self.env.get_invalid_action_mask(observations)
        sampling_probs = F.softmax(~invalid_action_mask * unnormalized_probs, dim=1)

        return Categorical(probs=sampling_probs).sample()

    def update(self, update_infos: Dict[str, object]) -> Dict[str, object]:
        # The return value of the update method is tracked as the loss
        # of the behavior agent. Since the loss doesn't really make sense
        # for a random agent, we just return a value of 0.
        return torch.tensor(0.0)

    def load(self):
        pass

    def save(self):
        pass
