from publish.utils import build_mlp, get_device
from torchtyping import TensorType
from typing import Dict
import torch


class RandomNetworkDistillation(torch.nn.Module):
    def __init__(self, config: Dict[str, object]):
        super().__init__()

        device = get_device()
        input_dim = 2 * config['env'].obs_dim
        self.learned_network = build_mlp(
            input_dim,
            config['embedding_dim'],
            config['hidden_layer_dim'],
            config['num_hidden_layers']
        ).to(device=device)

        self.fixed_network = build_mlp(
            input_dim,
            config['embedding_dim'],
            config['hidden_layer_dim'],
            config['num_hidden_layers']
        ).to(device=device)

    def parameters(self):
        return self.learned_network.parameters()

    def forward(
        self,
        update_infos: Dict[str, object]
    ) -> TensorType['batch_size', 'horizon']:
        states, dones = update_infos['states'], update_infos['dones']
        transition_input_embeddings = self._get_transition_input_embeddings(
            states
        )

        learned_embedding = self.learned_network(transition_input_embeddings)
        with torch.no_grad():
            fixed_embedding = self.fixed_network(transition_input_embeddings)

        l2_norms = torch.linalg.vector_norm(
            learned_embedding - fixed_embedding,
            dim=-1
        )

        return (1 - dones) * l2_norms

    def _get_transition_input_embeddings(
        self,
        states: TensorType['batch_size', 'horizon', 'obs_dim']
    ) -> TensorType['batch_size', 'horizon', 'two_times_obs_dim']:
        all_except_final_embeds = torch.cat(
            (states[:, :-1], states[:, 1:]),
            dim=-1
        )

        final_state_embeds = torch.cat(
            (states[:, -1], states[:, -1]),
            dim=-1
        ).unsqueeze(1)

        return torch.cat((all_except_final_embeds, final_state_embeds), dim=1)

    def loss(self, update_infos: Dict[str, object]) -> TensorType[float]:
        l2_norms = self.forward(update_infos).flatten()
        num_nonzero = (~torch.isclose(
            l2_norms,
            torch.zeros(1, device=l2_norms.device)
        )).sum()

        return l2_norms.sum() / num_nonzero
