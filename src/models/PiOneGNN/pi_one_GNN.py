import torch
import torch.nn as nn
import torch.nn.functional as F
from .edge_encoder import EdgeEncoder
from .connection_conv import ConnectionConv
from .wilson_readout import WilsonReadout

_ACTIVATIONS = {
    'relu':    nn.ReLU,
    'sigmoid': nn.Sigmoid,
    'tanh':    nn.Tanh,
    'gelu':    nn.GELU,
    'elu':     nn.ELU,
}

def _build_mlp(in_dim, mlp_dims, out_dim):
    """
    mlp_dims: lista de tuplas (hidden_dim, activation_str)
    Exemplo: [(64, 'relu'), (32, 'relu')]
    Se vazia, retorna um único Linear(in_dim, out_dim).
    """
    layers = []
    current_dim = in_dim
    for hidden_dim, act_name in mlp_dims:
        layers.append(nn.Linear(current_dim, hidden_dim))
        act_cls = _ACTIVATIONS.get(act_name.lower())
        if act_cls is None:
            raise ValueError(f"Ativação desconhecida: '{act_name}'. Opções: {list(_ACTIVATIONS)}")
        layers.append(act_cls())
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, out_dim))
    return nn.Sequential(*layers)


class PiOneGNN(nn.Module):
    def __init__(self, in_node_dim, in_edge_dim, d, num_layes, out_dim, num_powers, mlp_dims=None):
        super().__init__()
        self.node_embed = nn.Linear(in_node_dim, d)
        self.edge_encoder = EdgeEncoder(in_edge_dim, d)
        self.convs = nn.ModuleList([
            ConnectionConv(d, d)
            for _ in range(num_layes)
        ])
        self.readout = WilsonReadout(num_powers=num_powers)
        self.mlp = _build_mlp(1 + num_powers, mlp_dims or [], out_dim)

    def forward(self, data, cycle_edge_idxs, cycle_flip_masks, cycle_lengths, cycle_batch):
        x = data.x
        edge_attr = data.edge_attr
        edge_index = data.edge_index
        batch = data.batch

        x = self.node_embed(x)
        O_edges = self.edge_encoder(edge_attr)

        for conv in self.convs:
            x = F.relu(conv(x, edge_index, O_edges))

        out, holonomies = self.readout(x, O_edges, cycle_edge_idxs, cycle_flip_masks, cycle_lengths, cycle_batch, batch)

        return self.mlp(out), holonomies
