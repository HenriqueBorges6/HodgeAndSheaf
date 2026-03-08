import torch
import torch.nn as nn
import torch.nn.functional as F
from .edge_encoder import EdgeEncoder
from .connection_conv import ConnectionConv
from .wilson_readout import WilsonReadout

class PiOneGNN(nn.Module):
    def __init__(self, in_node_dim, in_edge_dim, d, num_layes, out_dim, num_powers):
        super().__init__()
        self.node_embed = nn.Linear(in_node_dim, d)
        self.edge_encoder = EdgeEncoder(in_edge_dim, d)
        self.convs = nn.ModuleList([
            ConnectionConv(d,d)
            for i in range(num_layes)
        ])
        self.readout = WilsonReadout(num_powers = num_powers)
        
        self.mlp = nn.Linear(1 + num_powers, out_dim)

    def forward(self, data, cycles, edge_map, cycle_batch):
        x = data.x
        edge_attr = data.edge_attr
        edge_index = data.edge_index
        batch = data.batch

        x = self.node_embed(x)
        O_edges = self.edge_encoder(edge_attr)

        for conv in self.convs:
            x = F.relu(conv(x, edge_index, O_edges))

        out, holonomies = self.readout(x, O_edges, cycles, edge_map, batch, cycle_batch)

        return self.mlp(out), holonomies # O_edges
