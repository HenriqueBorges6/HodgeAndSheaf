import torch
import torch.nn as nn

from torch_geometric.utils import degree
from torch_geometric.nn import MessagePassing

from .cosheaf_learner import CoSheafLearner

class CoSheafConv(MessagePassing):
    def __init__(self, in_channels, out_channels, d, hidden_dims):
        super().__init__(aggr= 'add', node_dim=0)
        self.d = d
        self.learner = CoSheafLearner(in_channels, hidden_dims, d)
        self.linear = nn.Linear(in_channels, out_channels)
                                
    def forward(self, x, edge_index, line_edge_index, signs):
        restriction_maps = self.learner(x,line_edge_index)
        x = self.linear(x)
        x = x.unsqueeze(1).expand(-1, self.d, -1).contiguous()
        num_edges = edge_index.shape[1]
        D = degree(line_edge_index[1], num_nodes=num_edges)  # (m,)
        D = D.unsqueeze(-1).unsqueeze(-1)
        agg = self.propagate(line_edge_index, x=x, maps=restriction_maps, signs=signs)
        return D * x - agg

    def message(self, x_j, maps_i, maps_j, signs):
        # x_j: (m, d, out_channels)
        # maps_j: (m, d, d) — F̂_e'
        # maps_i: (m, d, d) — F̂_e
        # signs: (m,)       — [v:e][v:e']
        out = maps_j @ x_j                                    # (m, d, out_channels)
        out = maps_i.transpose(-2, -1) @ out                  # (m, d, out_channels)
        return signs.unsqueeze(-1).unsqueeze(-1) * out        # (m, d, out_channels)
