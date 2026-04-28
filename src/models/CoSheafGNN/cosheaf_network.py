import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_add_pool

from .cosheaf_conv import CoSheafConv
from .utils import build_line_graph_edge_index

class CoSheafNetwork(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, d , hidden_dims, num_classes, learner_hidden):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.d = d 
        self.learner_hidden = learner_hidden
        self.linear = nn.Linear(edge_feat_dim + node_feat_dim, hidden_dims[0])
        self.convs = nn.ModuleList([
            CoSheafConv(hidden_dims[i], hidden_dims[i+1], self.d, self.learner_hidden) 
            for i in range(len(hidden_dims) - 1)
        ])
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x_node, x_edge, edge_index, batch=None):
        src, tgt = edge_index
        node_agg = (x_node[src] + x_node[tgt]) / 2.0   # (m, node_feat_dim)
        x = torch.cat([x_edge, node_agg], dim=1)         # (m, edge_feat_dim + node_feat_dim)
        x = F.relu(self.linear(x))                       # (m, hidden_dims[0])

        num_nodes = x_node.shape[0]
        line_edge_index, signs = build_line_graph_edge_index(num_nodes, edge_index)

        for conv in self.convs:
            x = F.relu(conv(x, edge_index, line_edge_index, signs))
            x = x.mean(dim=1)   # (m, d, h) → (m, h)

        x = global_add_pool(x, batch)

        z = self.classifier(x)
        return z


