import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_householder import torch_householder_orgqr

from torch_geometric.nn import SAGEConv

class Orthogonal(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, params):
        # params: (m, p) onde p = d*(d-1)//2
        m = params.shape[0]
        eye = torch.eye(self.d, device=params.device).unsqueeze(0).repeat(m, 1, 1)
        idx = torch.tril_indices(self.d, self.d, offset=-1, device=params.device)
        A = torch.zeros(m, self.d, self.d, device=params.device, dtype=params.dtype)
        A[:, idx[0], idx[1]] = params
        A = A + eye
        return torch_householder_orgqr(A)
    
class CoSheafLearner(nn.Module):
    def __init__(self, in_channels: int, hidden_dims: list[int], d: int):
        super().__init__()
        self.p = d * (d - 1) // 2
        self.linear = nn.Linear(in_channels, hidden_dims[0])
        self.conv = nn.ModuleList(
            [SAGEConv(hidden_dims[i], hidden_dims[i+1])
            for i in range(len(hidden_dims) - 1)]
        )
        self.conv_out = SAGEConv(hidden_dims[-1], self.p)
        self.orth = Orthogonal(d)

    def forward(self, x, line_edge_index):
        # x: (m, in_channels)
        # retorna: (m, d, d)
        print(f"  [L] linear input: {x.shape}")
        x = F.relu(self.linear(x))
        print(f"  [L] after linear: {x.shape}")
        for conv in self.conv:
            x = F.relu(conv(x, line_edge_index))
        print(f"  [L] before conv_out: {x.shape}, line_edge_index: {line_edge_index.shape}")
        x = self.conv_out(x, line_edge_index)
        print(f"  [L] after conv_out: {x.shape}")
        result = self.orth(x)
        print(f"  [L] after orth: {result.shape}")
        return result
