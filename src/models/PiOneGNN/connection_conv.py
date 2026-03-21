import torch
import torch.nn as nn
from torch_geometric.utils import degree
from torch_scatter import scatter_add

class ConnectionConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_self = nn.Linear(in_dim, out_dim)
        self.W_msg = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, O_edges):
        src = edge_index[0]
        dst = edge_index[1]
        deg = degree(dst, num_nodes=x.size(0)).clamp(min=1.0)

        m_u = self.W_msg(x[src])

        m_u = torch.unsqueeze(m_u, -1)
        m_u = torch.bmm(O_edges, m_u)
        m_u = torch.squeeze(m_u, -1)

        alpha = 1 / torch.sqrt(deg[src] * deg[dst])

        m_u = alpha.unsqueeze(-1) * m_u

        agg = scatter_add(m_u, dst, dim = 0, dim_size= x.size(0))

        h_v = self.W_self(x) + agg                        

        return h_v
