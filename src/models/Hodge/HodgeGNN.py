import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool


# Utils
def build_incidence_matrix(num_nodes, edge_index):
    num_edges = edge_index.shape[1]
    source = edge_index[0]
    target = edge_index[1]
    arangeE = torch.arange(num_edges, device=edge_index.device)
    B1 = torch.zeros(num_nodes, num_edges, device= edge_index.device)
    B1[source,arangeE] = -1
    B1[target,arangeE] = 1
    return B1

def hodge_laplacian(num_nodes, edge_index):
    B1 = build_incidence_matrix(num_nodes, edge_index)
    L1 = B1.T @ B1
    return L1

def normalize_laplacian(Laplacian, method = "none"):
    if method == "none":
        return Laplacian
    
    D = (torch.diagonal(Laplacian).clamp(min= 1e-6))

    if method == "symmetric":
        return D.pow(-0.5).unsqueeze(1) * Laplacian * D.pow(-0.5).unsqueeze(0)
    if method == "rw":
        return D.pow(-1).unsqueeze(1) * Laplacian

    raise ValueError(f"method deve ser 'none', 'symmetric' ou 'rw'. Recebido '{method}' ")


class HodgeConv(nn.Module):
    def __init__(self, in_dim, out_dim, bias = True, residual = False):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim)
        if residual:
            if in_dim == out_dim:
                self.residual = nn.Identity()
            if in_dim != out_dim:
                self.residual = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.residual = None

    def forward(self, L1_norm, H):
        out = L1_norm @ H 
        out = self.W(out)
        if self.residual is not None:
            out += self.residual(H)
        return out

class HodgeGNN(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, normalize, pooling, dropout, residual, mlp_dim = 64):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.normalize = normalize
        self.pooling = pooling
        self.dropout = dropout
        # self.residual = residual
        self.mlp_dim = mlp_dim

        self.edge_embed = nn.Linear(in_dim, hidden_dims[0])
        
        self.convs = nn.ModuleList([
            HodgeConv(hidden_dims[i], hidden_dims[i+1], residual= residual)
            for i in range(len(hidden_dims) - 1)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, out_dim)
        )
    
    def forward(self, data):
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        num_nodes = data.num_nodes
        batch = data.batch

        L1 = hodge_laplacian(num_nodes, edge_index)

        L1_norm = normalize_laplacian(L1, self.normalize)

        H = self.edge_embed(edge_attr)
        H = F.relu(H)

        for conv in self.convs:
            H = F.relu(conv(L1_norm, H))
        
        H = F.dropout(H, p = self.dropout, training=self.training)

        edge_batch = batch[edge_index[0]]
        if self.pooling == 'mean':
            H = global_mean_pool(H, edge_batch)
        elif self.pooling == 'sum':
            H = global_add_pool(H, edge_batch)
        elif self.pooling == 'max':
            H = global_max_pool(H, edge_batch)
        else:
            raise ValueError(f"pooling inválido: '{self.pooling}'")

        z = self.classifier(H)
        return z