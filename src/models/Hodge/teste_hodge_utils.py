from HodgeGNN import *
from torch_geometric.data import Data
import torch

# Grafo

edge_index = torch.tensor([[0, 0, 1, 1, 2],
                           [1, 2, 2, 0 ,3]])

data = Data(
    edge_index=edge_index,
    edge_attr=torch.randn(5, 4),  # 4 arestas, 4 features cada
    num_nodes=4,
    batch=torch.zeros(4, dtype=torch.long)
)

# print(build_incidence_matrix(4, edge_index))

# H = hodge_laplacian(4,edge_index)

# print(hodge_laplacian(4, edge_index))

# print(normalize_laplacian(H, "rw"))

model = HodgeGNN(
    in_dim=4,
    hidden_dims=[32, 32],
    out_dim=2,
    normalize='symmetric',
    pooling='mean',
    dropout=0.5,
    residual=False
)

print(model(data))