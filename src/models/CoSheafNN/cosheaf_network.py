import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool

from .cosheaf_conv import CoSheafConv
from .utils import build_line_graph_edge_index


class CoSheafNetwork(nn.Module):
    """
    Rede GNN baseada em co-feixes planos para classificacao de grafos.

    Pipeline:
        (x_edge, x_node) → embed → CoSheafConv * N → pool → classificador
    """

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        d: int,
        hidden_dims: list[int],
        num_classes: int,
        learner_hidden: list[int],
        mlp_dims: list[int] | None = None,
        dropout: float = 0.0,
        backbone: str = 'sage',
        orth_method: str = 'cayley',
    ):
        super().__init__()
        self.d = d
        # Embedding inicial: combina features de aresta com media dos nos
        self.embed = nn.Linear(edge_feat_dim + node_feat_dim, hidden_dims[0])
        # Camadas de convolucao sobre co-feixes + projecao para colapsar stalk
        self.convs = nn.ModuleList([
            CoSheafConv(hidden_dims[i], hidden_dims[i + 1], d, learner_hidden, backbone, orth_method)
            for i in range(len(hidden_dims) - 1)
        ])
        self.stalk_projections = nn.ModuleList([
            nn.Linear(d * hidden_dims[i + 1], hidden_dims[i + 1])
            for i in range(len(hidden_dims) - 1)
        ])
        # MLP classificador configuravel
        in_dim = hidden_dims[-1]
        layers: list[nn.Module] = []
        for h in (mlp_dims or []):
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(
        self,
        x_node: torch.Tensor,
        x_edge: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
        line_edge_index: torch.Tensor | None = None,
        line_signs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        src, tgt = edge_index

        # Combina features de aresta com a media dos nos endpoints
        node_agg = (x_node[src] + x_node[tgt]) / 2.0     # (m, node_feat_dim)
        x = torch.cat([x_edge, node_agg], dim=1)           # (m, edge_feat_dim + node_feat_dim)
        x = F.relu(self.embed(x))                          # (m, hidden_dims[0])

        # Grafo de linhas: precomputado (preferido) ou calculado on-the-fly (fallback)
        if line_edge_index is None:
            num_nodes = x_node.shape[0]
            line_edge_index, line_signs = build_line_graph_edge_index(num_nodes, edge_index)

        for conv, proj in zip(self.convs, self.stalk_projections):
            x = conv(x, edge_index, line_edge_index, line_signs)         # (m, d, h)
            m_edges = x.shape[0]
            x = proj(x.reshape(m_edges, -1))                             # (m, d*h) → (m, h)
            x = F.relu(x)

        # batch e indexado por nos; precisamos de um batch indexado por arestas
        # size e passado explicitamente para cobrir grafos sem arestas no batch
        if batch is not None:
            num_graphs = int(batch.max().item()) + 1
            edge_batch = batch[src]
        else:
            num_graphs = 1
            edge_batch = None
        x = global_add_pool(x, edge_batch, size=num_graphs)   # (graphs, h)
        return self.classifier(x)                              # (graphs, num_classes)
