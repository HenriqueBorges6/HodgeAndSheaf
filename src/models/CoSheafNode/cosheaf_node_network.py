import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree as pyg_degree

from src.models.CoSheafNN.cosheaf_conv import CoSheafConv, _sync
from src.models.CoSheafNN.cosheaf_conv_direct import CoSheafConvDirect
from src.models.CoSheafNN.utils import build_line_graph_edge_index


class CoSheafNodeNetwork(nn.Module):
    """
    Rede GNN baseada em co-feixes planos para classificacao de nos.

    Pipeline:
        x_node -> embed (no->aresta) -> CoSheafConv * N -> readout (aresta->no) -> classificador

    conv_mode:
        'linegraph' (default) — difusao via grafo de linhas L(G) (versao original)
        'direct' — difusao aresta->no->aresta sem construir L(G) (muito mais rapido)
    """

    def __init__(
        self,
        node_feat_dim: int,
        d: int,
        hidden_dims: list[int],
        num_classes: int,
        learner_hidden: list[int],
        mlp_dims: list[int] | None = None,
        dropout: float = 0.0,
        backbone: str = 'sage',
        orth_method: str = 'cayley',
        conv_mode: str = 'direct',
    ):
        super().__init__()
        self.d = d
        self.conv_mode = conv_mode

        # Embedding: concatena features dos endpoints de cada aresta
        self.embed = nn.Linear(2 * node_feat_dim, hidden_dims[0])

        # Camadas de convolucao
        if conv_mode == 'direct':
            self.convs = nn.ModuleList([
                CoSheafConvDirect(hidden_dims[i], hidden_dims[i + 1], d, learner_hidden, orth_method)
                for i in range(len(hidden_dims) - 1)
            ])
        elif conv_mode == 'linegraph':
            self.convs = nn.ModuleList([
                CoSheafConv(hidden_dims[i], hidden_dims[i + 1], d, learner_hidden, backbone, orth_method)
                for i in range(len(hidden_dims) - 1)
            ])
        else:
            raise ValueError(f"conv_mode deve ser 'direct' ou 'linegraph', recebeu '{conv_mode}'")

        # Projecao para colapsar stalk
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

        self._profile = False
        self._timings = {}

    def set_profiling(self, enabled: bool = True):
        self._profile = enabled
        for conv in self.convs:
            conv._profile = enabled

    def get_profile(self) -> dict:
        return dict(self._timings)

    def forward(
        self,
        x_node: torch.Tensor,
        edge_index: torch.Tensor,
        line_edge_index: torch.Tensor | None = None,
        line_signs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        p   = self._profile
        dev = x_node.device
        src, tgt = edge_index
        num_nodes = x_node.shape[0]

        # --- Embed (no -> aresta) ---
        if p: _sync(dev); t0 = time.time()
        x = torch.cat([x_node[src], x_node[tgt]], dim=1)
        x = F.relu(self.embed(x))
        if p: _sync(dev); self._timings['embed'] = time.time() - t0

        # --- CoSheafConv layers ---
        if self.conv_mode == 'linegraph' and line_edge_index is None:
            line_edge_index, line_signs = build_line_graph_edge_index(num_nodes, edge_index)

        for i, (conv, proj) in enumerate(zip(self.convs, self.stalk_projections)):
            if p: _sync(dev); t0 = time.time()

            if self.conv_mode == 'direct':
                x = conv(x, edge_index, num_nodes)
            else:
                x = conv(x, edge_index, line_edge_index, line_signs)

            if p:
                _sync(dev)
                conv_time = time.time() - t0
                self._timings[f'conv{i}'] = conv_time
                self._timings[f'conv{i}_learn_maps'] = conv._timings.get('learn_maps', 0)
                self._timings[f'conv{i}_stalk_embed'] = conv._timings.get('stalk_embed', 0)
                self._timings[f'conv{i}_diffusion'] = conv._timings.get('diffusion', 0)

            if p: t0 = time.time()
            m_edges = x.shape[0]
            x = proj(x.reshape(m_edges, -1))
            x = F.relu(x)
            if p: _sync(dev); self._timings[f'conv{i}_stalk_proj'] = time.time() - t0

        # --- Readout (aresta -> no) ---
        if p: _sync(dev); t0 = time.time()
        h = x.shape[1]
        node_feat = torch.zeros(num_nodes, h, device=dev, dtype=x.dtype)
        node_feat.scatter_add_(0, src.unsqueeze(1).expand(-1, h).contiguous(), x)
        node_feat.scatter_add_(0, tgt.unsqueeze(1).expand(-1, h).contiguous(), x)
        deg = pyg_degree(src, num_nodes) + pyg_degree(tgt, num_nodes)
        node_feat = node_feat / deg.clamp(min=1).unsqueeze(1)
        if p: _sync(dev); self._timings['readout'] = time.time() - t0

        # --- Classifier ---
        if p: t0 = time.time()
        out = self.classifier(node_feat)
        if p: _sync(dev); self._timings['classifier'] = time.time() - t0

        return out

