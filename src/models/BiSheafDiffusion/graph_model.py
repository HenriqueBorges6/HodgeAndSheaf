"""
BiSheaf Diffusion para classificação de grafos.

Diferença em relação ao node_model: após as camadas de difusão, aplica
global pooling sobre os nós de cada grafo antes do MLP.

Pipeline:
    x (n, node_feat_dim)
    → input_proj: Linear + ELU → (n, d·f)
    → reshape (n, d, f)
    → BiSheafDiffusionLayer × num_layers
    → reshape (n, d·f)
    → global_pool (mean | sum)  → (B, d·f)  onde B = batch size
    → LayerNorm
    → MLP → logits (B, num_classes)

O tensor `batch` (n,) mapeia cada nó ao seu grafo. É o padrão PyG.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool

from .diffusion import BiSheafDiffusionLayer
from .laplacian import compute_reverse_idx


_POOL = {
    'mean': global_mean_pool,
    'sum':  global_add_pool,
}


class BiSheafGraphClassifier(nn.Module):
    """
    Classificador de grafos via difusão de bisfeixe.

    Args:
        node_feat_dim:      features de entrada por nó
        d:                  dimensão do stalk
        hidden_channels:    número de canais f
        num_layers:         camadas de difusão
        num_classes:        classes de saída
        learner_hidden:     dimensão oculta dos learners
        learner_gnn_layers: camadas GNN nos learners
        dropout:            dropout por camada de difusão
        left_weights:       habilita W1
        right_weights:      habilita W2
        use_eps:            habilita resíduo ε aprendível
        norm_eps:           regularização para D^{-1/2}
        pooling:            'mean' ou 'sum'
        mlp_dims:           camadas ocultas do MLP final
    """

    def __init__(
        self,
        node_feat_dim: int,
        d: int,
        hidden_channels: int,
        num_layers: int,
        num_classes: int,
        learner_hidden: int = 64,
        learner_gnn_layers: int = 1,
        dropout: float = 0.0,
        left_weights: bool = True,
        right_weights: bool = True,
        use_eps: bool = True,
        norm_eps: float = 1e-3,
        pooling: str = 'sum',
        mlp_dims: list = None,
        backbone_F: str = 'node_edge_node',
        backbone_C: str = 'node_edge_node',
        map_type_F: str = 'general',
        map_type_C: str = 'general',
        orth_trans: str = 'householder',
        **backbone_kwargs,
    ):
        super().__init__()
        assert pooling in _POOL, f"pooling deve ser 'mean' ou 'sum', recebido: {pooling!r}"

        self.d    = d
        self.f    = hidden_channels
        self.pool = _POOL[pooling]
        in_ch     = d * hidden_channels

        self.input_proj = nn.Sequential(
            nn.Linear(node_feat_dim, in_ch),
            nn.ELU(),
        )

        self.layers = nn.ModuleList([
            BiSheafDiffusionLayer(
                in_ch=in_ch,
                d=d,
                f=hidden_channels,
                learner_hidden=learner_hidden,
                learner_gnn_layers=learner_gnn_layers,
                left_weights=left_weights,
                right_weights=right_weights,
                use_eps=use_eps,
                dropout=dropout,
                norm_eps=norm_eps,
                backbone_F=backbone_F,
                backbone_C=backbone_C,
                map_type_F=map_type_F,
                map_type_C=map_type_C,
                orth_trans=orth_trans,
                **backbone_kwargs,
            )
            for _ in range(num_layers)
        ])

        self.pre_mlp_norm = nn.LayerNorm(in_ch)

        dims   = [in_ch] + (mlp_dims or []) + [num_classes]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        reverse_idx: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x:           (n, node_feat_dim)
            edge_index:  (2, m) — arestas dentro do batch (índices globais)
            batch:       (n,) — grafo de cada nó, gerado pelo DataLoader do PyG
            reverse_idx: (m,) pré-computado; calculado automaticamente se None

        Retorna logits (B, num_classes).
        """
        if reverse_idx is None:
            reverse_idx = compute_reverse_idx(edge_index)

        n = x.shape[0]
        d, f = self.d, self.f

        x = self.input_proj(x).reshape(n, d, f)

        for layer in self.layers:
            x_flat = x.reshape(n, d * f)
            x = layer(x, x_flat, edge_index, reverse_idx)

        x_pooled = self.pool(x.reshape(n, d * f), batch)   # (B, d·f)

        return self.mlp(self.pre_mlp_norm(x_pooled))
