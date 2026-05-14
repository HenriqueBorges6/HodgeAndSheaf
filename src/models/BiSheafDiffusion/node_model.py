"""
BiSheaf Diffusion para classificação de nós.

Pipeline:
    x (n, node_feat_dim)
    → input_proj: Linear + ELU → (n, d·f)
    → reshape (n, d, f)
    → BiSheafDiffusionLayer × num_layers
    → reshape (n, d·f)
    → LayerNorm
    → MLP → logits (n, num_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion import BiSheafDiffusionLayer
from .laplacian import compute_reverse_idx


class BiSheafNodeClassifier(nn.Module):
    """
    Classificador de nós via difusão de bisfeixe.

    Args:
        node_feat_dim:      features de entrada por nó
        d:                  dimensão do stalk
        hidden_channels:    número de canais f (representação interna: d·f por nó)
        num_layers:         camadas de difusão
        num_classes:        classes de saída
        learner_hidden:     dimensão oculta dos learners de mapa
        learner_gnn_layers: camadas GNN nos learners
        dropout:            dropout aplicado por camada de difusão
        left_weights:       habilita W1 (mistura dimensões do stalk)
        right_weights:      habilita W2 (mistura canais)
        use_eps:            habilita resíduo ε aprendível
        norm_eps:           regularização para D^{-1/2}
        mlp_dims:           camadas ocultas do MLP final, ex: [64] ou [128, 64]
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
        mlp_dims: list = None,
        backbone_F: str = 'node_edge_node',
        backbone_C: str = 'node_edge_node',
        map_type_F: str = 'general',
        map_type_C: str = 'general',
        orth_trans: str = 'householder',
        sheaf_act:  str = 'id',
        inter_layer_norm: bool = False,
        **backbone_kwargs,
    ):
        super().__init__()
        self.d   = d
        self.f   = hidden_channels
        in_ch    = d * hidden_channels

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
                sheaf_act=sheaf_act,
                **backbone_kwargs,
            )
            for _ in range(num_layers)
        ])

        self.inter_norms = (
            nn.ModuleList([nn.LayerNorm(in_ch) for _ in range(num_layers - 1)])
            if inter_layer_norm and num_layers > 1 else None
        )

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
        reverse_idx: torch.Tensor = None,
        return_internals: bool = False,
    ) -> torch.Tensor | tuple:
        """
        Args:
            x:                (n, node_feat_dim)
            edge_index:       (2, m)
            reverse_idx:      (m,) pré-computado; calculado automaticamente se None
            return_internals: se True, retorna (logits, internals) onde internals é dict com:
                              'reps'   → list[(n, d*f)] por camada (após input_proj + após cada layer)
                              'maps_F' → list[(m, d, d)] por camada
                              'maps_C' → list[(m, d, d)] por camada

        Retorna logits (n, num_classes), ou (logits, internals) se return_internals=True.
        """
        if reverse_idx is None:
            reverse_idx = compute_reverse_idx(edge_index)

        n = x.shape[0]
        d, f = self.d, self.f

        x = self.input_proj(x).reshape(n, d, f)

        reps    = [x.reshape(n, d * f).detach()] if return_internals else None
        maps_Fs = [] if return_internals else None
        maps_Cs = [] if return_internals else None

        for i, layer in enumerate(self.layers):
            x_flat = x.reshape(n, d * f)
            if return_internals:
                x, mF, mC = layer(x, x_flat, edge_index, reverse_idx, return_maps=True)
                maps_Fs.append(mF.detach())
                maps_Cs.append(mC.detach())
            else:
                x = layer(x, x_flat, edge_index, reverse_idx)
            if self.inter_norms is not None and i < len(self.inter_norms):
                x = self.inter_norms[i](x.reshape(n, d * f)).reshape(n, d, f)
            if return_internals:
                reps.append(x.reshape(n, d * f).detach())

        logits = self.mlp(self.pre_mlp_norm(x.reshape(n, d * f)))

        if return_internals:
            return logits, {"reps": reps, "maps_F": maps_Fs, "maps_C": maps_Cs}
        return logits
