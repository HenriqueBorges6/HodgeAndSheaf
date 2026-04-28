"""
SheafNN — modelo de feixe celular para node classification.

Empilha camadas FlatBundleConv (mapas ortogonais) ou FlatGenSheafConv (mapas gerais)
sobre o grafo original para classificacao de nos.

Pipeline:
    x_raw (n, node_feat) -> input_proj -> (n, d*f) -> FlatConv x num_layers -> output_proj -> logits

Rode com:
    python -m src.experiments.sheaf_nn.train
"""

import torch
import torch.nn as nn
from src.models.SheafNN.sheaf import FlatBundleConv, FlatGenSheafConv


class SheafNN(nn.Module):
    """
    Rede neural de feixe para node classification.

    Args:
        node_feat_dim: dimensao das features de entrada
        d: dimensao do stalk (espaco de fibra por no)
        hidden_channels: numero de canais ocultos (dimensao f)
        num_layers: numero de camadas de difusao
        num_classes: numero de classes de saida
        conv_type: 'bundle' (mapas ortogonais) ou 'general' (mapas gerais d x d)
        orth_trans: metodo de transformacao ortogonal para conv_type='bundle'
                    'matrix_exp' | 'cayley' | 'householder' | 'euler' (d=2,3 apenas)
        gnn_type: backbone do learner de mapas ('SAGE', 'GCN', 'GAT', 'SumGNN')
        gnn_layers: camadas do GNN backbone
        gnn_hidden: dimensao oculta do GNN backbone
        dropout: probabilidade de dropout
        left_weights: aplicar pesos W1 na dimensao do stalk
        right_weights: aplicar pesos W2 na dimensao dos canais
        use_eps: usar residual epsilon aprendivel
    """

    def __init__(
        self,
        node_feat_dim: int,
        d: int,
        hidden_channels: int,
        num_layers: int,
        num_classes: int,
        conv_type: str = 'bundle',
        orth_trans: str = 'matrix_exp',
        gnn_type: str = 'SAGE',
        gnn_layers: int = 1,
        gnn_hidden: int = 64,
        dropout: float = 0.6,
        left_weights: bool = True,
        right_weights: bool = True,
        use_eps: bool = True,
        mlp_dims: list = None,   # ex: [128, 64] -> Linear(d*f,128)->ReLU->Linear(128,64)->ReLU->Linear(64,num_classes)
    ):
        super().__init__()
        assert conv_type in ('bundle', 'general'), f"conv_type deve ser 'bundle' ou 'general', recebido: {conv_type}"

        self.d = d
        self.f = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout

        # Projecao de entrada: features -> espaco de feixe (n, d*f)
        self.input_proj = nn.Sequential(
            nn.Linear(node_feat_dim, d * hidden_channels),
            nn.ELU(),
        )

        # Camadas de difusao
        conv_cls = FlatBundleConv if conv_type == 'bundle' else FlatGenSheafConv
        conv_kwargs = dict(
            stalk_dimension=d,
            left_weights=left_weights,
            right_weights=right_weights,
            use_eps=use_eps,
            dropout=dropout,
            gnn_type=gnn_type,
            gnn_layers=gnn_layers,
            gnn_hidden=gnn_hidden,
            linear_emb=True,
        )
        if conv_type == 'bundle':
            conv_kwargs['orth_trans'] = orth_trans

        self.convs = nn.ModuleList([
            conv_cls(hidden_channels, hidden_channels, **conv_kwargs)
            for _ in range(num_layers)
        ])

        # Normalizar embeddings do feixe antes da MLP (evita explosao de gradiente)
        self.pre_mlp_norm = nn.LayerNorm(d * hidden_channels)

        # MLP classificador com dims personalizaveis
        # mlp_dims=[128, 64] -> d*f->128->64->num_classes
        # mlp_dims=[] ou None -> d*f->num_classes (linear direto)
        dims = [d * hidden_channels] + (mlp_dims or []) + [num_classes]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:   # nao adiciona ReLU/Dropout na ultima camada
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: (n, node_feat_dim)
        edge_index: (2, m)
        Retorna: (n, num_classes) logits
        """
        # Projetar para espaco de feixe
        x = self.input_proj(x)   # (n, d*f)

        # Difusao por camadas
        for conv in self.convs:
            x, _ = conv(x, edge_index)   # (n, d*f)

        # Normalizar + classificar
        x = self.pre_mlp_norm(x)
        return self.mlp(x)               # (n, num_classes)
