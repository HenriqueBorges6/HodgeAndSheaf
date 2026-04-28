"""
CoSheafConvDirect — difusao do cofeixe sem construir o grafo de linhas.

Reformula Delta_1 x = B1^T B_tilde_1 x em dois passos:
    Passo 1 (aresta -> no):   agg_v = sum_{e ni v} [v:e] F_hat_e x_e
    Passo 2 (no -> aresta):   out_e = D_e x_e - F_hat_e^T ([src:e] agg_src + [tgt:e] agg_tgt)

Complexidade O(m d^2 C) ao inves de O(E_L d^2 C), onde E_L >> m tipicamente.
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cosheaf_learner import Orthogonal, HouseholderOrthogonal

_ORTH_MAP = {"cayley": Orthogonal, "householder": HouseholderOrthogonal}


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Learner direto: usa aresta->no->aresta ao inves de GNN em L(G)
# ---------------------------------------------------------------------------

class EdgeNodeEdgeLayer(nn.Module):
    """
    Camada de message-passing aresta->no->aresta sobre o grafo original G.
    Equivale a uma camada de agregacao em L(G), mas sem materializa-lo.

    Para cada aresta e=(u,v):
        node_agg_u = mean(features de arestas incidentes a u)
        node_agg_v = mean(features de arestas incidentes a v)
        out_e = W_neigh * (node_agg_u + node_agg_v) + W_self * x_e
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_self = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        # x: (m, C)  edge_index: (2, m)
        src, tgt = edge_index
        m, C = x.shape

        # Agregar arestas nos endpoints
        node_agg = torch.zeros(num_nodes, C, device=x.device, dtype=x.dtype)
        idx_src = src.unsqueeze(1).expand(-1, C).contiguous()
        idx_tgt = tgt.unsqueeze(1).expand(-1, C).contiguous()
        node_agg.scatter_add_(0, idx_src, x)
        node_agg.scatter_add_(0, idx_tgt, x)

        # Normalizar pelo grau
        deg = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
        deg.scatter_add_(0, src, torch.ones(m, device=x.device, dtype=x.dtype))
        deg.scatter_add_(0, tgt, torch.ones(m, device=x.device, dtype=x.dtype))
        node_agg = node_agg / deg.clamp(min=1).unsqueeze(1)

        # Coletar de volta para arestas
        neigh = node_agg[src] + node_agg[tgt]  # (m, C)
        return self.lin_neigh(neigh) + self.lin_self(x)


class CoSheafLearnerDirect(nn.Module):
    """
    Aprende mapas ortogonais F_hat_e via aresta->no->aresta (sem line graph).
    Arquitetura: Linear -> EdgeNodeEdgeLayer * N -> out -> Orthogonal
    """

    def __init__(self, in_channels: int, hidden_dims: list[int], d: int,
                 orth_method: str = 'cayley'):
        super().__init__()
        if orth_method not in _ORTH_MAP:
            raise ValueError(f"orth_method deve ser 'cayley' ou 'householder', recebeu '{orth_method}'")
        self.p = d * (d - 1) // 2
        self.linear = nn.Linear(in_channels, hidden_dims[0])
        self.convs = nn.ModuleList([
            EdgeNodeEdgeLayer(hidden_dims[i], hidden_dims[i + 1])
            for i in range(len(hidden_dims) - 1)
        ])
        self.conv_out = EdgeNodeEdgeLayer(hidden_dims[-1], self.p)
        self.orth = _ORTH_MAP[orth_method](d)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        # x: (m, in_channels) -> (m, d, d)
        x = F.relu(self.linear(x))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, num_nodes))
        x = self.conv_out(x, edge_index, num_nodes)  # (m, p)
        return self.orth(x)                            # (m, d, d)


# ---------------------------------------------------------------------------
# Convolucao direta: Delta_1 via aresta->no->aresta
# ---------------------------------------------------------------------------

class CoSheafConvDirect(nn.Module):
    """
    Convolucao de co-feixe plano SEM grafo de linhas.

    Reformula Delta_1 x como:
        1. Fx_e = F_hat_e @ x_e                              (m matmuls)
        2. agg_v = sum_{e ni v} [v:e] Fx_e                   (scatter sobre nos)
        3. gather_e = [src:e] agg_src + [tgt:e] agg_tgt      (index)
        4. out_e = D_e x_e - F_hat_e^T @ gather_e            (m matmuls)

    onde D_e = deg(src(e)) + deg(tgt(e)) no grafo original.
    """

    def __init__(self, in_channels: int, out_channels: int, d: int,
                 learner_hidden: list[int], orth_method: str = 'cayley'):
        super().__init__()
        self.d = d
        self.learner = CoSheafLearnerDirect(in_channels, learner_hidden, d, orth_method)
        # Stalk embed fundido: 1 Linear ao inves de d separados (otimizacao C1)
        self.stalk_linear = nn.Linear(in_channels, d * out_channels)
        self.out_channels = out_channels
        self._profile = False
        self._timings = {}

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        # x: (m, in_ch), edge_index: (2, m) -> retorna (m, d, out_ch)
        p = self._profile
        dev = x.device
        src, tgt = edge_index
        m = x.shape[0]
        d, C = self.d, self.out_channels

        # --- Learn maps ---
        if p: _sync(dev); t0 = time.time()
        maps = self.learner(x, edge_index, num_nodes)  # (m, d, d)
        if p: _sync(dev); self._timings['learn_maps'] = time.time() - t0

        # --- Stalk embed (fundido) ---
        if p: t0 = time.time()
        x = self.stalk_linear(x).reshape(m, d, C)     # (m, d, C)
        if p: _sync(dev); self._timings['stalk_embed'] = time.time() - t0

        # --- Difusao direta: aresta -> no -> aresta ---
        if p: t0 = time.time()

        # Passo 1: F_hat_e @ x_e para cada aresta
        Fx = torch.bmm(maps, x)                        # (m, d, C)

        # Passo 2: agregar nos vertices (B_tilde_1 x)
        # B1[src, e] = -1, B1[tgt, e] = +1
        node_agg = torch.zeros(num_nodes, d, C, device=dev, dtype=x.dtype)
        idx_src = src.view(-1, 1, 1).expand(-1, d, C).contiguous()
        idx_tgt = tgt.view(-1, 1, 1).expand(-1, d, C).contiguous()
        node_agg.scatter_add_(0, idx_src, -Fx)          # sign(src) = -1
        node_agg.scatter_add_(0, idx_tgt,  Fx)          # sign(tgt) = +1

        # Passo 3: coletar de volta para arestas (B1^T)
        # gather_e = [src:e] * agg[src] + [tgt:e] * agg[tgt]
        #          = -agg[src] + agg[tgt]
        gather = -node_agg[src] + node_agg[tgt]         # (m, d, C)

        # Passo 4: F_hat_e^T @ gather
        result = torch.bmm(maps.transpose(-2, -1), gather)  # (m, d, C)

        # Grau: D_e = deg(src(e)) + deg(tgt(e))
        deg = torch.zeros(num_nodes, device=dev, dtype=x.dtype)
        deg.scatter_add_(0, src, torch.ones(m, device=dev, dtype=x.dtype))
        deg.scatter_add_(0, tgt, torch.ones(m, device=dev, dtype=x.dtype))
        D_e = (deg[src] + deg[tgt]).view(-1, 1, 1)     # (m, 1, 1)

        if p: _sync(dev); self._timings['diffusion'] = time.time() - t0

        return D_e * x - result
