import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_householder import torch_householder_orgqr


def _sparse_adj(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    """Constroi matriz de adjacencia esparsa (n, n) a partir de edge_index."""
    src, tgt = edge_index
    ones = torch.ones(src.shape[0], device=edge_index.device)
    adj = torch.sparse_coo_tensor(
        torch.stack([tgt, src]), ones, (n, n)
    ).coalesce()
    return adj


class SAGELayer(nn.Module):
    """SAGEConv via torch.sparse.mm — usa kernels cusparse nativos do PyTorch."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_self  = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        adj = _sparse_adj(edge_index, n)                     # (n, n) esparsa
        agg = torch.sparse.mm(adj, x)                        # (n, C)
        deg = torch.sparse.sum(adj, dim=1).to_dense()        # (n,)
        deg = deg.clamp(min=1).unsqueeze(1)                   # (n, 1)
        return self.lin_neigh(agg / deg) + self.lin_self(x)


class Orthogonal(nn.Module):
    """
    Parametriza matrizes ortogonais d x d via mapa de Cayley:
        Q = (I - S)(I + S)^{-1}
    onde S e anti-simetrica. Mais rapido que matrix_exp para d pequeno.
    A base anti-simetrica e pre-computada como buffer.
    """

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        idx   = torch.tril_indices(d, d, offset=-1)
        basis = torch.zeros(idx.shape[1], d, d)
        basis[torch.arange(idx.shape[1]), idx[0], idx[1]] =  1.0
        basis[torch.arange(idx.shape[1]), idx[1], idx[0]] = -1.0
        self.register_buffer('basis', basis)              # (p, d, d)
        self.register_buffer('eye',   torch.eye(d))       # (d, d)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        # params: (batch, p)  →  retorna: (batch, d, d)
        S = torch.einsum('bp,pij->bij', params, self.basis)   # (batch, d, d) anti-simetrica
        I = self.eye.unsqueeze(0)                              # (1, d, d)
        # Q = (I - S)(I + S)^{-1}; para S anti-simetrica, comutam: = (I+S)^{-1}(I-S)
        return torch.linalg.solve(I + S, I - S)


class HouseholderOrthogonal(nn.Module):
    """
    Parametriza matrizes ortogonais d x d via reflexoes de Householder.
    Usa torch_householder_orgqr para converter vetores de reflexao em Q.
    """

    def __init__(self, d: int):
        super().__init__()
        self.d = d

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        # params: (batch, p) onde p = d*(d-1)//2  →  retorna: (batch, d, d)
        m = params.shape[0]
        eye = torch.eye(self.d, device=params.device, dtype=params.dtype)
        eye = eye.unsqueeze(0).expand(m, -1, -1).clone()
        idx = torch.tril_indices(self.d, self.d, offset=-1, device=params.device)
        A = torch.zeros(m, self.d, self.d, device=params.device, dtype=params.dtype)
        A[:, idx[0], idx[1]] = params
        A = A + eye
        return torch_householder_orgqr(A)


_ORTH_MAP = {"cayley": Orthogonal, "householder": HouseholderOrthogonal}


class GINLayer(nn.Module):
    """GINConv via torch.sparse.mm — usa kernels cusparse nativos do PyTorch."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        adj = _sparse_adj(edge_index, n)                     # (n, n) esparsa
        agg = torch.sparse.mm(adj, x)                        # (n, C)
        return self.mlp((1 + self.eps) * x + agg)


_BACKBONE = {"sage": SAGELayer, "gin": GINLayer}


class CoSheafLearner(nn.Module):
    """
    Aprende um mapa de restricao ortogonal F̂_e em O(d) para cada aresta,
    usando features das arestas propagadas pelo grafo de linhas L(G).

    Arquitetura: Linear → backbone* → backbone_out → Orthogonal

    backbone: 'sage' (default) | 'gin'
    """

    def __init__(self, in_channels: int, hidden_dims: list[int], d: int,
                 backbone: str = 'sage', orth_method: str = 'cayley'):
        super().__init__()
        if backbone not in _BACKBONE:
            raise ValueError(f"backbone deve ser 'sage' ou 'gin', recebeu '{backbone}'")
        if orth_method not in _ORTH_MAP:
            raise ValueError(f"orth_method deve ser 'cayley' ou 'householder', recebeu '{orth_method}'")
        self.p = d * (d - 1) // 2
        layer_cls = _BACKBONE[backbone]
        self.linear   = nn.Linear(in_channels, hidden_dims[0])
        self.convs    = nn.ModuleList(
            [layer_cls(hidden_dims[i], hidden_dims[i + 1])
             for i in range(len(hidden_dims) - 1)]
        )
        self.conv_out = layer_cls(hidden_dims[-1], self.p)
        self.orth     = _ORTH_MAP[orth_method](d)

    def forward(self, x: torch.Tensor, line_edge_index: torch.Tensor) -> torch.Tensor:
        # x: (m, in_channels)  →  retorna: (m, d, d)
        x = F.relu(self.linear(x))
        for conv in self.convs:
            x = F.relu(conv(x, line_edge_index))
        x = self.conv_out(x, line_edge_index)   # (m, p)
        return self.orth(x)                      # (m, d, d)