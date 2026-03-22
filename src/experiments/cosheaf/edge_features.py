"""
Features artificiais de aresta para datasets sem edge_attr.

Modos disponíveis:
    'structural'   — 4 features de grau e vizinhos em comum
    'node_derived' — 2*F features derivadas dos nós endpoint
    'all'          — concatenação das duas famílias acima

Uso em load_data:
    from src.experiments.cosheaf.edge_features import augment_edge_features
    data.edge_attr = augment_edge_features(data, mode='all')
"""

import torch
from torch_geometric.utils import degree as pyg_degree


# ---------------------------------------------------------------------------
# Família 1: features estruturais (sempre disponíveis)
# ---------------------------------------------------------------------------

def structural_edge_features(data: object) -> torch.Tensor:
    """
    4 features por aresta baseadas em grau e vizinhança:
        [deg_u_norm, deg_v_norm, deg_avg_norm, common_neighbors_norm]

    - deg_u_norm, deg_v_norm : grau de cada endpoint normalizado pelo grau máximo do grafo
    - deg_avg_norm           : média dos dois graus normalizada
    - common_neighbors_norm  : |N(u) ∩ N(v)| normalizado — mede participação em triângulos

    Implementação vetorizada: vizinhos em comum = (A²)[u,v], sem loops Python.
    """
    edge_index = data.edge_index
    src, tgt   = edge_index
    num_nodes  = int(edge_index.max().item()) + 1 if edge_index.numel() > 0 else 0
    m          = edge_index.shape[1]

    if m == 0:
        return torch.zeros(0, 4)

    deg     = pyg_degree(tgt, num_nodes=num_nodes).float()  # (N,)
    max_deg = deg.max().clamp(min=1.0)

    deg_u = deg[src] / max_deg          # (m,)
    deg_v = deg[tgt] / max_deg          # (m,)
    deg_avg = (deg_u + deg_v) / 2.0     # (m,)

    # Vizinhos em comum: A²[u,v]  (A = adjacência binária, simétrica)
    # Constrói A densa — ok para grafos moleculares pequenos (N ~ 30-50)
    A = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
    A[src, tgt] = 1.0
    A[tgt, src] = 1.0
    A2      = A @ A                                    # (N, N)
    common  = A2[src, tgt]                             # (m,)
    max_com = common.max().clamp(min=1.0)
    common  = common / max_com

    return torch.stack([deg_u, deg_v, deg_avg, common], dim=1)  # (m, 4)


# ---------------------------------------------------------------------------
# Família 2: features derivadas dos nós endpoint
# ---------------------------------------------------------------------------

def node_derived_edge_features(data: object) -> torch.Tensor | None:
    """
    2*F features por aresta a partir das features dos nós endpoint:
        [mean, abs_diff]

    - mean     = (x_u + x_v) / 2      — informação compartilhada
    - abs_diff = |x_u - x_v|          — divergência local

    Retorna None se data.x não estiver disponível.
    """
    if data.x is None:
        return None

    src, tgt = data.edge_index
    x = data.x.float()

    mean_feat = (x[src] + x[tgt]) / 2.0   # (m, F)
    diff_feat = (x[src] - x[tgt]).abs()   # (m, F)

    return torch.cat([mean_feat, diff_feat], dim=1)  # (m, 2F)


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

def augment_edge_features(data: object, mode: str = "all") -> torch.Tensor:
    """
    Computa features artificiais de aresta para um grafo.

    Parâmetros
    ----------
    data : objeto PyG com .edge_index e opcionalmente .x
    mode : 'structural' | 'node_derived' | 'all'

    Retorna
    -------
    Tensor (m, D) com D dependendo do modo:
        - 'structural'   : D = 4
        - 'node_derived' : D = 2*F  (exige data.x)
        - 'all'          : D = 4 + 2*F  (ou 4 se data.x ausente)
    """
    m = data.edge_index.shape[1]
    parts: list[torch.Tensor] = []

    if mode in ("structural", "all"):
        parts.append(structural_edge_features(data))

    if mode in ("node_derived", "all"):
        nd = node_derived_edge_features(data)
        if nd is not None:
            parts.append(nd)
        elif mode == "node_derived":
            # fallback se não há features de nó
            parts.append(torch.ones(m, 1, device=data.edge_index.device))

    if not parts:
        return torch.ones(m, 1, device=data.edge_index.device)

    return torch.cat(parts, dim=1)
