import torch


def build_incidence(num_nodes: int, edge_index: torch.Tensor) -> torch.Tensor:
    """
    Constroi a matriz de incidencia B1 em R^{n x m}.
    B1[v, e] = -1 se v eh a fonte de e, +1 se v eh o alvo, 0 caso contrario.
    """
    num_edges = edge_index.shape[1]
    arange_e = torch.arange(num_edges, device=edge_index.device)
    B1 = torch.zeros(num_nodes, num_edges, device=edge_index.device)
    B1[edge_index[0], arange_e] = -1.0
    B1[edge_index[1], arange_e] =  1.0
    return B1


def build_line_graph_edge_index(
    num_nodes: int,
    edge_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Constroi o edge_index do grafo de linhas L(G) e os sinais de incidencia.

    Versao vetorizada: A_L = B1^T B1.
    A_L[i, j] = sum_v B1[v,i] * B1[v,j] = sinal de incidencia entre arestas i e j.
    Os zeros fora da diagonal marcam pares sem vertice compartilhado.

    Retorna:
        line_edge_index : (2, E_L) — edge_index do grafo de linhas
        signs           : (E_L,)   — sinal de incidencia de cada aresta
    """
    num_edges = edge_index.shape[1]
    if num_edges == 0:
        return (
            torch.zeros(2, 0, dtype=torch.long, device=edge_index.device),
            torch.zeros(0, device=edge_index.device),
        )

    B1  = build_incidence(num_nodes, edge_index)
    A_L = B1.T @ B1                                    # (m, m)

    mask = A_L != 0
    mask.fill_diagonal_(False)
    line_edge_index = mask.nonzero(as_tuple=False).T.contiguous()   # (2, E_L)
    signs = A_L[line_edge_index[0], line_edge_index[1]]             # (E_L,)

    return line_edge_index, signs