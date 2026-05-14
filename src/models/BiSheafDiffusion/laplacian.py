"""
Operações do Laplaciano de Bisfeixe.

Implementa as três peças matemáticas centrais:

  1. sym_invsqrt(M)         — M^{-1/2} por decomposição espectral (batch)

  2. degree_F(F_tgt, ...)   — D^F_v = Σ_{e∋v} F_{v⊴e}^T F_{v⊴e}
     degree_C(C_tgt, ...)   — D^C_v = Σ_{e∋v} C_{v⊴e} C_{v⊴e}^T

     A diferença entre D^F e D^C reflete a direção de cada mapa:
       - F mapeia nó → aresta:  F^T F é o produto interno no espaço da aresta
       - C mapeia aresta → nó:  C C^T é o produto interno no espaço do nó

  3. bisheaf_lap_action(...) — L(F,C)·X sem normalização

     L(X)_v = [Σ_{u~v} C_{v⊴e} F_{v⊴e}] x_v - Σ_{u~v} C_{v⊴e} F_{u⊴e} x_u
               ^^^ bloco diagonal D_CF_v ^^^     ^^^ agregação off-diagonal ^^^

  4. compute_reverse_idx    — mapeia cada aresta (u,v) para seu índice reverso (v,u)

O Laplaciano normalizado é montado em diffusion.py:
    Δ(F,C) x = D_C^{-1/2} · L(F,C) · D_F^{-1/2} · x
"""

import torch


# ---------------------------------------------------------------------------
# Utilitário: raiz quadrada inversa matricial
# ---------------------------------------------------------------------------

def sym_invsqrt(M: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """
    M^{-1/2} para um batch de matrizes simétricas PSD.

    Estratégia: regularização M ← M + eps·I antes de eigh.
    Com eps=1e-3: todos eigenvalores ≥ eps, e inv_sqrt ≤ 1/√eps ≈ 31 (boundado).

    O backward de eigh envolve 1/(λ_i − λ_j). Com eigenvalores quase iguais
    isso pode explodir, por isso o chamador deve passar M.detach() quando
    D^F ou D^C não precisam de gradiente (análogo ao running_stats do BN).

    Args:
        M:   (n, d, d) simétricas PSD
        eps: regularização identidade

    Retorna (n, d, d).
    """
    d   = M.shape[-1]
    reg = eps * torch.eye(d, device=M.device, dtype=M.dtype).unsqueeze(0)
    lam, Q = torch.linalg.eigh(M + reg)          # lam ≥ eps
    return torch.bmm(Q * lam.rsqrt().unsqueeze(-2), Q.transpose(-2, -1))


# ---------------------------------------------------------------------------
# Matrizes de grau
# ---------------------------------------------------------------------------

def degree_F(maps_F_tgt: torch.Tensor, tgt: torch.Tensor, n: int) -> torch.Tensor:
    """
    D^F_v = Σ_{e∋v} F_{v⊴e}^T F_{v⊴e}    ∈ R^{d×d}, PSD.

    Para cada aresta i=(u,v), maps_F_tgt[i] = F_{v⊴e}. Scatter sobre tgt
    acumula um bloco por aresta incidente, cobrindo cada vizinho exatamente
    uma vez (porque ambas as direções estão em edge_index).

    Args:
        maps_F_tgt: (m, d, d) — F_{tgt[i]⊴e} por aresta
        tgt:        (m,) índices dos nós destino
        n:          número de nós

    Retorna (n, d, d).
    """
    d      = maps_F_tgt.shape[-1]
    blocks = torch.bmm(maps_F_tgt.transpose(-2, -1), maps_F_tgt)   # F^T F, (m, d, d)
    D      = torch.zeros(n, d, d, device=maps_F_tgt.device, dtype=maps_F_tgt.dtype)
    D.scatter_add_(0, tgt.view(-1, 1, 1).expand(-1, d, d), blocks)
    return D


def degree_C(maps_C_tgt: torch.Tensor, tgt: torch.Tensor, n: int) -> torch.Tensor:
    """
    D^C_v = Σ_{e∋v} C_{v⊴e} C_{v⊴e}^T    ∈ R^{d×d}, PSD.

    Atenção: CC^T (não C^T C).  C mapeia aresta → nó, por isso o produto
    externo CC^T é a métrica natural no espaço do nó (analogia: se F é uma
    matriz retangular d_nó × d_aresta, então F^T F e C C^T têm o mesmo shape
    d_nó × d_nó, mas refletem direções opostas de aplicação).

    Args:
        maps_C_tgt: (m, d, d) — C_{tgt[i]⊴e} por aresta
        tgt:        (m,) índices dos nós destino
        n:          número de nós

    Retorna (n, d, d).
    """
    d      = maps_C_tgt.shape[-1]
    blocks = torch.bmm(maps_C_tgt, maps_C_tgt.transpose(-2, -1))   # C C^T, (m, d, d)
    D      = torch.zeros(n, d, d, device=maps_C_tgt.device, dtype=maps_C_tgt.dtype)
    D.scatter_add_(0, tgt.view(-1, 1, 1).expand(-1, d, d), blocks)
    return D


# ---------------------------------------------------------------------------
# Ação do Laplaciano não-normalizado
# ---------------------------------------------------------------------------

def bisheaf_lap_action(
    maps_F_src: torch.Tensor,    # (m, d, d) — F_{src[i]⊴e}: restrição do nó de origem
    maps_F_tgt: torch.Tensor,    # (m, d, d) — F_{tgt[i]⊴e}: restrição do nó de destino
    maps_C_tgt: torch.Tensor,    # (m, d, d) — C_{tgt[i]⊴e}: co-restrição do nó de destino
    x: torch.Tensor,             # (n, d, f) — representações dos nós
    src: torch.Tensor,
    tgt: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """
    Computa L(F,C)·X sem normalização.

    L(X)_v = Σ_{i: tgt[i]=v} C_{v⊴e} ( F_{v⊴e} x_v − F_{u⊴e} x_u )

    Decomposto em dois termos:

      Diagonal:   D_CF_v x_v    onde D_CF_v = Σ C_{v⊴e} F_{v⊴e}  (d×d)
      Off-diag:   agg_v         onde agg_v  = Σ C_{v⊴e} F_{u⊴e} x_{src[i]}

    L(X)_v = D_CF_v x_v − agg_v

    Args:
        maps_F_src: (m, d, d) — F do nó de origem   (= maps_F direto do learner)
        maps_F_tgt: (m, d, d) — F do nó de destino  (= maps_F[reverse_idx])
        maps_C_tgt: (m, d, d) — C do nó de destino  (= maps_C[reverse_idx])
        x:          (n, d, f) representações
        src, tgt:   (m,)
        n:          número de nós

    Retorna (n, d, f).
    """
    d = x.shape[1]
    f = x.shape[2]

    # Termo off-diagonal: C_{v⊴e} F_{u⊴e} x_u  acumulado em v = tgt[i]
    Fx_src = torch.bmm(maps_F_src, x[src])               # F_{u⊴e} x_u,  (m, d, f)
    msg    = torch.bmm(maps_C_tgt, Fx_src)               # C_{v⊴e} F_{u⊴e} x_u, (m, d, f)

    agg = torch.zeros(n, d, f, device=x.device, dtype=x.dtype)
    agg.scatter_add_(0, tgt.view(-1, 1, 1).expand(-1, d, f), msg)

    # Bloco diagonal: D_CF_v = Σ_{i: tgt[i]=v} C_{v⊴e} F_{v⊴e}  (d×d)
    CF_block = torch.bmm(maps_C_tgt, maps_F_tgt)         # (m, d, d)
    D_CF     = torch.zeros(n, d, d, device=x.device, dtype=x.dtype)
    D_CF.scatter_add_(0, tgt.view(-1, 1, 1).expand(-1, d, d), CF_block)

    return torch.bmm(D_CF, x) - agg                      # (n, d, f)


# ---------------------------------------------------------------------------
# Utilitário: índice de aresta reversa
# ---------------------------------------------------------------------------

def compute_reverse_idx(edge_index: torch.Tensor) -> torch.Tensor:
    """
    Para cada aresta i = (u, v), retorna o índice j da aresta reversa (v, u).

    Necessário para recuperar F_{v⊴e} a partir do mapa aprendido para (u, v).
    Assume grafo não-direcionado (ambas as direções presentes em edge_index).

    Usa searchsorted vetorizado para suportar grafos grandes sem risco de
    colisão ou aresta ausente.
    """
    src, tgt = edge_index
    n = edge_index.max().item() + 1

    fwd = src.long() * n + tgt.long()
    rev = tgt.long() * n + src.long()

    sorted_fwd, perm = fwd.sort()
    pos = torch.searchsorted(sorted_fwd, rev).clamp(0, src.shape[0] - 1)
    return perm[pos]
