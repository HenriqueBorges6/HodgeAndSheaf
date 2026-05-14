"""
Camada de difusão de Bisfeixe (corpo central, task-agnostic).

Implementa uma etapa de difusão conforme:

    X(t+1) = (1 + tanh(ε)) · X(t)  −  GELU( Δ(F,C) · W1·X(t) · W2 )

onde

    Δ(F,C) = D_C^{-1/2} · L(F,C) · D_F^{-1/2}

e L(F,C) é o Laplaciano de bisfeixe não-normalizado. Esta camada não conhece
a tarefa: não faz pooling nem classificação. Os modelos de node e graph
classification montam o pipeline empilhando BiSheafDiffusionLayer e
adicionando a cabeça adequada.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .learners  import SheafMapLearner
from .laplacian import sym_invsqrt, degree_F, degree_C, bisheaf_lap_action


class BiSheafDiffusionLayer(nn.Module):
    """
    Uma camada de difusão de bisfeixe.

    Fluxo de execução:
        1. Aprende F e C  (um mapa d×d por aresta dirigida)
        2. Obtém os mapas pelo nó de destino via reverse_idx
        3. Computa D^F e D^C
        4. Calcula Δ(F,C) · W1·x · W2  sem materializar a matriz esparsa:
             a. z       = W1 · x                  (mistura dimensões do stalk)
             b. z_norm  = D_F^{-1/2} · z          (normalização direita)
             c. Lz      = L(F,C) · z_norm         (Laplaciano não-normalizado)
             d. delta   = D_C^{-1/2} · Lz         (normalização esquerda)
             e. delta   = W2 · delta               (mistura de canais)
        5. Residual: x = (1 + tanh(ε)) · x₀ − GELU(delta)

    Args:
        in_ch:              dimensão da representação (d·f, após input_proj)
        d:                  dimensão do stalk
        f:                  número de canais
        learner_hidden:     dimensão oculta dos learners
        learner_gnn_layers: camadas GNN nos learners
        left_weights:       habilita W1 (d×d, mistura dimensões do stalk)
        right_weights:      habilita W2 (f→f, mistura canais)
        use_eps:            habilita resíduo ε aprendível por dimensão do stalk
        dropout:            dropout aplicado na entrada e na saída
        norm_eps:           regularização para D^{-1/2}
    """

    def __init__(
        self,
        in_ch: int,
        d: int,
        f: int,
        learner_hidden: int = 64,
        learner_gnn_layers: int = 1,
        left_weights: bool = True,
        right_weights: bool = True,
        use_eps: bool = True,
        dropout: float = 0.0,
        norm_eps: float = 1e-3,
        backbone_F: str = 'node_edge_node',
        backbone_C: str = 'node_edge_node',
        map_type_F: str = 'general',
        map_type_C: str = 'general',
        orth_trans: str = 'householder',
        sheaf_act:  str = 'id',
        **backbone_kwargs,
    ):
        super().__init__()
        self.d        = d
        self.f        = f
        self.dropout  = dropout
        self.norm_eps = norm_eps

        self.learner_F = SheafMapLearner(
            in_ch, learner_hidden, learner_gnn_layers, d,
            backbone=backbone_F, map_type=map_type_F,
            orth_trans=orth_trans, sheaf_act=sheaf_act, **backbone_kwargs,
        )
        self.learner_C = SheafMapLearner(
            in_ch, learner_hidden, learner_gnn_layers, d,
            backbone=backbone_C, map_type=map_type_C,
            orth_trans=orth_trans, sheaf_act=sheaf_act, **backbone_kwargs,
        )

        self.use_left  = left_weights
        self.use_right = right_weights
        self.use_eps   = use_eps

        if left_weights:
            self.W1 = nn.Linear(d, d, bias=False)
            nn.init.eye_(self.W1.weight.data)

        if right_weights:
            self.W2 = nn.Linear(f, f)

        if use_eps:
            # um escalar aprendível por dimensão do stalk, shape (d, 1) para broadcast (n, d, f)
            self.eps = nn.Parameter(torch.zeros(d, 1))

    def get_maps(
        self,
        x_flat: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retorna (maps_F, maps_C), cada (m, d, d), sem side effects."""
        with torch.no_grad():
            maps_F = self.learner_F(x_flat, edge_index)
            maps_C = self.learner_C(x_flat, edge_index)
        return maps_F, maps_C

    def forward(
        self,
        x: torch.Tensor,          # (n, d, f)
        x_flat: torch.Tensor,     # (n, d·f) — view de x para os learners
        edge_index: torch.Tensor, # (2, m)
        reverse_idx: torch.Tensor,# (m,)
        return_maps: bool = False,
    ) -> torch.Tensor | tuple:
        """Retorna x atualizado (n, d, f), ou (x, maps_F, maps_C) se return_maps=True."""
        src, tgt = edge_index
        n = x.shape[0]
        d, f = self.d, self.f

        if self.training and self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=True)

        x0 = x

        # --- 1. Aprender mapas F e C ---
        # maps_F[i] = F_{src[i]⊴e},  maps_C[i] = C_{src[i]⊴e}
        maps_F = self.learner_F(x_flat, edge_index)   # (m, d, d)
        maps_C = self.learner_C(x_flat, edge_index)   # (m, d, d)

        # Mapas pelo nó de destino: maps_X_tgt[i] = X_{tgt[i]⊴e}
        maps_F_tgt = maps_F[reverse_idx]              # (m, d, d)
        maps_C_tgt = maps_C[reverse_idx]              # (m, d, d)

        # --- 2. Matrizes de grau ---
        # .detach() evita backward instável de eigh com eigenvalores próximos
        D_F = degree_F(maps_F_tgt, tgt, n)
        D_C = degree_C(maps_C_tgt, tgt, n)

        D_F_inv = sym_invsqrt(D_F.detach(), eps=self.norm_eps)   # (n, d, d)
        D_C_inv = sym_invsqrt(D_C.detach(), eps=self.norm_eps)   # (n, d, d)

        # --- 3. W1: mistura dimensões do stalk ---
        # W1 opera sobre a dimensão d: (n, d, f) → aplica W1 em cada canal
        if self.use_left:
            # x: (n, d, f) → reshape para (n·f, d), aplica W1, volta para (n, d, f)
            z = self.W1(x.transpose(1, 2).reshape(n * f, d)).reshape(n, f, d).transpose(1, 2)
        else:
            z = x

        # --- 4. Normalização direita: z_norm = D_F^{-1/2} · z ---
        z_norm = torch.bmm(D_F_inv, z)                # (n, d, f)

        # --- 5. Laplaciano não-normalizado: L(F,C) · z_norm ---
        Lz = bisheaf_lap_action(maps_F, maps_F_tgt, maps_C_tgt, z_norm, src, tgt, n)

        # --- 6. Normalização esquerda: delta = D_C^{-1/2} · Lz ---
        delta = torch.bmm(D_C_inv, Lz)                # (n, d, f)

        # --- 7. W2: mistura canais ---
        if self.use_right:
            delta = self.W2(delta.reshape(n * d, f)).reshape(n, d, f)

        if self.training and self.dropout > 0:
            delta = F.dropout(delta, p=self.dropout, training=True)

        delta = F.gelu(delta)

        # --- 8. Residual com epsilon aprendível ---
        if self.use_eps:
            scale = 1.0 + torch.tanh(self.eps)        # (d, 1) → broadcast (n, d, f)
            out = scale * x0 - delta
        else:
            out = x0 - delta

        if return_maps:
            return out, maps_F, maps_C
        return out
