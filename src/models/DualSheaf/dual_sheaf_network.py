"""
Dual Sheaf Diffusion Network.

Estende o Neural Sheaf Diffusion (Bodnar et al., NeurIPS 2022) com dois
learners independentes Phi_A e Phi_B, produzindo mapas de restrição
assimétricos F^A e F^B por aresta.

Laplaciano resultante para cada par de nós v, u adjacentes:

    L_{vu} x_u = -(F^A_v)^T  F^B_u  x_u     (termo off-diagonal)
    D_{vv}  x_v =  sum_u (F^A_v)^T  F^B_v  x_v  (termo diagonal)

O sinal completo de difusão normalizado por camada t é:

    X(t+1) = (1 + tanh(eps_t)) * X(t)  -  GELU( D^{-1/2} L_F D^{-1/2} W1 X(t) W2 )

onde D^{-1/2} L D^{-1/2} é aproximado por normalização escalar 1/sqrt(deg_u * deg_v).

Variantes de laplaciano (laplacian_mode):
    'dual'       : F^A^T ( F^B x_v  -  F^B x_u )   — mapas completamente assimétricos
    'mixed_self' : F^A^T ( F^A x_v  -  F^B x_u )   — self-term usa F^A, neighbor usa F^B
    'mixed_nbr'  : F^A^T ( F^B x_v  -  F^A x_u )   — self-term usa F^B, neighbor usa F^A
    'symmetric'  : F^A^T ( F^A x_v  -  F^A x_u )   — feixe clássico de um único learner

Variante de co-feixe (cosheaf_b=True):
    F^B é aprendido por uma GNN que opera no grafo de linhas L(G), onde cada
    aresta de G se torna um nó e a vizinhança reflete adjacências entre arestas.
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.SheafNN.sheaf import Orthogonal


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Utilitários de grafo
# ---------------------------------------------------------------------------

def compute_line_graph(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Constrói o grafo de linhas L(G) a partir de edge_index.

    No grafo de linhas, cada aresta de G vira um nó. Dois nós de L(G) são
    conectados quando as arestas originais compartilham um nó intermediário:

        e1 = (u → v)  e  e2 = (v → w)  →  aresta (e1, e2) em L(G)

    Retorna line_edge_index de shape (2, k).
    """
    src, tgt = edge_index
    line_srcs, line_tgts = [], []

    for v in range(num_nodes):
        incoming = (tgt == v).nonzero(as_tuple=True)[0]
        outgoing = (src == v).nonzero(as_tuple=True)[0]
        if incoming.numel() == 0 or outgoing.numel() == 0:
            continue
        e1 = incoming.unsqueeze(1).expand(-1, outgoing.numel()).reshape(-1)
        e2 = outgoing.unsqueeze(0).expand(incoming.numel(), -1).reshape(-1)
        mask = e1 != e2
        line_srcs.append(e1[mask])
        line_tgts.append(e2[mask])

    if not line_srcs:
        return torch.zeros(2, 0, dtype=torch.long, device=edge_index.device)

    return torch.stack([torch.cat(line_srcs), torch.cat(line_tgts)], dim=0)


def compute_reverse_idx(edge_index: torch.Tensor) -> torch.Tensor:
    """
    Para cada aresta i = (u, v), retorna o índice j da aresta reversa (v, u).

    Necessário para recuperar os mapas F^A_v a partir do índice da aresta
    (u → v), pois os learners produzem mapas indexados pela direção src→tgt.
    Assume que o grafo é não-direcionado (ambas as direções estão em edge_index).

    Usa searchsorted vetorizado em vez de dict para suportar grafos grandes
    sem risco de colisão ou aresta ausente.
    """
    src, tgt = edge_index
    n = edge_index.max().item() + 1

    fwd = src.long() * n + tgt.long()   # hash de cada aresta
    rev = tgt.long() * n + src.long()   # hash da aresta reversa

    # Ordenar hashes forward e buscar onde cada reverso cai
    sorted_fwd, perm = fwd.sort()
    pos = torch.searchsorted(sorted_fwd, rev)
    pos = pos.clamp(0, src.shape[0] - 1)
    return perm[pos]


# ---------------------------------------------------------------------------
# Camada de message passing nó → aresta → nó
# ---------------------------------------------------------------------------

class NodeEdgeNodeLayer(nn.Module):
    """
    Camada de message passing que agrega informação de arestas nos nós.

    Para cada nó v:
        h_e  = W_edge (x_src + x_tgt)      para cada aresta e incidente
        agg_v = mean(h_e para todas as arestas incidentes a v)
        out_v = W_self x_v + agg_v

    Usada tanto no SheafLearner (propaga features de nós em G) quanto no
    CosheafEdgeLearner (propaga features de arestas em L(G)).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin_edge = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_self = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, tgt = edge_index
        n = x.shape[0]
        out_C = self.lin_edge.out_features

        edge_feat = self.lin_edge(x[src] + x[tgt])    # (m, out_C)

        node_agg = torch.zeros(n, out_C, device=x.device, dtype=x.dtype)
        node_agg.scatter_add_(0, src.unsqueeze(1).expand(-1, out_C).contiguous(), edge_feat)
        node_agg.scatter_add_(0, tgt.unsqueeze(1).expand(-1, out_C).contiguous(), edge_feat)

        m = src.shape[0]
        deg = torch.zeros(n, device=x.device, dtype=x.dtype)
        deg.scatter_add_(0, src, torch.ones(m, device=x.device, dtype=x.dtype))
        deg.scatter_add_(0, tgt, torch.ones(m, device=x.device, dtype=x.dtype))
        node_agg = node_agg / deg.clamp(min=1).unsqueeze(1)

        return self.lin_self(x) + node_agg


# ---------------------------------------------------------------------------
# Learners de mapas de restrição
# ---------------------------------------------------------------------------

class SheafLearner(nn.Module):
    """
    Aprende um mapa de restrição F_{v ◁ e} para cada aresta dirigida,
    propagando features de nós no grafo G.

    Pipeline:
        x (n, C)  →  emb  →  NodeEdgeNodeLayer × L  →  concat(h_src, h_tgt)  →  (m, out_dim)

    out_dim depende de orth_trans:
        None          →  d*d           (matriz geral)
        euler/householder  →  d*(d-1)//2   (parâmetros de rotação)
        matrix_exp/cayley  →  d*(d+1)//2   (parâmetros skew-simétricos)
    """

    def __init__(self, in_channels: int, d: int, hidden_dim: int = 64,
                 gnn_layers: int = 1, out_dim: int = None):
        super().__init__()
        self.d = d
        out_dim = out_dim if out_dim is not None else d * d

        self.emb   = nn.Linear(in_channels, hidden_dim)
        self.convs = nn.ModuleList([NodeEdgeNodeLayer(hidden_dim, hidden_dim) for _ in range(gnn_layers)])
        self.readout = nn.Linear(2 * hidden_dim, out_dim)

    def forward(self, x_flat: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, tgt = edge_index
        h = F.elu(self.emb(x_flat))
        for conv in self.convs:
            h = F.elu(conv(h, edge_index))
        return self.readout(torch.cat([h[src], h[tgt]], dim=1))   # (m, out_dim)


class CosheafEdgeLearner(nn.Module):
    """
    Aprende mapas de restrição F^B_e operando no grafo de linhas L(G).

    Diferente do SheafLearner, que agrega features de nós, este módulo
    inicializa features de arestas (concat dos endpoints) e as propaga
    na estrutura de adjacência entre arestas definida por L(G).

    Esse domínio é o natural para co-feixes: seções são definidas em arestas,
    e a coerência é medida entre arestas que se encontram em um nó comum.

    Pipeline:
        concat(x_src, x_tgt) (m, 2C)  →  emb  →  NodeEdgeNodeLayer[L(G)] × L  →  (m, out_dim)
    """

    def __init__(self, node_channels: int, hidden_dim: int = 64,
                 gnn_layers: int = 1, out_dim: int = None, d: int = 2):
        super().__init__()
        self.d = d
        out_dim = out_dim if out_dim is not None else d * d

        self.emb     = nn.Linear(2 * node_channels, hidden_dim)
        self.convs   = nn.ModuleList([NodeEdgeNodeLayer(hidden_dim, hidden_dim) for _ in range(gnn_layers)])
        self.readout = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_flat: torch.Tensor, edge_index: torch.Tensor,
                line_edge_index: torch.Tensor) -> torch.Tensor:
        src, tgt = edge_index
        edge_feat = torch.cat([x_flat[src], x_flat[tgt]], dim=1)   # (m, 2C)
        h = F.elu(self.emb(edge_feat))
        for conv in self.convs:
            h = F.elu(conv(h, line_edge_index))
        return self.readout(h)                                       # (m, out_dim)


# ---------------------------------------------------------------------------
# Modelo principal
# ---------------------------------------------------------------------------

class DualSheafDiffusion(nn.Module):
    """
    Dual Sheaf Diffusion para classificação de nós.

    Cada camada de difusão executa:
        1. Aprende F^A e F^B (um mapa d×d por aresta dirigida)
        2. Monta o Laplaciano assimétrico L_F com normalização 1/sqrt(deg_u * deg_v)
        3. Aplica: X' = (1 + tanh(eps)) * X  -  GELU( L_F W1 X W2 )

    Ao final, LayerNorm + MLP produz os logits de classificação.

    Args:
        node_feat_dim:      dimensão das features de entrada dos nós
        d:                  dimensão do stalk (espaço local por nó)
        hidden_channels:    número de canais f na representação interna (n, d, f)
        num_layers:         número de camadas de difusão
        num_classes:        número de classes de saída
        learner_hidden:     dimensão oculta dos learners de mapa
        learner_gnn_layers: camadas GNN nos learners
        dropout:            dropout aplicado na entrada e na saída de cada camada
        left_weights:       habilita W1 (d×d) que mistura dimensões do stalk
        right_weights:      habilita W2 (f→f) que mistura canais
        use_eps:            habilita resíduo epsilon aprendível por camada
        single_learner:     se True, F^B = F^A (feixe clássico, sem dualidade)
        orth_trans:         parametrização ortogonal dos mapas —
                            None (geral) | 'euler' | 'householder' | 'matrix_exp' | 'cayley'
        laplacian_mode:     como F^A e F^B são combinados no Laplaciano (ver módulo docstring)
        cosheaf_b:          se True, F^B é aprendido no grafo de linhas L(G)
        mlp_dims:           dimensões das camadas ocultas do classificador final,
                            ex: [64] ou [128, 64]. [] = projeção linear direta.
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
        single_learner: bool = False,
        orth_trans: str = None,
        laplacian_mode: str = 'dual',
        cosheaf_b: bool = False,
        mlp_dims: list = None,
    ):
        super().__init__()
        assert orth_trans in (None, 'euler', 'householder', 'matrix_exp', 'cayley'), \
            f"orth_trans inválido: {orth_trans}"
        assert laplacian_mode in ('dual', 'mixed_self', 'mixed_nbr', 'symmetric'), \
            f"laplacian_mode inválido: {laplacian_mode}"

        self.d               = d
        self.f               = hidden_channels
        self.num_layers      = num_layers
        self.use_left_weights  = left_weights
        self.use_right_weights = right_weights
        self.use_eps         = use_eps
        self.single_learner  = single_learner
        self.orth_trans      = orth_trans
        self.laplacian_mode  = laplacian_mode
        self.cosheaf_b       = cosheaf_b and not single_learner

        # Dimensão de saída dos learners depende da parametrização escolhida
        if orth_trans is None:
            learner_out_dim = d * d
        elif orth_trans in ('euler', 'householder'):
            learner_out_dim = d * (d - 1) // 2
        else:
            learner_out_dim = d * (d + 1) // 2

        in_ch = d * hidden_channels

        self.input_proj = nn.Sequential(
            nn.Linear(node_feat_dim, in_ch),
            nn.ELU(),
        )

        self.learners_A = nn.ModuleList([
            SheafLearner(in_ch, d, learner_hidden, learner_gnn_layers, out_dim=learner_out_dim)
            for _ in range(num_layers)
        ])

        if not single_learner:
            if cosheaf_b:
                self.learners_B = nn.ModuleList([
                    CosheafEdgeLearner(in_ch, learner_hidden, learner_gnn_layers,
                                       out_dim=learner_out_dim, d=d)
                    for _ in range(num_layers)
                ])
            else:
                self.learners_B = nn.ModuleList([
                    SheafLearner(in_ch, d, learner_hidden, learner_gnn_layers, out_dim=learner_out_dim)
                    for _ in range(num_layers)
                ])

        if orth_trans is not None:
            self.orth_transform = Orthogonal(d=d, orthogonal_map=orth_trans)

        if left_weights:
            self.W1 = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(num_layers)])
            for lin in self.W1:
                nn.init.eye_(lin.weight.data)   # inicializa como identidade para não perturbar o início

        if right_weights:
            self.W2 = nn.ModuleList([
                nn.Linear(hidden_channels, hidden_channels) for _ in range(num_layers)
            ])

        if use_eps:
            self.epsilons = nn.ParameterList([
                nn.Parameter(torch.zeros(d, 1)) for _ in range(num_layers)
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

        self.dropout = dropout
        self._profile = False
        self._timings = {}

    def set_profiling(self, enabled: bool = True):
        self._profile = enabled

    def get_profile(self) -> dict:
        return dict(self._timings)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        reverse_idx: torch.Tensor,
        line_edge_index: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x:               (n, node_feat_dim)  features dos nós
            edge_index:      (2, m)              arestas dirigidas (grafo não-direcionado)
            reverse_idx:     (m,)                índice da aresta reversa para cada aresta
            line_edge_index: (2, k)              arestas de L(G); obrigatório se cosheaf_b=True

        Retorna logits (n, num_classes).
        """
        if self.cosheaf_b and line_edge_index is None:
            raise ValueError("cosheaf_b=True requer line_edge_index.")

        p   = self._profile
        dev = x.device
        n   = x.shape[0]
        src, tgt = edge_index
        m   = src.shape[0]
        d, f = self.d, self.f

        if p: _sync(dev); t0 = time.time()
        x = self.input_proj(x).reshape(n, d, f)    # (n, d, f)
        if p: _sync(dev); self._timings['input_proj'] = time.time() - t0

        # Normalização simétrica: escala cada aresta por 1/sqrt(deg_src * deg_tgt)
        deg = torch.zeros(n, device=dev, dtype=x.dtype)
        deg.scatter_add_(0, tgt, torch.ones(m, device=dev, dtype=x.dtype))
        edge_norm = (1.0 / (deg[src] * deg[tgt]).clamp(min=1).sqrt()).view(-1, 1, 1)

        for t in range(self.num_layers):
            if p: _sync(dev); t0 = time.time()

            if self.training and self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=True)

            x0     = x
            x_flat = x.reshape(n, d * f)

            # --- Aprender mapas de restrição ---
            raw_A  = self.learners_A[t](x_flat, edge_index)
            maps_A = self.orth_transform(raw_A) if self.orth_trans else raw_A.reshape(-1, d, d)

            if self.single_learner:
                maps_B = maps_A
            elif self.cosheaf_b:
                raw_B  = self.learners_B[t](x_flat, edge_index, line_edge_index)
                maps_B = self.orth_transform(raw_B) if self.orth_trans else raw_B.reshape(-1, d, d)
            else:
                raw_B  = self.learners_B[t](x_flat, edge_index)
                maps_B = self.orth_transform(raw_B) if self.orth_trans else raw_B.reshape(-1, d, d)

            # maps_A_tgt[i] = F^A do nó tgt[i] — obtido via aresta reversa
            maps_A_tgt = maps_A[reverse_idx]
            maps_B_tgt = maps_B[reverse_idx]

            if p: _sync(dev); self._timings[f'layer{t}_learn'] = time.time() - t0
            if p: t0 = time.time()

            # --- Left weights: mistura dimensões do stalk ---
            if self.use_left_weights:
                x_w = self.W1[t](x.transpose(1, 2).reshape(n * f, d)).reshape(n, f, d).transpose(1, 2)
            else:
                x_w = x

            # Selecionar qual mapa entra no self-term e no neighbor-term
            # conforme laplacian_mode (ver docstring do módulo)
            mode      = self.laplacian_mode
            maps_self = maps_A_tgt if mode == 'mixed_self' else maps_B_tgt
            maps_nbr  = maps_A     if mode in ('mixed_nbr', 'symmetric') else maps_B

            # --- Termo off-diagonal: mensagem de src para tgt ---
            # msg_{u→v} = (F^A_v)^T  F^B_u  x_u / sqrt(deg_v * deg_u)
            Nx  = torch.bmm(maps_nbr, x_w[src])                       # (m, d, f)
            msg = torch.bmm(maps_A_tgt.transpose(-2, -1), Nx)         # (m, d, f)
            msg = msg * edge_norm

            agg = torch.zeros(n, d, f, device=dev, dtype=x.dtype)
            agg.scatter_add_(0, tgt.view(-1, 1, 1).expand(-1, d, f).contiguous(), msg)

            # --- Termo diagonal: D_v = sum_u (F^A_v)^T F^B_v / sqrt(deg_v * deg_u) ---
            D_block = torch.bmm(maps_A_tgt.transpose(-2, -1), maps_self) * edge_norm   # (m, d, d)
            D = torch.zeros(n, d, d, device=dev, dtype=x.dtype)
            D.scatter_add_(0, tgt.view(-1, 1, 1).expand(-1, d, d).contiguous(), D_block)
            Dx = torch.bmm(D, x_w)                                    # (n, d, f)

            Lx = Dx - agg   # L_F x

            if self.use_right_weights:
                Lx = self.W2[t](Lx.reshape(n * d, f)).reshape(n, d, f)

            if p: _sync(dev); self._timings[f'layer{t}_diffusion'] = time.time() - t0

            if self.training and self.dropout > 0:
                Lx = F.dropout(Lx, p=self.dropout, training=True)
            Lx = F.gelu(Lx)

            # --- Residual com epsilon aprendível por dimensão do stalk ---
            if self.use_eps:
                scale = (1.0 + torch.tanh(self.epsilons[t])).unsqueeze(0)  # (1, d, 1)
                x = scale * x0 - Lx
            else:
                x = x0 - Lx

        if p: _sync(dev); t0 = time.time()
        out = self.mlp(self.pre_mlp_norm(x.reshape(n, d * f)))
        if p: _sync(dev); self._timings['output_proj'] = time.time() - t0

        return out
