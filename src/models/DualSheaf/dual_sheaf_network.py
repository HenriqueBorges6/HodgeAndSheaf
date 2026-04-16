"""
Dual Sheaf Diffusion — duas redes aprendendo mapas de restricao independentes.

Baseado no Neural Sheaf Diffusion (Bodnar et al., NeurIPS 2022), mas com dois
learners separados Phi_A e Phi_B. O Laplaciano do feixe e construido com:

    L_{vu} = -(F^A_{v tri e})^T  F^B_{u tri e}
    D_{vv} =  sum_{v tri e} (F^A_{v tri e})^T  F^B_{v tri e}

Difusao com epsilon residual aprendivel (inspirado no FlatBundleConv):
    Lx = D^{-1/2} L_F(t) D^{-1/2} @ W1 @ X(t) @ W2
    X(t+1) = (1 + tanh(eps)) * X(t) - dropout(sigma(Lx))   [per-layer skip]

Melhorias vs v1:
    - Epsilon residual per-layer (x0 = input da camada, nao global)
    - Normalizacao simetrica D^{-1/2} L D^{-1/2} (ao inves de D^{-1} L)
    - Dropout no input dos learners (regulariza aprendizado dos mapas)
    - Left weights W1 (d x d) para mistura de dimensoes do stalk
    - Learner com GNN no->aresta->no (ao inves de MLP puro)
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.SheafNN.sheaf import Orthogonal


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def compute_line_graph(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Constroi o grafo de linhas L(G) de edge_index.

    No grafo de linhas, cada aresta de G vira um no. Dois nos de L(G)
    sao conectados se as arestas originais compartilham um no em G:
        e1=(u,v) e e2=(v,w)  →  aresta (e1, e2) em L(G)

    Retorna: line_edge_index (2, k) — arestas de L(G)
    """
    src, tgt = edge_index                  # (m,) cada
    m = src.shape[0]

    # Para cada no v, listar arestas que chegam (como tgt) e saem (como src)
    # Aresta e1=(u,v) conecta com aresta e2=(v,w): tgt[e1] == src[e2]
    # Implementacao vetorizada: para cada no v, produto cartesiano de
    # arestas incidentes como tgt x arestas incidentes como src
    line_srcs, line_tgts = [], []
    for v in range(num_nodes):
        incoming = (tgt == v).nonzero(as_tuple=True)[0]   # arestas com tgt=v
        outgoing = (src == v).nonzero(as_tuple=True)[0]   # arestas com src=v
        if incoming.numel() == 0 or outgoing.numel() == 0:
            continue
        # produto cartesiano: cada aresta incoming conecta com cada outgoing
        e1 = incoming.unsqueeze(1).expand(-1, outgoing.numel()).reshape(-1)
        e2 = outgoing.unsqueeze(0).expand(incoming.numel(), -1).reshape(-1)
        # excluir self-loops (e1 == e2) e reversa imediata
        mask = e1 != e2
        line_srcs.append(e1[mask])
        line_tgts.append(e2[mask])

    if not line_srcs:
        return torch.zeros(2, 0, dtype=torch.long, device=edge_index.device)

    line_src = torch.cat(line_srcs)
    line_tgt = torch.cat(line_tgts)
    return torch.stack([line_src, line_tgt], dim=0)


def compute_reverse_idx(edge_index: torch.Tensor) -> torch.Tensor:
    """
    Para cada aresta dirigida i = (src_i, tgt_i), encontra o indice j
    da aresta reversa (tgt_i, src_i). Assume grafo nao-direcionado (ambas
    direcoes presentes em edge_index).
    """
    src, tgt = edge_index
    m = src.shape[0]
    n = max(src.max().item(), tgt.max().item()) + 1
    edge_hash = src.long() * n + tgt.long()
    hash_to_idx = {}
    for i in range(m):
        hash_to_idx[edge_hash[i].item()] = i
    reverse_hash = tgt.long() * n + src.long()
    reverse_idx = torch.tensor(
        [hash_to_idx[reverse_hash[i].item()] for i in range(m)],
        dtype=torch.long, device=edge_index.device
    )
    return reverse_idx


# ---------------------------------------------------------------------------
# GNN layer no->aresta->no (sem MessagePassing do PyG)
# ---------------------------------------------------------------------------

class NodeEdgeNodeLayer(nn.Module):
    """
    Message-passing no->aresta->no sobre o grafo original.
    Para cada no v:
        edge_msg_e = W_edge @ (x[src] + x[tgt])   para cada aresta e=(src,tgt)
        agg_v = mean(edge_msg de arestas incidentes a v)
        out_v = W_self @ x_v + agg_v
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin_edge = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_self = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x: (n, C)  edge_index: (2, m)
        src, tgt = edge_index
        n, C = x.shape
        m = src.shape[0]
        out_C = self.lin_edge.out_features

        # Computar features por aresta
        edge_feat = self.lin_edge(x[src] + x[tgt])    # (m, out_C)

        # Agregar arestas nos endpoints (cada aresta contribui para src e tgt)
        node_agg = torch.zeros(n, out_C, device=x.device, dtype=x.dtype)
        idx_src = src.unsqueeze(1).expand(-1, out_C).contiguous()
        idx_tgt = tgt.unsqueeze(1).expand(-1, out_C).contiguous()
        node_agg.scatter_add_(0, idx_src, edge_feat)
        node_agg.scatter_add_(0, idx_tgt, edge_feat)

        # Normalizar pelo grau
        deg = torch.zeros(n, device=x.device, dtype=x.dtype)
        deg.scatter_add_(0, src, torch.ones(m, device=x.device, dtype=x.dtype))
        deg.scatter_add_(0, tgt, torch.ones(m, device=x.device, dtype=x.dtype))
        node_agg = node_agg / deg.clamp(min=1).unsqueeze(1)

        return self.lin_self(x) + node_agg


# ---------------------------------------------------------------------------
# Sheaf Learner com GNN backbone
# ---------------------------------------------------------------------------

class SheafLearner(nn.Module):
    """
    Aprende mapas de restricao F_{v tri e} para cada aresta.

    Arquitetura:
        emb(x_v) -> NodeEdgeNodeLayer * gnn_layers -> concat(h_src, h_tgt) -> out -> (out_dim,)

    out_dim depende do tipo de mapa:
        - geral: d*d parametros, retorna (m, d, d)
        - ortogonal (euler/householder): d*(d-1)//2 parametros brutos, transformados pelo Orthogonal
        - ortogonal (matrix_exp/cayley): d*(d+1)//2 parametros brutos
    """

    def __init__(self, in_channels: int, d: int, hidden_dim: int = 64,
                 gnn_layers: int = 1, out_dim: int = None):
        super().__init__()
        self.d = d
        out_dim = out_dim if out_dim is not None else d * d

        self.emb = nn.Linear(in_channels, hidden_dim)
        self.convs = nn.ModuleList([
            NodeEdgeNodeLayer(hidden_dim, hidden_dim)
            for _ in range(gnn_layers)
        ])
        self.readout = nn.Linear(2 * hidden_dim, out_dim)

    def forward(self, x_flat: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x_flat: (n, C)  edge_index: (2, m)
        Retorna: (m, out_dim) — parametros brutos por aresta
        """
        src, tgt = edge_index
        h = F.elu(self.emb(x_flat))
        for conv in self.convs:
            h = F.elu(conv(h, edge_index))
        edge_feat = torch.cat([h[src], h[tgt]], dim=1)  # (m, 2*hidden)
        return self.readout(edge_feat)                    # (m, out_dim)


# ---------------------------------------------------------------------------
# Co-Sheaf Edge Learner — aprende mapas operando no grafo de linhas L(G)
# ---------------------------------------------------------------------------

class CosheafEdgeLearner(nn.Module):
    """
    Aprende mapas de restricao F^B_{e} a partir de features de ARESTAS,
    propagando no grafo de linhas L(G) (dominio natural do co-feixe).

    Contrasta com SheafLearner que propaga features de NOS no grafo G.

    Arquitetura:
        edge_feat_e = concat(x_src, x_tgt)    para cada aresta e=(src,tgt)
        emb(edge_feat) -> EdgeGNN on L(G) * gnn_layers -> readout -> (out_dim,)

    O grafo de linhas L(G) conecta arestas que compartilham um no:
        e1=(u,v) -- e2=(v,w)  <=> tgt[e1] == src[e2]
    """

    def __init__(self, node_channels: int, hidden_dim: int = 64,
                 gnn_layers: int = 1, out_dim: int = None, d: int = 2):
        super().__init__()
        self.d = d
        out_dim = out_dim if out_dim is not None else d * d
        edge_in = 2 * node_channels   # concat(x_src, x_tgt)

        self.emb = nn.Linear(edge_in, hidden_dim)
        # GNN no grafo de linhas: cada "no" e uma aresta de G
        self.convs = nn.ModuleList([
            NodeEdgeNodeLayer(hidden_dim, hidden_dim)
            for _ in range(gnn_layers)
        ])
        self.readout = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                line_edge_index: torch.Tensor) -> torch.Tensor:
        """
        x:               (n, C)  features de nos
        edge_index:      (2, m)  arestas do grafo original
        line_edge_index: (2, k)  arestas do grafo de linhas L(G)

        Retorna: (m, out_dim) — parametros brutos por aresta
        """
        src, tgt = edge_index
        # Derivar features de arestas a partir dos nos endpoints
        edge_feat = torch.cat([x[src], x[tgt]], dim=1)   # (m, 2*C)

        # Propagar no grafo de linhas (arestas sao os "nos" agora)
        h = F.elu(self.emb(edge_feat))                    # (m, hidden)
        for conv in self.convs:
            h = F.elu(conv(h, line_edge_index))            # (m, hidden)

        return self.readout(h)                             # (m, out_dim)


# ---------------------------------------------------------------------------
# Modelo principal
# ---------------------------------------------------------------------------

class DualSheafDiffusion(nn.Module):
    """
    Dual Sheaf Diffusion para node classification.

    Pipeline:
        x_raw -> input_proj -> (n, d, f)
        for each layer t:
            Phi_A(X(t)) -> F^A     (learner A — restricao "saida")
            Phi_B(X(t)) -> F^B     (learner B — restricao "entrada"; se single_learner=True, F^B = F^A)
            Lx = L_F @ (W1 @ X) @ W2
            X(t+1) = (1 + tanh(eps_t)) * X(t) - GELU(Lx)   [se use_eps=True]
                   = X(t) - GELU(Lx)                         [se use_eps=False]
        LayerNorm -> MLP -> logits

    Args:
        node_feat_dim:      dimensao das features de entrada
        d:                  dimensao do stalk por no
        hidden_channels:    numero de canais ocultos f
        num_layers:         numero de camadas de difusao
        num_classes:        numero de classes de saida
        learner_hidden:     dimensao oculta dos learners de mapa
        learner_gnn_layers: camadas GNN no backbone dos learners
        dropout:            probabilidade de dropout
        left_weights:       aplicar W1 (d x d) na dimensao do stalk
        right_weights:      aplicar W2 (f -> f) na dimensao dos canais
        use_eps:            usar residual epsilon aprendivel
        single_learner:     usar um unico learner (F^A = F^B) — caso classico
        mlp_dims:           dimensoes da MLP classificadora, ex: [64] ou [128, 64]
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
            f"orth_trans deve ser None, 'euler', 'householder', 'matrix_exp' ou 'cayley'. Recebido: {orth_trans}"
        assert laplacian_mode in ('dual', 'mixed_self', 'mixed_nbr', 'symmetric'), \
            f"laplacian_mode deve ser 'dual', 'mixed_self', 'mixed_nbr' ou 'symmetric'. Recebido: {laplacian_mode}"

        self.d = d
        self.f = hidden_channels
        self.num_layers = num_layers
        self.use_left_weights = left_weights
        self.use_right_weights = right_weights
        self.use_eps = use_eps
        self.single_learner = single_learner
        self.orth_trans = orth_trans
        self.laplacian_mode = laplacian_mode
        self.cosheaf_b = cosheaf_b and not single_learner

        # Numero de parametros que o learner precisa produzir por aresta
        # None (geral): d*d  |  euler/householder: d*(d-1)//2  |  matrix_exp/cayley: d*(d+1)//2
        if orth_trans is None:
            learner_out_dim = d * d
        elif orth_trans in ('euler', 'householder'):
            learner_out_dim = d * (d - 1) // 2
        else:  # matrix_exp, cayley
            learner_out_dim = d * (d + 1) // 2

        # Input: features brutas -> (d * f) dimensional
        self.input_proj = nn.Sequential(
            nn.Linear(node_feat_dim, d * hidden_channels),
            nn.ELU(),
        )

        in_ch = d * hidden_channels

        # Learner A — sempre presente
        self.learners_A = nn.ModuleList([
            SheafLearner(in_ch, d, learner_hidden, learner_gnn_layers, out_dim=learner_out_dim)
            for _ in range(num_layers)
        ])

        # Learner B — apenas se dual (single_learner=False)
        if not single_learner:
            if cosheaf_b:
                # Co-feixe: opera em features de arestas propagadas no grafo de linhas L(G)
                self.learners_B = nn.ModuleList([
                    CosheafEdgeLearner(d * hidden_channels, learner_hidden,
                                       learner_gnn_layers, out_dim=learner_out_dim, d=d)
                    for _ in range(num_layers)
                ])
            else:
                self.learners_B = nn.ModuleList([
                    SheafLearner(in_ch, d, learner_hidden, learner_gnn_layers, out_dim=learner_out_dim)
                    for _ in range(num_layers)
                ])

        # Transform ortogonal — converte parametros brutos em matrizes ortogonais
        if orth_trans is not None:
            self.orth_transform = Orthogonal(d=d, orthogonal_map=orth_trans)

        # Left weights W1 (d x d): mistura dimensoes do stalk
        if left_weights:
            self.W1 = nn.ModuleList([
                nn.Linear(d, d, bias=False) for _ in range(num_layers)
            ])
            for lin in self.W1:
                nn.init.eye_(lin.weight.data)

        # Right weights W2 (f -> f): mistura canais
        if right_weights:
            self.W2 = nn.ModuleList([
                nn.Linear(hidden_channels, hidden_channels)
                for _ in range(num_layers)
            ])

        # Epsilon residual aprendivel por camada
        if use_eps:
            self.epsilons = nn.ParameterList([
                nn.Parameter(torch.zeros(d, 1)) for _ in range(num_layers)
            ])

        # Normalizacao + MLP classificador
        # mlp_dims=[]  -> LayerNorm -> Linear(d*f, num_classes)
        # mlp_dims=[64] -> LayerNorm -> Linear(d*f,64) -> ReLU -> Dropout -> Linear(64, num_classes)
        self.pre_mlp_norm = nn.LayerNorm(d * hidden_channels)
        dims = [d * hidden_channels] + (mlp_dims or []) + [num_classes]
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
        x:               (n, node_feat_dim) features brutas dos nos
        edge_index:      (2, m) arestas dirigidas (grafo nao-direcionado)
        reverse_idx:     (m,) indice da aresta reversa para cada aresta
        line_edge_index: (2, k) arestas do grafo de linhas L(G) — obrigatorio se cosheaf_b=True
        """
        if self.cosheaf_b and line_edge_index is None:
            raise ValueError("cosheaf_b=True requer line_edge_index no forward.")
        p = self._profile
        dev = x.device
        n = x.shape[0]
        src, tgt = edge_index
        m = src.shape[0]
        d, f = self.d, self.f

        # --- Input projection ---
        if p: _sync(dev); t0 = time.time()
        x = self.input_proj(x)            # (n, d*f)
        x = x.reshape(n, d, f)            # (n, d, f)
        if p: _sync(dev); self._timings['input_proj'] = time.time() - t0

        # Pre-computar grau para normalizacao simetrica por aresta
        deg = torch.zeros(n, device=dev, dtype=x.dtype)
        deg.scatter_add_(0, tgt, torch.ones(m, device=dev, dtype=x.dtype))
        # Escala por aresta: 1/sqrt(deg_src * deg_tgt)
        edge_norm = (1.0 / (deg[src] * deg[tgt]).clamp(min=1).sqrt()).view(-1, 1, 1)  # (m, 1, 1)

        # --- Diffusion layers ---
        for t in range(self.num_layers):
            if p: _sync(dev); t0 = time.time()

            # (1) Dropout no input da camada (como na referencia FlatBundleConv)
            if self.training and self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=True)

            # Salvar x0 per-layer para epsilon residual
            x0 = x

            x_flat = x.reshape(n, d * f)   # (n, C) para os learners

            # Aprender mapas: um por aresta dirigida
            raw_A = self.learners_A[t](x_flat, edge_index)            # (m, out_dim)
            if self.orth_trans is not None:
                maps_A = self.orth_transform(raw_A)                    # (m, d, d) ortogonal
            else:
                maps_A = raw_A.reshape(-1, d, d)                       # (m, d, d) geral

            # single_learner: F^B = F^A (caso classico de um feixe so)
            if self.single_learner:
                maps_B = maps_A
            elif self.cosheaf_b:
                # Co-feixe: learner B opera em features de arestas no grafo de linhas
                raw_B = self.learners_B[t](x_flat, edge_index, line_edge_index)  # (m, out_dim)
                if self.orth_trans is not None:
                    maps_B = self.orth_transform(raw_B)
                else:
                    maps_B = raw_B.reshape(-1, d, d)
            else:
                raw_B = self.learners_B[t](x_flat, edge_index)        # (m, out_dim)
                if self.orth_trans is not None:
                    maps_B = self.orth_transform(raw_B)
                else:
                    maps_B = raw_B.reshape(-1, d, d)

            # Mapas do lado tgt via aresta reversa
            maps_A_tgt = maps_A[reverse_idx]  # (m, d, d)
            maps_B_tgt = maps_B[reverse_idx]  # (m, d, d)

            if p: _sync(dev); self._timings[f'layer{t}_learn'] = time.time() - t0

            if p: t0 = time.time()

            # --- Aplicar left weights W1 (d x d) ---
            if self.use_left_weights:
                x_w = x.transpose(1, 2).reshape(n * f, d)         # (n*f, d)
                x_w = self.W1[t](x_w).reshape(n, f, d).transpose(1, 2)  # (n, d, f)
            else:
                x_w = x

            # Selecionar mapas para self-term e neighbor-term segundo laplacian_mode:
            #   'dual'       : F^A^T( F^B·x_v  -  F^B·x_u )
            #   'mixed_self' : F^A^T( F^A·x_v  -  F^B·x_u )
            #   'mixed_nbr'  : F^A^T( F^B·x_v  -  F^A·x_u )
            #   'symmetric'  : F^A^T( F^A·x_v  -  F^A·x_u )  (= single sheaf classico)
            mode = self.laplacian_mode
            maps_self = maps_A_tgt if mode == 'mixed_self' else maps_B_tgt   # mapa aplicado em x_v
            maps_nbr  = maps_A     if mode == 'mixed_nbr'  else (            # mapa aplicado em x_u
                        maps_A     if mode == 'symmetric'  else maps_B)

            # --- Off-diagonal: mensagem src -> tgt ---
            # msg = (F^A_tgt)^T @ maps_nbr_src @ x_src
            x_src = x_w[src]                                          # (m, d, f)
            Nx = torch.bmm(maps_nbr, x_src)                          # (m, d, f)
            msg = torch.bmm(maps_A_tgt.transpose(-2, -1), Nx)        # (m, d, f)
            msg = msg * edge_norm                                     # norm simetrica por aresta

            # Agregar mensagens nos nos alvo
            idx = tgt.view(-1, 1, 1).expand(-1, d, f).contiguous()
            agg = torch.zeros(n, d, f, device=dev, dtype=x.dtype)
            agg.scatter_add_(0, idx, msg)

            # --- Diagonal: D_v = sum_{v tri e} (F^A_v)^T @ maps_self_v ---
            D_block = torch.bmm(maps_A_tgt.transpose(-2, -1), maps_self)  # (m, d, d)
            D_block = D_block * edge_norm                             # norm simetrica por aresta
            D = torch.zeros(n, d, d, device=dev, dtype=x.dtype)
            D.scatter_add_(0, tgt.view(-1, 1, 1).expand(-1, d, d).contiguous(), D_block)
            Dx = torch.bmm(D, x_w)                                   # (n, d, f)

            # L_F x = D x - agg
            Lx = Dx - agg

            # Aplicar right weights W2 (mistura canais f -> f)
            if self.use_right_weights:
                Lx = self.W2[t](Lx.reshape(n * d, f)).reshape(n, d, f)

            if p: _sync(dev); self._timings[f'layer{t}_diffusion'] = time.time() - t0

            # (2) Dropout no Lx + GELU
            if self.training and self.dropout > 0:
                Lx = F.dropout(Lx, p=self.dropout, training=True)
            Lx = F.gelu(Lx)

            # --- Residual ---
            if self.use_eps:
                eps = self.epsilons[t]                             # (d, 1)
                scale = (1.0 + torch.tanh(eps)).unsqueeze(0)       # (1, d, 1)
                x = scale * x0 - Lx
            else:
                x = x0 - Lx

        # --- Output ---
        if p: _sync(dev); t0 = time.time()
        h = self.pre_mlp_norm(x.reshape(n, d * f))
        out = self.mlp(h)                              # (n, num_classes)
        if p: _sync(dev); self._timings['output_proj'] = time.time() - t0

        return out
