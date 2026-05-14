"""
Learners de mapas de restrição (F) e co-restrição (C).

Estrutura fixa do SheafMapLearner:

    x (n, in_channels)
    → emb: Linear + ELU  → (n, hidden_dim)
    → backbone × gnn_layers → (n, hidden_dim)
    → readout: concat(h_src, h_tgt) → tanh → reshape → (m, d, d)

O backbone é configurável via string. Novos backbones são registrados com
@register_backbone('nome') — nenhuma outra mudança é necessária.

Backbones disponíveis:
    'node_edge_node'  agrega representações simétricas de arestas (padrão, custom)
    'edge_aware'      NodeEdgeNode + edge_attr na mensagem       (custom)
    'sage'            SAGEConv do PyG
    'gcn'             GCNConv do PyG
    'gat'             GATConv do PyG  (concat=False, heads configurável)
    'gin'             GINConv do PyG

Para adicionar um novo backbone:

    @register_backbone('meu_backbone')
    def _(hidden_dim, **kwargs):
        return MinhaConv(hidden_dim, hidden_dim)

Isso é suficiente — o SheafMapLearner usa o registry automaticamente.
"""

import inspect
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Backbones customizados (não disponíveis diretamente no PyG)
# ---------------------------------------------------------------------------

class NodeEdgeNode(nn.Module):
    """
    out_v = W_self · x_v + mean_{e ∋ v}( W_edge · (x_src + x_tgt) )

    Agrega representações simétricas de arestas. Diferente do SAGEConv:
    a mensagem mistura os dois endpoints antes de agregar, não propaga
    cada vizinho separadamente.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin_edge = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_self = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, tgt = edge_index
        n = x.shape[0]
        C = self.lin_edge.out_features

        edge_feat = self.lin_edge(x[src] + x[tgt])

        agg = torch.zeros(n, C, device=x.device, dtype=x.dtype)
        agg.scatter_add_(0, src.unsqueeze(1).expand(-1, C), edge_feat)
        agg.scatter_add_(0, tgt.unsqueeze(1).expand(-1, C), edge_feat)

        m   = src.shape[0]
        deg = torch.zeros(n, device=x.device, dtype=x.dtype)
        deg.scatter_add_(0, src, torch.ones(m, device=x.device, dtype=x.dtype))
        deg.scatter_add_(0, tgt, torch.ones(m, device=x.device, dtype=x.dtype))

        return self.lin_self(x) + agg / deg.clamp(min=1).unsqueeze(1)


class EdgeAwareConv(nn.Module):
    """
    out_v = W_self · x_v + mean_{e ∋ v}( W_edge · (x_src + x_tgt) + W_attr · edge_attr_e )

    NodeEdgeNode enriquecido com edge_attr na mensagem. Útil quando o dataset
    tem features de aresta informativas (ex: tipo de ligação no MUTAG).
    Para o learner de C, aproxima o domínio natural do co-feixe sem precisar
    construir o grafo de linhas.
    """

    def __init__(self, in_channels: int, out_channels: int, edge_attr_dim: int):
        super().__init__()
        self.lin_edge = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_attr = nn.Linear(edge_attr_dim, out_channels, bias=False)
        self.lin_self = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        src, tgt = edge_index
        n = x.shape[0]
        C = self.lin_edge.out_features

        edge_feat = self.lin_edge(x[src] + x[tgt]) + self.lin_attr(edge_attr)

        agg = torch.zeros(n, C, device=x.device, dtype=x.dtype)
        agg.scatter_add_(0, src.unsqueeze(1).expand(-1, C), edge_feat)
        agg.scatter_add_(0, tgt.unsqueeze(1).expand(-1, C), edge_feat)

        m   = src.shape[0]
        deg = torch.zeros(n, device=x.device, dtype=x.dtype)
        deg.scatter_add_(0, src, torch.ones(m, device=x.device, dtype=x.dtype))
        deg.scatter_add_(0, tgt, torch.ones(m, device=x.device, dtype=x.dtype))

        return self.lin_self(x) + agg / deg.clamp(min=1).unsqueeze(1)


class MLPConv(nn.Module):
    """
    out_v = MLP( x_v )

    Não usa estrutura do grafo — cada nó é transformado independentemente.
    Ignora vizinhos e arestas por completo. Útil como baseline para medir
    quanto da capacidade do modelo vem da topologia versus das features.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, edge_index=None, **_) -> torch.Tensor:
        return self.mlp(x)


class MLPEdgeConv(nn.Module):
    """
    out_v = MLP( x_v + mean_{e ∋ v}( W_attr · edge_attr_e ) )

    MLP com agregação de edge features nos nós. Usa o conteúdo das arestas
    mas ignora os embeddings dos vizinhos — não propaga x_u. A diferença em
    relação ao EdgeAwareConv é que aqui não há mistura de endpoints na mensagem:
    cada nó recebe só a média das features das suas arestas, não dos seus vizinhos.
    """

    def __init__(self, in_channels: int, out_channels: int, edge_attr_dim: int):
        super().__init__()
        self.lin_attr = nn.Linear(edge_attr_dim, in_channels, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        src, tgt = edge_index
        n = x.shape[0]
        C = x.shape[1]

        attr_proj = self.lin_attr(edge_attr)                    # (m, in_channels)

        agg = torch.zeros(n, C, device=x.device, dtype=x.dtype)
        agg.scatter_add_(0, src.unsqueeze(1).expand(-1, C), attr_proj)
        agg.scatter_add_(0, tgt.unsqueeze(1).expand(-1, C), attr_proj)

        m   = src.shape[0]
        deg = torch.zeros(n, device=x.device, dtype=x.dtype)
        deg.scatter_add_(0, src, torch.ones(m, device=x.device, dtype=x.dtype))
        deg.scatter_add_(0, tgt, torch.ones(m, device=x.device, dtype=x.dtype))

        return self.mlp(x + agg / deg.clamp(min=1).unsqueeze(1))


# ---------------------------------------------------------------------------
# Transformação ortogonal (auto-contida, sem depender de sheaf.py)
# ---------------------------------------------------------------------------

class Orthogonal(nn.Module):
    """
    Transforma parâmetros livres em uma matriz ortogonal d×d.

    Métodos disponíveis:
        'householder'  — produto de reflexões de Householder via torch_householder_orgqr
                         d(d-1)/2 parâmetros
        'matrix_exp'   — exponencial de matriz anti-simétrica e^A
                         d(d+1)/2 parâmetros
        'cayley'       — transformação de Cayley (I+A/2)^{-1}(I-A/2)
                         d(d+1)/2 parâmetros
        'euler'        — ângulos de Euler (só d=2 ou d=3)
                         d(d-1)/2 parâmetros

    O import de torch_householder é lazy — só carregado quando orth_trans='householder',
    sem exigir a dependência para os outros métodos.
    """

    def __init__(self, d: int, orthogonal_map: str):
        super().__init__()
        assert orthogonal_map in ('householder', 'matrix_exp', 'cayley', 'euler'), \
            f"orth_trans inválido: {orthogonal_map!r}"
        self.d              = d
        self.orthogonal_map = orthogonal_map

    def _euler_2d(self, params: torch.Tensor) -> torch.Tensor:
        theta = params[:, 0:1] * 2 * math.pi
        sin, cos = torch.sin(theta), torch.cos(theta)
        return torch.cat([cos, -sin, sin, cos], dim=1).view(-1, 2, 2)

    def _euler_3d(self, params: torch.Tensor) -> torch.Tensor:
        a = params[:, 0:1] * 2 * math.pi
        b = params[:, 1:2] * 2 * math.pi
        g = params[:, 2:3] * 2 * math.pi
        sa, ca = torch.sin(a), torch.cos(a)
        sb, cb = torch.sin(b), torch.cos(b)
        sg, cg = torch.sin(g), torch.cos(g)
        return torch.cat([
            ca*cb,  ca*sb*sg - sa*cg,  ca*sb*cg + sa*sg,
            sa*cb,  sa*sb*sg + ca*cg,  sa*sb*cg - ca*sg,
            -sb,    cb*sg,             cb*cg,
        ], dim=1).view(-1, 3, 3)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        d = self.d

        if self.orthogonal_map == 'euler':
            assert 2 <= d <= 3, "euler só suporta d=2 ou d=3"
            return self._euler_2d(params) if d == 2 else self._euler_3d(params)

        # Preenche triangular inferior com os parâmetros livres
        offset     = -1 if self.orthogonal_map == 'householder' else 0
        tril_idx   = torch.tril_indices(d, d, offset=offset, device=params.device)
        mat        = torch.zeros(params.size(0), d, d, dtype=params.dtype, device=params.device)
        mat[:, tril_idx[0], tril_idx[1]] = params

        if self.orthogonal_map == 'householder':
            from torch_householder import torch_householder_orgqr
            eye = torch.eye(d, device=params.device).unsqueeze(0).expand(params.size(0), -1, -1)
            return torch_householder_orgqr(mat.tril(diagonal=-1) + eye)

        # matrix_exp e cayley: constrói A anti-simétrica a partir do tril
        A = mat.tril() - mat.tril().transpose(-2, -1)
        if self.orthogonal_map == 'matrix_exp':
            return torch.matrix_exp(A)

        # cayley: (I - A/2)^{-1} (I + A/2)
        Id = torch.eye(d, dtype=A.dtype, device=A.device)
        return torch.linalg.solve(Id - A / 2, Id + A / 2)


# ---------------------------------------------------------------------------
# Funções auxiliares para tipos de mapa
# ---------------------------------------------------------------------------

def _sym_transform(raw: torch.Tensor, d: int) -> torch.Tensor:
    """Preenche triangular inferior e simetriza → matriz simétrica d×d."""
    m = raw.shape[0]
    idx = torch.tril_indices(d, d, device=raw.device)
    mat = torch.zeros(m, d, d, device=raw.device, dtype=raw.dtype)
    mat[:, idx[0], idx[1]] = raw
    # mat + mat^T - diag(mat): soma triângulo inferior + superior, diagonal contada uma vez
    diag = torch.diagonal(mat, dim1=-2, dim2=-1)
    return torch.tanh(mat + mat.transpose(-1, -2) - torch.diag_embed(diag))


def _sinkhorn(raw: torch.Tensor, d: int, n_iter: int = 5) -> torch.Tensor:
    """Normalização de Sinkhorn: linha/coluna alternada → duplamente estocástica."""
    mat = torch.exp(raw.reshape(-1, d, d))
    for _ in range(n_iter):
        mat = mat / mat.sum(-1, keepdim=True).clamp(min=1e-9)
        mat = mat / mat.sum(-2, keepdim=True).clamp(min=1e-9)
    return mat


# ---------------------------------------------------------------------------
# Registry de tipos de mapa
# ---------------------------------------------------------------------------

# Cada entrada define como o readout produz uma matriz d×d por aresta.
# Campos:
#   param_size : callable(d) → int   — quantos escalares o readout produz
#   init       : str                 — estratégia de init do bias do readout
#   transform  : callable(raw, d, module) → (m, d, d)
#                module é nn.Module|None — usado apenas para 'orthogonal'

_MAP_REGISTRY: dict = {
    'general': {
        'param_size': lambda d: d * d,
        'init': 'identity',
        'transform': lambda raw, d, _: torch.tanh(raw).reshape(-1, d, d),
    },
    'diagonal': {
        # Apenas d parâmetros — mapas diagonais, eixos do stalk independentes
        'param_size': lambda d: d,
        'init': 'ones',
        'transform': lambda raw, d, _: torch.diag_embed(torch.tanh(raw)),
    },
    'symmetric': {
        # F = F^T — d(d+1)/2 parâmetros
        'param_size': lambda d: d * (d + 1) // 2,
        'init': 'tril_identity',
        'transform': lambda raw, d, _: _sym_transform(raw, d),
    },
    'orthogonal': {
        # F^T F = I — usa classe Orthogonal de SheafNN/sheaf.py
        # param_size depende de orth_trans → definido em SheafMapLearner.__init__
        'param_size': None,
        'init': 'zero',
        'transform': lambda raw, d, module: module(raw),
    },
    'stochastic': {
        # Linhas somam 1 via softmax — generaliza permutações
        'param_size': lambda d: d * d,
        'init': 'zero',
        'transform': lambda raw, d, _: torch.softmax(raw.reshape(-1, d, d), dim=-1),
    },
    'permutation': {
        # Aproxima matriz de permutação via Sinkhorn — duplamente estocástica
        'param_size': lambda d: d * d,
        'init': 'zero',
        'transform': lambda raw, d, _: _sinkhorn(raw, d),
    },
}


def _init_readout(readout: nn.Linear, map_type: str, d: int) -> None:
    """Inicializa o readout de acordo com o tipo de mapa."""
    init = _MAP_REGISTRY[map_type]['init']
    nn.init.normal_(readout.weight.data, std=1e-3)

    if init == 'identity':
        readout.bias.data.copy_(torch.eye(d).reshape(-1))
    elif init == 'ones':
        # diagonal: tanh(1) ≈ 0.76 → D^F_v ≈ 0.58·deg·I, bem condicionado
        readout.bias.data.fill_(1.0)
    elif init == 'tril_identity':
        # triangular inferior de I_d → após simetrização produz I_d
        idx = torch.tril_indices(d, d)
        bias = torch.zeros(d * (d + 1) // 2)
        # elementos diagonais (onde idx[0] == idx[1]) recebem 1
        diag_mask = idx[0] == idx[1]
        bias[diag_mask] = 1.0
        readout.bias.data.copy_(bias)
    elif init == 'zero':
        nn.init.zeros_(readout.bias.data)


# ---------------------------------------------------------------------------
# Registry de backbones
# ---------------------------------------------------------------------------

_BACKBONE_REGISTRY: dict = {}


def register_backbone(name: str):
    """
    Decorator que registra uma fábrica de backbone.

    A função decorada recebe (hidden_dim, **kwargs) e retorna UMA camada.
    O SheafMapLearner instancia gnn_layers cópias independentes.

    Exemplo:
        @register_backbone('minha_conv')
        def _(hidden_dim, **kwargs):
            return MinhaConv(hidden_dim, hidden_dim)
    """
    def decorator(fn):
        _BACKBONE_REGISTRY[name] = fn
        return fn
    return decorator


@register_backbone('node_edge_node')
def _(hidden_dim: int, **_) -> nn.Module:
    return NodeEdgeNode(hidden_dim, hidden_dim)


@register_backbone('edge_aware')
def _(hidden_dim: int, edge_attr_dim: int = None, **_) -> nn.Module:
    assert edge_attr_dim is not None, "backbone='edge_aware' requer edge_attr_dim"
    return EdgeAwareConv(hidden_dim, hidden_dim, edge_attr_dim)


@register_backbone('sage')
def _(hidden_dim: int, **_) -> nn.Module:
    from torch_geometric.nn import SAGEConv
    return SAGEConv(hidden_dim, hidden_dim)


@register_backbone('gcn')
def _(hidden_dim: int, **_) -> nn.Module:
    from torch_geometric.nn import GCNConv
    return GCNConv(hidden_dim, hidden_dim)


@register_backbone('gat')
def _(hidden_dim: int, gat_heads: int = 1, **_) -> nn.Module:
    from torch_geometric.nn import GATConv
    # concat=False: a saída é a média das heads → mantém dim = hidden_dim
    return GATConv(hidden_dim, hidden_dim, heads=gat_heads, concat=False)


@register_backbone('gin')
def _(hidden_dim: int, **_) -> nn.Module:
    from torch_geometric.nn import GINConv
    return GINConv(nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
    ))


@register_backbone('mlp')
def _(hidden_dim: int, **_) -> nn.Module:
    return MLPConv(hidden_dim, hidden_dim)


@register_backbone('mlp_edge')
def _(hidden_dim: int, edge_attr_dim: int = None, **_) -> nn.Module:
    assert edge_attr_dim is not None, "backbone='mlp_edge' requer edge_attr_dim"
    return MLPEdgeConv(hidden_dim, hidden_dim, edge_attr_dim)


# ---------------------------------------------------------------------------
# Utilitário: chamada uniforme independente de assinatura
# ---------------------------------------------------------------------------

def _call_conv(
    conv: nn.Module,
    h: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
) -> torch.Tensor:
    """
    Chama conv(h, edge_index) ou conv(h, edge_index, edge_attr=edge_attr)
    conforme a assinatura do módulo.

    Isso permite misturar backbones PyG (que aceitam edge_attr) com backbones
    customizados (que não aceitam) sem if/else no forward principal.
    """
    sig = inspect.signature(conv.forward)
    if edge_attr is not None and 'edge_attr' in sig.parameters:
        return conv(h, edge_index, edge_attr=edge_attr)
    return conv(h, edge_index)


# ---------------------------------------------------------------------------
# Fábrica de backbone
# ---------------------------------------------------------------------------

def _build_backbone(
    name: str,
    hidden_dim: int,
    gnn_layers: int,
    **kwargs,
) -> nn.ModuleList:
    if name not in _BACKBONE_REGISTRY:
        raise ValueError(
            f"backbone desconhecido: {name!r}. "
            f"Registrados: {sorted(_BACKBONE_REGISTRY)}"
        )
    factory = _BACKBONE_REGISTRY[name]
    return nn.ModuleList([factory(hidden_dim, **kwargs) for _ in range(gnn_layers)])


# ---------------------------------------------------------------------------
# Learner de mapas de restrição / co-restrição
# ---------------------------------------------------------------------------

class SheafMapLearner(nn.Module):
    """
    Aprende um mapa d×d por aresta dirigida.

    Pipeline:
        x (n, in_channels)
        → emb               → (n, hidden_dim)
        → backbone × layers → (n, hidden_dim)
        → readout           → raw params por aresta
        → map_transform     → (m, d, d)

    Args:
        in_channels:      dimensão de entrada (d · f após input_proj)
        hidden_dim:       dimensão oculta do backbone
        gnn_layers:       número de camadas do backbone
        d:                dimensão do stalk
        backbone:         nome registrado em _BACKBONE_REGISTRY
        map_type:         tipo do mapa de restrição — chave de _MAP_REGISTRY
                          'general' | 'diagonal' | 'symmetric' |
                          'orthogonal' | 'stochastic' | 'permutation'
        orth_trans:       parametrização ortogonal; usado só quando map_type='orthogonal'
                          'householder' | 'cayley' | 'matrix_exp' | 'euler'
        **backbone_kwargs: repassados à fábrica do backbone
                          ex: edge_attr_dim=4 para 'edge_aware', gat_heads=4 para 'gat'
    """

    _SHEAF_ACTS = {
        'id':      lambda x: x,
        'tanh':    torch.tanh,
        'relu':    F.relu,
        'gelu':    F.gelu,
        'sigmoid': torch.sigmoid,
        'elu':     F.elu,
    }

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        gnn_layers: int,
        d: int,
        backbone: str = 'node_edge_node',
        map_type: str = 'general',
        orth_trans: str = 'householder',
        sheaf_act: str = 'id',
        **backbone_kwargs,
    ):
        super().__init__()
        assert map_type in _MAP_REGISTRY, (
            f"map_type inválido: {map_type!r}. Opções: {sorted(_MAP_REGISTRY)}"
        )
        assert sheaf_act in self._SHEAF_ACTS, (
            f"sheaf_act inválido: {sheaf_act!r}. Opções: {sorted(self._SHEAF_ACTS)}"
        )
        self.d        = d
        self.map_type = map_type
        self._sheaf_act = self._SHEAF_ACTS[sheaf_act]

        self.emb   = nn.Linear(in_channels, hidden_dim)
        self.convs = _build_backbone(backbone, hidden_dim, gnn_layers, **backbone_kwargs)

        # Determina quantos parâmetros o readout deve produzir
        map_info = _MAP_REGISTRY[map_type]
        if map_type == 'orthogonal':
            self.map_module = Orthogonal(d=d, orthogonal_map=orth_trans)
            if orth_trans in ('euler', 'householder'):
                param_size = d * (d - 1) // 2
            else:
                param_size = d * (d + 1) // 2
        else:
            self.map_module = None
            param_size = map_info['param_size'](d)

        self.readout  = nn.Linear(2 * hidden_dim, param_size)
        self._transform = map_info['transform']
        _init_readout(self.readout, map_type, d)

    def forward(
        self,
        x_flat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x_flat:     (n, in_channels)
            edge_index: (2, m)
            edge_attr:  (m, edge_attr_dim) ou None

        Retorna (m, d, d) — um mapa por aresta dirigida.
        """
        src, tgt = edge_index
        h = F.elu(self.emb(x_flat))
        for conv in self.convs:
            h = F.elu(_call_conv(conv, h, edge_index, edge_attr))
        raw = self.readout(torch.cat([h[src], h[tgt]], dim=1))
        raw = self._sheaf_act(raw)
        return self._transform(raw, self.d, self.map_module)
