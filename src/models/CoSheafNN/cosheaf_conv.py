import time

import torch
import torch.nn as nn

from .cosheaf_learner import CoSheafLearner


def _sync(device):
    """Sincroniza CUDA antes de medir tempo (noop em CPU)."""
    if device.type == 'cuda':
        torch.cuda.synchronize()


class CoSheafConv(nn.Module):
    """
    Camada de convolucao para co-feixes planos.

    Implementa uma iteracao do Laplaciano do co-feixe:
        x'_e = D_e * x_e - sum_{e' ~ e} sign(e,e') * F̂_e^T F̂_e' x_e'

    Opera sobre arestas do grafo original (= nos do grafo de linhas).
    Usa scatter_add manual com .contiguous() para evitar crash no kernel C++.
    """

    def __init__(self, in_channels: int, out_channels: int, d: int, learner_hidden: list[int],
                 backbone: str = 'sage', orth_method: str = 'cayley'):
        super().__init__()
        self.d = d
        self.learner = CoSheafLearner(in_channels, learner_hidden, d, backbone, orth_method)
        # d projecoes independentes: cada dimensao do stalk recebe embedding proprio
        self.stalk_linears = nn.ModuleList([
            nn.Linear(in_channels, out_channels) for _ in range(d)
        ])
        self._profile = False
        self._timings = {}

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        line_edge_index: torch.Tensor,
        signs: torch.Tensor,
    ) -> torch.Tensor:
        # x               : (m, in_channels)
        # edge_index       : (2, m)
        # line_edge_index  : (2, E_L)
        # signs            : (E_L,)
        # retorna          : (m, d, out_channels)
        p   = self._profile
        dev = x.device

        if p: _sync(dev); t0 = time.time()
        maps = self.learner(x, line_edge_index)                            # (m, d, d)
        if p: _sync(dev); self._timings['learn_maps'] = time.time() - t0

        if p: t0 = time.time()
        # Cada dimensao do stalk recebe projecao propria (quebra simetria)
        x    = torch.stack([lin(x) for lin in self.stalk_linears], dim=1)  # (m, d, out_channels)
        if p: _sync(dev); self._timings['stalk_embed'] = time.time() - t0

        if p: t0 = time.time()
        num_edges = edge_index.shape[1]
        src, tgt  = line_edge_index

        x_j    = x[src]                                                    # (E_L, d, out_channels)
        maps_j = maps[src]                                                 # (E_L, d, d)
        maps_i = maps[tgt]                                                 # (E_L, d, d)

        msg  = maps_i.transpose(-2, -1) @ (maps_j @ x_j)                  # (E_L, d, out_channels)
        msg  = signs.view(-1, 1, 1) * msg

        C    = x.shape[-1]
        idx  = tgt.view(-1, 1, 1).expand(-1, self.d, C).contiguous()      # contiguous evita crash no kernel C++
        agg  = torch.zeros(num_edges, self.d, C, device=x.device, dtype=x.dtype).scatter_add(0, idx, msg)
        D    = torch.zeros(num_edges, 1, 1, device=x.device, dtype=x.dtype).scatter_add(
                   0, tgt.view(-1, 1, 1), torch.ones(tgt.shape[0], 1, 1, device=x.device, dtype=x.dtype)
               )
        if p: _sync(dev); self._timings['diffusion'] = time.time() - t0

        return D * x - agg
