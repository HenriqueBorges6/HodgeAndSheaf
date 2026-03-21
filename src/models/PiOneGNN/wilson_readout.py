import torch
import torch.nn as nn
from torch_scatter import scatter_add


class WilsonReadout(nn.Module):
    def __init__(self, num_powers=3):
        super().__init__()
        self.num_powers = num_powers

    def forward(self, x, O_edges, cycle_edge_idxs, cycle_flip_masks, cycle_lengths, cycle_batch, batch):
        device = O_edges.device
        d = O_edges.shape[-1]

        # --- Normas nodais ---
        nodal_norms = (x ** 2).sum(dim=-1)
        nodal_norms = scatter_add(nodal_norms, batch, dim=0)  # [B]

        total_cycles = cycle_edge_idxs.shape[0]

        if total_cycles > 0:
            max_len = cycle_edge_idxs.shape[1]

            # Coleta matrizes O para todos os ciclos: [total_cycles, max_len, d, d]
            O_all = O_edges[cycle_edge_idxs]

            # Aplica transposta onde a aresta é reversa
            flip_exp = cycle_flip_masks.unsqueeze(-1).unsqueeze(-1)  # [total_cycles, max_len, 1, 1]
            O_all = torch.where(flip_exp, O_all.transpose(-1, -2), O_all)

            # Substitui posições de padding por identidade
            pad_mask = torch.arange(max_len, device=device).unsqueeze(0) >= cycle_lengths.unsqueeze(1)
            I = torch.eye(d, device=device)
            O_all = torch.where(pad_mask.unsqueeze(-1).unsqueeze(-1), I, O_all)

            # Produto sequencial: H = O_L @ ... @ O_1  — loop sobre max_len (não sobre ciclos)
            H = I.unsqueeze(0).expand(total_cycles, -1, -1).clone()  # [total_cycles, d, d]
            for step in range(max_len):
                H = torch.bmm(O_all[:, step], H)

            # Momentos espectrais: tr(H^1), tr(H^2), ..., tr(H^num_powers)
            traces_list = []
            H_power = H.clone()
            for k in range(self.num_powers):
                traces_list.append(torch.diagonal(H_power, dim1=-2, dim2=-1).sum(-1))  # [total_cycles]
                H_power = torch.bmm(H_power, H)

            wilson_loops = torch.stack(traces_list, dim=-1)  # [total_cycles, num_powers]
            wilson_agg = scatter_add(wilson_loops, cycle_batch, dim=0, dim_size=batch.max() + 1)  # [B, num_powers]
        else:
            H = torch.zeros(0, d, d, device=device)
            wilson_agg = torch.zeros(batch.max() + 1, self.num_powers, device=device)

        nodal_norms = nodal_norms.unsqueeze(-1)  # [B, 1]
        return torch.cat([nodal_norms, wilson_agg], dim=-1), H  # [B, 1 + num_powers], [total_cycles, d, d]
