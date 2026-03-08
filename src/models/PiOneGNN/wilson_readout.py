import torch
import torch.nn as nn
from torch_scatter import scatter_add

class WilsonReadout(nn.Module):
    def __init__(self, num_powers = 3):
        super().__init__()
        self.num_powers = num_powers

    def forward(self, x, O_edges, cycles, edge_map, batch, cycle_batch):
        nodal_norms = (torch.pow(x, 2)).sum(dim=-1)
        
        nodal_norms =  scatter_add(nodal_norms,batch, dim=0)
        holonomies = []
        wilson_loops = []
        for cycle in cycles:            

            H = torch.eye(O_edges.shape[2],device=O_edges.device)
            for (src,dst) in cycle:
                idx, is_reversed = edge_map[(src,dst)]
                O = O_edges[idx]
                if is_reversed:
                    O = O.T
                
                H = O @ H
            
            holonomies.append(H)

            H_power = H.clone()
            traces = []
            for k in range(self.num_powers):
                H_power = H_power @ H
                traces.append(torch.trace(H_power))
            wilson_loops.append(torch.stack(traces))

        
        wilson_loops = torch.stack(wilson_loops)           
        wilson_agg = scatter_add(wilson_loops, cycle_batch, dim=0, dim_size=batch.max()+1)  
        nodal_norms = nodal_norms.unsqueeze(-1)  

        return torch.cat([nodal_norms, wilson_agg], dim=-1),  holonomies  
