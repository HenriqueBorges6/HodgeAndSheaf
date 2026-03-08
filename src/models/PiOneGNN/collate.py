import torch
from torch_geometric.data import Batch


def bundle_collate(data_list):
    sizes = torch.tensor([data.num_nodes for data in data_list])
    offsets = torch.cat([torch.tensor([0]), torch.cumsum(sizes, dim=0)[:-1]])

    all_cycles = []
    cycle_batch = []
    for k, (data, offset) in enumerate(zip(data_list, offsets)):
        off = offset.item()  
        for cycle in data.cycles:
            shifted_cycle = [(src + off, dst + off) for src, dst in cycle]
            all_cycles.append(shifted_cycle)
            cycle_batch.append(k)
    edge_sizes = torch.tensor([data.edge_index.shape[1] for data in data_list])
    edge_offsets = torch.cat([torch.tensor([0]), torch.cumsum(edge_sizes, dim=0)[:-1]])
    global_edge_map = {}
    for (data, off_n, off_e) in zip(data_list, offsets, edge_offsets):
        off_n = off_n.item()
        off_e = off_e.item()
        for (src, dst), (idx, is_reversed) in data.edge_map.items():
            global_edge_map[(src + off_n, dst + off_n)] = (idx + off_e, is_reversed)
    batch = Batch.from_data_list(data_list, exclude_keys=['cycles', 'edge_map'])
    cycle_batch = torch.tensor(cycle_batch)
    return batch, all_cycles, global_edge_map, cycle_batch
