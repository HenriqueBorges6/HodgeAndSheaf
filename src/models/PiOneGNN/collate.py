import torch
from torch_geometric.data import Batch


def bundle_collate(data_list):
    edge_sizes = torch.tensor([data.edge_index.shape[1] for data in data_list])
    edge_offsets = torch.cat([torch.tensor([0]), torch.cumsum(edge_sizes, dim=0)[:-1]])

    all_idxs = []    # lista de tensores [L_c] com índices de arestas (shiftados)
    all_flips = []   # lista de tensores [L_c] com flags de inversão
    cycle_batch = []

    for k, (data, off_e) in enumerate(zip(data_list, edge_offsets)):
        off_e = off_e.item()
        for idxs, flips in zip(data.cycle_idxs, data.cycle_flips):
            all_idxs.append(idxs + off_e)
            all_flips.append(flips)
            cycle_batch.append(k)

    if all_idxs:
        cycle_lengths = torch.tensor([t.shape[0] for t in all_idxs])
        max_len = cycle_lengths.max().item()
        total_cycles = len(all_idxs)

        padded_idxs = torch.zeros(total_cycles, max_len, dtype=torch.long)
        padded_flips = torch.zeros(total_cycles, max_len, dtype=torch.bool)
        for i, (idxs, flips) in enumerate(zip(all_idxs, all_flips)):
            L = idxs.shape[0]
            padded_idxs[i, :L] = idxs
            padded_flips[i, :L] = flips
    else:
        cycle_lengths = torch.zeros(0, dtype=torch.long)
        padded_idxs = torch.zeros(0, 0, dtype=torch.long)
        padded_flips = torch.zeros(0, 0, dtype=torch.bool)

    batch = Batch.from_data_list(data_list, exclude_keys=['cycle_idxs', 'cycle_flips'])
    cycle_batch = torch.tensor(cycle_batch, dtype=torch.long)

    return batch, padded_idxs, padded_flips, cycle_lengths, cycle_batch
