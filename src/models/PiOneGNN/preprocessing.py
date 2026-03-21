import warnings
warnings.filterwarnings("ignore")

import torch
import networkx as nx


def extract_cycle_basis(data):
    edges = data.edge_index.t().tolist()
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    G.add_edges_from(edges)
    return nx.cycle_basis(G)


def cycle_nodes_to_edges(cycle):
    return list(zip(cycle, cycle[1:] + [cycle[0]]))


def build_edge_index_map(edge_index):
    edge_map = dict()
    for i, j in enumerate(edge_index.t().tolist()):
        edge = tuple(j)
        reverse_edge = tuple(j[::-1])
        edge_map[edge] = (i, False)
        edge_map[reverse_edge] = (i, True)
    return edge_map


def preprocess_dataset(dataset):
    for data in dataset:
        cycles_nodes = extract_cycle_basis(data)
        edge_map = build_edge_index_map(data.edge_index)

        cycle_idxs = []   # lista de tensores [L_c], índices de arestas
        cycle_flips = []  # lista de tensores [L_c], bool (transpor O?)

        for cycle_nodes in cycles_nodes:
            cycle_edges = cycle_nodes_to_edges(cycle_nodes)
            idxs, flips = [], []
            for (src, dst) in cycle_edges:
                idx, is_rev = edge_map[(src, dst)]
                idxs.append(idx)
                flips.append(is_rev)
            cycle_idxs.append(torch.tensor(idxs, dtype=torch.long))
            cycle_flips.append(torch.tensor(flips, dtype=torch.bool))

        data.cycle_idxs = cycle_idxs
        data.cycle_flips = cycle_flips
