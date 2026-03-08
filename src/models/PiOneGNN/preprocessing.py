import warnings
warnings.filterwarnings("ignore")

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
        cycles = extract_cycle_basis(data)
        data.cycles = []
        for cycle in cycles:
            data.cycles.append(cycle_nodes_to_edges(cycle))
        data.edge_map = build_edge_index_map(data.edge_index)  
