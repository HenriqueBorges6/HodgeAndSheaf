import torch

def build_incidence(num_nodes, edge_index):
    num_edges = edge_index.shape[1]
    source = edge_index[0]
    target = edge_index[1]
    arangeE = torch.arange(num_edges, device=edge_index.device)
    B1 = torch.zeros(num_nodes, num_edges, device= edge_index.device)
    B1[source,arangeE] = -1
    B1[target,arangeE] = 1
    return B1

def build_line_graph_edge_index(num_nodes, edge_index):
    B1 = build_incidence(num_nodes,edge_index)
    line_edge_index = []
    signs = []
    for w in range(num_nodes):
        edge = torch.nonzero(B1[w, :], as_tuple= False).squeeze(1)
        k = edge.shape[0]
        if k < 2:
            continue

        ej = edge.unsqueeze(1).expand(-1, k)
        ek = edge.unsqueeze(0).expand(k, -1)

        ej_filtered = ej.reshape(-1)
        ek_filtered = ek.reshape(-1)
        mask = ej_filtered != ek_filtered
        ej_filtered = ej_filtered[mask]
        ek_filtered = ek_filtered[mask]
        line_edge_index.append(torch.stack([ej_filtered, ek_filtered], dim=0))  # (2, k_pares)

        sign = B1[w, ej_filtered] * B1[w, ek_filtered]
        signs.append(sign)
    return torch.cat(line_edge_index, dim=1), torch.cat(signs)

