"""
Diagnóstico: verifica inconsistências em cycle_idxs vs edge_index
para datasets que causam CUDA assert.

Uso:
    python -m src.experiments.pi_one.diagnose --dataset PROTEINS
    python -m src.experiments.pi_one.diagnose --dataset ENZYMES
"""

import argparse
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree as pyg_degree

from src.models.PiOneGNN.preprocessing import (
    extract_cycle_basis, build_edge_index_map, cycle_nodes_to_edges
)


def diagnose(dataset_name):
    print(f"\n=== Diagnóstico: {dataset_name} ===\n")

    raw = TUDataset(root=f'data/{dataset_name}', name=dataset_name)
    dataset = list(raw)
    print(f"Total de grafos: {len(dataset)}")

    # Mesma normalização de load_data: remap edge_index para 0-based por grafo
    for data in dataset:
        if data.edge_index.numel() == 0:
            continue
        unique_nodes, new_edge_index = torch.unique(data.edge_index, return_inverse=True)
        new_edge_index = new_edge_index.reshape(2, -1)
        n_unique = unique_nodes.shape[0]
        if data.x is not None and data.x.shape[0] != n_unique:
            old_n = data.x.shape[0]
            x_expanded = torch.zeros(n_unique, data.x.shape[1], dtype=data.x.dtype)
            mask = unique_nodes < old_n
            x_expanded[mask] = data.x[unique_nodes[mask]]
            data.x = x_expanded
        data.edge_index = new_edge_index
        data.num_nodes = n_unique

    # Mesmo tratamento que load_data
    for data in dataset:
        if data.x is None:
            deg = pyg_degree(data.edge_index[1], num_nodes=data.num_nodes).unsqueeze(-1).float()
            data.x = deg

    in_edge_dim = 1
    for data in dataset:
        if data.edge_attr is not None and data.edge_attr.shape[0] == data.edge_index.shape[1]:
            in_edge_dim = data.edge_attr.shape[1]
            break

    print(f"in_edge_dim detectado: {in_edge_dim}")
    print(f"in_node_dim: {dataset[0].x.shape[1]}")

    problems = []
    edge_attr_mismatch = 0
    node_idx_oob = 0

    for i, data in enumerate(dataset):
        num_edges = data.edge_index.shape[1]
        num_nodes_x = data.x.shape[0]
        num_nodes_declared = data.num_nodes

        # 1. Mismatch em edge_attr
        if data.edge_attr is not None and data.edge_attr.shape[0] != num_edges:
            edge_attr_mismatch += 1
            problems.append(
                f"  Grafo {i}: edge_attr.shape[0]={data.edge_attr.shape[0]} "
                f"!= edge_index.shape[1]={num_edges}"
            )

        # 2. Nó fora do range em edge_index (causa degree() e scatter a quebrarem)
        if num_edges > 0:
            max_node_idx = data.edge_index.max().item()
            if max_node_idx >= num_nodes_x:
                node_idx_oob += 1
                problems.append(
                    f"  Grafo {i}: edge_index.max()={max_node_idx} >= x.shape[0]={num_nodes_x} "
                    f"(num_nodes_declared={num_nodes_declared})"
                )

        # 3. num_nodes declarado != x.shape[0]
        if num_nodes_declared is not None and num_nodes_declared != num_nodes_x:
            problems.append(
                f"  Grafo {i}: num_nodes={num_nodes_declared} != x.shape[0]={num_nodes_x}"
            )

        # 4. cycle_idxs fora do range
        edge_map = build_edge_index_map(data.edge_index)
        cycles = extract_cycle_basis(data)
        for c_idx, cycle_nodes in enumerate(cycles):
            for (src, dst) in cycle_nodes_to_edges(cycle_nodes):
                if (src, dst) not in edge_map:
                    problems.append(
                        f"  Grafo {i}, ciclo {c_idx}: aresta ({src},{dst}) não está no edge_map!"
                    )
                    continue
                idx, _ = edge_map[(src, dst)]
                if idx < 0 or idx >= num_edges:
                    problems.append(
                        f"  Grafo {i}, ciclo {c_idx}: idx={idx} fora de [0, {num_edges})"
                    )

    print(f"\nGrafos com edge_attr.shape[0] != edge_index.shape[1]: {edge_attr_mismatch}")
    print(f"Grafos com edge_index.max() >= x.shape[0]          : {node_idx_oob}")
    print(f"Total de problemas encontrados                      : {len(problems)}")

    if problems:
        print("\nDetalhes (primeiros 20):")
        for p in problems[:20]:
            print(p)
    else:
        print("\nNenhum problema encontrado.")
        print("Tente rodar com CUDA_LAUNCH_BLOCKING=1 para ver o traceback real.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="PROTEINS")
    args = parser.parse_args()
    diagnose(args.dataset)
