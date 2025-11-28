"""
Data loading utilities for graph datasets
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import kagglehub


def load_mutag_dataset():
    """
    Load MUTAG dataset from Kaggle

    Returns:
    --------
    data : dict
        Dictionary containing:
        - edges: np.ndarray, shape (num_edges, 2)
        - graph_indicator: np.ndarray, shape (num_nodes,)
        - graph_labels: np.ndarray, shape (num_graphs,)
        - node_labels: np.ndarray, shape (num_nodes,)
        - edge_labels: np.ndarray, shape (num_edges,)
    """
    # Download dataset
    path = kagglehub.dataset_download("tammodukker/mutag-dataset")
    print(f"Dataset path: {path}")

    # Load files
    edges = np.loadtxt(f"{path}/MUTAG/MUTAG_A.txt", delimiter=',', dtype=int)
    graph_indicator = np.loadtxt(f"{path}/MUTAG/MUTAG_graph_indicator.txt", dtype=int)
    graph_labels = np.loadtxt(f"{path}/MUTAG/MUTAG_graph_labels.txt", dtype=int)
    node_labels = np.loadtxt(f"{path}/MUTAG/MUTAG_node_labels.txt", dtype=int)
    edge_labels = np.loadtxt(f"{path}/MUTAG/MUTAG_edge_labels.txt", dtype=int)

    # Print dataset info
    print(f"\nMUTAG Dataset Statistics:")
    print(f"  Number of graphs: {len(graph_labels)}")
    print(f"  Number of nodes: {len(graph_indicator)}")
    print(f"  Number of edges: {len(edges)}")
    print(f"  Classes: {np.unique(graph_labels)} (-1: non-mutagenic, 1: mutagenic)")
    print(f"  Node label types: {len(np.unique(node_labels))}")
    print(f"  Edge label types: {len(np.unique(edge_labels))}")

    return {
        'edges': edges,
        'graph_indicator': graph_indicator,
        'graph_labels': graph_labels,
        'node_labels': node_labels,
        'edge_labels': edge_labels
    }


def extract_single_graph(graph_id, edges, graph_indicator, edge_labels, node_labels=None):
    """
    Extract a single graph from the dataset

    Parameters:
    -----------
    graph_id : int
        Graph ID (1-indexed)
    edges : np.ndarray
        All edges in dataset
    graph_indicator : np.ndarray
        Node to graph assignment
    edge_labels : np.ndarray
        Edge labels
    node_labels : np.ndarray, optional
        Node labels

    Returns:
    --------
    graph_data : dict
        Dictionary containing:
        - edges: np.ndarray, shape (num_graph_edges, 2), 0-indexed
        - edge_features: np.ndarray, shape (num_graph_edges,)
        - num_nodes: int
        - node_features: np.ndarray or None
    """
    # Find nodes belonging to this graph
    node_mask = (graph_indicator == graph_id)
    # IMPORTANT: node IDs in MUTAG are 1-indexed, so add +1
    node_ids = np.where(node_mask)[0] + 1
    num_nodes = len(node_ids)

    # Create mapping from old node IDs (1-indexed) to new (0-indexed)
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(node_ids)}

    # Find edges belonging to this graph
    edge_mask = np.isin(edges[:, 0], node_ids) & np.isin(edges[:, 1], node_ids)
    graph_edges_original = edges[edge_mask]
    graph_edge_labels = edge_labels[edge_mask]

    # Remap edge indices to new node IDs
    graph_edges = np.array([
        [old_to_new[src], old_to_new[tgt]]
        for src, tgt in graph_edges_original
    ])

    result = {
        'edges': graph_edges,
        'edge_features': graph_edge_labels,
        'num_nodes': num_nodes
    }

    # Add node features if available
    if node_labels is not None:
        graph_node_labels = node_labels[node_mask]
        result['node_features'] = graph_node_labels

    return result


class MUTAGDataset(Dataset):
    """
    PyTorch Dataset for MUTAG graphs
    """

    def __init__(self, mutag_data, transform=None):
        """
        Parameters:
        -----------
        mutag_data : dict
            Output from load_mutag_dataset()
        transform : callable, optional
            Optional transform to be applied on a sample
        """
        self.edges = mutag_data['edges']
        self.graph_indicator = mutag_data['graph_indicator']
        self.graph_labels = mutag_data['graph_labels']
        self.edge_labels = mutag_data['edge_labels']
        self.node_labels = mutag_data['node_labels']
        self.transform = transform

        # Convert labels to binary (0, 1)
        self.binary_labels = ((self.graph_labels + 1) // 2).astype(int)

        self.num_graphs = len(self.graph_labels)

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, idx):
        """
        Get a single graph

        Returns:
        --------
        sample : dict
            - graph_data: dict from extract_single_graph
            - label: int (0 or 1)
        """
        graph_id = idx + 1  # Graph IDs are 1-indexed

        graph_data = extract_single_graph(
            graph_id,
            self.edges,
            self.graph_indicator,
            self.edge_labels,
            self.node_labels
        )

        sample = {
            'graph_data': graph_data,
            'label': self.binary_labels[idx]
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
