"""
Hodge Laplacian utilities
Based on Park et al., MICCAI 2023
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import coo_matrix


def compute_hodge_laplacian_L1(edges, num_nodes):
    """
    Compute Hodge 1-Laplacian L1 = B1^T B1

    L1(i,j) captures if edges ei and ej share a node:
    - L1(i,i) = 2 (self-loop)
    - L1(i,j) = -2 if ei=(u,v), ej=(v,u) (reverse edge)
    - L1(i,j) = 1 if edges share exactly one node with same orientation
    - L1(i,j) = -1 if edges share exactly one node with opposite orientation
    - L1(i,j) = 0 otherwise

    Parameters:
    -----------
    edges : np.ndarray, shape (num_edges, 2)
        Edge list where each row is [source, target]
    num_nodes : int
        Number of nodes in the graph

    Returns:
    --------
    L1 : np.ndarray, shape (num_edges, num_edges)
        Hodge 1-Laplacian matrix
    """
    num_edges = len(edges)

    # Build incidence matrix B1: nodes x edges
    # B1(v, e) = +1 if v is target of e, -1 if v is source of e
    row_idx = []
    col_idx = []
    data = []

    for e_idx, (u, v) in enumerate(edges):
        # Source node u
        row_idx.append(u)
        col_idx.append(e_idx)
        data.append(-1)

        # Target node v
        row_idx.append(v)
        col_idx.append(e_idx)
        data.append(+1)

    B1 = coo_matrix((data, (row_idx, col_idx)),
                    shape=(num_nodes, num_edges)).tocsr()

    # Compute L1 = B1^T @ B1
    L1 = (B1.T @ B1).toarray()

    return L1


def prepare_edge_features(edge_labels, num_edge_types):
    """
    Convert edge labels to one-hot encoded features

    Parameters:
    -----------
    edge_labels : np.ndarray, shape (num_edges,)
        Edge type labels (already 0-indexed: 0, 1, 2, 3, ...)
    num_edge_types : int
        Number of distinct edge types

    Returns:
    --------
    features : torch.Tensor, shape (num_edges, num_edge_types)
        One-hot encoded edge features
    """
    # Edge labels in MUTAG are already 0-indexed [0, 1, 2, 3]
    # No conversion needed
    edge_labels_tensor = torch.tensor(edge_labels, dtype=torch.long)

    # One-hot encode
    features = F.one_hot(edge_labels_tensor, num_classes=num_edge_types).float()

    return features


def normalize_hodge_laplacian(L1, method='symmetric'):
    """
    Normalize Hodge Laplacian

    Parameters:
    -----------
    L1 : np.ndarray or torch.Tensor
        Hodge Laplacian matrix
    method : str
        'symmetric': D^(-1/2) L1 D^(-1/2)
        'random_walk': D^(-1) L1

    Returns:
    --------
    L1_norm : same type as input
        Normalized Laplacian
    """
    is_numpy = isinstance(L1, np.ndarray)

    if is_numpy:
        L1_t = torch.from_numpy(L1).float()
    else:
        L1_t = L1

    # Compute degree matrix
    D = torch.diag(L1_t.diagonal())
    D_inv_sqrt = torch.diag(1.0 / (torch.sqrt(D.diagonal()) + 1e-8))

    if method == 'symmetric':
        L1_norm = D_inv_sqrt @ L1_t @ D_inv_sqrt
    elif method == 'random_walk':
        D_inv = torch.diag(1.0 / (D.diagonal() + 1e-8))
        L1_norm = D_inv @ L1_t
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    if is_numpy:
        return L1_norm.numpy()
    else:
        return L1_norm
