"""
Sheaf Convolutional Network (SCN)
Based on "Neural Sheaf Diffusion" (Bodnar et al., NeurIPS 2022)

Implementation of Equation 4 from the paper:
Y = σ((I_nd - ΔF)(I_n ⊗ W1)XW2)

where:
- ΔF is the normalized sheaf Laplacian
- W1 is a d×d weight matrix (stalk transformation)
- W2 is an f1×f2 weight matrix (feature channels)
- σ is a non-linearity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SheafConvolutionalLayer(nn.Module):
    """
    Single Sheaf Convolutional Layer

    Implements: Y = σ((I_nd - ΔF)(I_n ⊗ W1)XW2)
    """

    def __init__(self, in_channels, out_channels, stalk_dim, dropout=0.0):
        """
        Parameters:
        -----------
        in_channels : int
            Number of input feature channels (f1)
        out_channels : int
            Number of output feature channels (f2)
        stalk_dim : int
            Dimension of stalks (d)
        dropout : float
            Dropout rate
        """
        super(SheafConvolutionalLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stalk_dim = stalk_dim

        # W1: d × d matrix for stalk transformations
        self.W1 = nn.Parameter(torch.Tensor(stalk_dim, stalk_dim))

        # W2: f1 × f2 matrix for feature channel projection
        self.W2 = nn.Parameter(torch.Tensor(in_channels, out_channels))

        # Dropout
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)

    def forward(self, X, Delta_F):
        """
        Forward pass

        Parameters:
        -----------
        X : torch.Tensor, shape (num_edges * d, in_channels)
            Input features on edges
        Delta_F : torch.Tensor, shape (num_edges * d, num_edges * d)
            Normalized sheaf Laplacian

        Returns:
        --------
        Y : torch.Tensor, shape (num_edges * d, out_channels)
            Output features
        """
        num_edges_d = X.shape[0]
        num_edges = num_edges_d // self.stalk_dim
        d = self.stalk_dim

        # (I_n ⊗ W1)X
        # Reshape X to (num_edges, d, in_channels)
        X_reshaped = X.reshape(num_edges, d, self.in_channels)

        # Apply W1 via batch matrix multiplication
        # W1^T @ X_v for each edge v
        X_transformed = torch.bmm(
            self.W1.T.unsqueeze(0).expand(num_edges, -1, -1),  # (num_edges, d, d)
            X_reshaped  # (num_edges, d, in_channels)
        )

        # Flatten back: (num_edges * d, in_channels)
        X_transformed = X_transformed.reshape(num_edges_d, self.in_channels)

        # Apply W2: (num_edges * d, in_channels) @ (in_channels, out_channels)
        X_transformed = X_transformed @ self.W2

        # (I_nd - ΔF) @ X_transformed
        I_nd = torch.eye(num_edges_d, device=X.device)
        conv_matrix = I_nd - Delta_F
        Y = conv_matrix @ X_transformed

        # Activation
        Y = F.elu(Y)

        # Dropout
        Y = self.dropout(Y)

        return Y


class SheafBuilder(nn.Module):
    """
    Learn restriction maps from edge features

    Implements Φ(x_v, x_u) to learn F_v⊆e restriction maps
    """

    def __init__(self, edge_feature_dim, stalk_dim, sheaf_type='diagonal'):
        """
        Parameters:
        -----------
        edge_feature_dim : int
            Dimension of edge features
        stalk_dim : int
            Dimension of stalks (d)
        sheaf_type : str
            Type of sheaf: 'diagonal', 'orthogonal', 'general'
        """
        super(SheafBuilder, self).__init__()

        self.edge_feature_dim = edge_feature_dim
        self.stalk_dim = stalk_dim
        self.sheaf_type = sheaf_type

        # Output dimension
        if sheaf_type == 'diagonal':
            output_dim = stalk_dim
        elif sheaf_type == 'orthogonal':
            output_dim = stalk_dim
        else:  # general
            output_dim = stalk_dim * stalk_dim

        # MLP: Φ(concat(x_v, x_u)) -> restriction map parameters
        self.mlp = nn.Sequential(
            nn.Linear(2 * edge_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

        if sheaf_type == 'orthogonal':
            self.mlp.add_module('tanh', nn.Tanh())

    def _householder_matrix(self, v):
        """
        Build orthogonal matrix via Householder reflection
        H = I - 2vv^T / ||v||^2

        Parameters:
        -----------
        v : torch.Tensor, shape (batch, stalk_dim)

        Returns:
        --------
        H : torch.Tensor, shape (batch, stalk_dim, stalk_dim)
        """
        batch_size = v.shape[0]

        # Normalize v
        v_norm = torch.norm(v, dim=1, keepdim=True).clamp(min=1e-6)
        v_normalized = v / v_norm

        # H = I - 2vv^T
        I = torch.eye(self.stalk_dim, device=v.device).unsqueeze(0).expand(batch_size, -1, -1)
        vvT = torch.bmm(v_normalized.unsqueeze(2), v_normalized.unsqueeze(1))
        H = I - 2 * vvT

        return H

    def forward(self, edge_features, adjacency_matrix):
        """
        Build restriction maps

        Parameters:
        -----------
        edge_features : torch.Tensor, shape (num_edges, edge_feature_dim)
            Features for each edge
        adjacency_matrix : torch.Tensor, shape (num_edges, num_edges)
            Adjacency matrix between edges (from L1)

        Returns:
        --------
        restriction_maps : dict
            (i, j) -> F_ij matrix (stalk_dim, stalk_dim)
        """
        num_edges = edge_features.shape[0]

        # Find adjacent edge pairs
        adj_indices = torch.nonzero(torch.abs(adjacency_matrix) > 0, as_tuple=False)

        # Remove self-loops
        mask = adj_indices[:, 0] != adj_indices[:, 1]
        adj_indices = adj_indices[mask]

        # Concatenate features for each pair
        i_features = edge_features[adj_indices[:, 0]]
        j_features = edge_features[adj_indices[:, 1]]
        pair_features = torch.cat([i_features, j_features], dim=1)

        # Generate parameters via MLP
        all_params = self.mlp(pair_features)

        # Build restriction maps
        restriction_maps = {}

        for idx, (i, j) in enumerate(adj_indices):
            i_idx, j_idx = i.item(), j.item()
            params = all_params[idx]

            if self.sheaf_type == 'diagonal':
                F_ij = torch.diag(params)

            elif self.sheaf_type == 'orthogonal':
                F_ij = self._householder_matrix(params.unsqueeze(0)).squeeze(0)

            else:  # general
                F_ij = params.reshape(self.stalk_dim, self.stalk_dim)

            restriction_maps[(i_idx, j_idx)] = F_ij

        return restriction_maps


class SheafLaplacianBuilder(nn.Module):
    """
    Build normalized sheaf Laplacian ΔF from restriction maps

    Implements Definition 2 from the paper
    """

    def __init__(self, stalk_dim):
        super(SheafLaplacianBuilder, self).__init__()
        self.stalk_dim = stalk_dim

    def forward(self, restriction_maps, num_edges, device):
        """
        Build sheaf Laplacian

        Parameters:
        -----------
        restriction_maps : dict
            (i, j) -> F_ij matrices
        num_edges : int
            Number of edges in the graph
        device : torch.device
            Device for tensors

        Returns:
        --------
        Delta_F : torch.Tensor, shape (num_edges * d, num_edges * d)
            Normalized sheaf Laplacian
        """
        d = self.stalk_dim
        n = num_edges

        # Initialize Laplacian
        LF = torch.zeros(n * d, n * d, device=device)

        # Build LF (unnormalized)
        # Diagonal blocks: LF_vv = Σ_u F_v⊆e^T F_v⊆e
        for v in range(n):
            diag_block = torch.zeros(d, d, device=device)

            for (i, j), F_ij in restriction_maps.items():
                if i == v:
                    diag_block += F_ij.T @ F_ij

            LF[v*d:(v+1)*d, v*d:(v+1)*d] = diag_block

        # Off-diagonal blocks: LF_vu = -F_u⊆e^T F_v⊆e
        for (v, u), F_vu in restriction_maps.items():
            if (u, v) in restriction_maps:
                F_uv = restriction_maps[(u, v)]
                LF[v*d:(v+1)*d, u*d:(u+1)*d] = -F_uv.T @ F_vu

        # Normalize: ΔF = D^(-1/2) LF D^(-1/2)
        # Extract diagonal blocks from LF
        D_blocks = torch.zeros(n, d, d, device=device)
        for v in range(n):
            D_blocks[v] = LF[v*d:(v+1)*d, v*d:(v+1)*d]

        # D^(-1/2) via eigendecomposition
        D_inv_sqrt = torch.zeros(n * d, n * d, device=device)

        for v in range(n):
            D_v = D_blocks[v] + 1e-5 * torch.eye(d, device=device)

            try:
                eigvals, eigvecs = torch.linalg.eigh(D_v)
                eigvals_inv_sqrt = 1.0 / torch.sqrt(eigvals.clamp(min=1e-6))
                D_v_inv_sqrt = eigvecs @ torch.diag(eigvals_inv_sqrt) @ eigvecs.T
            except:
                D_v_inv_sqrt = torch.eye(d, device=device) / torch.sqrt(torch.trace(D_v).clamp(min=1e-6))

            D_inv_sqrt[v*d:(v+1)*d, v*d:(v+1)*d] = D_v_inv_sqrt

        # ΔF = D^(-1/2) LF D^(-1/2)
        Delta_F = D_inv_sqrt @ LF @ D_inv_sqrt

        return Delta_F


class SheafConvolutionalNetwork(nn.Module):
    """
    Sheaf Convolutional Network for graph classification

    Based on "Neural Sheaf Diffusion" (Bodnar et al., NeurIPS 2022)
    """

    def __init__(self,
                 edge_feature_dim,
                 hidden_dims=[32, 32],
                 stalk_dim=2,
                 num_classes=2,
                 sheaf_type='diagonal',
                 mlp_hidden_dim=128,
                 dropout=0.5):
        """
        Parameters:
        -----------
        edge_feature_dim : int
            Dimension of edge features
        hidden_dims : list of int
            Hidden dimensions for each SCN layer
        stalk_dim : int
            Dimension of stalks (d in paper)
        num_classes : int
            Number of output classes
        sheaf_type : str
            'diagonal', 'orthogonal', or 'general'
        mlp_hidden_dim : int
            Hidden dimension for final classifier MLP
        dropout : float
            Dropout rate
        """
        super(SheafConvolutionalNetwork, self).__init__()

        self.edge_feature_dim = edge_feature_dim
        self.stalk_dim = stalk_dim
        self.sheaf_type = sheaf_type

        # Initial embedding
        self.input_embedding = nn.Linear(edge_feature_dim, hidden_dims[0])

        # Sheaf builder
        self.sheaf_builder = SheafBuilder(edge_feature_dim, stalk_dim, sheaf_type)

        # Sheaf Laplacian builder
        self.laplacian_builder = SheafLaplacianBuilder(stalk_dim)

        # SCN layers
        self.scn_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.scn_layers.append(
                SheafConvolutionalLayer(
                    hidden_dims[i],
                    hidden_dims[i+1],
                    stalk_dim,
                    dropout
                )
            )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, num_classes)
        )

    def forward(self, L1, edge_features):
        """
        Forward pass

        Parameters:
        -----------
        L1 : torch.Tensor, shape (num_edges, num_edges)
            Edge-to-edge Hodge Laplacian L1
        edge_features : torch.Tensor, shape (num_edges, edge_feature_dim)
            Features on edges

        Returns:
        --------
        logits : torch.Tensor, shape (num_classes,)
            Classification logits
        """
        num_edges = edge_features.shape[0]
        device = edge_features.device

        # Initial embedding
        H = F.relu(self.input_embedding(edge_features))

        # Expand to stalks: (num_edges, stalk_dim, hidden_dim)
        X = H.unsqueeze(1).expand(-1, self.stalk_dim, -1)
        X = X.reshape(num_edges * self.stalk_dim, -1)

        # Learn sheaf structure
        restriction_maps = self.sheaf_builder(edge_features, L1)
        Delta_F = self.laplacian_builder(restriction_maps, num_edges, device)

        # Apply SCN layers
        for layer in self.scn_layers:
            X = layer(X, Delta_F)

        # Aggregate over stalks
        X_reshaped = X.reshape(num_edges, self.stalk_dim, -1)
        H = X_reshaped.mean(dim=1)  # (num_edges, hidden_dim)

        # Global pooling over edges
        graph_embedding = H.mean(dim=0)  # (hidden_dim,)

        # Classification
        logits = self.classifier(graph_embedding)

        return logits
