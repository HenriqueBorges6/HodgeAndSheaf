"""
Hodge-GNN: Graph Neural Network using Hodge 1-Laplacian
Based on Park et al., MICCAI 2023

Key idea: Perform convolution directly on graph edges using Hodge Laplacian L1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HodgeConvLayer(nn.Module):
    """
    Single Hodge convolution layer operating on edges

    H^(l+1) = σ(L1 @ H^(l) @ W^(l))

    where:
    - L1 is the Hodge 1-Laplacian (num_edges x num_edges)
    - H^(l) is edge features (num_edges x in_features)
    - W^(l) is learnable weight matrix (in_features x out_features)
    """

    def __init__(self, in_features, out_features, activation=True):
        super(HodgeConvLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        # Use nn.Linear like in reference implementation
        self.linear = nn.Linear(in_features, out_features)

        # Proper weight initialization
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, L1, H):
        """
        Forward pass

        Parameters:
        -----------
        L1 : torch.Tensor, shape (num_edges, num_edges)
            Hodge 1-Laplacian
        H : torch.Tensor, shape (num_edges, in_features)
            Features on edges

        Returns:
        --------
        output : torch.Tensor, shape (num_edges, out_features)
            Transformed edge features
        """
        # Normalize L1: use absolute values and normalize by row
        L1_adj = torch.abs(L1)
        row_sum = L1_adj.sum(dim=1, keepdim=True)
        L1_normalized = L1_adj / (row_sum + 1e-8)

        # Aggregate via normalized Hodge Laplacian
        aggregated = torch.matmul(L1_normalized, H)

        # Add skip connection before transformation (CRITICAL!)
        combined = aggregated + H

        # Transform features using nn.Linear (like reference implementation)
        output = self.linear(combined)

        if self.activation:
            output = F.relu(output)

        return output


class HodgeGNN(nn.Module):
    """
    Hodge Graph Neural Network for graph classification

    Architecture:
    1. Initial edge embedding
    2. Multiple Hodge convolution layers
    3. Global pooling over edges
    4. MLP classifier
    """

    def __init__(self,
                 input_dim=None,  # For backward compatibility
                 edge_feature_dim=None,  # New parameter name
                 hidden_dims=[32, 32, 32, 32],
                 num_classes=2,
                 mlp_hidden_dim=128,
                 pooling_type='mean',
                 dropout=0.5):
        """
        Parameters:
        -----------
        edge_feature_dim : int
            Dimension of input edge features
        hidden_dims : list of int
            Hidden dimensions for Hodge conv layers, e.g., [32, 64]
        num_classes : int
            Number of output classes
        mlp_hidden_dim : int
            Hidden dimension for final MLP classifier
        pooling_type : str
            Type of pooling: 'mean', 'sum', 'max'
        dropout : float
            Dropout rate
        """
        super(HodgeGNN, self).__init__()

        # Handle both parameter names for backward compatibility
        if input_dim is not None and edge_feature_dim is None:
            edge_feature_dim = input_dim
        elif edge_feature_dim is None and input_dim is None:
            raise ValueError("Either input_dim or edge_feature_dim must be provided")

        self.edge_feature_dim = edge_feature_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.pooling_type = pooling_type
        self.dropout = dropout

        # Initial edge embedding
        self.edge_embedding = nn.Linear(edge_feature_dim, hidden_dims[0])

        # Hodge convolution layers
        self.hodge_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hodge_layers.append(
                HodgeConvLayer(hidden_dims[i], hidden_dims[i + 1])
            )

        # MLP classifier (simplified like reference)
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
            Hodge 1-Laplacian
        edge_features : torch.Tensor, shape (num_edges, edge_feature_dim)
            Initial edge features

        Returns:
        --------
        logits : torch.Tensor, shape (num_classes,)
            Class logits for graph classification
        """
        # Initial embedding
        h = self.edge_embedding(edge_features)
        h = F.relu(h)

        # Hodge convolution layers (note: L1 first, then h)
        # Activation is inside the layer, no extra ReLU/Dropout needed
        for layer in self.hodge_layers:
            h = layer(L1, h)

        # Global pooling over edges
        if self.pooling_type == 'mean':
            graph_embedding = torch.mean(h, dim=0)  # (hidden_dim,)
        elif self.pooling_type == 'sum':
            graph_embedding = torch.sum(h, dim=0)
        elif self.pooling_type == 'max':
            graph_embedding = torch.max(h, dim=0)[0]
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

        # Classification
        logits = self.classifier(graph_embedding)

        return logits

    def forward_batch(self, L1_list, edge_features_list):
        """
        Forward pass for a batch of graphs

        Parameters:
        -----------
        L1_list : list of torch.Tensor
            List of Hodge Laplacians
        edge_features_list : list of torch.Tensor
            List of edge feature matrices

        Returns:
        --------
        logits : torch.Tensor, shape (batch_size, num_classes)
            Class logits for each graph
        """
        batch_logits = []

        for L1, edge_features in zip(L1_list, edge_features_list):
            logits = self.forward(L1, edge_features)
            batch_logits.append(logits)

        return torch.stack(batch_logits)
