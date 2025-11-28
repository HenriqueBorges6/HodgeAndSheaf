"""
Neural Sheaf Diffusion Networks
Based on Bodnar et al., NeurIPS 2022

Key innovations:
- Learn restriction maps from data
- Operate on d-dimensional stalks
- Avoid oversmoothing via sheaf geometry
- OPTIMIZED: Vectorized operations instead of loops
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SheafBuilder(nn.Module):
    """
    Learn restriction maps Fv⊆e: F(v) -> F(e) from edge features

    OPTIMIZED VERSION: Vectorized batch processing
    """

    def __init__(self, edge_feature_dim, stalk_dim, sheaf_type='diagonal'):
        """
        Parameters:
        -----------
        edge_feature_dim : int
            Dimension of edge features
        stalk_dim : int
            Dimension of stalks (d in paper)
        sheaf_type : str
            'diagonal', 'orthogonal', 'general'
        """
        super(SheafBuilder, self).__init__()

        self.edge_feature_dim = edge_feature_dim
        self.stalk_dim = stalk_dim
        self.sheaf_type = sheaf_type

        # Output dimension based on sheaf type
        if sheaf_type == 'diagonal':
            output_dim = stalk_dim
        elif sheaf_type == 'orthogonal':
            output_dim = stalk_dim  # Householder reflections
        else:  # general
            output_dim = stalk_dim * stalk_dim

        # MLP to learn restriction maps
        # Input: concatenated features of two adjacent edges
        # OPTIMIZATION: Use LayerNorm for stable training
        self.mlp = nn.Sequential(
            nn.Linear(2 * edge_feature_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),  # Mild regularization
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, output_dim)
        )

        # For orthogonal maps, add output normalization
        if sheaf_type == 'orthogonal':
            # Add tanh to bound outputs to [-1, 1] for numerical stability
            self.mlp.add_module('output_tanh', nn.Tanh())

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize MLP weights using Xavier uniform"""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _build_householder(self, v):
        """
        Build orthogonal matrix from Householder reflection (STABLE VERSION)
        H = I - 2vv^T / ||v||^2

        Parameters:
        -----------
        v : torch.Tensor, shape (batch, stalk_dim)

        Returns:
        --------
        H : torch.Tensor, shape (batch, stalk_dim, stalk_dim)

        Numerical stability improvements:
        - Clamp input to reasonable range to avoid extreme values
        - Use robust normalization with larger epsilon
        - Add gradient clipping implicitly via tanh
        """
        batch_size = v.shape[0]

        # STABILITY FIX 1: Clamp to prevent extreme values
        v = torch.clamp(v, -10.0, 10.0)

        # STABILITY FIX 2: Robust normalization
        v_norm_val = torch.norm(v, dim=1, keepdim=True)
        # Avoid division by very small numbers
        v_normalized = v / torch.clamp(v_norm_val, min=1e-4)

        # H = I - 2vv^T
        I = torch.eye(self.stalk_dim, device=v.device).unsqueeze(0).expand(batch_size, -1, -1)
        vvT = torch.bmm(v_normalized.unsqueeze(2), v_normalized.unsqueeze(1))
        H = I - 2 * vvT

        return H

    def forward(self, edge_features, L1):
        """
        Build restriction maps for all adjacent edge pairs

        OPTIMIZED: Vectorized batch processing

        Parameters:
        -----------
        edge_features : torch.Tensor, shape (num_edges, edge_feature_dim)
        L1 : torch.Tensor, shape (num_edges, num_edges)

        Returns:
        --------
        restriction_maps : dict
            (i, j) -> F_ij matrix (stalk_dim x stalk_dim)
        """
        num_edges = edge_features.shape[0]

        # Find all adjacent edge pairs efficiently
        adjacency = (torch.abs(L1) > 0).float()

        # Get indices of non-zero entries (i, j pairs)
        edge_indices = torch.nonzero(adjacency, as_tuple=False)  # (num_pairs, 2)

        # Remove self-loops
        mask = edge_indices[:, 0] != edge_indices[:, 1]
        edge_indices = edge_indices[mask]

        # Vectorized feature concatenation
        i_features = edge_features[edge_indices[:, 0]]  # (num_pairs, dim)
        j_features = edge_features[edge_indices[:, 1]]  # (num_pairs, dim)
        edge_pairs = torch.cat([i_features, j_features], dim=1)  # (num_pairs, 2*dim)

        # Single MLP forward pass for ALL pairs
        all_params = self.mlp(edge_pairs)  # (num_pairs, output_dim)

        # Build restriction maps
        restriction_maps = {}

        for idx, (i, j) in enumerate(edge_indices):
            i, j = i.item(), j.item()
            params = all_params[idx]

            if self.sheaf_type == 'diagonal':
                F_ij = torch.diag(params)

            elif self.sheaf_type == 'orthogonal':
                F_ij = self._build_householder(params.unsqueeze(0)).squeeze(0)

            else:  # general
                F_ij = params.reshape(self.stalk_dim, self.stalk_dim)

            restriction_maps[(i, j)] = F_ij

        return restriction_maps


class SheafLaplacian(nn.Module):
    """
    Construct Sheaf Laplacian ΔF from restriction maps

    ΔF = D^(-1/2) LF D^(-1/2)
    """

    def __init__(self, stalk_dim):
        super(SheafLaplacian, self).__init__()
        self.stalk_dim = stalk_dim

    def forward(self, restriction_maps, num_edges, L1):
        """
        Build sheaf Laplacian

        Parameters:
        -----------
        restriction_maps : dict
        num_edges : int
        L1 : torch.Tensor

        Returns:
        --------
        Delta_F : torch.Tensor, shape (num_edges*d, num_edges*d)
        """
        d = self.stalk_dim
        n = num_edges
        device = L1.device

        # Initialize sheaf Laplacian as block matrix
        LF = torch.zeros(n * d, n * d, device=device)

        # Diagonal blocks: LF_vv = Σ F^T_v⊆e F_v⊆e
        for v in range(n):
            block_sum = torch.zeros(d, d, device=device)

            adjacency = (torch.abs(L1[v]) > 0).float()
            neighbors = torch.where(adjacency > 0)[0]

            for u in neighbors:
                u_idx = u.item()
                if (v, u_idx) in restriction_maps:
                    F_vu = restriction_maps[(v, u_idx)]
                    block_sum += F_vu.T @ F_vu

            LF[v*d:(v+1)*d, v*d:(v+1)*d] = block_sum

        # Off-diagonal blocks: LF_vu = -F^T_u⊆e F_v⊆e (note: original paper is for v,u nodes)
        for (v, u), F_vu in restriction_maps.items():
            if (u, v) in restriction_maps:
                F_uv = restriction_maps[(u, v)]
                # Corrected from -F_vu.T @ F_uv
                LF[v*d:(v+1)*d, u*d:(u+1)*d] = -F_uv.T @ F_vu

        # Normalization: D^(-1/2) LF D^(-1/2)
        D_blocks = torch.zeros(n, d, d, device=device)
        for v in range(n):
            D_blocks[v] = LF[v*d:(v+1)*d, v*d:(v+1)*d]

        # D^(-1/2) via eigenvalue decomposition (GRADIENT-SAFE VERSION)
        # CRITICAL FIX: torch.linalg.eigh backward can produce NaN gradients
        # Solution: Use detach() or simpler normalization
        D_inv_sqrt = torch.zeros_like(LF)
        for v in range(n):
            D_v = D_blocks[v] + 1e-3 * torch.eye(d, device=device)

            # Compute D^(-1/2) without gradients through eigendecomposition
            # This prevents NaN in backward pass while keeping forward accurate
            with torch.no_grad():
                try:
                    eigvals, eigvecs = torch.linalg.eigh(D_v)
                    eigvals_clamped = torch.clamp(eigvals, min=1e-4)
                    eigvals_inv_sqrt = 1.0 / torch.sqrt(eigvals_clamped)
                    D_v_inv_sqrt_nograd = eigvecs @ torch.diag(eigvals_inv_sqrt) @ eigvecs.T
                except:
                    D_v_inv_sqrt_nograd = torch.eye(d, device=device) / (torch.norm(D_v) + 1e-3)

            # Reattach to computation graph via simple operation
            # This allows gradients to flow through LF but not through the normalization
            D_v_inv_sqrt = D_v_inv_sqrt_nograd.detach()

            D_inv_sqrt[v*d:(v+1)*d, v*d:(v+1)*d] = D_v_inv_sqrt

        # Normalized sheaf Laplacian
        Delta_F = D_inv_sqrt @ LF @ D_inv_sqrt

        return Delta_F


class SheafDiffusionLayer(nn.Module):
    """
    Sheaf diffusion layer with learnable weights

    X(t+1) = X(t) - σ(ΔF (I ⊗ W1) X(t) W2)
    """

    def __init__(self, in_features, out_features, stalk_dim):
        super(SheafDiffusionLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.stalk_dim = stalk_dim

        # Learnable weights
        self.W1 = nn.Parameter(torch.Tensor(stalk_dim, stalk_dim))
        self.W2 = nn.Parameter(torch.Tensor(in_features, out_features))

        # Residual projection
        self.res_proj = nn.Linear(in_features, out_features)

        # Epsilon for controlling diffusion magnitude
        self.epsilon = nn.Parameter(torch.zeros(stalk_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        nn.init.xavier_uniform_(self.res_proj.weight)
        if self.res_proj.bias is not None:
            nn.init.zeros_(self.res_proj.bias)

    def forward(self, X, Delta_F):
        """
        Parameters:
        -----------
        X : torch.Tensor, shape (num_edges*d, in_features)
        Delta_F : torch.Tensor, shape (num_edges*d, num_edges*d)

        Returns:
        --------
        X_next : torch.Tensor, shape (num_edges*d, out_features)
        """
        num_edges_d = X.shape[0]
        num_edges = num_edges_d // self.stalk_dim
        d = self.stalk_dim

        # OPTIMIZATION: Vectorized Kronecker product (I ⊗ W1)
        # Reshape X to (num_edges, d, in_features)
        X_reshaped = X.reshape(num_edges, d, self.in_features)

        # Apply W1 using batched matrix multiplication
        # (num_edges, d, in_features) @ (d, d)^T = (num_edges, d, in_features)
        X_transformed = torch.bmm(
            self.W1.unsqueeze(0).expand(num_edges, -1, -1),  # (num_edges, d, d)
            X_reshaped  # (num_edges, d, in_features)
        )

        # Flatten back to (num_edges*d, in_features)
        X_transformed = X_transformed.reshape(num_edges_d, self.in_features)

        # Apply W2: project features
        X_transformed = X_transformed @ self.W2  # (num_edges*d, out_features)

        # Diffusion
        diffused = Delta_F @ X_transformed

        # Residual connection
        X_proj = self.res_proj(X)

        # Combine with epsilon modulation
        eps_expanded = self.epsilon.repeat(num_edges).unsqueeze(1)
        X_next = (1 + eps_expanded) * X_proj - F.elu(diffused)

        return X_next


class SheafNeuralNetwork(nn.Module):
    """
    Complete Sheaf Neural Network for graph classification
    """

    def __init__(self,
                 input_dim=None,  # For backward compatibility
                 edge_feature_dim=None,  # New parameter name
                 stalk_dim=2,
                 hidden_dims=[32, 64],
                 num_classes=2,
                 sheaf_type='diagonal',
                 mlp_hidden_dim=128,
                 dropout=0.5):
        super(SheafNeuralNetwork, self).__init__()

        # Handle both parameter names for backward compatibility
        if input_dim is not None and edge_feature_dim is None:
            edge_feature_dim = input_dim
        elif edge_feature_dim is None and input_dim is None:
            raise ValueError("Either input_dim or edge_feature_dim must be provided")

        self.edge_feature_dim = edge_feature_dim
        self.stalk_dim = stalk_dim
        self.sheaf_type = sheaf_type

        # Initial embedding
        self.edge_embedding = nn.Linear(edge_feature_dim, hidden_dims[0])

        # Sheaf builder
        self.sheaf_builder = SheafBuilder(edge_feature_dim, stalk_dim, sheaf_type)

        # Sheaf Laplacian builder
        self.sheaf_laplacian = SheafLaplacian(stalk_dim)

        # Diffusion layers
        self.diffusion_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.diffusion_layers.append(
                SheafDiffusionLayer(hidden_dims[i], hidden_dims[i+1], stalk_dim)
            )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim // 2, num_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform"""
        # Edge embedding
        nn.init.xavier_uniform_(self.edge_embedding.weight)
        if self.edge_embedding.bias is not None:
            nn.init.zeros_(self.edge_embedding.bias)

        # Classifier
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, L1, edge_features):
        """
        Parameters:
        -----------
        L1 : torch.Tensor, shape (num_edges, num_edges)
        edge_features : torch.Tensor, shape (num_edges, edge_feature_dim)

        Returns:
        --------
        logits : torch.Tensor, shape (num_classes,)
        """
        num_edges = edge_features.shape[0]

        # Initial embedding
        H = F.relu(self.edge_embedding(edge_features))

        # Expand to stalks: (num_edges, d, hidden_dim)
        X = H.unsqueeze(1).expand(-1, self.stalk_dim, -1)
        X = X.reshape(num_edges * self.stalk_dim, -1)

        # Learn sheaf structure once (based on original edge features)
        restriction_maps = self.sheaf_builder(edge_features, L1)
        Delta_F = self.sheaf_laplacian(restriction_maps, num_edges, L1)

        # Diffusion layers
        for layer in self.diffusion_layers:
            # Diffuse
            X = layer(X, Delta_F)

            # Update H (aggregate over stalks)
            X_reshaped = X.reshape(num_edges, self.stalk_dim, -1)
            H = X_reshaped.mean(dim=1)

        # Global pooling
        graph_embedding = H.mean(dim=0)

        # Classification
        logits = self.classifier(graph_embedding)

        return logits
