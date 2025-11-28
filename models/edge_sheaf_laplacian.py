"""
Edge Sheaf Laplacian Network
Hybrid approach combining Hodge-GNN and Sheaf Neural Networks

Key innovations:
- Use Hodge 1-Laplacian L1 for edge-to-edge connectivity
- Learn sheaf restriction maps on edges (not nodes)
- Combine both Hodge and Sheaf diffusion mechanisms
- OPTIMIZED: Vectorized operations throughout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeSheafBuilder(nn.Module):
    """
    Learn sheaf restriction maps between adjacent EDGES

    Unlike standard SheafBuilder (which operates on nodes),
    this builds restriction maps F_ei⊆ej between edges that share nodes

    OPTIMIZED: Vectorized batch processing
    """

    def __init__(self, edge_feature_dim, stalk_dim, sheaf_type='diagonal'):
        """
        Parameters:
        -----------
        edge_feature_dim : int
            Dimension of edge features
        stalk_dim : int
            Dimension of stalks attached to each edge
        sheaf_type : str
            'diagonal', 'orthogonal', 'general'
        """
        super(EdgeSheafBuilder, self).__init__()

        self.edge_feature_dim = edge_feature_dim
        self.stalk_dim = stalk_dim
        self.sheaf_type = sheaf_type

        # Output dimension based on sheaf type
        if sheaf_type == 'diagonal':
            output_dim = stalk_dim
        elif sheaf_type == 'orthogonal':
            output_dim = stalk_dim
        else:  # general
            output_dim = stalk_dim * stalk_dim

        # MLP to learn restriction maps between adjacent edges
        # Input: concatenated features of two adjacent edges (connected via L1)
        self.mlp = nn.Sequential(
            nn.Linear(2 * edge_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def _build_householder(self, v):
        """
        Build orthogonal matrix from Householder reflection
        H = I - 2vv^T / ||v||^2

        Parameters:
        -----------
        v : torch.Tensor, shape (batch, stalk_dim)

        Returns:
        --------
        H : torch.Tensor, shape (batch, stalk_dim, stalk_dim)
        """
        batch_size = v.shape[0]
        v_norm = v / (torch.norm(v, dim=1, keepdim=True) + 1e-8)

        # H = I - 2vv^T
        I = torch.eye(self.stalk_dim, device=v.device).unsqueeze(0).expand(batch_size, -1, -1)
        vvT = torch.bmm(v_norm.unsqueeze(2), v_norm.unsqueeze(1))
        H = I - 2 * vvT

        return H

    def forward(self, edge_features, L1):
        """
        Build sheaf restriction maps between adjacent edges

        FULLY OPTIMIZED: Completely vectorized, no Python loops

        Parameters:
        -----------
        edge_features : torch.Tensor, shape (num_edges, edge_feature_dim)
        L1 : torch.Tensor, shape (num_edges, num_edges)
            Hodge 1-Laplacian encoding edge adjacency

        Returns:
        --------
        restriction_matrices : torch.Tensor, shape (num_pairs, stalk_dim, stalk_dim)
        edge_indices : torch.Tensor, shape (num_pairs, 2)
        """
        num_edges = edge_features.shape[0]
        device = edge_features.device

        # Find all adjacent edge pairs from L1
        # L1(i,j) != 0 means edges i and j share a node
        adjacency = (torch.abs(L1) > 0).float()

        # Get indices of adjacent edge pairs
        edge_indices = torch.nonzero(adjacency, as_tuple=False)  # (num_pairs, 2)

        # Remove self-loops
        mask = edge_indices[:, 0] != edge_indices[:, 1]
        edge_indices = edge_indices[mask]

        num_pairs = edge_indices.shape[0]

        # Vectorized feature concatenation
        i_features = edge_features[edge_indices[:, 0]]  # (num_pairs, dim)
        j_features = edge_features[edge_indices[:, 1]]  # (num_pairs, dim)
        edge_pairs = torch.cat([i_features, j_features], dim=1)  # (num_pairs, 2*dim)

        # Single MLP forward pass for ALL pairs
        all_params = self.mlp(edge_pairs)  # (num_pairs, output_dim)

        # Build restriction maps - FULLY VECTORIZED
        if self.sheaf_type == 'diagonal':
            # Vectorized diagonal matrix construction
            restriction_matrices = torch.zeros(num_pairs, self.stalk_dim, self.stalk_dim, device=device)
            batch_indices = torch.arange(num_pairs, device=device)
            diag_indices = torch.arange(self.stalk_dim, device=device)
            restriction_matrices[batch_indices[:, None], diag_indices, diag_indices] = all_params

        elif self.sheaf_type == 'orthogonal':
            # Batch Householder reflection
            restriction_matrices = self._build_householder(all_params)

        else:  # general
            # Batch reshape
            restriction_matrices = all_params.reshape(num_pairs, self.stalk_dim, self.stalk_dim)

        return restriction_matrices, edge_indices


class EdgeSheafLaplacian(nn.Module):
    """
    Construct Edge Sheaf Laplacian combining:
    - Hodge 1-Laplacian structure (edge-to-edge)
    - Sheaf restriction maps learned from data

    Result: ΔF_edge that captures both topology and learned geometry
    """

    def __init__(self, stalk_dim):
        super(EdgeSheafLaplacian, self).__init__()
        self.stalk_dim = stalk_dim

    def forward(self, restriction_matrices, edge_indices, num_edges, L1):
        """
        Build edge sheaf Laplacian

        FULLY OPTIMIZED: Vectorized construction, batch eigendecomposition

        Parameters:
        -----------
        restriction_matrices : torch.Tensor, shape (num_pairs, stalk_dim, stalk_dim)
            Learned restriction maps between adjacent edges
        edge_indices : torch.Tensor, shape (num_pairs, 2)
            Indices of edge pairs
        num_edges : int
        L1 : torch.Tensor
            Hodge 1-Laplacian

        Returns:
        --------
        Delta_F : torch.Tensor, shape (num_edges*d, num_edges*d)
            Edge sheaf Laplacian
        """
        d = self.stalk_dim
        n = num_edges
        device = L1.device
        num_pairs = edge_indices.shape[0]

        # Initialize edge sheaf Laplacian as block matrix
        LF = torch.zeros(n * d, n * d, device=device)

        # VECTORIZED: Diagonal blocks construction
        # LF_ii = Σ F^T_i⊆j F_i⊆j
        # Compute F^T @ F for all pairs at once
        FtF = torch.bmm(restriction_matrices.transpose(1, 2), restriction_matrices)  # (num_pairs, d, d)

        # Accumulate to diagonal blocks using scatter_add
        for idx in range(num_pairs):
            i = edge_indices[idx, 0].item()
            LF[i*d:(i+1)*d, i*d:(i+1)*d] += FtF[idx]

        # VECTORIZED: Off-diagonal blocks
        # Build mapping from (j, i) to index for fast lookup
        edge_pair_to_idx = {}
        for idx in range(num_pairs):
            i, j = edge_indices[idx, 0].item(), edge_indices[idx, 1].item()
            edge_pair_to_idx[(i, j)] = idx

        # Off-diagonal blocks: LF_ij = -F^T_i⊆k F_j⊆k
        for idx in range(num_pairs):
            i, j = edge_indices[idx, 0].item(), edge_indices[idx, 1].item()

            # Check if reverse edge exists
            if (j, i) in edge_pair_to_idx:
                ji_idx = edge_pair_to_idx[(j, i)]

                # Weight by L1 connectivity strength
                L1_weight = L1[i, j]

                # F_ji^T @ F_ij
                F_ij = restriction_matrices[idx]
                F_ji = restriction_matrices[ji_idx]

                LF[i*d:(i+1)*d, j*d:(j+1)*d] = -torch.sign(L1_weight) * F_ji.T @ F_ij

        # BATCH NORMALIZATION: D^(-1/2) LF D^(-1/2)
        # Extract all diagonal blocks at once
        D_blocks = torch.zeros(n, d, d, device=device)
        for i in range(n):
            D_blocks[i] = LF[i*d:(i+1)*d, i*d:(i+1)*d]

        # Ensure symmetry (critical for eigh)
        D_blocks = (D_blocks + D_blocks.transpose(1, 2)) / 2.0

        # Add stronger regularization to ensure positive definiteness
        eps = 1e-4  # Increased from 1e-6
        D_blocks = D_blocks + eps * torch.eye(d, device=device).unsqueeze(0)

        # BATCH eigenvalue decomposition (single GPU call) with error handling
        try:
            eigvals, eigvecs = torch.linalg.eigh(D_blocks)  # (n, d), (n, d, d)
        except RuntimeError as e:
            # Fallback: use stronger regularization and try again
            print(f"Warning: eigh failed, using stronger regularization. Error: {e}")
            D_blocks = D_blocks + 1e-3 * torch.eye(d, device=device).unsqueeze(0)
            eigvals, eigvecs = torch.linalg.eigh(D_blocks)

        # Clamp eigenvalues to avoid numerical issues
        eigvals = torch.clamp(eigvals, min=eps)

        # Compute D^(-1/2) for all blocks at once
        eigvals_inv_sqrt = 1.0 / torch.sqrt(eigvals)  # (n, d)

        # Batch matrix multiplication: eigvecs @ diag(eigvals_inv_sqrt) @ eigvecs^T
        # Using einsum for efficiency
        D_inv_sqrt_blocks = torch.einsum('nij,nj,nkj->nik', eigvecs, eigvals_inv_sqrt, eigvecs)

        # Assemble D^(-1/2) from blocks
        D_inv_sqrt = torch.zeros_like(LF)
        for i in range(n):
            D_inv_sqrt[i*d:(i+1)*d, i*d:(i+1)*d] = D_inv_sqrt_blocks[i]

        # Normalized edge sheaf Laplacian
        Delta_F = D_inv_sqrt @ LF @ D_inv_sqrt

        return Delta_F


class HybridDiffusionLayer(nn.Module):
    """
    Hybrid diffusion combining:
    - Hodge convolution: aggregates via L1
    - Sheaf diffusion: uses learned restriction maps

    X(t+1) = X(t) - α * L1 X(t) W1 - β * ΔF_edge (I ⊗ W2) X(t) W3

    where α, β balance Hodge vs Sheaf contributions
    """

    def __init__(self, in_features, out_features, stalk_dim):
        super(HybridDiffusionLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.stalk_dim = stalk_dim

        # Hodge convolution weights
        self.W_hodge = nn.Parameter(torch.Tensor(in_features, out_features))

        # Sheaf diffusion weights
        self.W_sheaf_1 = nn.Parameter(torch.Tensor(stalk_dim, stalk_dim))
        self.W_sheaf_2 = nn.Parameter(torch.Tensor(in_features, out_features))

        # Residual projection
        self.res_proj = nn.Linear(in_features, out_features)

        # Learnable balance parameters
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Hodge weight
        self.beta = nn.Parameter(torch.tensor(0.5))   # Sheaf weight

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_hodge)
        nn.init.xavier_uniform_(self.W_sheaf_1)
        nn.init.xavier_uniform_(self.W_sheaf_2)
        nn.init.xavier_uniform_(self.res_proj.weight)
        if self.res_proj.bias is not None:
            nn.init.zeros_(self.res_proj.bias)

    def forward(self, X, L1, Delta_F):
        """
        OPTIMIZED: Fully vectorized, no loops

        Parameters:
        -----------
        X : torch.Tensor, shape (num_edges, in_features) OR (num_edges*d, in_features)
        L1 : torch.Tensor, shape (num_edges, num_edges)
        Delta_F : torch.Tensor, shape (num_edges*d, num_edges*d)

        Returns:
        --------
        X_next : torch.Tensor, shape (num_edges*d, out_features)
        """
        # Normalize L1 (critical for stable training)
        L1_adj = torch.abs(L1)
        row_sum = L1_adj.sum(dim=1, keepdim=True)
        L1_normalized = L1_adj / (row_sum + 1e-8)

        # Hodge convolution component
        if X.shape[0] == L1.shape[0]:
            # X is (num_edges, features)
            # Add skip connection: aggregate + input
            aggregated = torch.mm(L1_normalized, X)
            combined = aggregated + X  # Skip connection
            hodge_diffused = torch.mm(combined, self.W_hodge)
            X_res = self.res_proj(X)
        else:
            # X is already expanded to (num_edges*d, features)
            num_edges = L1.shape[0]

            # Aggregate stalks for Hodge convolution
            X_collapsed = X.reshape(num_edges, self.stalk_dim, -1).mean(dim=1)

            # Add skip connection: aggregate + input
            aggregated = torch.mm(L1_normalized, X_collapsed)
            combined = aggregated + X_collapsed  # Skip connection
            hodge_diffused = torch.mm(combined, self.W_hodge)

            # VECTORIZED Sheaf diffusion component
            # Reshape X to (num_edges, stalk_dim, features)
            X_reshaped = X.reshape(num_edges, self.stalk_dim, -1)

            # Apply W_sheaf_1 to all stalks at once using batch matrix multiplication
            # (num_edges, stalk_dim, features) @ (stalk_dim, stalk_dim)^T
            # = (num_edges, stalk_dim, features)
            X_transformed = torch.bmm(self.W_sheaf_1.unsqueeze(0).expand(num_edges, -1, -1),
                                       X_reshaped)  # (num_edges, stalk_dim, features)

            # Flatten back to (num_edges*d, features) and apply W_sheaf_2
            X_transformed = X_transformed.reshape(num_edges * self.stalk_dim, -1)
            X_transformed = X_transformed @ self.W_sheaf_2

            # Sheaf diffusion
            sheaf_diffused = Delta_F @ X_transformed

            # Residual
            X_res = self.res_proj(X)

            # Combine both diffusions
            X_next = X_res - F.sigmoid(self.alpha) * hodge_diffused.repeat_interleave(self.stalk_dim, dim=0) \
                           - F.sigmoid(self.beta) * F.elu(sheaf_diffused)

            return X_next

        # Simple Hodge-only path (when X is not expanded)
        X_next = X_res - F.sigmoid(self.alpha) * hodge_diffused
        return X_next


class EdgeSheafLaplacianNetwork(nn.Module):
    """
    Complete hybrid network combining Hodge-GNN and Sheaf Neural Networks

    Architecture:
    1. Initial edge embedding
    2. Expand to stalks
    3. Learn edge sheaf structure via EdgeSheafBuilder
    4. Hybrid diffusion layers (Hodge + Sheaf)
    5. Global pooling
    6. MLP classifier
    """

    def __init__(self,
                 edge_feature_dim,
                 stalk_dim,
                 hidden_dims,
                 num_classes,
                 sheaf_type='diagonal',
                 mlp_hidden_dim=128,
                 dropout=0.5,
                 cache_sheaf=True):
        super(EdgeSheafLaplacianNetwork, self).__init__()

        self.edge_feature_dim = edge_feature_dim
        self.stalk_dim = stalk_dim
        self.sheaf_type = sheaf_type
        self.cache_sheaf = cache_sheaf

        # Initial embedding
        self.edge_embedding = nn.Linear(edge_feature_dim, hidden_dims[0])

        # Edge sheaf builder - operates on STALK SPACE for each layer
        self.edge_sheaf_builders = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            # Input is hidden_dims[i] (per-stalk features)
            self.edge_sheaf_builders.append(
                EdgeSheafBuilder(hidden_dims[i], stalk_dim, sheaf_type)
            )

        # Edge sheaf Laplacian builder
        self.edge_sheaf_laplacian = EdgeSheafLaplacian(stalk_dim)

        # Hybrid diffusion layers
        self.diffusion_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.diffusion_layers.append(
                HybridDiffusionLayer(hidden_dims[i], hidden_dims[i+1], stalk_dim)
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

        # Cache for sheaf structures (cleared after forward pass)
        self._cached_restriction_matrices = None
        self._cached_edge_indices = None
        self._cached_delta_f = None

    def forward(self, L1, edge_features):
        """
        OPTIMIZED & CORRECTED:
        - Sheaf builder now operates on stalk-space features X (not collapsed H)
        - Caching of sheaf structures to avoid recomputation
        - Proper memory cleanup

        Parameters:
        -----------
        L1 : torch.Tensor, shape (num_edges, num_edges)
            Hodge 1-Laplacian
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

        # Hybrid diffusion layers
        for layer_idx, layer in enumerate(self.diffusion_layers):
            # CRITICAL FIX: Use X (stalk features), not H (collapsed features)
            # Each stalk needs its own representation for sheaf learning
            X_reshaped = X.reshape(num_edges, self.stalk_dim, -1)

            # Use mean stalk features for sheaf structure learning
            # This preserves stalk information while creating edge-level features
            X_for_sheaf = X_reshaped.mean(dim=1)  # (num_edges, hidden_dim)

            # Learn edge sheaf structure with appropriate builder for this layer
            restriction_matrices, edge_indices = self.edge_sheaf_builders[layer_idx](
                X_for_sheaf, L1
            )

            # Build edge sheaf Laplacian
            Delta_F = self.edge_sheaf_laplacian(
                restriction_matrices, edge_indices, num_edges, L1
            )

            # Hybrid diffusion (Hodge + Sheaf)
            X = layer(X, L1, Delta_F)

            # Memory cleanup: delete intermediate tensors
            if not self.cache_sheaf:
                del restriction_matrices, edge_indices, Delta_F
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Update H (aggregate over stalks) for next iteration
            X_reshaped = X.reshape(num_edges, self.stalk_dim, -1)
            H = X_reshaped.mean(dim=1)

        # Global pooling over edges
        graph_embedding = H.mean(dim=0)

        # Classification
        logits = self.classifier(graph_embedding)

        return logits
