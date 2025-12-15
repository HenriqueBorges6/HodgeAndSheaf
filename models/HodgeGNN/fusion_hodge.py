import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

def compute_hodge_operators(edge_index, num_nodes):
    """
    Boundary Operators
    B1: Incidence Matrix
    L0: Laplaciano do Grafo
    L1: Laplaciano das Arestas
    """
    num_edges = edge_index.shape[1]
    device = edge_index.device
    B1 = torch.zeros(num_nodes, num_edges, device=device)
    
    for edge_idx in range(num_edges):
        source = edge_index[0, edge_idx]  # Nó de origem
        target = edge_index[1, edge_idx]  # Nó de destino
        
        B1[source, edge_idx] = 1   # Aresta SAI (+1)
        B1[target, edge_idx] = -1  # Aresta CHEGA (-1)
    
    # === Computar Laplacianos ===
    L0 = B1 @ B1.T  # Graph Laplacian (node space)
    L1 = B1.T @ B1  # Hodge Laplacian (edge space)
    
    return L0, L1, B1

def normalize_laplacian(L):
    """ 
    L_norm = D^(-1/2) @ L @ D^(-1/2)
    """
    # Grau: soma dos valores absolutos em cada linha
    degree = torch.abs(L).sum(dim=1).clamp(min=1e-6)  # Evita divisão por zero
    
    # D^(-1/2)
    deg_inv_sqrt = torch.pow(degree, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.  # Remove infinitos
    
    # L_norm = D^(-1/2) @ L @ D^(-1/2)
    L_norm = deg_inv_sqrt.unsqueeze(1) * L * deg_inv_sqrt.unsqueeze(0)
    
    return L_norm


class DualMessagePassingHodgeGNN(nn.Module):
    """
    Dual Message Passing Hodge GNN
    
    Arquitetura:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Em cada layer:
    
    1. WITHIN-DOMAIN AGGREGATION:
       - Nós agregam de nós vizinhos (via L₀)
       - Arestas agregam de arestas vizinhas (via L₁)
    
    2. CROSS-DOMAIN MESSAGE PASSING:
       - Nós → Arestas (via B₁)
       - Arestas → Nós (via B₁ᵀ)
    
    3. UPDATE:
       - BatchNorm + ReLU + Dropout
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Isso permite que informação flua tanto dentro de cada
    espaço quanto ENTRE espaços!
    """
    
    def __init__(self, 
                 num_node_features,
                 num_edge_features=0,
                 num_classes=2,
                 hidden_dim=64,
                 num_layers=3,
                 pooling='mean',
                 dropout=0.5,
                 normalize_L=True):
        """
        Args:
            num_node_features: Dimensão das node features (7 para MUTAG)
            num_edge_features: Dimensão das edge features (4 para MUTAG)
            num_classes: Número de classes
            hidden_dim: Dimensão das camadas ocultas
            num_layers: Número de camadas de message passing
            pooling: 'mean', 'max', ou 'sum'
            dropout: Taxa de dropout
            normalize_L: Se deve normalizar L₀ e L₁
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.pooling = pooling
        self.dropout = dropout
        self.normalize_L = normalize_L
        
        print("🏗️  Construindo DualMessagePassingHodgeGNN...")
        print(f"   Node features: {num_node_features}")
        print(f"   Edge features: {num_edge_features}")
        print(f"   Hidden dim: {hidden_dim}")
        print(f"   Num layers: {num_layers}")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # NODE PROCESSING (L₀ space)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.node_linears = nn.ModuleList()
        for i in range(num_layers):
            in_dim = num_node_features if i == 0 else hidden_dim
            self.node_linears.append(nn.Linear(in_dim, hidden_dim))
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # EDGE PROCESSING (L₁ space)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.edge_linears = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                # Primeira camada: concatenação de node features + edge_attr
                in_dim = 2 * num_node_features + num_edge_features
            else:
                in_dim = hidden_dim
            self.edge_linears.append(nn.Linear(in_dim, hidden_dim))
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # CROSS-DOMAIN MESSAGE PASSING
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        # Nós → Arestas
        self.node_to_edge = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Arestas → Nós
        self.edge_to_node = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # NORMALIZATION LAYERS
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.node_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) 
            for _ in range(num_layers)
        ])
        
        self.edge_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) 
            for _ in range(num_layers)
        ])
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # FUSION & CLASSIFICATION
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),  # Concatena node + edge
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Contar parâmetros
        total_params = sum(p.numel() for p in self.parameters())
        print(f"   Total de parâmetros: {total_params:,}")
        print("✅ Modelo construído!\n")
    
    def forward(self, x, edge_index, batch, edge_attr=None):
        """
        Forward pass com Dual Message Passing.
        
        Args:
            x: [num_nodes, num_node_features]
            edge_index: [2, num_edges]
            batch: [num_nodes] - indica qual nó pertence a qual grafo
            edge_attr: [num_edges, num_edge_features] (opcional)
        
        Returns:
            logits: [batch_size, num_classes]
        """
        num_nodes = x.shape[0]
        num_edges = edge_index.shape[1]
        device = x.device
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # STEP 1: Computar Operadores Hodge
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        L0, L1, B1 = compute_hodge_operators(edge_index, num_nodes)
        
        # Normalizar Laplacians
        if self.normalize_L:
            L0 = normalize_laplacian(L0)
            L1 = normalize_laplacian(L1)
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # STEP 2: Inicializar Features
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        # Node features (já temos)
        h_node = x  # [num_nodes, num_node_features]
        
        # Edge features: concatenar features dos nós nas pontas
        h_edge = torch.cat([
            x[edge_index[0]],  # Source node features
            x[edge_index[1]]   # Target node features
        ], dim=1)  # [num_edges, 2*num_node_features]
        
        # Adicionar edge_attr se disponível
        if edge_attr is not None:
            h_edge = torch.cat([h_edge, edge_attr.float()], dim=1)
            # [num_edges, 2*num_node_features + num_edge_features]
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # STEP 3: Message Passing Layers
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        for layer in range(self.num_layers):
            # ┌─────────────────────────────────────────┐
            # │ 3.1: WITHIN-DOMAIN AGGREGATION          │
            # └─────────────────────────────────────────┘
            
            # Nós agregam de nós vizinhos (via L₀)
            h_node_agg = L0 @ h_node  # [num_nodes, hidden_dim]
            h_node_new = self.node_linears[layer](h_node_agg)
            
            # Arestas agregam de arestas vizinhas (via L₁)
            h_edge_agg = L1 @ h_edge  # [num_edges, hidden_dim]
            h_edge_new = self.edge_linears[layer](h_edge_agg)
            
            # ┌─────────────────────────────────────────┐
            # │ 3.2: CROSS-DOMAIN MESSAGE PASSING       │
            # └─────────────────────────────────────────┘
            
            # Nós → Arestas (via estrutura do grafo)
            # Cada aresta (u,v) recebe mensagem dos nós u e v
            node_msg = self.node_to_edge[layer](h_node_new)
            edge_receives_from_nodes = (
                node_msg[edge_index[0]] +  # Mensagem do source
                node_msg[edge_index[1]]    # Mensagem do target
            ) / 2  # Média
            h_edge_new = h_edge_new + edge_receives_from_nodes
            
            # Arestas → Nós (via B₁ᵀ)
            # Cada nó agrega mensagens de suas arestas incidentes
            edge_msg = self.edge_to_node[layer](h_edge_new)
            node_receives_from_edges = B1 @ edge_msg  # B₁: [num_nodes, num_edges]
            h_node_new = h_node_new + node_receives_from_edges
            
            # ┌─────────────────────────────────────────┐
            # │ 3.3: NORMALIZATION & ACTIVATION         │
            # └─────────────────────────────────────────┘
            
            # Nodes
            h_node_new = self.node_norms[layer](h_node_new)
            h_node_new = F.relu(h_node_new)
            h_node_new = F.dropout(h_node_new, p=self.dropout, training=self.training)
            
            # Edges
            h_edge_new = self.edge_norms[layer](h_edge_new)
            h_edge_new = F.relu(h_edge_new)
            h_edge_new = F.dropout(h_edge_new, p=self.dropout, training=self.training)
            
            # Update para próxima iteração
            h_node = h_node_new
            h_edge = h_edge_new
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # STEP 4: Graph-Level Pooling
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        # Node pooling
        if self.pooling == 'mean':
            h_node_graph = global_mean_pool(h_node, batch)
        elif self.pooling == 'max':
            h_node_graph = global_max_pool(h_node, batch)
        elif self.pooling == 'sum':
            h_node_graph = global_add_pool(h_node, batch)
        
        # Edge pooling
        edge_batch = batch[edge_index[0]]  # Qual grafo cada aresta pertence
        if self.pooling == 'mean':
            h_edge_graph = global_mean_pool(h_edge, edge_batch)
        elif self.pooling == 'max':
            h_edge_graph = global_max_pool(h_edge, edge_batch)
        elif self.pooling == 'sum':
            h_edge_graph = global_add_pool(h_edge, edge_batch)
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # STEP 5: Fusion & Classification
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        # Concatenar representações de nós e arestas
        h_graph = torch.cat([h_node_graph, h_edge_graph], dim=1)
        # [batch_size, 2*hidden_dim]
        
        # Classificação
        logits = self.classifier(h_graph)
        # [batch_size, num_classes]
        
        return logits



