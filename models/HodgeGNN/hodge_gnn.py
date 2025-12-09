import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

def compute_hodge_laplacian_L1(edge_index, num_nodes):
    """
    Calcula o Hodge Laplacian L1 = B1^T @ B1
    
    Explicação:
    - B1 é a matriz de incidência [num_nodes, num_edges]
    - L1 conecta arestas que compartilham nós
    
    Args:
        edge_index: [2, num_edges] - [source_nodes, target_nodes]
        num_nodes: int - número de nós no grafo
    
    Returns:
        L1: [num_edges, num_edges] - Hodge Laplacian
        B1: [num_nodes, num_edges] - Matriz de incidência
    """
    num_edges = edge_index.shape[1]
    
    # Construir B1: matriz de incidência
    # B1[node, edge] = +1 se aresta SAI do node
    # B1[node, edge] = -1 se aresta CHEGA no node
    B1 = torch.zeros(num_nodes, num_edges, device=edge_index.device)
    
    for edge_idx in range(num_edges):
        source = edge_index[0, edge_idx]
        target = edge_index[1, edge_idx]
        
        B1[source, edge_idx] = 1   # Aresta sai
        B1[target, edge_idx] = -1  # Aresta chega
    
    # L1 = B1^T @ B1
    # Isso captura as relações entre arestas que compartilham nós
    L1 = B1.T @ B1
    
    return L1, B1


def normalize_hodge_laplacian(L1):
    """
    Normaliza o Hodge Laplacian (similar à normalização simétrica em GCN)
    
    L1_norm = D^(-1/2) @ L1 @ D^(-1/2)
    onde D é a matriz diagonal com graus
    
    Isso ajuda na estabilidade numérica durante o treinamento.
    """
    # Grau de cada "nó" (que é uma aresta no espaço L1)
    # degree[i] = número de arestas conectadas à aresta i
    degree = torch.abs(L1).sum(dim=1)  # [num_edges]
    
    # Evitar divisão por zero
    degree = torch.clamp(degree, min=1e-6)
    
    # D^(-1/2)
    deg_inv_sqrt = torch.pow(degree, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.
    
    # D^(-1/2) @ L1 @ D^(-1/2)
    L1_norm = deg_inv_sqrt.unsqueeze(1) * L1 * deg_inv_sqrt.unsqueeze(0)
    
    return L1_norm

class HodgeConvLayer(nn.Module):
    """
    Camada de convolução sobre arestas usando Hodge Laplacian.
    
    Funcionamento:
    1. Agrega features de arestas vizinhas usando L1
    2. Aplica transformação linear
    3. Aplica ativação e normalização
    
    É como uma GCNConv, mas opera em arestas!
    """
    def __init__(self, in_features, out_features, 
                 use_bias=True, activation='relu', 
                 batch_norm=True, dropout=0.0):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Transformação linear (como em GCN)
        self.linear = nn.Linear(in_features, out_features, bias=use_bias)
        
        # Batch normalization (opcional)
        self.batch_norm = nn.BatchNorm1d(out_features) if batch_norm else None
        
        # Função de ativação
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'elu':
            self.activation = F.elu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        else:
            self.activation = lambda x: x
        
        self.dropout = dropout
        
    def forward(self, edge_features, L1):
        """
        Args:
            edge_features: [num_edges, in_features]
            L1: [num_edges, num_edges] - Hodge Laplacian (já normalizado)
        
        Returns:
            out: [num_edges, out_features]
        """
        # 1. Agregação: cada aresta agrega info de arestas vizinhas
        aggregated = L1 @ edge_features  # [num_edges, in_features]
        
        # 2. Transformação linear
        out = self.linear(aggregated)  # [num_edges, out_features]
        
        # 3. Batch normalization (se configurado)
        if self.batch_norm is not None:
            out = self.batch_norm(out)
        
        # 4. Ativação
        out = self.activation(out)
        
        # 5. Dropout (apenas durante treinamento)
        out = F.dropout(out, p=self.dropout, training=self.training)
        
        return out
    

class HodgeGNN(nn.Module):
    def __init__(self, 
                 num_classes=2,
                 hidden_dims=[64, 64, 32],          # Convoluções
                 mlp_hidden_dims=[128, 64],         
                 pooling='mean',
                 dropout=0.5,
                 batch_norm=True,
                 activation='relu',
                 normalize_L1=True):
        super().__init__()
        
        self.pooling = pooling
        self.normalize_L1 = normalize_L1
        
        # Convoluções (igual antes)
        self.convs = nn.ModuleList()
        self.convs.append(
            HodgeConvLayer(1, hidden_dims[0], 
                          batch_norm=batch_norm,
                          activation=activation,
                          dropout=dropout)
        )
        for i in range(len(hidden_dims) - 1):
            self.convs.append(
                HodgeConvLayer(hidden_dims[i], hidden_dims[i+1],
                              batch_norm=batch_norm,
                              activation=activation,
                              dropout=dropout)
            )
        
        # MLP Classificador - CUSTOMIZÁVEL! ✨
        mlp_layers = []
        
        # Input do MLP: saída das convoluções após pooling
        input_dim = hidden_dims[-1]
        
        # Camadas ocultas do MLP
        for mlp_dim in mlp_hidden_dims:
            mlp_layers.append(nn.Linear(input_dim, mlp_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            input_dim = mlp_dim
        
        # Camada final: última hidden_dim → num_classes
        mlp_layers.append(nn.Linear(input_dim, num_classes))
        
        self.classifier = nn.Sequential(*mlp_layers)
                
    def forward(self, x, edge_index, batch):
        """
        Args:
            x: [num_nodes, num_node_features] - NÃO USADO! 
               (HodgeGNN opera apenas em arestas)
            edge_index: [2, num_edges] - conectividade
            batch: [num_nodes] - indica qual nó pertence a qual grafo
        
        Returns:
            logits: [batch_size, num_classes]
        """
        device = edge_index.device
        num_nodes = x.shape[0]
        num_edges = edge_index.shape[1]
        
        # Extrair edge_batch: qual aresta pertence a qual grafo
        # edge_batch[i] = índice do grafo ao qual a aresta i pertence
        edge_batch = batch[edge_index[0]]  # [num_edges]
        
        # ===== PASSO 1: Calcular Hodge Laplacian L1 =====
        L1, B1 = compute_hodge_laplacian_L1(edge_index, num_nodes)
        
        # Normalizar L1 (opcional, mas geralmente melhora)
        if self.normalize_L1:
            L1 = normalize_hodge_laplacian(L1)
        
        # ===== PASSO 2: Edge Features Iniciais =====
        # No paper, eles usam os pesos das arestas
        # Como MUTAG tem edge_attr, podemos usar
        # Por simplicidade, vamos usar "grau" das arestas como feature inicial
        edge_features = torch.ones(num_edges, 1, device=device)  # [num_edges, 1]
        
        # ===== PASSO 3: Convoluções em Arestas =====
        h = edge_features
        for conv in self.convs:
            h = conv(h, L1)
        # h agora é [num_edges, hidden_dims[-1]]
        
        # ===== PASSO 4: Pooling (agregar arestas → grafo) =====
        if self.pooling == 'mean':
            h_graph = global_mean_pool(h, edge_batch)
        elif self.pooling == 'max':
            h_graph = global_max_pool(h, edge_batch)
        elif self.pooling == 'sum':
            h_graph = global_add_pool(h, edge_batch)
        # h_graph agora é [batch_size, hidden_dims[-1]]
        
        # ===== PASSO 5: Classificação =====
        logits = self.classifier(h_graph)
        
        return logits