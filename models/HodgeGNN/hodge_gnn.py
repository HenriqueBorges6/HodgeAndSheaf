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
    
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
class HodgeGNN(nn.Module):
    def __init__(self, 
                 num_edge_features=4,  # ← NOVO! MUTAG tem 4
                 num_classes=2,
                 hidden_dims=[64, 64, 32],
                 mlp_hidden_dims=[128, 64],
                 pooling='mean',
                 dropout=0.5,
                 batch_norm=True,
                 activation='relu',
                 normalize_L1=True,
                 use_edge_attr=True):  # ← NOVO!
        super().__init__()
        
        self.pooling = pooling
        self.normalize_L1 = normalize_L1
        self.use_edge_attr = use_edge_attr
        
        # Primeira camada de convolução
        # Se usar edge_attr: entrada tem num_edge_features dimensões
        # Se não: entrada tem 1 dimensão (só 1s)
        input_dim = num_edge_features if use_edge_attr else 1
        
        self.convs = nn.ModuleList()
        self.convs.append(
            HodgeConvLayer(input_dim, hidden_dims[0],  # ← Mudou de 1 para input_dim
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
        
        # MLP (igual antes)
        mlp_layers = []
        input_dim = hidden_dims[-1]
        for mlp_dim in mlp_hidden_dims:
            mlp_layers.extend([
                nn.Linear(input_dim, mlp_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = mlp_dim
        mlp_layers.append(nn.Linear(input_dim, num_classes))
        self.classifier = nn.Sequential(*mlp_layers)
        
    def forward(self, x, edge_index, batch, edge_attr=None):  # ← NOVO argumento!
        """
        Args:
            x: [num_nodes, num_node_features] - não usado (por enquanto)
            edge_index: [2, num_edges]
            batch: [num_nodes]
            edge_attr: [num_edges, num_edge_features] ← NOVO!
        """
        device = edge_index.device
        num_nodes = x.shape[0]
        num_edges = edge_index.shape[1]
        
        edge_batch = batch[edge_index[0]]
        
        # 1. Calcular L1
        L1, B1 = compute_hodge_laplacian_L1(edge_index, num_nodes)
        if self.normalize_L1:
            L1 = normalize_hodge_laplacian(L1)
        
        # 2. Edge Features Iniciais
        if self.use_edge_attr and edge_attr is not None:
            # Usar edge attributes do dataset! ✅
            h = edge_attr.float()  # [num_edges, num_edge_features]
        else:
            # Fallback: apenas 1s
            h = torch.ones(num_edges, 1, device=device)
        
        # 3. Convoluções
        for conv in self.convs:
            h = conv(h, L1)
        
        # 4. Pooling
        if self.pooling == 'mean':
            h_graph = global_mean_pool(h, edge_batch)
        elif self.pooling == 'max':
            h_graph = global_max_pool(h, edge_batch)
        elif self.pooling == 'sum':
            h_graph = global_add_pool(h, edge_batch)
        
        # 5. Classificação
        logits = self.classifier(h_graph)
        
        return logits