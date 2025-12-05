import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

class GCN(nn.Module):
    def __init__(self, 
                 num_features,
                 num_classes,
                 hidden_dims=[64, 32],  # Lista com dimensões de cada camada oculta
                 conv_type='GCN',  # Tipo de camada: 'GCN', 'GAT', 'GraphSAGE'
                 activation='relu',  # Função de ativação
                 dropout=0.5,  # Taxa de dropout
                 batch_norm=True,  # Se usa batch normalization
                 residual=False,  # Se usa conexões residuais
                 pooling='none',  # Para classificação de grafos: 'mean', 'max', 'add', 'none'
                 jk_mode=None):  # Jumping Knowledge: 'concat', 'max', 'lstm', None
        super(GCN, self).__init__()
        
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.residual = residual
        self.pooling = pooling
        self.jk_mode = jk_mode
        
        # Selecionando a função de ativação
        self.activation = self._get_activation(activation)
        
        # Construindo a lista de camadas convolucionais
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None
        
        # Dimensões de entrada e saída para cada camada
        input_dim = num_features
        for hidden_dim in hidden_dims:
            # Adiciona a camada convolucional apropriada
            conv = self._get_conv_layer(conv_type, input_dim, hidden_dim)
            self.convs.append(conv)
            
            # Adiciona batch normalization se solicitado
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
            input_dim = hidden_dim
        
        # Se estamos usando Jumping Knowledge, precisamos ajustar a dimensão final
        if jk_mode == 'concat':
            # Concatena as saídas de todas as camadas
            final_dim = num_features + sum(hidden_dims)
        elif jk_mode in ['max', 'lstm']:
            # Usa apenas a dimensão da última camada
            final_dim = hidden_dims[-1] if hidden_dims else num_features
        else:
            # Modo padrão: apenas a última camada
            final_dim = hidden_dims[-1] if hidden_dims else num_features
        
        # Camada final de classificação
        self.classifier = nn.Linear(final_dim, num_classes)
        
        # Para conexões residuais, precisamos garantir que as dimensões sejam compatíveis
        if residual:
            self.residual_transforms = nn.ModuleList()
            input_dim = num_features
            for hidden_dim in hidden_dims:
                if input_dim != hidden_dim:
                    # Se as dimensões não batem, usamos uma projeção linear
                    self.residual_transforms.append(nn.Linear(input_dim, hidden_dim))
                else:
                    # Se as dimensões batem, não precisamos de transformação
                    self.residual_transforms.append(nn.Identity())
                input_dim = hidden_dim
        
        # Para Jumping Knowledge com LSTM
        if jk_mode == 'lstm':
            # O LSTM processa a sequência de embeddings de cada camada
            self.jk_lstm = nn.LSTM(hidden_dims[-1], hidden_dims[-1], batch_first=True)
    
    def _get_activation(self, name):
        """Retorna a função de ativação apropriada"""
        activations = {
            'relu': F.relu,
            'elu': F.elu,
            'leaky_relu': F.leaky_relu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
        }
        if name not in activations:
            raise ValueError(f"Ativação {name} não suportada")
        return activations[name]
    
    def _get_conv_layer(self, conv_type, in_dim, out_dim):
        """Retorna a camada convolucional apropriada"""
        if conv_type == 'GCN':
            return GCNConv(in_dim, out_dim)
        elif conv_type == 'GAT':
            # GAT usa atenção multi-head, vamos usar 8 heads por padrão
            # A dimensão de saída é multiplicada pelo número de heads
            return GATConv(in_dim, out_dim // 8, heads=8, concat=True)
        elif conv_type == 'GraphSAGE':
            return SAGEConv(in_dim, out_dim)
        else:
            raise ValueError(f"Tipo de convolução {conv_type} não suportado")
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass pela rede
        
        Args:
            x: Features dos nós [num_nodes, num_features]
            edge_index: Conectividade do grafo [2, num_edges]
            batch: Tensor indicando a qual grafo cada nó pertence (para batch de grafos)
        
        Returns:
            Logits de classificação
        """
        # Guardamos as representações de cada camada para Jumping Knowledge
        layer_outputs = [x]
        
        # Passamos pelas camadas convolucionais
        for i, conv in enumerate(self.convs):
            # Aplicamos a convolução
            x_new = conv(x, edge_index)
            
            # Aplicamos batch normalization se configurado
            if self.batch_norms is not None:
                x_new = self.batch_norms[i](x_new)
            
            # Aplicamos a função de ativação
            x_new = self.activation(x_new)
            
            # Aplicamos dropout (apenas durante treinamento)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            
            # Conexão residual se configurado
            if self.residual:
                # Transformamos x para ter a mesma dimensão que x_new se necessário
                x_residual = self.residual_transforms[i](x)
                x_new = x_new + x_residual
            
            # Guardamos a saída desta camada
            layer_outputs.append(x_new)
            
            # Atualizamos x para a próxima iteração
            x = x_new
        
        # Aplicamos Jumping Knowledge se configurado
        if self.jk_mode == 'concat':
            # Concatena todas as representações
            x = torch.cat(layer_outputs, dim=-1)
        elif self.jk_mode == 'max':
            # Pega o máximo elemento-wise através das camadas
            x = torch.stack(layer_outputs[1:], dim=0).max(dim=0)[0]
        elif self.jk_mode == 'lstm':
            # Usa um LSTM para combinar as representações
            # Empilha as saídas: [num_layers, num_nodes, hidden_dim]
            stacked = torch.stack(layer_outputs[1:], dim=0)
            # Transpõe para [num_nodes, num_layers, hidden_dim]
            stacked = stacked.transpose(0, 1)
            # Passa pelo LSTM e pega o último estado oculto
            _, (x, _) = self.jk_lstm(stacked)
            x = x.squeeze(0)
        # Caso contrário, mantemos apenas a última camada (já está em x)
        
        # Pooling para classificação de grafos inteiros
        if self.pooling != 'none' and batch is not None:
            if self.pooling == 'mean':
                x = global_mean_pool(x, batch)
            elif self.pooling == 'max':
                x = global_max_pool(x, batch)
            elif self.pooling == 'add':
                x = global_add_pool(x, batch)
        
        # Camada final de classificação
        x = self.classifier(x)
        
        return x