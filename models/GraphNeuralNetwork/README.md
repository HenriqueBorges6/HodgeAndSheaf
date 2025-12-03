# Graph Neural Network

Utilizado uma GNN nos seguintes Datasets:
* MUTAG

## Implementation

Descrever modelo
```python
class GNN(nn.Module):
    def __init__(self, dataset, hidden_channels):
        super(GNN, self).__init__()
        torch.manual_seed(0)

        input_dim = dataset.num_node_features
        output_dim = dataset.num_classes

        self.conv1 = GCNConv(input_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, output_dim) 


    def forward(self, node_features, edge_index, batch):
        node_features = self.conv1(node_features, edge_index)
        node_features = F.relu(node_features)

        node_features = self.conv2(node_features, edge_index)
        node_features = F.relu(node_features)

        node_features = global_mean_pool(node_features, batch)

        node_features = F.dropout(node_features, p=0.5, training=self.training)
        node_features = self.lin(node_features)
        return node_features
```

## Experimentos:
O primeiro experimento foi realizado sem a presença de Dropout
```python
Seed fixada em 0
Modelo pronto no dispositivo: cpu
GNN(
  (conv1): GCNConv(7, 64)
  (conv2): GCNConv(64, 64)
  (lin): Linear(in_features=64, out_features=2, bias=True)
)
Iniciando treinamento...
Época: 020, Loss: 0.5461, Acurácia Treino: 0.7267, Acurácia Teste: 0.8158
Época: 040, Loss: 0.4964, Acurácia Treino: 0.7667, Acurácia Teste: 0.7895
Época: 060, Loss: 0.5051, Acurácia Treino: 0.7267, Acurácia Teste: 0.7632
Época: 080, Loss: 0.5440, Acurácia Treino: 0.7600, Acurácia Teste: 0.7105
Época: 100, Loss: 0.5489, Acurácia Treino: 0.7667, Acurácia Teste: 0.7632
Época: 120, Loss: 0.4847, Acurácia Treino: 0.7600, Acurácia Teste: 0.7368
Época: 140, Loss: 0.4785, Acurácia Treino: 0.7733, Acurácia Teste: 0.7632
Época: 160, Loss: 0.4803, Acurácia Treino: 0.7867, Acurácia Teste: 0.7632
Época: 180, Loss: 0.4379, Acurácia Treino: 0.7800, Acurácia Teste: 0.7895
Época: 200, Loss: 0.4239, Acurácia Treino: 0.7867, Acurácia Teste: 0.7895
Época: 220, Loss: 0.4444, Acurácia Treino: 0.8133, Acurácia Teste: 0.8158
Época: 240, Loss: 0.4177, Acurácia Treino: 0.8067, Acurácia Teste: 0.7895
Época: 260, Loss: 0.4367, Acurácia Treino: 0.8200, Acurácia Teste: 0.8158
Época: 280, Loss: 0.3937, Acurácia Treino: 0.8333, Acurácia Teste: 0.7895
Época: 300, Loss: 0.4361, Acurácia Treino: 0.8533, Acurácia Teste: 0.8158
Época: 320, Loss: 0.4172, Acurácia Treino: 0.8333, Acurácia Teste: 0.8158
Época: 340, Loss: 0.3690, Acurácia Treino: 0.8067, Acurácia Teste: 0.7895
Época: 360, Loss: 0.4291, Acurácia Treino: 0.8333, Acurácia Teste: 0.7632
Época: 380, Loss: 0.3652, Acurácia Treino: 0.8533, Acurácia Teste: 0.8421
Época: 400, Loss: 0.4022, Acurácia Treino: 0.8667, Acurácia Teste: 0.7895
Época: 420, Loss: 0.4075, Acurácia Treino: 0.8533, Acurácia Teste: 0.8421
Época: 440, Loss: 0.5085, Acurácia Treino: 0.8400, Acurácia Teste: 0.7895
Época: 460, Loss: 0.3856, Acurácia Treino: 0.8600, Acurácia Teste: 0.8158
Época: 480, Loss: 0.4318, Acurácia Treino: 0.8667, Acurácia Teste: 0.8158
Época: 500, Loss: 0.3901, Acurácia Treino: 0.8467, Acurácia Teste: 0.8421

--- Resultados Finais ---
Matriz de Confusão (Array):
[[ 9  4]
 [ 2 23]]


 Verdadeiros Negativos : 9
 Verdadeiros Positivos : 23
 Falsos Positivos : 4
 Falsos Negativos : 2
```

Já utilizando um dropout de 0.5 temos os seguintes resultados:
```python
Seed fixada em 0
Modelo pronto no dispositivo: cpu
GNN(
  (conv1): GCNConv(7, 64)
  (conv2): GCNConv(64, 64)
  (lin): Linear(in_features=64, out_features=2, bias=True)
)
Iniciando treinamento...
Época: 020, Loss: 0.5352, Acurácia Treino: 0.7067, Acurácia Teste: 0.8158
Época: 040, Loss: 0.4789, Acurácia Treino: 0.7333, Acurácia Teste: 0.7895
Época: 060, Loss: 0.5364, Acurácia Treino: 0.7600, Acurácia Teste: 0.7632
Época: 080, Loss: 0.5407, Acurácia Treino: 0.7467, Acurácia Teste: 0.7632
Época: 100, Loss: 0.5083, Acurácia Treino: 0.7600, Acurácia Teste: 0.7632
Época: 120, Loss: 0.5573, Acurácia Treino: 0.7667, Acurácia Teste: 0.7632
Época: 140, Loss: 0.5024, Acurácia Treino: 0.7533, Acurácia Teste: 0.7368
Época: 160, Loss: 0.4906, Acurácia Treino: 0.7533, Acurácia Teste: 0.7632
Época: 180, Loss: 0.5207, Acurácia Treino: 0.7533, Acurácia Teste: 0.7632
Época: 200, Loss: 0.5308, Acurácia Treino: 0.7533, Acurácia Teste: 0.7632
Época: 220, Loss: 0.5023, Acurácia Treino: 0.7467, Acurácia Teste: 0.7632
Época: 240, Loss: 0.5045, Acurácia Treino: 0.7533, Acurácia Teste: 0.7632
Época: 260, Loss: 0.4864, Acurácia Treino: 0.7533, Acurácia Teste: 0.7632
Época: 280, Loss: 0.4590, Acurácia Treino: 0.7533, Acurácia Teste: 0.7368
Época: 300, Loss: 0.4619, Acurácia Treino: 0.7533, Acurácia Teste: 0.7632
Época: 320, Loss: 0.4976, Acurácia Treino: 0.7867, Acurácia Teste: 0.7632
Época: 340, Loss: 0.4621, Acurácia Treino: 0.7533, Acurácia Teste: 0.7895
Época: 360, Loss: 0.5217, Acurácia Treino: 0.7867, Acurácia Teste: 0.7895
Época: 380, Loss: 0.4308, Acurácia Treino: 0.7867, Acurácia Teste: 0.8158
Época: 400, Loss: 0.4389, Acurácia Treino: 0.7667, Acurácia Teste: 0.7368
Época: 420, Loss: 0.4464, Acurácia Treino: 0.8067, Acurácia Teste: 0.8158
Época: 440, Loss: 0.4672, Acurácia Treino: 0.7867, Acurácia Teste: 0.7895
Época: 460, Loss: 0.4784, Acurácia Treino: 0.7867, Acurácia Teste: 0.7368
Época: 480, Loss: 0.4225, Acurácia Treino: 0.8067, Acurácia Teste: 0.8158
Época: 500, Loss: 0.4859, Acurácia Treino: 0.7933, Acurácia Teste: 0.7632

--- Resultados Finais ---
Matriz de Confusão (Array):
[[ 6  7]
 [ 2 23]]


 Verdadeiros Negativos : 6
 Verdadeiros Positivos : 23
 Falsos Positivos : 7
 Falsos Negativos : 2
```

O Dropout parece afetar bastante a performance do modelo, ao menos no MUTAG

```python
# Sem dropout

Matriz de Confusão (Array):
[[ 9  4]
 [ 2 23]]

 Verdadeiros Negativos : 9
 Verdadeiros Positivos : 23
 Falsos Positivos : 4
 Falsos Negativos : 2

# Com dropout p=0.5

Matriz de Confusão (Array):
[[ 6  7]
 [ 2 23]]

 Verdadeiros Negativos : 6
 Verdadeiros Positivos : 23
 Falsos Positivos : 7
 Falsos Negativos : 2
```