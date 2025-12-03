import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool # 

import random
import numpy as np
import os

def set_seed(seed=42):
    # 1. Python nativo
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2. Numpy (muito usado pelo Scikit-Learn e DataLoaders)
    np.random.seed(seed)
    
    # 3. PyTorch (CPU e GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # Se usar multi-GPU
    
    # 4. Forçar algoritmos determinísticos (CuDNN)
    # Isso deixa um pouco mais lento, mas garante que a GPU não use atalhos randômicos
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Seed fixada em {seed}")

# --- CHAME ISSO ANTES DE TUDO ---
set_seed(0)

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
    

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset

# 1. Carregar e Embaralhar os Dados
# O dataset MUTAG vem ordenado, então o shuffle é obrigatório!
dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG').shuffle()

# 2. Divisão de Treino e Teste (Split manual)
# Vamos usar os primeiros 150 para treinar e o resto para testar
train_dataset = dataset[:150]
test_dataset = dataset[150:]

# 3. Criar os Carregadores (Loaders)
# Eles cuidam de pegar grafos pequenos e montar o "super grafo" (batch) automaticamente
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 4. Instanciar o Modelo e Ferramentas
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Usa GPU se tiver

model = GNN(dataset, hidden_channels=64).to(device) # Joga o modelo pra GPU
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # Quem atualiza os pesos
criterion = torch.nn.CrossEntropyLoss() # Quem calcula a nota do erro

print(f"Modelo pronto no dispositivo: {device}")
print(model)

def train():
    model.train() # 1. Avisa que vamos treinar (liga Dropout, etc)
    total_loss = 0
    
    # Itera sobre os batches (lotes de grafos)
    for data in train_loader:
        data = data.to(device) # Move os dados para GPU/CPU
        
        optimizer.zero_grad() # 2. Zera os gradientes antigos (limpa a memória)
        
        # 3. Forward Pass: O modelo tenta adivinhar
        # Note que usamos data.x, data.edge_index e data.batch
        out = model(data.x, data.edge_index, data.batch) 
        
        # 4. Calcula o erro (Loss) comparando com a resposta real (data.y)
        loss = criterion(out, data.y)
        
        # 5. Backward Pass: Calcula quem errou e quanto (Gradientes)
        loss.backward()
        
        # 6. Atualiza os pesos
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def test(loader):
    model.eval() # 1. Avisa que é teste (desliga Dropout)
    correct = 0
    
    for data in loader:
        data = data.to(device)
        
        # No teste não precisamos calcular gradientes (economiza memória)
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.batch)
            
        # Pega a classe com maior probabilidade (argmax)
        pred = out.argmax(dim=1) 
        
        # Soma quantos acertos tivemos
        correct += int((pred == data.y).sum()) 
        
    return correct / len(loader.dataset)

# --- O Loop Principal (Épocas) ---
print("Iniciando treinamento...")

for epoch in range(1, 501): 
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    
    if epoch % 20 == 0: 
        print(f'Época: {epoch:03d}, Loss: {loss:.4f}, Acurácia Treino: {train_acc:.4f}, Acurácia Teste: {test_acc:.4f}')

from sklearn.metrics import confusion_matrix

# --- Função Auxiliar para pegar todas as previsões ---
@torch.no_grad()
def get_all_preds(loader):
    model.eval()
    all_preds = []
    all_labels = []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        all_preds.append(pred.cpu())
        all_labels.append(data.y.cpu())
    return torch.cat(all_preds), torch.cat(all_labels)

# --- Execução e Print ---
print("\n--- Resultados Finais ---")
preds, labels = get_all_preds(test_loader)

# Calcula a matriz
cm = confusion_matrix(labels, preds)

# 1. Print da Matriz crua (formato array)
print("Matriz de Confusão (Array):")
print(cm)

# 2. Print Detalhado (Extraindo os valores)
# .ravel() funciona bem para binário (achata a matriz 2x2 em 4 números)
tn, fp, fn, tp = cm.ravel()

print("\n")
print(f" Verdadeiros Negativos : {tn}")
print(f" Verdadeiros Positivos : {tp}")
print(f" Falsos Positivos : {fp}")
print(f" Falsos Negativos : {fn}")