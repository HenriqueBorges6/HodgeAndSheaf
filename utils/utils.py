import torch
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

from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader
import numpy as np
def create_data_splits(dataset, n_splits=10, batch_size=32, random_seed=42):
    """
    Cria splits para validação cruzada k-fold estratificada.
    
    A estratificação garante que cada fold tenha aproximadamente a mesma
    proporção de classes que o dataset completo. Isso é importante porque
    temos um desbalanceamento de classes no MUTAG.
    """
    # Extraindo os labels para estratificação
    y = np.array([data.y.item() for data in dataset])
    
    # Criando o objeto de validação cruzada
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    
    # Lista para armazenar os loaders de cada fold
    fold_loaders = []
    
    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        # Dividimos train_val_idx em treino e validação (80/20)
        # Usamos estratificação novamente aqui
        inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
        train_idx, val_idx = next(inner_skf.split(
            np.zeros(len(train_val_idx)), 
            y[train_val_idx]
        ))
        
        # Convertendo índices relativos para absolutos
        train_idx = train_val_idx[train_idx]
        val_idx = train_val_idx[val_idx]
        
        # Criando subsets
        train_dataset = dataset[train_idx.tolist()]
        val_dataset = dataset[val_idx.tolist()]
        test_dataset = dataset[test_idx.tolist()]
        
        # Criando DataLoaders
        # O DataLoader do PyG automaticamente faz o batching de múltiplos grafos
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        fold_loaders.append({
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'test_size': len(test_dataset)
        })
        
        if fold_idx == 0:
            print(f'Fold {fold_idx + 1}:')
            print(f'  Treino: {len(train_dataset)} grafos')
            print(f'  Validação: {len(val_dataset)} grafos')
            print(f'  Teste: {len(test_dataset)} grafos')
    
    return fold_loaders
