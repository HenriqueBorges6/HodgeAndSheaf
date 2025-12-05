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