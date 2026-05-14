"""
Configuracao — BiSheafNodeClassifier em classificacao de nos.

Datasets suportados:
  Planetoid: Cora, Citeseer, PubMed
  WebKB:     Texas, Cornell, Wisconsin  (split_idx 0-9)

Rode com:
    python -m src.experiments.bisheaf_diffusion.train_node
    python -m src.experiments.bisheaf_diffusion.train_node --dataset Cora
    python -m src.experiments.bisheaf_diffusion.train_node --dataset Texas --all-splits
"""

CONFIG = {
    "dataset": "Texas",
    "epochs": 2000,
    "seed": 1,
    "lr": 1e-4,
    "weight_decay": 5e-4,
    "patience": 0,          # early stopping; 0 = desabilitado
    "train_on_trainval": False, # True: treina com train+val, avalia no test
    "print_every": 50,

    # Texas:   183 nos, 309 arestas, 5 classes, 10 splits 48/27/109 (train/val/test)
    # Cornell: 183 nos, 295 arestas, 5 classes, 10 splits
    # Cora:    2708 nos, 10556 arestas, 7 classes, split 140/500/1000
    "model": {
        "d": 5,
        "hidden_channels": 128,
        "num_layers": 3,
        "learner_hidden": 128,
        "learner_gnn_layers": 3,
        "dropout": 0.5,
        "left_weights": True,
        "right_weights": True,
        "use_eps": True,
        "norm_eps": 1e-3,
        "mlp_dims": [128],
        # backbone: 'node_edge_node' | 'sage' | 'gcn' | 'gat' | 'gin' | 'mlp' | 'edge_aware' | 'mlp_edge'
        "backbone_F": "node_edge_node",
        "backbone_C": "node_edge_node",
        # map_type: 'general' | 'diagonal' | 'symmetric' | 'orthogonal' | 'stochastic' | 'permutation'
        "map_type_F": "general",
        "map_type_C": "general",
        # orth_trans: usado só quando map_type='orthogonal'
        # 'householder' | 'cayley' | 'matrix_exp' | 'euler'
        # NOTA: 'householder' depende do pacote externo torch_householder, que
        # trava em Windows com algumas versoes de CUDA (deadlock no primeiro
        # forward). 'cayley' e 'matrix_exp' sao puro PyTorch e equivalentes.
        "orth_trans": "cayley",
    },
}
