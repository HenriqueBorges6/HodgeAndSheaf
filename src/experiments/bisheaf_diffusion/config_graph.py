"""
Configuracao — BiSheafGraphClassifier em classificacao de grafos.

Datasets suportados (TUDataset):
  MUTAG:   188 grafos, 2 classes, 7 features de no, ~17 nos/grafo
  PROTEINS: 1113 grafos, 2 classes, 3 features de no
  IMDB-B:  1000 grafos, 2 classes, sem features de no (usa grau)

Rode com:
    python -m src.experiments.bisheaf_diffusion.train_graph
    python -m src.experiments.bisheaf_diffusion.train_graph --dataset PROTEINS
"""

CONFIG = {
    "dataset": "MUTAG",
    "epochs": 200,
    "seed": 1,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "n_folds": 10,     # k-fold estratificado
    "print_every": 50,

    # MUTAG: baseline maioria ~66%, GNN tipico ~85-89%
    "model": {
        "d": 2,
        "hidden_channels": 64,
        "num_layers": 2,
        "learner_hidden": 64,
        "learner_gnn_layers": 1,
        "dropout": 0.5,
        "left_weights": True,
        "right_weights": True,
        "use_eps": True,
        "norm_eps": 1e-3,
        "pooling": "sum",
        "mlp_dims": [64],
        "backbone_F": "node_edge_node",
        "backbone_C": "node_edge_node",
        "map_type_F": "general",
        "map_type_C": "general",
        "orth_trans": "householder",
    },
}
