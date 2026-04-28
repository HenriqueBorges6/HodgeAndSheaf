"""
Configuracao para DualSheafDiffusion no Cora.

Cora: 2708 nos, 10556 arestas, 1433 features de no, 7 classes.
      Split fixo: 140 train / 500 val / 1000 test.
      Baseline: ~55% (maioria), GCN tipico ~81%.

Rode com:
    python -m src.experiments.dual_sheaf.train
    python -m src.experiments.dual_sheaf.train --dataset Citeseer
"""

CONFIG = {
    "dataset": "Texas",
    "epochs": 2000,
    "seed": 0,
    "lr": 0.001,
    "weight_decay": 5e-4,
    "train_on_trainval": True,   # True: treina com train+val, avalia no test

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
        "single_learner": False,  # False = dual (Phi_A != Phi_B), True = classico (Phi_A = Phi_B)
        "orth_trans": 'householder',       # None = mapas gerais R^{d×d} | 'euler'/'householder'/'matrix_exp'/'cayley'
        "laplacian_mode": "dual",   # 'dual'       : F^A^T(F^B·xv - F^B·xu)
                                    # 'mixed_self' : F^A^T(F^A·xv - F^B·xu)
                                    # 'mixed_nbr'  : F^A^T(F^B·xv - F^A·xu)
                                    # 'symmetric'  : F^A^T(F^A·xv - F^A·xu) (classico)
        "cosheaf_b": False,         # True: F^B via grafo de linhas L(G) (co-feixe)
        "mlp_dims": [64],
    }
}

# Datasets suportados:
#   Planetoid: Cora, Citeseer, PubMed
#   WebKB:     Texas, Cornell, Wisconsin  (split_idx 0-9)
#
# Exemplos:
#   python -m src.experiments.dual_sheaf.train
#   python -m src.experiments.dual_sheaf.train --dataset Cornell
#   python -m src.experiments.dual_sheaf.train --dataset Texas --split 3
#   python -m src.experiments.dual_sheaf.train --all-splits

