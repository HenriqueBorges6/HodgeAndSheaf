"""
Configuracao para SheafNN no Cora.

Cora: 2708 nos, 10556 arestas, 1433 features de no, 7 classes.
      Split fixo: 140 train / 500 val / 1000 test.
      Baseline: ~55% (maioria), GCN tipico ~81%.

Rode com:
    python -m src.experiments.sheaf_nn.train
    python -m src.experiments.sheaf_nn.train --dataset Citeseer
    python -m src.experiments.sheaf_nn.train --conv general
"""

CONFIG = {
    "dataset": "Cornell",
    "epochs": 2000,
    "seed": 0,
    "lr": 0.01,
    "weight_decay": 5e-4,
    "train_on_trainval": True,   # True: treina com train+val, avalia no test

    "model": {
        "conv_type": "bundle",
        "orth_trans": "householder",   # euler com d=2: 1 param por no (angulo)
        "d": 2,
        "hidden_channels": 64,
        "num_layers": 2,
        "gnn_type": "SAGE",
        "gnn_layers": 1,
        "gnn_hidden": 64,
        "dropout": 0.5,
        "left_weights": True,
        "right_weights": True,
        "use_eps": True,
        "mlp_dims": [64],   # [] = linear direto, [64] = d*f->64->classes
    }
}

# Notas sobre datasets suportados:
#   Planetoid: Cora, Citeseer, PubMed
#   WebKB:     Texas, Cornell, Wisconsin  (heterophilous, split_idx 0-9)
#
# Exemplos de uso:
#   python -m src.experiments.sheaf_nn.train
#   python -m src.experiments.sheaf_nn.train --dataset Cora
#   python -m src.experiments.sheaf_nn.train --dataset Texas --split 3
#   python -m src.experiments.sheaf_nn.train --baseline
