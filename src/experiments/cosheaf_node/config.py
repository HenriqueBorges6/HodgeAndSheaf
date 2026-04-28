"""
Configuracao para CoSheafNodeNetwork no Cora.

Cora: 2708 nos, 10556 arestas, 1433 features de no, 7 classes.
      Sem edge features nativas — usa edge_feat_mode para gerar.
      Split fixo: 140 train / 500 val / 1000 test.
      Baseline: ~55% (maioria), GCN tipico ~81%.

Rode com:
    python -m src.experiments.cosheaf_node.train
    python -m src.experiments.cosheaf_node.train --dataset Citeseer
"""

CONFIG = {
    "dataset": "Cora",        # Cora | Citeseer | PubMed
    "epochs": 500,
    "seed": 1,
    "lr": 0.0001,
    "weight_decay": 0,
    # Features de aresta: 'none' (ones) | 'structural' (4) | 'node_derived' (2*F) | 'all' (4+2*F)
    "edge_feat_mode": "node_derived",

    "model": {
        # node_feat_dim e num_classes sao detectados automaticamente
        "d": 4,
        "hidden_dims": [128,32],
        "learner_hidden": [128],
        "mlp_dims": [128],           # classificador linear direto
        "dropout": 0.5,
        "backbone": "sage",          # usado apenas com conv_mode='linegraph'
        "orth_method": "cayley",
        "conv_mode": "direct",       # 'direct' (sem line graph) | 'linegraph' (original)
    }
}
