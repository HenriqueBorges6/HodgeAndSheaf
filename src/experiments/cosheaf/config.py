CONFIG = {
    "dataset": "MUTAG",   # MUTAG | PTC_MR | ENZYMES | PROTEINS
    "batch_size": 64,
    "epochs": 300,
    "seed": 1,
    "lr": 0.001,
    "weight_decay": 0,
    # Features artificiais de aresta para datasets sem edge_attr (ENZYMES, PROTEINS).
    # Ignorado se o dataset já tem edge_attr nativo (ex: MUTAG, PTC_MR).
    # Opções: 'none' | 'structural' | 'node_derived' | 'all'
    "edge_feat_mode": "none",

    "model": {
        # node_feat_dim, edge_feat_dim e num_classes sao detectados automaticamente
        "d": 2,                       # dimensao do stalk (espaco de secoes por aresta)
        "hidden_dims": [64, 64],      # dimensoes das camadas CoSheafConv
        "learner_hidden": [64],       # dimensoes internas do CoSheafLearner
        "mlp_dims": [64],             # camadas intermediarias do classificador ([] = linear direto)
        "dropout": 0.0,               # dropout entre camadas do classificador
        "backbone": "gin",            # backbone do CoSheafLearner: 'sage' | 'gin'
    }
}