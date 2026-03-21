CONFIG = {
    "dataset": "MUTAG",
    "batch_size": 32,
    "epochs": 200,
    "seed": 1,
    "lr": 0.001,
    "weight_decay": 0,
    "lambda_hol": 0.01,   # peso da regularização de holonomia

    "model": {
        "in_node_dim": 7,
        "in_edge_dim": 4,
        "d": 4,
        "num_layes": 3,
        "out_dim": 2,
        "num_powers": 3,
        "mlp_dims": [(64, "relu")],   # lista de (hidden_dim, activation) — [] para Linear direto
    }
}
