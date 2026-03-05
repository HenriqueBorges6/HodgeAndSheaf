CONFIG = {
    "dataset": "MUTAG",
    "batch_size": 32,
    "epochs": 100,
    "seed": 1,
    "lr" : 0.001,
    "weight_decay" : 0,

    "model": {
        "in_dim": 4,
        "hidden_dims": [64, 64, 64],
        "out_dim": 2,
        "normalize": "symmetric",
        "pooling": "sum",
        "residual": False,
        "mlp_dim": 64,
        "dropout": 0.15,
    }
}
