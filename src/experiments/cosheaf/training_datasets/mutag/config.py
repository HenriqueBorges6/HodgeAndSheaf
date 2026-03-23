"""
Grid search para CoSheafNetwork no MUTAG.

MUTAG: 188 grafos moleculares, 2 classes (mutagenicos vs nao),
       ~17 nos/grafo, 7 features de no, 4 features de aresta.
       Baseline maioria: ~0.66

Rode com:
    python -m src.experiments.cosheaf.training_datasets.mutag.train
    python -m src.experiments.cosheaf.training_datasets.mutag.train --dry-run
    python -m src.experiments.cosheaf.training_datasets.mutag.train --out-dir runs/mutag_sweep
"""

DATASET = "MUTAG"
SEED = 1
EPOCHS = 300
BATCH_SIZE = 64
N_FOLDS = 5

GRID = {
    "d":               [2, 3],
    "hidden_dims":     [[64, 64], [128, 128]],
    "learner_hidden":  [[32], [64]],
    "mlp_dims":        [[64],],
    "dropout":         [0.0,],
    "backbone":        ["sage", "gin"],
    "orth_method":     ["cayley"],
    "lr":              [0.01, 0.001],
    "weight_decay":    [0.0, 1e-4],
    "edge_feat_mode":  ["none"],
}
