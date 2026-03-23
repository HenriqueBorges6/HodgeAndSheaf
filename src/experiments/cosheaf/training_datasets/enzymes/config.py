"""
Grid search para CoSheafNetwork no ENZYMES.

ENZYMES: 600 grafos de proteinas, 6 classes (tipos de enzima),
         ~33 nos/grafo, 3 features de no, SEM features de aresta nativas.
         Baseline maioria: ~0.167

Rode com:
    python -m src.experiments.cosheaf.training_datasets.enzymes.train
    python -m src.experiments.cosheaf.training_datasets.enzymes.train --dry-run
    python -m src.experiments.cosheaf.training_datasets.enzymes.train --out-dir runs/enzymes_sweep
"""

DATASET = "ENZYMES"
SEED = 1
EPOCHS = 300
BATCH_SIZE = 64
N_FOLDS = 10

GRID = {
    "d":               [2, 3],
    "hidden_dims":     [[64, 64], [128, 128]],
    "learner_hidden":  [[32], [64]],
    "mlp_dims":        [[64], [128]],
    "dropout":         [0.0],
    "backbone":        ["sage", "gin"],
    "orth_method":     ["cayley"],
    "lr":              [0.01, 0.001],
    "weight_decay":    [0.0],
    "edge_feat_mode":  ["none", "all"],
}