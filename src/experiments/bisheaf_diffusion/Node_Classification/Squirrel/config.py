"""Configuração e grid para Squirrel (Wikipedia — ~5k nós, 5 classes, 10 splits)."""

import copy
from src.experiments.bisheaf_diffusion.config_node import CONFIG as _BASE

BASE_CONFIG = copy.deepcopy(_BASE)
BASE_CONFIG.update({
    "dataset":           "Squirrel",
    "train_on_trainval": False,
    "epochs":            2000,
    "patience":          200,
    "lr":                1e-3,
    "weight_decay":      5e-4,
    "seed":              0,
})
BASE_CONFIG["model"].update({
    "mlp_dims":   [64],
    "orth_trans": "cayley",
})

# 3×2×2×2×2×2 = 96 configs × 10 splits = 960 jobs
GRID = {
    "model": {
        "d":               [1, 2, 3],
        "hidden_channels": [32, 64],
        "backbone":        ["sage", "gin"],
        "map_type_F":      ["general", "orthogonal"],
        "map_type_C":      ["general", "orthogonal"],
        "dropout":         [0.3, 0.5],
    },
}