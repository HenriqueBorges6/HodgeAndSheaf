"""
Configuração base e grid de hiperparâmetros para Texas.

BASE_CONFIG herda de config_node.py e sobrescreve dataset, épocas,
paciência e transformação ortogonal (cayley). train_on_trainval=False
para usar validação no early stopping e reportar test@best_val.

GRID — produto cartesiano de model × train:
  Modelo (82 944 combinações = 4×3×2×2×2×4×3×2×2×3×3×2):
    d                : dimensão do stalk [2, 3, 4, 5]
    num_layers       : camadas de difusão [2, 3, 4]
    hidden_channels  : canais internos por dimensão do stalk [64, 128]
    map_type_F/C     : tipo dos mapas F e C [general, orthogonal]
    backbone         : backbone dos learners [sage, gin, gcn, mlp]
    learner_gnn_layers: profundidade do backbone GNN [1, 2, 3]
    learner_hidden   : dimensão oculta dos learners [64, 128]
    mlp_dims         : camadas ocultas do MLP final [[64], [128]]
    dropout          : dropout nas camadas de difusão [0, 0.25, 0.5]
    sheaf_act        : ativação nos parâmetros brutos antes da transformação
                       dos mapas [gelu, tanh, relu]  (id = sem ativação, não testado)
    inter_layer_norm : LayerNorm(d·f) entre camadas de difusão [True, False]

  Treino (4 combinações):
    lr               : taxa de aprendizado [1e-3, 1e-4]
    weight_decay     : regularização L2 [0, 5e-4]

  Total: 331 776 configs × 10 splits ≈ 3,3M jobs.
  Recomenda-se random search (n_samples) para exploração inicial.
"""

import copy
from src.experiments.bisheaf_diffusion.config_node import CONFIG as _BASE

BASE_CONFIG = copy.deepcopy(_BASE)
BASE_CONFIG.update({
    "dataset":           "Texas",
    "train_on_trainval": False,
    "epochs":            2000,
    "patience":          200,
    "seed":              0,
    "print_every":       100,
})
BASE_CONFIG["model"].update({
    "mlp_dims":   [64],
    "orth_trans": "cayley",
})


GRID = {
    "model": {
        "d":            [2, 3, 4],
        "num_layers":   [2, 3, 4],
        "hidden_channels": [64, 128],
        "map_type_F":   ["general", "orthogonal"],
        "map_type_C":   ["general", "orthogonal"],
        
        "backbone":     ["sage", "gin", "gcn", "mlp"],
        "learner_gnn_layers": [ 2, 3],
        "learner_hidden": [64, 128],
        
        "mlp_dims": [[64], [128]],
        "dropout":      [0, 0.25 ,0.5],
        "sheaf_act":    ["gelu", "tanh", "relu", "id"],
        "inter_layer_norm": [True, False],
    },
    "train": {
        "lr":           [0.001],
        "weight_decay": [5e-4],
    },
}
