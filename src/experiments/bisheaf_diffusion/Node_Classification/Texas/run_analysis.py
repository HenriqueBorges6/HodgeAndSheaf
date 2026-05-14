"""
Análise pós-sweep: re-treina as top-N configs e roda análise completa.

Para cada configuração selecionada:
  1. Lê grid_results.json → identifica o split com melhor val_acc
  2. Se os pesos já existem em weights/, carrega direto
     Senão, re-treina aquele (config, split) e salva os pesos
  3. Roda analyze_config → salva JSON em results/grid/analysis/

Uso:
    python -m src.experiments.bisheaf_diffusion.Node_Classification.Texas.run_analysis
    python -m src.experiments.bisheaf_diffusion.Node_Classification.Texas.run_analysis --top-n 20
    python -m src.experiments.bisheaf_diffusion.Node_Classification.Texas.run_analysis --force-retrain
    python -m src.experiments.bisheaf_diffusion.Node_Classification.Texas.run_analysis --no-full-spectrum
"""

import argparse
import json
import os
import shutil

import torch

from src.experiments.bisheaf_diffusion.Node_Classification.Texas._analysis_lib import analyze_config
from src.experiments.bisheaf_diffusion.Node_Classification.Texas.sweep import (
    _RESULTS_DIR,
    _TRAIN_KEYS,
    BASE_CONFIG,
    _generate_grid,
)
from src.experiments.bisheaf_diffusion.train_node import load_data, main as train_main
from src.models.BiSheafDiffusion import BiSheafNodeClassifier

_ANALYSIS_DIR = os.path.join(_RESULTS_DIR, "analysis")
_WEIGHTS_DIR  = os.path.join(_RESULTS_DIR, "weights")


# ---------------------------------------------------------------------------
# Leitura dos resultados
# ---------------------------------------------------------------------------

def _load_results(results_file: str = None):
    if results_file:
        path = results_file if os.path.isabs(results_file) else os.path.join(_RESULTS_DIR, results_file)
    else:
        # Tenta grid_results.json, depois sample_results.json, depois grid_partial.json
        for name in ("grid_results.json", "sample_results.json",
                     "grid_partial.json", "sample_partial.json"):
            path = os.path.join(_RESULTS_DIR, name)
            if os.path.exists(path):
                break
        else:
            raise FileNotFoundError(
                f"Nenhum arquivo de resultados encontrado em {_RESULTS_DIR}. "
                "Rode o sweep antes de rodar a análise."
            )
    print(f"  Lendo resultados de: {os.path.basename(path)}")
    with open(path) as f:
        return json.load(f)


def _aggregate_raw(raw: list) -> dict:
    from collections import defaultdict
    import numpy as np
    buckets: dict = defaultdict(list)
    for r in raw:
        if r.get("error") is None and r.get("test_acc") is not None:
            buckets[r["config"]].append(r["test_acc"])
    return {
        cfg: {"mean": float(np.array(accs).mean() * 100),
              "std":  float(np.array(accs).std()  * 100),
              "n":    len(accs)}
        for cfg, accs in buckets.items()
    }


def _get_top_configs(results, top_n):
    agg = results.get("aggregated") or _aggregate_raw(results.get("raw", []))
    valid = [(cfg, v) for cfg, v in agg.items() if v.get("mean") is not None]
    valid.sort(key=lambda x: -x[1]["mean"])
    return valid[:top_n]


def _find_best_split(results, cfg_name):
    """
    Retorna (split_idx, val_acc, cfg_override) do split com maior val_acc.
    cfg_override pode ser None se o sweep foi gerado sem essa chave.
    """
    raw = [
        r for r in results.get("raw", [])
        if r["config"] == cfg_name and r.get("val_acc") is not None
    ]
    if not raw:
        return None, None, None
    best = max(raw, key=lambda r: r["val_acc"])
    return best["split_idx"], best["val_acc"], best.get("cfg_override")


def _cfg_override_from_grid(cfg_name, split_idx):
    """
    Fallback: reconstrói cfg_override a partir do grid se não estiver nos resultados.
    Retorna None se cfg_name não for encontrado no grid atual.
    """
    grid = {name: params for name, params in _generate_grid()}
    if cfg_name not in grid:
        return None
    params          = grid[cfg_name]
    model_overrides = {k: v for k, v in params.items() if k not in _TRAIN_KEYS}
    train_overrides = {k: v for k, v in params.items() if k in _TRAIN_KEYS}
    merged_model    = {**BASE_CONFIG["model"], **model_overrides}
    return {
        "dataset":           "Texas",
        "split_idx":         split_idx,
        "train_on_trainval": False,
        "epochs":            BASE_CONFIG["epochs"],
        "patience":          BASE_CONFIG["patience"],
        "lr":                train_overrides.get("lr", BASE_CONFIG["lr"]),
        "weight_decay":      train_overrides.get("weight_decay", BASE_CONFIG["weight_decay"]),
        "seed":              BASE_CONFIG["seed"],
        "model":             merged_model,
    }


# ---------------------------------------------------------------------------
# Reconstrução do modelo
# ---------------------------------------------------------------------------

def _rebuild_model(cfg_override, num_classes, node_feat_dim, device):
    mcfg = cfg_override["model"]
    return BiSheafNodeClassifier(
        node_feat_dim      = node_feat_dim,
        d                  = mcfg["d"],
        hidden_channels    = mcfg["hidden_channels"],
        num_layers         = mcfg["num_layers"],
        num_classes        = num_classes,
        learner_hidden     = mcfg.get("learner_hidden", 64),
        learner_gnn_layers = mcfg.get("learner_gnn_layers", 1),
        dropout            = mcfg.get("dropout", 0.0),
        left_weights       = mcfg.get("left_weights", True),
        right_weights      = mcfg.get("right_weights", True),
        use_eps            = mcfg.get("use_eps", True),
        norm_eps           = mcfg.get("norm_eps", 1e-3),
        mlp_dims           = mcfg.get("mlp_dims"),
        backbone_F         = mcfg.get("backbone_F", "node_edge_node"),
        backbone_C         = mcfg.get("backbone_C", "node_edge_node"),
        map_type_F         = mcfg.get("map_type_F", "general"),
        map_type_C         = mcfg.get("map_type_C", "general"),
        orth_trans         = mcfg.get("orth_trans", "householder"),
        sheaf_act          = mcfg.get("sheaf_act", "id"),
        inter_layer_norm   = mcfg.get("inter_layer_norm", False),
    ).to(device)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def main(top_n=10, force_retrain=False, full_spectrum=True, results_file=None):
    """
    Analisa as top_n configs por test acc médio.

    Args:
        top_n:         número de configs a analisar
        force_retrain: ignora pesos salvos e re-treina sempre
        full_spectrum: inclui listas completas de autovalores no JSON
        results_file:  caminho para JSON de resultados (padrão: auto-detectado)
    """
    results    = _load_results(results_file)
    top_cfgs   = _get_top_configs(results, top_n)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(_ANALYSIS_DIR, exist_ok=True)
    os.makedirs(_WEIGHTS_DIR,  exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Análise pós-sweep — Texas")
    print(f"  Top {len(top_cfgs)} configs  |  device={device}")
    print(f"  Análises → {_ANALYSIS_DIR}")
    print(f"  Pesos   → {_WEIGHTS_DIR}")
    print(f"{'='*60}\n")

    for cfg_name, stats in top_cfgs:
        mean_str = f"{stats['mean']:.1f}±{stats['std']:.1f}% ({stats['n']} splits)"
        print(f"--- {cfg_name}  test={mean_str}")

        out_path = os.path.join(_ANALYSIS_DIR, f"{cfg_name}.json")
        if os.path.exists(out_path) and not force_retrain:
            print(f"  [skip] análise já existe\n")
            continue

        split_idx, val_acc, cfg_override = _find_best_split(results, cfg_name)
        if split_idx is None:
            print(f"  [skip] nenhum split válido\n")
            continue

        if cfg_override is None:
            cfg_override = _cfg_override_from_grid(cfg_name, split_idx)
        if cfg_override is None:
            print(f"  [skip] cfg_override não encontrado — sweep gerado com grid antigo\n")
            continue

        cfg_override = {**cfg_override, "split_idx": split_idx}
        print(f"  Melhor split: {split_idx}  (val={val_acc*100:.1f}%)")

        weights_file = os.path.join(_WEIGHTS_DIR, f"{cfg_name}_split{split_idx}.pt")
        data, num_classes = load_data("Texas", split_idx=split_idx)
        node_feat_dim     = data.x.shape[1]

        if os.path.exists(weights_file) and not force_retrain:
            print(f"  Carregando pesos: {os.path.basename(weights_file)}")
            model = _rebuild_model(cfg_override, num_classes, node_feat_dim, device)
            model.load_state_dict(torch.load(weights_file, map_location=device, weights_only=True))
        else:
            print(f"  Re-treinando split {split_idx}...")
            tmp_dir = os.path.join(_WEIGHTS_DIR, f"_tmp_{cfg_name}")
            train_main(save_dir=tmp_dir, cfg_override=cfg_override, quiet=True)
            saved = os.path.join(tmp_dir, "best_model.pt")
            if not os.path.exists(saved):
                print(f"  [skip] treino não gerou pesos (NaN ou sem melhora)\n")
                shutil.rmtree(tmp_dir, ignore_errors=True)
                continue
            model = _rebuild_model(cfg_override, num_classes, node_feat_dim, device)
            model.load_state_dict(torch.load(saved, map_location=device, weights_only=True))
            torch.save(model.state_dict(), weights_file)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"  Pesos salvos: {os.path.basename(weights_file)}")

        data = data.to(device)
        print(f"  Rodando análise...")
        analyze_config(model, data, cfg_name, _ANALYSIS_DIR,
                       device=device, full_spectrum=full_spectrum)
        print(f"  Salvo: {cfg_name}.json\n")

    print(f"Concluído. Análises em: {_ANALYSIS_DIR}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Análise pós-sweep — Texas")
    parser.add_argument("--top-n",          type=int, default=10,
                        help="Número de configs a analisar (default 10).")
    parser.add_argument("--force-retrain",  action="store_true",
                        help="Re-treina mesmo se pesos já existirem.")
    parser.add_argument("--no-full-spectrum", action="store_true",
                        help="Não armazena listas completas de autovalores (economiza espaço).")
    parser.add_argument("--results-file",   type=str, default=None,
                        help="Caminho para JSON de resultados (grid_results.json, "
                             "sample_results.json, grid_partial.json, etc.). "
                             "Auto-detectado se omitido.")
    args = parser.parse_args()
    main(
        top_n         = args.top_n,
        force_retrain = args.force_retrain,
        full_spectrum = not args.no_full_spectrum,
        results_file  = args.results_file,
    )
