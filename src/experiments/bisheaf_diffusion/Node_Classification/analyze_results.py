"""
Análise dos resultados do grid search — todos os datasets.

Uso:
    python -m src.experiments.bisheaf_diffusion.Node_Classification.analyze_results
    python -m src.experiments.bisheaf_diffusion.Node_Classification.analyze_results --top 5
    python -m src.experiments.bisheaf_diffusion.Node_Classification.analyze_results --datasets Texas Wisconsin
    python -m src.experiments.bisheaf_diffusion.Node_Classification.analyze_results --results-file Texas/results/grid/sample_results.json
    python -m src.experiments.bisheaf_diffusion.Node_Classification.analyze_results --results-file Texas/results/grid/grid_partial.json
"""

import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(__file__)
_DATASETS = ["Texas", "Cornell", "Wisconsin", "Film", "Squirrel", "Chameleon", "Cora", "Citeseer"]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _load(dataset: str, results_file: str = None) -> dict | None:
    if results_file:
        path = results_file if os.path.isabs(results_file) else os.path.join(_HERE, results_file)
    else:
        path = os.path.join(_HERE, dataset, "results", "grid_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Parse do nome da config
# ---------------------------------------------------------------------------

_CFG_RE = re.compile(
    r"d(?P<d>\d+)"
    r"_h(?P<h>\d+)"
    r"_F(?P<mF>gen|ort)"
    r"_C(?P<mC>gen|ort)"
    r"(?:_(?P<bb>sage|gin|gcn|mlp|gat|node))?"
    r"(?:_dr(?P<dr>\d+))?"
)

def _parse(cfg_name: str) -> dict:
    m = _CFG_RE.search(cfg_name)
    if not m:
        return {}
    dr_raw = m.group("dr")
    return {
        "d":       int(m.group("d")),
        "h":       int(m.group("h")),
        "bb":      m.group("bb") or "?",
        "map_F":   m.group("mF"),
        "map_C":   m.group("mC"),
        "dropout": float("0." + dr_raw) if dr_raw else None,
    }


# ---------------------------------------------------------------------------
# Análise por dataset
# ---------------------------------------------------------------------------

def _group_by(agg: dict, key_fn) -> dict[str, list[float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for cfg, v in agg.items():
        if v["mean"] is None:
            continue
        k = key_fn(_parse(cfg))
        if k is not None:
            groups[k].append(v["mean"])
    return groups


def _group_stats(groups: dict[str, list[float]]) -> dict[str, dict]:
    return {
        k: {"mean": np.mean(vs), "std": np.std(vs), "n": len(vs)}
        for k, vs in groups.items()
    }


def _aggregate_raw(raw: list) -> dict:
    from collections import defaultdict
    buckets: dict = defaultdict(list)
    errors:  dict = defaultdict(list)
    for r in raw:
        cfg = r["config"]
        if r.get("error"):
            errors[cfg].append(r["error"])
        elif r.get("test_acc") is not None:
            buckets[cfg].append(r["test_acc"])
    agg = {}
    for cfg, accs in buckets.items():
        arr = np.array(accs) * 100
        agg[cfg] = {"mean": float(arr.mean()), "std": float(arr.std()),
                    "n": len(arr), "errors": errors.get(cfg, [])}
    for cfg, errs in errors.items():
        if cfg not in agg:
            agg[cfg] = {"mean": None, "std": None, "n": 0, "errors": errs}
    return agg


def analyze_dataset(dataset: str, data: dict, top_n: int) -> dict:
    agg = data.get("aggregated") or _aggregate_raw(data.get("raw", []))
    valid = {cfg: v for cfg, v in agg.items() if v["mean"] is not None}
    errors = {cfg: v for cfg, v in agg.items() if v["mean"] is None}

    ranked = sorted(valid.items(), key=lambda x: -x[1]["mean"])

    # Grupos por hiperparâmetro
    by_d       = _group_stats(_group_by(valid, lambda p: str(p.get("d"))))
    by_h       = _group_stats(_group_by(valid, lambda p: str(p.get("h"))))
    by_bb      = _group_stats(_group_by(valid, lambda p: p.get("bb")))
    by_mF      = _group_stats(_group_by(valid, lambda p: p.get("map_F")))
    by_mC      = _group_stats(_group_by(valid, lambda p: p.get("map_C")))
    by_dropout = _group_stats(_group_by(valid, lambda p: str(p.get("dropout"))))
    by_mFmC    = _group_stats(_group_by(valid,
                    lambda p: f"F{p.get('map_F','?')}_C{p.get('map_C','?')}"))

    return {
        "dataset":  dataset,
        "n_configs": len(valid),
        "n_errors":  len(errors),
        "ranked":    ranked,
        "by_d":      by_d,
        "by_h":      by_h,
        "by_bb":     by_bb,
        "by_mF":     by_mF,
        "by_mC":     by_mC,
        "by_dropout": by_dropout,
        "by_mFmC":   by_mFmC,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _fmt_group(stats: dict, label: str) -> str:
    entries = sorted(stats.items(), key=lambda x: -x[1]["mean"])
    parts = [f"{k}: {v['mean']:.1f}±{v['std']:.1f}" for k, v in entries]
    return f"  {label:<14} " + "   ".join(parts)


def print_dataset_report(r: dict, top_n: int) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  {r['dataset']}  —  {r['n_configs']} configs  "
          f"({r['n_errors']} erros)")
    print(sep)

    # Top configs
    print(f"\n  Top {min(top_n, len(r['ranked']))} configs:")
    print(f"  {'#':<3}  {'Config':<32}  {'Mean%':>8}  {'Std%':>6}  {'n':>4}")
    print(f"  {'-'*58}")
    for i, (cfg, v) in enumerate(r["ranked"][:top_n], 1):
        print(f"  {i:<3}  {cfg:<32}  {v['mean']:>7.2f}  {v['std']:>5.2f}  {v['n']:>4}")

    # Análise marginal
    print(f"\n  Efeito marginal (média sobre todas as outras dimensões):")
    if r["by_bb"]:
        print(_fmt_group(r["by_bb"],      "backbone"))
    print(_fmt_group(r["by_d"],           "d"))
    print(_fmt_group(r["by_h"],           "hidden"))
    print(_fmt_group(r["by_mFmC"],        "map F×C"))
    print(_fmt_group(r["by_mF"],          "map_F"))
    print(_fmt_group(r["by_mC"],          "map_C"))
    print(_fmt_group(r["by_dropout"],     "dropout"))


def print_cross_dataset_summary(reports: list[dict], top_n: int) -> None:
    sep = "=" * 70
    print(f"\n\n{sep}")
    print(f"  RESUMO CROSS-DATASET")
    print(sep)

    # Melhor config por dataset
    print(f"\n  Melhor config por dataset:")
    print(f"  {'Dataset':<12}  {'Config':<32}  {'Mean%':>8}  {'Std%':>6}  {'n':>4}")
    print(f"  {'-'*64}")
    for r in reports:
        if not r["ranked"]:
            print(f"  {r['dataset']:<12}  {'(sem resultados)':<32}")
            continue
        cfg, v = r["ranked"][0]
        print(f"  {r['dataset']:<12}  {cfg:<32}  {v['mean']:>7.2f}  {v['std']:>5.2f}  {v['n']:>4}")

    # Backbone: melhor dataset a dataset
    if any(r["by_bb"] for r in reports):
        print(f"\n  backbone  (média por dataset):")
        for r in reports:
            if not r["by_bb"]:
                continue
            line = "   ".join(
                f"{k}: {v['mean']:.1f}"
                for k, v in sorted(r["by_bb"].items(), key=lambda x: -x[1]["mean"])
            )
            print(f"    {r['dataset']:<12}  {line}")

    # map_type: F e C combinados
    print(f"\n  map F×C  (média por dataset):")
    for r in reports:
        if not r["by_mFmC"]:
            continue
        entries = sorted(r["by_mFmC"].items(), key=lambda x: -x[1]["mean"])
        line = "   ".join(f"{k}: {v['mean']:.1f}" for k, v in entries)
        print(f"    {r['dataset']:<12}  {line}")

    # d: melhor por dataset
    print(f"\n  d  (média por dataset):")
    for r in reports:
        if not r["by_d"]:
            continue
        entries = sorted(r["by_d"].items(), key=lambda x: -x[1]["mean"])
        line = "   ".join(f"d={k}: {v['mean']:.1f}" for k, v in entries)
        print(f"    {r['dataset']:<12}  {line}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(datasets: list[str] = None, top_n: int = 10, results_file: str = None) -> None:
    if results_file:
        # Arquivo único: infere o dataset pelo campo "dataset" dentro do JSON
        path = results_file if os.path.isabs(results_file) else os.path.join(_HERE, results_file)
        if not os.path.exists(path):
            print(f"  Arquivo não encontrado: {path}")
            return
        with open(path) as f:
            data = json.load(f)
        ds = data.get("dataset", os.path.basename(path))
        r  = analyze_dataset(ds, data, top_n)
        print_dataset_report(r, top_n)
        return

    datasets = datasets or _DATASETS
    reports  = []
    for ds in datasets:
        data = _load(ds)
        if data is None:
            print(f"\n  [{ds}] grid_results.json não encontrado — pulando.")
            continue
        r = analyze_dataset(ds, data, top_n)
        reports.append(r)
        print_dataset_report(r, top_n)

    if len(reports) > 1:
        print_cross_dataset_summary(reports, top_n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Análise dos resultados do grid search — BiSheafDiffusion"
    )
    parser.add_argument("--datasets",     nargs="+", default=None,
                        help="Datasets a analisar. Padrão: todos disponíveis.")
    parser.add_argument("--top",          type=int,  default=10,
                        help="Quantas configs mostrar no ranking (default 10).")
    parser.add_argument("--results-file", type=str,  default=None,
                        help="Caminho para um JSON específico (grid_results.json, "
                             "sample_results.json, grid_partial.json, etc.).")
    args = parser.parse_args()
    main(datasets=args.datasets, top_n=args.top, results_file=args.results_file)