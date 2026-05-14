"""
Random search sobre o grid Wisconsin — amostra N configs aleatórias.

Mantém toda a infraestrutura do sweep.py (workers, pool, resume, telemetria)
e substitui o produto cartesiano por amostragem aleatória sem reposição.

Uso:
    python -m src.experiments.bisheaf_diffusion.Node_Classification.Wisconsin.sweep_sample
    python -m src.experiments.bisheaf_diffusion.Node_Classification.Wisconsin.sweep_sample --n-samples 500
    python -m src.experiments.bisheaf_diffusion.Node_Classification.Wisconsin.sweep_sample --n-samples 200 --seed 42
    python -m src.experiments.bisheaf_diffusion.Node_Classification.Wisconsin.sweep_sample --resume results/grid/sample_partial.json
    python -m src.experiments.bisheaf_diffusion.Node_Classification.Wisconsin.sweep_sample --dry-run
"""

import argparse
import itertools
import json
import os
import random
import threading
import time
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from src.experiments.bisheaf_diffusion.Node_Classification.Wisconsin.config import BASE_CONFIG, GRID
from src.experiments.bisheaf_diffusion.Node_Classification.Wisconsin.sweep import (
    _TRAIN_KEYS,
    Telemetry,
    _decide_workers,
    _make_config_name,
    _pool_error_result,
    _print_config_summary,
    _print_efficiency_report,
    _print_job_line,
    _print_ranking,
    _run_job,
    _run_pool,
    _LOCK,
)

_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "grid")


# ---------------------------------------------------------------------------
# Geração da amostra aleatória
# ---------------------------------------------------------------------------

def _full_grid_list() -> list[dict]:
    """Retorna todos os parâmetros do produto cartesiano como lista de dicts."""
    model_grid = GRID.get("model", {})
    train_grid = GRID.get("train", {})
    all_grid   = {}
    for k, v in {**model_grid, **train_grid}.items():
        all_grid[k] = v if isinstance(v, list) else [v]
    keys   = list(all_grid.keys())
    values = [all_grid[k] for k in keys]
    result = []
    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        if "backbone" in params:
            backbone = params.pop("backbone")
            params["backbone_F"] = backbone
            params["backbone_C"] = backbone
        result.append(params)
    return result


def _generate_sample(n_samples: int, seed: int) -> list[tuple[str, dict]]:
    """
    Amostra n_samples configs aleatórias sem reposição do produto cartesiano.
    Se n_samples >= total de configs, retorna o grid completo embaralhado.
    """
    rng  = random.Random(seed)
    pool = _full_grid_list()
    if n_samples >= len(pool):
        rng.shuffle(pool)
        sample = pool
    else:
        sample = rng.sample(pool, n_samples)
    return [(_make_config_name(p), p) for p in sample]


# ---------------------------------------------------------------------------
# Construção dos jobs
# ---------------------------------------------------------------------------

def _build_jobs_sample(
    n_samples: int,
    n_splits: int,
    seed: int,
    show_progress: bool,
    skip_done: set = None,
) -> list[dict]:
    skip_done = skip_done or set()
    jobs = []
    for cfg_name, params in _generate_sample(n_samples, seed):
        model_overrides = {k: v for k, v in params.items() if k not in _TRAIN_KEYS}
        train_overrides = {k: v for k, v in params.items() if k in _TRAIN_KEYS}
        merged_model    = {**BASE_CONFIG["model"], **model_overrides}
        for split_idx in range(n_splits):
            if (cfg_name, split_idx) in skip_done:
                continue
            cfg_override = {
                "dataset":           "Wisconsin",
                "split_idx":         split_idx,
                "train_on_trainval": False,
                "epochs":            BASE_CONFIG["epochs"],
                "patience":          BASE_CONFIG["patience"],
                "lr":                train_overrides.get("lr", BASE_CONFIG["lr"]),
                "weight_decay":      train_overrides.get("weight_decay", BASE_CONFIG["weight_decay"]),
                "seed":              BASE_CONFIG["seed"],
                "model":             merged_model,
            }
            jobs.append({
                "config":        cfg_name,
                "split_idx":     split_idx,
                "cfg_override":  cfg_override,
                "show_progress": show_progress,
            })
    return jobs


# ---------------------------------------------------------------------------
# Persistência
# ---------------------------------------------------------------------------

def _save_partial(raw_results: list[dict]) -> None:
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    path = os.path.join(_RESULTS_DIR, "sample_partial.json")
    tmp  = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"dataset": "Wisconsin", "raw": raw_results}, f, indent=2)
    os.replace(tmp, path)


def _save_final(raw_results: list[dict], aggregated: dict, telemetry: Telemetry) -> None:
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    with open(os.path.join(_RESULTS_DIR, "sample_results.json"), "w") as f:
        json.dump({"dataset": "Wisconsin", "raw": raw_results, "aggregated": aggregated}, f, indent=2)
    with open(os.path.join(_RESULTS_DIR, "sample_efficiency.json"), "w") as f:
        json.dump(telemetry.summary(), f, indent=2)


def _load_done_from_file(path: str) -> set:
    """Retorna conjunto de (config, split_idx) já concluídos num arquivo de resultados."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return set()
    done = set()
    for r in data.get("raw", []):
        if r.get("error") is None and r.get("test_acc") is not None:
            done.add((r["config"], r["split_idx"]))
    return done


def _load_done(partial_path: str) -> tuple[set, list]:
    try:
        with open(partial_path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return set(), []
    done = set()
    for r in data.get("raw", []):
        if r.get("error") is None and r.get("test_acc") is not None:
            done.add((r["config"], r["split_idx"]))
    return done, data.get("raw", [])


def _load_all_done() -> set:
    """
    Agrega todos os resultados já existentes no diretório de resultados:
    grid_results.json, grid_partial.json, sample_results.json, sample_partial.json.
    Usado para não re-rodar configs já concluídas em qualquer run anterior.
    """
    done = set()
    for name in ("grid_results.json", "grid_partial.json",
                 "sample_results.json", "sample_partial.json"):
        path = os.path.join(_RESULTS_DIR, name)
        done |= _load_done_from_file(path)
    return done


# ---------------------------------------------------------------------------
# Agregação
# ---------------------------------------------------------------------------

def _aggregate(raw_results: list[dict]) -> dict:
    buckets = defaultdict(list)
    errors  = defaultdict(list)
    for r in raw_results:
        cfg = r["config"]
        if r["error"]:
            errors[cfg].append(r["error"])
        elif r["test_acc"] is not None:
            buckets[cfg].append(r["test_acc"])
    agg = {}
    for cfg, accs in buckets.items():
        arr = np.array(accs) * 100
        agg[cfg] = {"mean": float(arr.mean()), "std": float(arr.std()), "n": len(arr),
                    "errors": errors.get(cfg, [])}
    for cfg, errs in errors.items():
        if cfg not in agg:
            agg[cfg] = {"mean": None, "std": None, "n": 0, "errors": errs}
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    n_samples:         int   = 300,
    n_splits:          int   = 10,
    seed:              int   = 0,
    workers:           int   = None,
    max_gpu_mb:        float = 5120,
    mem_per_job_mb:    float = 512,
    ram_per_worker_mb: float = 1500,
    resume_path:       str   = None,
    dry_run:           bool  = False,
    save_every:        int   = 100,
) -> dict:
    skip_done: set      = _load_all_done()
    prev_raw:  list     = []
    if resume_path:
        resume_done, prev_raw = _load_done(resume_path)
        skip_done |= resume_done
        print(f"  [resume] {len(resume_done)} jobs no arquivo de resume.")
    if skip_done:
        print(f"  [skip] {len(skip_done)} jobs já concluídos (em qualquer run anterior) — pulados.")

    show_progress = (workers == 1)
    jobs = _build_jobs_sample(n_samples, n_splits, seed, show_progress, skip_done)

    total_grid   = len(_full_grid_list())
    sample_size  = min(n_samples, total_grid)

    if dry_run:
        sample = _generate_sample(n_samples, seed)
        print(f"\n{'='*60}")
        print(f"  DRY RUN — Wisconsin Random Search")
        print(f"{'='*60}\n")
        print(f"  Grid total      : {total_grid:,} configs")
        print(f"  Amostra         : {sample_size} configs  (seed={seed})")
        print(f"  Splits          : {n_splits}")
        print(f"  Jobs totais     : {len(jobs)}\n")
        print(f"  {'Config':<28}  Overrides")
        print(f"  {'-'*56}")
        for cfg_name, params in sample:
            print(f"  {cfg_name:<28}  {params}")
        print()
        return {}

    if not jobs:
        print("  Nenhum job pendente.")
        return {}

    n_workers, info = _decide_workers(
        n_jobs=len(jobs),
        user_workers=workers,
        max_gpu_mb=max_gpu_mb,
        mem_per_job_mb=mem_per_job_mb,
        ram_per_worker_mb=ram_per_worker_mb,
    )

    print(f"\n{'='*60}")
    print(f"  BiSheafDiffusion — Wisconsin Random Search")
    print(f"  Grid total      : {total_grid:,} configs")
    print(f"  Amostra         : {sample_size} configs  (seed={seed})")
    print(f"  Splits          : {n_splits}")
    print(f"  Jobs pendentes  : {len(jobs)}")
    print(f"  Workers         : {n_workers}  ({info['source']})")
    if info['source'] == "auto":
        print(f"    GPU livre     : {info['gpu_free']:.0f} MB  -> n_gpu={info['n_gpu']}")
        print(f"    RAM livre     : {info['ram_free']:.0f} MB  -> n_ram={info['n_ram']}")
    print(f"{'='*60}\n")

    expected: dict = defaultdict(int)
    for j in jobs:
        expected[j["config"]] += 1

    partial:     dict       = defaultdict(list)
    printed_cfg: set        = set()
    raw_results: list[dict] = list(prev_raw)
    pbar = tqdm(total=len(jobs), desc="Jobs", unit="job", dynamic_ncols=True)

    def on_result(result, done, total):
        _print_job_line(result, done, total)
        cfg = result["config"]
        partial[cfg].append(result)
        raw_results.append(result)
        if cfg not in printed_cfg and len(partial[cfg]) == expected[cfg]:
            _print_config_summary(cfg, partial[cfg])
            printed_cfg.add(cfg)
        if done % save_every == 0 or done == total:
            _save_partial(raw_results)
        pbar.update(1)

    telemetry = Telemetry(interval_s=2.0)
    telemetry.start()
    t0 = time.time()
    try:
        _run_pool(jobs, n_workers, on_result, telemetry)
    finally:
        telemetry.stop()
    pbar.close()
    total_time = time.time() - t0

    aggregated = _aggregate(raw_results)
    n_errors   = sum(1 for r in raw_results if r.get("error"))

    _print_ranking(aggregated, top_n=10)
    _print_efficiency_report(telemetry, total_time, len(raw_results), n_errors, n_workers)
    _save_final(raw_results, aggregated, telemetry)

    print(f"  Resultados salvos em: {_RESULTS_DIR}")
    return aggregated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wisconsin Random Search — BiSheafNodeClassifier")
    parser.add_argument("--n-samples",       type=int,   default=300,
                        help="Número de configs amostradas (default 300).")
    parser.add_argument("--n-splits",        type=int,   default=10,
                        help="Splits por config (default 10).")
    parser.add_argument("--seed",            type=int,   default=0,
                        help="Semente para reprodutibilidade da amostra (default 0).")
    parser.add_argument("--workers",         type=int,   default=None,
                        help="Workers paralelos (auto se omitido).")
    parser.add_argument("--max-gpu-mb",      type=float, default=5120)
    parser.add_argument("--mem-per-job",     type=float, default=512)
    parser.add_argument("--ram-per-worker",  type=float, default=1500)
    parser.add_argument("--resume",          type=str,   default=None,
                        help="Caminho para sample_partial.json para retomar run.")
    parser.add_argument("--dry-run",         action="store_true",
                        help="Mostra configs amostradas sem executar.")
    parser.add_argument("--save-every",      type=int,   default=100,
                        help="Salva sample_partial.json a cada N jobs (default 100).")
    args = parser.parse_args()

    main(
        n_samples        = args.n_samples,
        n_splits         = args.n_splits,
        seed             = args.seed,
        workers          = args.workers,
        max_gpu_mb       = args.max_gpu_mb,
        mem_per_job_mb   = args.mem_per_job,
        ram_per_worker_mb= args.ram_per_worker,
        resume_path      = args.resume,
        dry_run          = args.dry_run,
        save_every       = args.save_every,
    )
