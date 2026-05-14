"""
Mini-sweep paralelizado do BiSheafNodeClassifier em classificação de nós.

Roda múltiplas configurações em paralelo, reporta val/test por split,
exibe k-fold CV ao final de cada (dataset, config) e salva resultados.

Uso:
    python -m src.experiments.bisheaf_diffusion.sweep_node
    python -m src.experiments.bisheaf_diffusion.sweep_node --datasets Texas Wisconsin
    python -m src.experiments.bisheaf_diffusion.sweep_node --configs general orthogonal
    python -m src.experiments.bisheaf_diffusion.sweep_node --workers 4
    python -m src.experiments.bisheaf_diffusion.sweep_node --dry-run
    python -m src.experiments.bisheaf_diffusion.sweep_node --save-dir runs/sweep
"""

import argparse
import copy
import json
import multiprocessing as mp
import os
import subprocess
import threading
import time
import warnings
from collections import defaultdict
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    wait,
)
from concurrent.futures.process import BrokenProcessPool

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

# Timeout (segundos) — se nenhum job completar nesse intervalo, consideramos
# que o pool travou (Windows swap, deadlock CUDA, worker morto sem exception).
_JOB_HANG_TIMEOUT_S = 900


# ---------------------------------------------------------------------------
# Datasets e configs do sweep
# ---------------------------------------------------------------------------

DATASETS = {
    "Texas":     {"n_splits": 10},
    "Wisconsin": {"n_splits": 10},
    "Cornell":   {"n_splits": 10},
    "Film":      {"n_splits": 10},
    "Squirrel":  {"n_splits": 10},
    "Chameleon": {"n_splits": 10},
    "Cora":      {"n_splits": 1},
    "Citeseer":  {"n_splits": 1},
    "PubMed":    {"n_splits": 1},
}

SWEEP_CONFIGS = {
    "general": {
        "map_type_F": "general",
        "map_type_C": "general",
        "backbone_F": "sage",
        "backbone_C": "sage",
    },
    "orthogonal": {
        "map_type_F": "orthogonal",
        "map_type_C": "orthogonal",
        "orth_trans": "householder",
        "backbone_F": "sage",
        "backbone_C": "sage",
    },
}


# ---------------------------------------------------------------------------
# Telemetria — coletor em thread daemon, NUNCA bloqueia o scheduler
# ---------------------------------------------------------------------------

def _gpu_query() -> tuple[float, float]:
    """(free_mb, util_pct) via nvidia-smi."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            parts = r.stdout.strip().split("\n")[0].split(",")
            return float(parts[0].strip()), float(parts[1].strip())
    except Exception:
        pass
    return float("inf"), 0.0


def _ram_query() -> tuple[float, float]:
    """(free_mb, used_pct) via psutil."""
    try:
        import psutil
        m = psutil.virtual_memory()
        return m.available / 1024 ** 2, m.percent
    except Exception:
        return float("inf"), 0.0


class Telemetry:
    """Amostra GPU/RAM em thread daemon. O scheduler nunca chama nvidia-smi."""

    def __init__(self, interval_s: float = 2.0):
        self.interval_s = interval_s
        self.samples: list[dict] = []
        self.peak_mbs: list[float] = []
        self.t_start = time.time()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.is_set():
            gpu_free, gpu_util = _gpu_query()
            ram_free, ram_pct  = _ram_query()
            self.samples.append({
                "t":         time.time() - self.t_start,
                "gpu_free":  gpu_free,
                "gpu_util":  gpu_util,
                "ram_free":  ram_free,
                "ram_used":  ram_pct,
            })
            if self._stop.wait(self.interval_s):
                break

    def record_peak(self, peak_mb: float):
        self.peak_mbs.append(peak_mb or 0.0)

    def summary(self) -> dict:
        s = self.samples
        n = len(s)
        ram_ok = n > 0 and s[0]["ram_free"] < float("inf")
        gpu_ok = n > 0

        def _arr(key):
            return np.array([x[key] for x in s])

        bottleneck = "—"
        if n > 0:
            high_ram = ram_ok and float(_arr("ram_used").mean()) > 85
            idle_gpu = float(_arr("gpu_free").mean()) > 1000
            bottleneck = "RAM" if (high_ram and idle_gpu) else "GPU"

        return {
            "samples":     n,
            "duration_s":  round(s[-1]["t"], 1) if n else 0.0,
            "bottleneck":  bottleneck,
            "gpu": {
                "mean_free_mb":  round(float(_arr("gpu_free").mean()), 1) if gpu_ok else None,
                "mean_util_pct": round(float(_arr("gpu_util").mean()), 1) if gpu_ok else None,
                "max_peak_mb":   round(float(max(self.peak_mbs)), 1) if self.peak_mbs else 0.0,
            },
            "ram": {
                "mean_free_mb":  round(float(_arr("ram_free").mean()), 1) if ram_ok else None,
                "min_free_mb":   round(float(_arr("ram_free").min()), 1) if ram_ok else None,
                "mean_used_pct": round(float(_arr("ram_used").mean()), 1) if ram_ok else None,
                "peak_used_pct": round(float(_arr("ram_used").max()), 1) if ram_ok else None,
            },
        }


# ---------------------------------------------------------------------------
# Worker — executa em processo separado
# ---------------------------------------------------------------------------

# Cache de dataset por processo. Cada worker carrega cada dataset 1x e reusa
# em todos os jobs subsequentes do mesmo dataset (workers sao reusados pelo Pool).
_WORKER_CACHE: dict = {}


def _run_job(job: dict) -> dict:
    """Roda um (dataset, split, config). Cache de dataset por worker."""
    import gc
    import warnings
    warnings.filterwarnings("ignore")

    import torch
    from src.experiments.bisheaf_diffusion.train_node import main, load_data_shared

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    dataset_name = job["cfg_override"]["dataset"]
    if dataset_name not in _WORKER_CACHE:
        _WORKER_CACHE[dataset_name] = load_data_shared(dataset_name)
    preloaded, num_classes = _WORKER_CACHE[dataset_name]

    try:
        result = main(
            cfg_override=job["cfg_override"],
            quiet=True,
            preloaded_data=preloaded,
            num_classes_hint=num_classes,
        )
        peak_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                   if torch.cuda.is_available() else 0.0)
        return {
            "dataset":   job["dataset"],
            "config":    job["config"],
            "split_idx": job["split_idx"],
            "test_acc":  result["test_at_best_val"],
            "val_acc":   result["best_val_acc"],
            "time_s":    result["total_time_s"],
            "peak_mb":   peak_mb,
            "error":     None,
        }
    except Exception:
        import traceback
        return {
            "dataset":   job["dataset"],
            "config":    job["config"],
            "split_idx": job["split_idx"],
            "test_acc":  None,
            "val_acc":   None,
            "time_s":    None,
            "peak_mb":   None,
            "error":     traceback.format_exc(),
        }
    finally:
        # Higiene de memoria entre jobs no mesmo worker.
        # Sem isso, o cache CUDA do PyTorch cresce a cada job e a RAM acumula
        # objetos do modelo anterior ate o GC do Python disparar — sob carga,
        # isso vira swap no Windows e os workers efetivamente travam.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.synchronize()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Construção dos jobs
# ---------------------------------------------------------------------------

def _build_jobs(datasets: list, configs: list, n_splits_override: int,
                base_cfg: dict) -> list[dict]:
    jobs = []
    for dataset in datasets:
        n_splits = min(n_splits_override, DATASETS[dataset]["n_splits"])
        for cfg_name in configs:
            for split_idx in range(n_splits):
                model_overrides = SWEEP_CONFIGS[cfg_name]
                cfg_override = {
                    "dataset":           dataset,
                    "split_idx":         split_idx,
                    "train_on_trainval": False,
                    "model": {**base_cfg.get("model", {}), **model_overrides},
                }
                for k in ("epochs", "lr", "weight_decay", "patience", "seed"):
                    if k in base_cfg:
                        cfg_override[k] = base_cfg[k]
                jobs.append({
                    "dataset":     dataset,
                    "config":      cfg_name,
                    "split_idx":   split_idx,
                    "cfg_override": cfg_override,
                })
    return jobs


# ---------------------------------------------------------------------------
# Decisão de n_workers — UMA VEZ no início, fixo durante todo o run
# ---------------------------------------------------------------------------

def _decide_workers(
    n_jobs: int,
    user_workers,
    max_gpu_mb: float,
    mem_per_job_mb: float,
    ram_per_worker_mb: float,
    hard_cap: int = 16,
) -> tuple[int, dict]:
    if user_workers is not None:
        n = max(1, min(user_workers, hard_cap, n_jobs))
        return n, {"source": "user", "n": n}

    gpu_free, _ = _gpu_query()
    ram_free, _ = _ram_query()

    n_gpu = max(1, int(min(gpu_free, max_gpu_mb) / mem_per_job_mb))
    n_ram = (max(1, int(ram_free / ram_per_worker_mb))
             if ram_free < float("inf") else hard_cap)
    n     = max(1, min(n_gpu, n_ram, hard_cap, n_jobs))

    return n, {
        "source":   "auto",
        "n":        n,
        "n_gpu":    n_gpu,
        "n_ram":    n_ram,
        "gpu_free": gpu_free,
        "ram_free": ram_free,
    }


# ---------------------------------------------------------------------------
# Pool runner — submete tudo, processa por as_completed
# ---------------------------------------------------------------------------

def _pool_error_result(job: dict, msg: str) -> dict:
    """Resultado de erro padronizado quando o pool falha (crash, timeout)."""
    return {
        "dataset":   job["dataset"],
        "config":    job["config"],
        "split_idx": job["split_idx"],
        "test_acc":  None,
        "val_acc":   None,
        "time_s":    None,
        "peak_mb":   None,
        "error":     msg,
    }


def _run_pool(jobs: list[dict], n_workers: int, on_result, telemetry: Telemetry) -> list[dict]:
    """
    Executa todos os jobs com submissao em fluxo:
      - mantem no maximo `n_workers * 2` futures em voo
      - detecta hang via `wait(timeout=...)`: se nada completar dentro do
        intervalo, marca os jobs em voo como erro e segue
      - captura BrokenProcessPool: se um worker morre, recria o pool e
        continua com os jobs restantes (em vez de perder o sweep inteiro)
    """
    ctx          = mp.get_context("spawn")
    job_iter     = iter(jobs)
    total        = len(jobs)
    results: list[dict] = []
    in_flight: dict     = {}  # future -> job

    def _submit_next(executor: ProcessPoolExecutor) -> bool:
        try:
            j = next(job_iter)
        except StopIteration:
            return False
        in_flight[executor.submit(_run_job, j)] = j
        return True

    def _drain_as_errors(msg: str) -> None:
        """Falha em massa: reporta todos os in_flight como erro e limpa."""
        for fut, job in list(in_flight.items()):
            fut.cancel()
            res = _pool_error_result(job, msg)
            results.append(res)
            telemetry.record_peak(0.0)
            on_result(res, len(results), total)
        in_flight.clear()

    executor = ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx)
    try:
        # Prime o pool com 2x workers em voo.
        for _ in range(n_workers * 2):
            if not _submit_next(executor):
                break

        while in_flight:
            done, _ = wait(
                list(in_flight.keys()),
                timeout=_JOB_HANG_TIMEOUT_S,
                return_when=FIRST_COMPLETED,
            )

            if not done:
                # Nenhum job completou no intervalo de timeout. Considerar hang.
                msg = (f"Pool hang: nenhum job completou em "
                       f"{_JOB_HANG_TIMEOUT_S}s. Workers travados ou em swap.")
                with _LOCK:
                    print(f"\n  AVISO: {msg}")
                    print(f"  Cancelando {len(in_flight)} jobs em voo e recriando o pool.\n")
                _drain_as_errors(msg)
                executor.shutdown(wait=False, cancel_futures=True)
                executor = ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx)
                # Repopular com jobs restantes.
                for _ in range(n_workers * 2):
                    if not _submit_next(executor):
                        break
                continue

            for fut in done:
                job = in_flight.pop(fut)
                try:
                    result = fut.result()
                except BrokenProcessPool as e:
                    # Pool morreu (worker crashed). Reportar este job e
                    # qualquer outro in_flight como erro, recriar o pool.
                    msg = f"BrokenProcessPool: {e}"
                    with _LOCK:
                        print(f"\n  AVISO: worker morreu ({msg}). "
                              f"Recriando o pool com os jobs restantes.\n")
                    res = _pool_error_result(job, msg)
                    results.append(res)
                    telemetry.record_peak(0.0)
                    on_result(res, len(results), total)
                    _drain_as_errors(msg)
                    executor.shutdown(wait=False, cancel_futures=True)
                    executor = ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx)
                    for _ in range(n_workers * 2):
                        if not _submit_next(executor):
                            break
                    break  # sai do for done — in_flight foi limpo
                except Exception as e:
                    # Erro inesperado ao recuperar o resultado.
                    res = _pool_error_result(job, f"fut.result() falhou: {e!r}")
                    results.append(res)
                    telemetry.record_peak(0.0)
                    on_result(res, len(results), total)
                else:
                    results.append(result)
                    telemetry.record_peak(result.get("peak_mb") or 0.0)
                    on_result(result, len(results), total)

                _submit_next(executor)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    return results


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

_LOCK = threading.Lock()


def _print_job_result(result: dict, idx: int, total: int) -> None:
    if result["error"]:
        last = [l.strip() for l in result["error"].splitlines() if l.strip()]
        brief = last[-1][:70] if last else "erro"
        status = f"ERRO: {brief}"
    else:
        status = (f"val={result['val_acc']*100:5.1f}%  "
                  f"test={result['test_acc']*100:5.1f}%  "
                  f"peak={result['peak_mb']:.0f}MB")
    with _LOCK:
        print(f"  [{idx:3d}/{total}]  {result['dataset']:<12} {result['config']:<14}"
              f"  split={result['split_idx']}  {status}")


def _print_config_summary(dataset: str, config: str, results: list[dict]) -> None:
    valid = [r for r in results if r["test_acc"] is not None]
    if not valid:
        return
    val_arr  = np.array([r["val_acc"]  for r in valid]) * 100
    test_arr = np.array([r["test_acc"] for r in valid]) * 100
    sep = "-" * 56
    with _LOCK:
        print(f"\n  {sep}")
        print(f"  {dataset} / {config}  ({len(valid)} splits)")
        print(f"  Val : {val_arr.mean():.1f} +- {val_arr.std():.1f} %")
        print(f"  Test: {test_arr.mean():.1f} +- {test_arr.std():.1f} %")
        print(f"  {sep}\n")


def _print_final_table(aggregated: dict, datasets: list, configs: list) -> None:
    col_w  = 16
    name_w = 12
    header = f"{'Dataset':<{name_w}}" + "".join(f"{c:^{col_w}}" for c in configs)
    sep    = "=" * len(header)
    print(f"\n{sep}")
    print("  RESULTADO FINAL — Test acc (mean +- std %)")
    print(f"{sep}")
    print(header)
    print("-" * len(header))
    for dataset in datasets:
        row = f"{dataset:<{name_w}}"
        for cfg in configs:
            entry = aggregated.get((dataset, cfg))
            if entry is None:
                row += f"{'—':^{col_w}}"
            elif entry["mean"] is None:
                row += f"{'ERR':^{col_w}}"
            else:
                cell = f"{entry['mean']:.1f}+-{entry['std']:.1f}"
                row += f"{cell:^{col_w}}"
        print(row)
    print(sep)


def _print_efficiency_report(telemetry: Telemetry, total_time_s: float,
                             n_jobs: int, n_workers: int) -> None:
    s   = telemetry.summary()
    gpu = s["gpu"]
    ram = s["ram"]

    def _fmt(v, fmt=".0f"):
        return f"{v:{fmt}}" if v is not None else "n/a"

    sep = "=" * 50
    print(f"\n{sep}")
    print(f"  Relatorio de Eficiencia")
    print(f"{sep}")
    print(f"  Jobs concluidos   : {n_jobs}")
    print(f"  Tempo total       : {total_time_s:.0f}s ({total_time_s/60:.1f} min)")
    print(f"  Throughput medio  : {n_jobs / (total_time_s / 60):.2f} job/min")
    print(f"  Workers           : {n_workers} (fixo)")
    print(f"  Gargalo detectado : {s['bottleneck']}")
    print(f"  -- GPU {'-'*36}")
    print(f"  VRAM pico maximo  : {_fmt(gpu['max_peak_mb'])} MB")
    print(f"  VRAM livre (media): {_fmt(gpu['mean_free_mb'])} MB")
    print(f"  Utilizacao GPU    : {_fmt(gpu['mean_util_pct'])} %")
    print(f"  -- RAM {'-'*36}")
    print(f"  RAM livre (media) : {_fmt(ram['mean_free_mb'])} MB")
    print(f"  RAM livre (min)   : {_fmt(ram['min_free_mb'])} MB")
    print(f"  RAM usada (pico)  : {_fmt(ram['peak_used_pct'])} %")
    print(f"{sep}")


# ---------------------------------------------------------------------------
# Agregação
# ---------------------------------------------------------------------------

def _aggregate(raw_results: list[dict]) -> dict:
    buckets = defaultdict(list)
    errors  = defaultdict(list)
    for r in raw_results:
        key = (r["dataset"], r["config"])
        if r["error"]:
            errors[key].append(r["error"])
        elif r["test_acc"] is not None:
            buckets[key].append(r["test_acc"])
    aggregated = {}
    for key, accs in buckets.items():
        arr = np.array(accs) * 100
        aggregated[key] = {
            "mean":   float(arr.mean()),
            "std":    float(arr.std()),
            "n":      len(arr),
            "errors": errors.get(key, []),
        }
    for key, errs in errors.items():
        if key not in aggregated:
            aggregated[key] = {"mean": None, "std": None, "n": 0, "errors": errs}
    return aggregated


# ---------------------------------------------------------------------------
# Dry run e pre-download
# ---------------------------------------------------------------------------

def _dry_run(jobs: list[dict], datasets: list, configs: list) -> None:
    print(f"\n{'='*60}")
    print(f"  DRY RUN — nenhum treino sera executado")
    print(f"{'='*60}\n")
    print(f"  {'#':<5} {'Dataset':<12} {'Config':<16} {'Split'}")
    print(f"  {'-'*45}")
    for i, job in enumerate(jobs, 1):
        print(f"  {i:<5} {job['dataset']:<12} {job['config']:<16} {job['split_idx']}")
    print(f"\n  Total de jobs: {len(jobs)}\n")
    print(f"  Configs e hiperparametros de modelo:")
    for name, overrides in SWEEP_CONFIGS.items():
        if name in configs:
            print(f"\n  [{name}]")
            for k, v in overrides.items():
                print(f"    {k} = {v!r}")
    print(f"\n  Datasets e splits:")
    for ds in datasets:
        n = len([j for j in jobs if j["dataset"] == ds and j["config"] == configs[0]])
        print(f"    {ds:<12} -> {n} split(s)")
    print()


def _predownload_datasets(datasets: list) -> None:
    """
    Baixa cada dataset sequencialmente antes de lancar workers paralelos.
    Evita race condition onde multiplos processos tentam escrever o mesmo
    arquivo de cache ao mesmo tempo.
    """
    from src.experiments.bisheaf_diffusion.train_node import load_data
    print("  Verificando datasets...")
    for ds in datasets:
        try:
            load_data(ds, split_idx=0)
            print(f"    {ds:<12} OK")
        except Exception as e:
            print(f"    {ds:<12} AVISO: {e}")
    print()


# ---------------------------------------------------------------------------
# Persistência
# ---------------------------------------------------------------------------

def _build_output(raw_results, aggregated, datasets, configs):
    return {
        "datasets":   datasets,
        "configs":    configs,
        "raw":        raw_results,
        "aggregated": {f"{k[0]}/{k[1]}": v for k, v in aggregated.items()},
    }


def _incremental_save(save_dir, raw_results, datasets, configs):
    agg = _aggregate(raw_results)
    out = _build_output(raw_results, agg, datasets, configs)
    with open(os.path.join(save_dir, "sweep_partial.json"), "w") as f:
        json.dump(out, f, indent=2)


def _save_results(save_dir, raw_results, aggregated, datasets, configs):
    out = _build_output(raw_results, aggregated, datasets, configs)
    with open(os.path.join(save_dir, "sweep_results.json"), "w") as f:
        json.dump(out, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    datasets:          list  = None,
    configs:           list  = None,
    n_splits:          int   = 10,
    workers:           int   = None,
    max_gpu_mb:        float = 5120,
    mem_per_job_mb:    float = 512,
    ram_per_worker_mb: float = 1500,
    save_dir:          str   = None,
    dry_run:           bool  = False,
    base_cfg_override: dict  = None,
) -> dict:
    from src.experiments.bisheaf_diffusion.config_node import CONFIG

    datasets = datasets or list(DATASETS.keys())
    configs  = configs  or list(SWEEP_CONFIGS.keys())

    unknown_d = set(datasets) - set(DATASETS)
    unknown_c = set(configs)  - set(SWEEP_CONFIGS)
    if unknown_d: raise ValueError(f"Datasets desconhecidos: {unknown_d}")
    if unknown_c: raise ValueError(f"Configs desconhecidas:  {unknown_c}")

    base_cfg = copy.deepcopy(CONFIG)
    if base_cfg_override:
        base_cfg.update(base_cfg_override)

    jobs = _build_jobs(datasets, configs, n_splits, base_cfg)

    if dry_run:
        _dry_run(jobs, datasets, configs)
        return {}

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    _predownload_datasets(datasets)

    n_workers, info = _decide_workers(
        n_jobs=len(jobs),
        user_workers=workers,
        max_gpu_mb=max_gpu_mb,
        mem_per_job_mb=mem_per_job_mb,
        ram_per_worker_mb=ram_per_worker_mb,
    )

    print(f"\n{'='*60}")
    print(f"  BiSheafDiffusion — Node Classification Sweep")
    print(f"  Datasets        : {datasets}")
    print(f"  Configs         : {configs}")
    print(f"  Jobs            : {len(jobs)}")
    print(f"  Workers         : {n_workers}  ({info['source']})")
    if info['source'] == "auto":
        print(f"    GPU livre     : {info['gpu_free']:.0f} MB  -> n_gpu={info['n_gpu']}")
        print(f"    RAM livre     : {info['ram_free']:.0f} MB  -> n_ram={info['n_ram']}")
    print(f"  Dataset cache   : por worker (1 carga por processo por dataset)")
    print(f"  Protocolo       : train | val | test@best_val")
    print(f"{'='*60}\n")

    expected: dict = defaultdict(int)
    for j in jobs:
        expected[(j["dataset"], j["config"])] += 1

    raw_results: list[dict] = []
    partial: dict = defaultdict(list)
    printed_summary = set()

    def on_result(result, idx, total):
        _print_job_result(result, idx, total)
        key = (result["dataset"], result["config"])
        partial[key].append(result)
        raw_results.append(result)
        if key not in printed_summary and len(partial[key]) == expected[key]:
            _print_config_summary(result["dataset"], result["config"], partial[key])
            printed_summary.add(key)
            if save_dir:
                _incremental_save(save_dir, raw_results, datasets, configs)

    telemetry = Telemetry(interval_s=2.0)
    telemetry.start()
    t0 = time.time()
    try:
        _run_pool(jobs, n_workers, on_result, telemetry)
    finally:
        telemetry.stop()
    total_time = time.time() - t0

    aggregated = _aggregate(raw_results)
    _print_final_table(aggregated, datasets, configs)
    _print_efficiency_report(telemetry, total_time, len(raw_results), n_workers)

    failed = [r for r in raw_results if r["error"]]
    if failed:
        print(f"\n{'='*60}")
        print(f"  ERROS COMPLETOS ({len(failed)} jobs falharam)")
        print(f"{'='*60}")
        seen = set()
        for r in failed:
            key = (r["dataset"], r["config"])
            if key not in seen:
                seen.add(key)
                print(f"\n  -- {r['dataset']} / {r['config']} (split {r['split_idx']}) --")
                print(r["error"])

    if save_dir:
        _save_results(save_dir, raw_results, aggregated, datasets, configs)
        with open(os.path.join(save_dir, "efficiency.json"), "w") as f:
            json.dump(telemetry.summary(), f, indent=2)
        print(f"  Resultados salvos em: {save_dir}")

    return aggregated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets",       nargs="+", default=None,
                        help=f"Subset de datasets. Opcoes: {list(DATASETS)}")
    parser.add_argument("--configs",        nargs="+", default=None,
                        help=f"Subset de configs. Opcoes: {list(SWEEP_CONFIGS)}")
    parser.add_argument("--n-splits",       type=int,   default=10,
                        help="Splits por dataset multisplit (max 10).")
    parser.add_argument("--workers",        type=int,   default=None,
                        help="Workers paralelos (override). Auto se omitido.")
    parser.add_argument("--max-gpu-mb",     type=float, default=5120,
                        help="VRAM maxima a usar em MB (default 5120).")
    parser.add_argument("--mem-per-job",    type=float, default=512,
                        help="Estimativa de VRAM por job em MB (default 512).")
    parser.add_argument("--ram-per-worker", type=float, default=1500,
                        help="Estimativa de RAM por worker em MB (default 1500).")
    parser.add_argument("--save-dir",       type=str,   default=None)
    parser.add_argument("--dry-run",        action="store_true",
                        help="Mostra jobs e configs sem executar.")
    args = parser.parse_args()

    main(
        datasets=args.datasets,
        configs=args.configs,
        n_splits=args.n_splits,
        workers=args.workers,
        max_gpu_mb=args.max_gpu_mb,
        mem_per_job_mb=args.mem_per_job,
        ram_per_worker_mb=args.ram_per_worker,
        save_dir=args.save_dir,
        dry_run=args.dry_run,
    )
