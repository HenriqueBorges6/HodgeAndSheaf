"""
Lógica compartilhada de grid search para todos os datasets de classificação de nós.

Não use diretamente — importe via Dataset/sweep.py.
"""

import json
import multiprocessing as mp
import os
import subprocess
import threading
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool

import numpy as np
from tqdm import tqdm

_LOCK = threading.Lock()

# Timeout por pool — 15 min sem nenhum job completar é considerado hang.
_JOB_HANG_TIMEOUT_S = 900


# ---------------------------------------------------------------------------
# Telemetria — thread daemon, nunca bloqueia o scheduler
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
    """Amostra GPU/RAM a cada interval_s em thread daemon."""

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
                "t":        time.time() - self.t_start,
                "gpu_free": gpu_free,
                "gpu_util": gpu_util,
                "ram_free": ram_free,
                "ram_used": ram_pct,
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
            "samples":    n,
            "duration_s": round(s[-1]["t"], 1) if n else 0.0,
            "bottleneck": bottleneck,
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

_WORKER_CACHE: dict = {}


def _run_job(job: dict) -> dict:
    """Roda um (dataset, config, split). Cache de dataset por worker."""
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

    quiet = not job.get("show_progress", False)
    try:
        result = main(
            cfg_override=job["cfg_override"],
            quiet=quiet,
            preloaded_data=preloaded,
            num_classes_hint=num_classes,
        )
        peak_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                   if torch.cuda.is_available() else 0.0)
        return {
            "config":    job["config"],
            "split_idx": job["split_idx"],
            "val_acc":   result["best_val_acc"],
            "test_acc":  result["test_at_best_val"],
            "time_s":    result["total_time_s"],
            "peak_mb":   peak_mb,
            "error":     None,
        }
    except Exception:
        import traceback
        return {
            "config":    job["config"],
            "split_idx": job["split_idx"],
            "val_acc":   None,
            "test_acc":  None,
            "time_s":    None,
            "peak_mb":   None,
            "error":     traceback.format_exc(),
        }
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.synchronize()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

def make_config_name(params: dict) -> str:
    """
    Nome legível para um ponto do grid.
    Exemplo: d2_h64_Fgen_Cort_dr5
    """
    d   = params["d"]
    h   = params["hidden_channels"]
    mF  = params["map_type_F"][:3]   # gen | ort | dia | sym
    mC  = params["map_type_C"][:3]
    name = f"d{d}_h{h}_F{mF}_C{mC}"
    if "dropout" in params:
        dr = str(params["dropout"]).replace("0.", "")
        name += f"_dr{dr}"
    if "backbone_F" in params:
        name += f"_{params['backbone_F'][:4]}"
    return name


def generate_grid(grid: dict) -> list[tuple[str, dict]]:
    """
    Gera lista de (config_name, model_overrides) a partir do GRID do dataset.

    - Se 'backbone' estiver no grid, expande em backbone_F e backbone_C.
    - map_type_F e map_type_C devem estar explicitamente no grid (não são
      mais fixados aqui).
    """
    model_grid = grid["model"]
    keys       = list(model_grid.keys())
    values     = [model_grid[k] for k in keys]
    configs    = []
    for combo in __import__("itertools").product(*values):
        params = dict(zip(keys, combo))
        if "backbone" in params:
            backbone = params.pop("backbone")
            params["backbone_F"] = backbone
            params["backbone_C"] = backbone
        configs.append((make_config_name(params), params))
    return configs


def build_jobs(
    dataset: str,
    base_config: dict,
    grid: dict,
    n_splits: int,
    show_progress: bool,
    skip_done: set = None,
) -> list[dict]:
    skip_done = skip_done or set()
    jobs = []
    for cfg_name, model_overrides in generate_grid(grid):
        merged_model = {**base_config["model"], **model_overrides}
        for split_idx in range(n_splits):
            if (cfg_name, split_idx) in skip_done:
                continue
            cfg_override = {
                "dataset":           dataset,
                "split_idx":         split_idx,
                "train_on_trainval": False,
                "epochs":            base_config["epochs"],
                "patience":          base_config["patience"],
                "lr":                base_config["lr"],
                "weight_decay":      base_config["weight_decay"],
                "seed":              base_config["seed"],
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
# Decisão de workers — uma vez no início
# ---------------------------------------------------------------------------

def decide_workers(
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
# Pool runner
# ---------------------------------------------------------------------------

def _pool_error_result(job: dict, msg: str) -> dict:
    return {
        "config":    job["config"],
        "split_idx": job["split_idx"],
        "val_acc":   None,
        "test_acc":  None,
        "time_s":    None,
        "peak_mb":   None,
        "error":     msg,
    }


def run_pool(jobs: list[dict], n_workers: int, on_result, telemetry: Telemetry) -> list[dict]:
    """
    Executa jobs com n_workers fixo.
    - Submissão em fluxo: n_workers*2 em voo, depois 1 por 1
    - Hang detection: se nada completar em _JOB_HANG_TIMEOUT_S, recria pool
    - BrokenProcessPool: recria pool e continua com jobs restantes
    """
    ctx      = mp.get_context("spawn")
    job_iter = iter(jobs)
    total    = len(jobs)
    results: list[dict] = []
    in_flight: dict     = {}

    def _submit_next(executor: ProcessPoolExecutor) -> bool:
        try:
            j = next(job_iter)
        except StopIteration:
            return False
        in_flight[executor.submit(_run_job, j)] = j
        return True

    def _drain_as_errors(msg: str) -> None:
        for fut, job in list(in_flight.items()):
            fut.cancel()
            res = _pool_error_result(job, msg)
            results.append(res)
            telemetry.record_peak(0.0)
            on_result(res, len(results), total)
        in_flight.clear()

    executor = ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx)
    try:
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
                msg = (f"Pool hang: nenhum job completou em "
                       f"{_JOB_HANG_TIMEOUT_S}s. Workers travados ou em swap.")
                with _LOCK:
                    print(f"\n  AVISO: {msg}")
                    print(f"  Cancelando {len(in_flight)} jobs e recriando o pool.\n")
                _drain_as_errors(msg)
                executor.shutdown(wait=False, cancel_futures=True)
                executor = ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx)
                for _ in range(n_workers * 2):
                    if not _submit_next(executor):
                        break
                continue

            for fut in done:
                job = in_flight.pop(fut)
                try:
                    result = fut.result()
                except BrokenProcessPool as e:
                    msg = f"BrokenProcessPool: {e}"
                    with _LOCK:
                        print(f"\n  AVISO: worker morreu ({msg}). Recriando pool.\n")
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
                    break
                except Exception as e:
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

def print_job_line(result: dict, done: int, total: int) -> None:
    cfg   = result["config"]
    split = result["split_idx"]
    if result["error"]:
        last = [l.strip() for l in result["error"].splitlines() if l.strip()]
        brief = last[-1][:60] if last else "erro"
        status = f"ERRO: {brief}"
    else:
        status = (f"val={result['val_acc']*100:5.1f}%  "
                  f"test={result['test_acc']*100:5.1f}%  "
                  f"peak={result['peak_mb']:.0f}MB")
    with _LOCK:
        print(f"  [{done:4d}/{total}]  {cfg:<24}  split={split}  {status}")


def print_config_summary(cfg_name: str, results: list[dict]) -> None:
    valid = [r for r in results if r["test_acc"] is not None]
    if not valid:
        return
    arr = np.array([r["test_acc"] for r in valid]) * 100
    with _LOCK:
        print(f"\n  -- {cfg_name}  ({len(valid)} splits):"
              f"  test = {arr.mean():.1f} +- {arr.std():.1f} %\n")


def print_ranking(dataset: str, aggregated: dict, top_n: int = 10) -> None:
    entries = [(cfg, v) for cfg, v in aggregated.items() if v["mean"] is not None]
    entries.sort(key=lambda x: -x[1]["mean"])
    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  Top {min(top_n, len(entries))} configs — {dataset} / test acc (mean +- std %)")
    print(f"{sep}")
    print(f"  {'#':<4}  {'Config':<28}  {'Test%':>10}  {'Std%':>6}  {'Splits':>6}")
    print(f"  {'-'*60}")
    for i, (cfg, v) in enumerate(entries[:top_n], 1):
        print(f"  {i:<4}  {cfg:<28}  {v['mean']:>9.1f}  {v['std']:>5.1f}  {v['n']:>6}")
    print(f"{sep}\n")


def print_efficiency_report(telemetry: Telemetry, total_time_s: float,
                            n_jobs: int, n_errors: int, n_workers: int) -> None:
    s   = telemetry.summary()
    gpu = s["gpu"]
    ram = s["ram"]
    h, rem = divmod(int(total_time_s), 3600)
    m       = rem // 60

    def _fmt(v, fmt=".0f"):
        return f"{v:{fmt}}" if v is not None else "n/a"

    sep = "=" * 50
    print(f"\n{sep}")
    print(f"  Relatorio de Eficiencia")
    print(f"{sep}")
    print(f"  Jobs concluidos   : {n_jobs}  (erros: {n_errors})")
    print(f"  Tempo total       : {h}h {m:02d}m")
    print(f"  Throughput        : {n_jobs / (total_time_s / 60):.2f} job/min")
    print(f"  Workers           : {n_workers} (fixo)")
    print(f"  Gargalo           : {s['bottleneck']}")
    print(f"  -- GPU {'-'*36}")
    print(f"  VRAM pico max     : {_fmt(gpu['max_peak_mb'])} MB")
    print(f"  VRAM livre (media): {_fmt(gpu['mean_free_mb'])} MB")
    print(f"  Utilizacao GPU    : {_fmt(gpu['mean_util_pct'])} %")
    print(f"  -- RAM {'-'*36}")
    print(f"  RAM livre (media) : {_fmt(ram['mean_free_mb'])} MB")
    print(f"  RAM livre (min)   : {_fmt(ram['min_free_mb'])} MB")
    print(f"  RAM usada (pico)  : {_fmt(ram['peak_used_pct'])} %")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Agregação
# ---------------------------------------------------------------------------

def aggregate(raw_results: list[dict]) -> dict:
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
        agg[cfg] = {
            "mean":   float(arr.mean()),
            "std":    float(arr.std()),
            "n":      len(arr),
            "errors": errors.get(cfg, []),
        }
    for cfg, errs in errors.items():
        if cfg not in agg:
            agg[cfg] = {"mean": None, "std": None, "n": 0, "errors": errs}
    return agg


# ---------------------------------------------------------------------------
# Persistência
# ---------------------------------------------------------------------------

def save_partial(results_dir: str, dataset: str, raw_results: list[dict]) -> None:
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "grid_partial.json"), "w") as f:
        json.dump({"dataset": dataset, "raw": raw_results}, f, indent=2)


def save_final(results_dir: str, dataset: str,
               raw_results: list[dict], aggregated: dict, telemetry: Telemetry) -> None:
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "grid_results.json"), "w") as f:
        json.dump({"dataset": dataset, "raw": raw_results,
                   "aggregated": aggregated}, f, indent=2)
    with open(os.path.join(results_dir, "efficiency.json"), "w") as f:
        json.dump(telemetry.summary(), f, indent=2)


def load_done(partial_path: str) -> tuple[set, list[dict]]:
    """Retorna (skip_done, prev_raw) de um grid_partial.json."""
    with open(partial_path) as f:
        data = json.load(f)
    done = set()
    for r in data.get("raw", []):
        if r.get("error") is None and r.get("test_acc") is not None:
            done.add((r["config"], r["split_idx"]))
    return done, data.get("raw", [])


# ---------------------------------------------------------------------------
# Entry point compartilhado
# ---------------------------------------------------------------------------

def run_sweep(
    dataset: str,
    base_config: dict,
    grid: dict,
    results_dir: str,
    n_splits: int = 10,
    workers: int = None,
    max_gpu_mb: float = 5120,
    mem_per_job_mb: float = 512,
    ram_per_worker_mb: float = 1500,
    resume_path: str = None,
    dry_run: bool = False,
) -> dict:
    """
    Roda o grid search completo para um dataset.
    Chamado pelo sweep.py de cada dataset.
    """
    skip_done: set = set()
    prev_raw: list[dict] = []
    if resume_path:
        skip_done, prev_raw = load_done(resume_path)
        print(f"  [resume] {len(skip_done)} (config, split) ja concluidos — pulados.")

    show_progress = (workers == 1)
    jobs = build_jobs(dataset, base_config, grid, n_splits, show_progress, skip_done)

    grid_pts = generate_grid(grid)

    if dry_run:
        print(f"\n{'='*60}")
        print(f"  DRY RUN — {dataset} Grid Search")
        print(f"{'='*60}\n")
        print(f"  Configs : {len(grid_pts)}")
        print(f"  Splits  : {n_splits}")
        print(f"  Jobs    : {len(jobs)}\n")
        print(f"  {'Config':<28}  Overrides")
        print(f"  {'-'*56}")
        for cfg_name, params in grid_pts:
            ov = dict(params)
            if "backbone_F" in ov:
                ov["backbone"] = ov.pop("backbone_F")
                ov.pop("backbone_C", None)
            print(f"  {cfg_name:<28}  {ov}")
        print()
        return {}

    if not jobs:
        print("  Nenhum job pendente.")
        return {}

    n_workers, info = decide_workers(
        n_jobs=len(jobs),
        user_workers=workers,
        max_gpu_mb=max_gpu_mb,
        mem_per_job_mb=mem_per_job_mb,
        ram_per_worker_mb=ram_per_worker_mb,
    )

    print(f"\n{'='*60}")
    print(f"  BiSheafDiffusion — {dataset} Grid Search")
    print(f"  Configs         : {len(grid_pts)}")
    print(f"  Splits          : {n_splits}")
    print(f"  Jobs pendentes  : {len(jobs)}")
    print(f"  Workers         : {n_workers}  ({info['source']})")
    if info["source"] == "auto":
        print(f"    GPU livre     : {info['gpu_free']:.0f} MB  -> n_gpu={info['n_gpu']}")
        print(f"    RAM livre     : {info['ram_free']:.0f} MB  -> n_ram={info['n_ram']}")
    print(f"  Dataset cache   : por worker (1 carga por processo)")
    print(f"  Protocolo       : train | val | test@best_val")
    print(f"{'='*60}\n")

    expected: dict = defaultdict(int)
    for j in jobs:
        expected[j["config"]] += 1

    partial: dict = defaultdict(list)
    printed_cfg   = set()
    raw_results: list[dict] = list(prev_raw)
    pbar = tqdm(total=len(jobs), desc="Jobs", unit="job", dynamic_ncols=True)

    def on_result(result, done, total):
        print_job_line(result, done, total)
        cfg = result["config"]
        partial[cfg].append(result)
        raw_results.append(result)
        if cfg not in printed_cfg and len(partial[cfg]) == expected[cfg]:
            print_config_summary(cfg, partial[cfg])
            printed_cfg.add(cfg)
        save_partial(results_dir, dataset, raw_results)
        pbar.update(1)

    telemetry = Telemetry(interval_s=2.0)
    telemetry.start()
    t0 = time.time()
    try:
        run_pool(jobs, n_workers, on_result, telemetry)
    finally:
        telemetry.stop()
    pbar.close()
    total_time = time.time() - t0

    aggregated = aggregate(raw_results)
    n_errors   = sum(1 for r in raw_results if r.get("error"))

    print_ranking(dataset, aggregated, top_n=10)
    print_efficiency_report(telemetry, total_time, len(raw_results), n_errors, n_workers)
    save_final(results_dir, dataset, raw_results, aggregated, telemetry)

    print(f"  Resultados salvos em: {results_dir}")
    return aggregated
