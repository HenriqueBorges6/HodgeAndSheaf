"""
Mini sweep paralelo — laplacian_mode x cosheaf_b.

Diferencas vs grid_search.py:
  - Execucao paralela com N workers (ProcessPoolExecutor + spawn)
  - 3 splits WebKB ao inves de 10
  - Early stopping (patience configuravel)
  - Foco so em laplacian_mode x cosheaf_b
  - train_on_trainval=False (usa val para early stopping)

Uso:
  python -m src.experiments.dual_sheaf.minisweep
  python -m src.experiments.dual_sheaf.minisweep --datasets Texas Cornell --workers 3
  python -m src.experiments.dual_sheaf.minisweep --resume runs/dual_sheaf_mini/<run_id>
  python -m src.experiments.dual_sheaf.minisweep --no-cosheaf
"""

import argparse
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import torch.multiprocessing as mp

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Grade e config base
# ---------------------------------------------------------------------------

LAPLACIAN_MODES = ["dual", "mixed_self", "mixed_nbr", "symmetric"]
WEBKB_DATASETS  = {"texas", "cornell", "wisconsin"}
N_SPLITS_DEFAULT = 3

BASE_CFG = {
    "epochs":            1000,
    "lr":                0.001,
    "weight_decay":      5e-4,
    "train_on_trainval": False,   # False: usa val para early stopping
    "patience":          150,     # early stopping
    "print_every":       99999,   # silencia prints periodicos nos workers
    "model": {
        "d":                2,
        "hidden_channels":  64,
        "num_layers":       2,
        "learner_hidden":   64,
        "learner_gnn_layers": 1,
        "dropout":          0.5,
        "left_weights":     True,
        "right_weights":    True,
        "use_eps":          True,
        "single_learner":   False,
        "orth_trans":       None,
        "mlp_dims":         [64],
    },
}


# ---------------------------------------------------------------------------
# Construcao dos jobs
# ---------------------------------------------------------------------------

def build_jobs(datasets, n_splits, include_cosheaf):
    """4 modos x 2 cosheaf x n_splits x datasets."""
    jobs = []
    for dataset in datasets:
        is_webkb = dataset.lower() in WEBKB_DATASETS
        cosheaf_options = [False, True] if include_cosheaf else [False]
        for mode in LAPLACIAN_MODES:
            for cb in cosheaf_options:
                for s in range(n_splits):
                    jobs.append({
                        "dataset":       dataset,
                        "laplacian_mode": mode,
                        "cosheaf_b":     cb,
                        "split_or_seed": s,
                        "is_webkb":      is_webkb,
                    })
    return jobs


def job_key(job):
    return (job["dataset"], job["laplacian_mode"], job["cosheaf_b"], job["split_or_seed"])


# ---------------------------------------------------------------------------
# Execucao de um job (roda em processo separado)
# ---------------------------------------------------------------------------

def run_job(job):
    """Executado em processo filho — cada um tem seu proprio contexto CUDA."""
    import warnings
    warnings.filterwarnings("ignore")

    from src.experiments.dual_sheaf.train import main as train_main

    cfg = {
        **BASE_CFG,
        "dataset": job["dataset"],
        "model": {
            **BASE_CFG["model"],
            "laplacian_mode": job["laplacian_mode"],
            "cosheaf_b":      job["cosheaf_b"],
        },
    }
    if job["is_webkb"]:
        cfg["split_idx"] = job["split_or_seed"]
        cfg["seed"] = 0
    else:
        cfg["seed"] = job["split_or_seed"]

    t0 = time.time()
    try:
        r = train_main(cfg_override=cfg, quiet=True)
        return {
            **job,
            "test_acc":   r["test_at_best_val"] * 100,
            "val_acc":    r["best_val_acc"] * 100,
            "epochs_run": r.get("epochs_run", cfg["epochs"]),
            "elapsed_s":  round(time.time() - t0, 1),
            "status":     "ok",
        }
    except Exception as e:
        import traceback
        return {
            **job,
            "test_acc":   None,
            "val_acc":    None,
            "epochs_run": None,
            "elapsed_s":  round(time.time() - t0, 1),
            "status":     f"error: {traceback.format_exc()}",
        }


# ---------------------------------------------------------------------------
# IO progressivo
# ---------------------------------------------------------------------------

def load_done_keys(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    done.add(tuple(job_key(r)))
                except Exception:
                    pass
    return done


def append_result(path, result):
    with open(path, "a") as f:
        f.write(json.dumps(result) + "\n")


def append_fold(path, fold):
    with open(path, "a") as f:
        f.write(json.dumps(fold) + "\n")


def load_all_results(path):
    results = []
    if not os.path.exists(path):
        return results
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except Exception:
                    pass
    return results


# ---------------------------------------------------------------------------
# Agregacao e tabela
# ---------------------------------------------------------------------------

def aggregate(results):
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        if r["status"] != "ok":
            continue
        key = (r["dataset"], r["laplacian_mode"], r["cosheaf_b"])
        groups[key].append(r["test_acc"])

    agg = {}
    for key, accs in groups.items():
        arr = np.array(accs)
        agg[key] = {
            "dataset":       key[0],
            "laplacian_mode": key[1],
            "cosheaf_b":     key[2],
            "n":     len(accs),
            "mean":  round(float(arr.mean()), 2),
            "std":   round(float(arr.std()),  2),
            "accs":  accs,
        }
    return agg


def print_summary(agg, datasets):
    col_w = 16
    ds_cols = list(datasets)
    header = f"  {'Config':<26}" + "".join(f"  {d:>{col_w}}" for d in ds_cols)
    sep = "-" * len(header)

    print(f"\n{'='*72}")
    print(f"  Resultados finais (train→val→test, early stopping)")
    print(f"{'='*72}")
    print(header)
    print(sep)

    # Agrupar por (mode, cosheaf_b)
    configs = {}
    for (ds, mode, cb), v in agg.items():
        k = (mode, cb)
        configs.setdefault(k, {})[ds] = v

    for (mode, cb), ds_results in sorted(configs.items()):
        cb_str = "+cosheaf" if cb else "        "
        label = f"{mode:<14} {cb_str}"
        row = ""
        for d in ds_cols:
            if d in ds_results:
                r = ds_results[d]
                cell = f"{r['mean']:>5.1f}±{r['std']:<4.1f} (n={r['n']})"
            else:
                cell = "---"
            row += f"  {cell:>{col_w}}"
        print(f"  {label}  {row}")

    print(sep)
    print(f"  {'Melhor:':<26}", end="")
    for d in ds_cols:
        ds_rows = [v for (ds, _, _), v in agg.items() if ds == d]
        if ds_rows:
            best = max(ds_rows, key=lambda r: r["mean"])
            cb_str = "C" if best["cosheaf_b"] else " "
            cell = f"{best['mean']:.1f}[{best['laplacian_mode'][:4]}{cb_str}]"
        else:
            cell = "---"
        print(f"  {cell:>{col_w}}", end="")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets",   nargs="+", default=["Texas", "Cornell"])
    parser.add_argument("--splits",     type=int,  default=N_SPLITS_DEFAULT,
                        help="Numero de splits WebKB (default 3)")
    parser.add_argument("--workers",    type=int,  default=3,
                        help="Workers paralelos (default 3)")
    parser.add_argument("--patience",   type=int,  default=150,
                        help="Early stopping patience (default 150)")
    parser.add_argument("--epochs",     type=int,  default=1000)
    parser.add_argument("--no-cosheaf", action="store_true")
    parser.add_argument("--resume",     type=str,  default=None)
    args = parser.parse_args()

    # Atualizar BASE_CFG com args
    BASE_CFG["patience"] = args.patience
    BASE_CFG["epochs"]   = args.epochs

    # Diretorio de saida
    if args.resume:
        out_dir = args.resume
        print(f"\nRetomando sweep: {out_dir}")
    else:
        run_id  = datetime.now().strftime("%Y-%m-%d_%H-%M")
        out_dir = os.path.join("runs", "dual_sheaf_mini", run_id)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nNovo mini-sweep: {out_dir}")

    results_path      = os.path.join(out_dir, "results.jsonl")
    fold_summary_path = os.path.join(out_dir, "fold_summary.jsonl")
    summary_path      = os.path.join(out_dir, "summary.json")

    # Construir jobs
    jobs = build_jobs(
        datasets=args.datasets,
        n_splits=args.splits,
        include_cosheaf=not args.no_cosheaf,
    )
    total = len(jobs)

    done_keys = load_done_keys(results_path)
    pending   = [j for j in jobs if tuple(job_key(j)) not in done_keys]

    n_configs = len(LAPLACIAN_MODES) * (2 if not args.no_cosheaf else 1)
    print(f"  Grade: {len(LAPLACIAN_MODES)} modos x {'2 cosheaf' if not args.no_cosheaf else '1'} = {n_configs} configs")
    print(f"  Splits por config: {args.splits}  |  Workers: {args.workers}")
    print(f"  Early stopping: patience={args.patience}")
    print(f"  Jobs total: {total}  |  Concluidos: {total - len(pending)}  |  Pendentes: {len(pending)}")

    if not pending:
        print("  Todos os jobs ja concluidos!")
    else:
        print(f"\n  Iniciando em 3s... (Ctrl+C para interromper)\n")
        time.sleep(3)

        sweep_start  = time.time()
        completed    = total - len(pending)
        n_ok, n_err  = 0, 0

        # Rastrear accs por config para detectar quando um fold completa
        # config_key -> lista de test_acc
        from collections import defaultdict
        fold_accs = defaultdict(list)

        # Pre-carregar splits ja concluidos (para retomada correta)
        for r in load_all_results(results_path):
            if r["status"] == "ok":
                ck = (r["dataset"], r["laplacian_mode"], r["cosheaf_b"])
                fold_accs[ck].append(r["test_acc"])

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as executor:
            futures = {executor.submit(run_job, job): job for job in pending}

            for future in as_completed(futures):
                result = future.result()
                append_result(results_path, result)
                completed += 1

                elapsed = time.time() - sweep_start
                done_so_far = completed - (total - len(pending))
                avg_s   = elapsed / done_so_far if done_so_far > 0 else 0
                rem_s   = avg_s * (total - completed)
                eta_str = f"ETA ~{int(rem_s//3600)}h{int((rem_s%3600)//60):02d}m"

                job = futures[future]
                cb_str   = "+C" if job["cosheaf_b"] else "  "
                mode_str = job["laplacian_mode"]
                ck       = (job["dataset"], job["laplacian_mode"], job["cosheaf_b"])

                if result["status"] == "ok":
                    n_ok += 1
                    fold_accs[ck].append(result["test_acc"])
                    ep_str = f"ep={result['epochs_run']}" if result['epochs_run'] else ""
                    print(
                        f"  [{completed:3d}/{total}] {eta_str}"
                        f"  {job['dataset']:<8} {mode_str:<12} {cb_str}"
                        f"  split={job['split_or_seed']}"
                        f"  test={result['test_acc']:.1f}%  {ep_str}"
                        f"  ({result['elapsed_s']:.0f}s)"
                    )

                    # Fold completo: todos os splits desta config terminaram
                    if len(fold_accs[ck]) == args.splits:
                        arr  = np.array(fold_accs[ck])
                        fold = {
                            "dataset":        job["dataset"],
                            "laplacian_mode": job["laplacian_mode"],
                            "cosheaf_b":      job["cosheaf_b"],
                            "n_splits":       args.splits,
                            "mean":           round(float(arr.mean()), 2),
                            "std":            round(float(arr.std()),  2),
                            "accs":           fold_accs[ck],
                        }
                        append_fold(fold_summary_path, fold)
                        cb_label = "+cosheaf" if job["cosheaf_b"] else ""
                        print(
                            f"\n  *** FOLD COMPLETO ***"
                            f"  {job['dataset']} | {mode_str} {cb_label}"
                            f"  →  {fold['mean']:.1f} ± {fold['std']:.1f}%"
                            f"  (splits: {[f'{a:.1f}' for a in fold_accs[ck]]})\n"
                        )
                else:
                    n_err += 1
                    print(
                        f"  [{completed:3d}/{total}] ERRO"
                        f"  {job['dataset']:<8} {mode_str:<12} {cb_str}"
                        f"  split={job['split_or_seed']}"
                        f"  → {result['status'][:80]}"
                    )

        total_elapsed = time.time() - sweep_start
        print(f"\n  Sweep concluido: {n_ok} ok, {n_err} erros, {total_elapsed/60:.1f}min")

    # Sumario final
    all_results = load_all_results(results_path)
    agg = aggregate(all_results)

    with open(summary_path, "w") as f:
        json.dump(list(agg.values()), f, indent=2)
    print(f"  Sumario: {summary_path}")

    print_summary(agg, args.datasets)


if __name__ == "__main__":
    main()
