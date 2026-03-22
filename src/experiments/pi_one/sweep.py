"""
Bateria de experimentos: grid search de hiperparâmetros × múltiplos datasets.

Cada run é executada em um subprocesso isolado via torch.multiprocessing.
Isso garante que erros de CUDA (device-side assert, OOM) não contaminem
as runs seguintes — o processo principal continua independentemente.

Uso:
    python -m src.experiments.pi_one.sweep
    python -m src.experiments.pi_one.sweep --out-dir runs/sweep1 --dry-run
"""

import argparse
import csv
import itertools
import os
import time
import traceback

import torch.multiprocessing as mp

# ---------------------------------------------------------------------------
# Datasets e Grid
# ---------------------------------------------------------------------------

DATASETS = ["MUTAG", "PTC_MR", "ENZYMES", "PROTEINS" ] # "MUTAG", "PTC_MR","PROTEINS","ENZYMES"

GRID = {
    "d":           [2, 3, 4],
    "num_powers":  [1, 2, 3],
    "lambda_hol":  [0.0, 0.1],
    "lr":          [0.01, 0.001],
    "weight_decay":[0, 0.001],
}

FIXED = {
    "epochs":     300,
    "batch_size": 128,
    "seed":       1,
    "mlp_dims":   [(64, "relu")],
    "num_layes":  3,
}

CSV_COLS = ["dataset", "d", "num_powers", "lambda_hol", "lr", "weight_decay",
            "mean_acc", "std_acc", "status"]

# ---------------------------------------------------------------------------
# Worker (deve ser função de módulo para ser picklable com spawn)
# ---------------------------------------------------------------------------

def _worker(cfg_override, save_dir, result_queue):
    """Executa uma run de treino e coloca o resultado na queue."""
    try:
        from src.experiments.pi_one.train import main as train_main
        result = train_main(save_dir=save_dir, cfg_override=cfg_override)
        result_queue.put(("ok", result))
    except Exception as e:
        result_queue.put(("error", str(e) + "\n" + traceback.format_exc()))


def run_in_subprocess(cfg_override, run_dir):
    """
    Lança _worker num processo filho separado.
    Retorna ('ok', result_dict) ou ('error', mensagem).
    CUDA errors no filho não afetam o processo pai.
    """
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(cfg_override, run_dir, q))
    p.start()
    p.join()

    if not q.empty():
        return q.get()
    return ("error", f"processo filho encerrou com código {p.exitcode}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_configs():
    keys = list(GRID.keys())
    configs = []
    for values in itertools.product(*GRID.values()):
        params = dict(zip(keys, values))
        for dataset in DATASETS:
            cfg_override = {
                "dataset":      dataset,
                "epochs":       FIXED["epochs"],
                "batch_size":   FIXED["batch_size"],
                "seed":         FIXED["seed"],
                "lr":           params["lr"],
                "weight_decay": params["weight_decay"],
                "lambda_hol":   params["lambda_hol"],
                "model": {
                    "d":          params["d"],
                    "num_powers": params["num_powers"],
                    "num_layes":  FIXED["num_layes"],
                    "mlp_dims":   FIXED["mlp_dims"],
                },
            }
            run_id = (
                f"{dataset}_d{params['d']}_np{params['num_powers']}"
                f"_lhol{params['lambda_hol']}_lr{params['lr']}_wd{params['weight_decay']}"
            )
            configs.append((run_id, dataset, params, cfg_override))
    return configs


def run_label(dataset, params):
    return (f"{dataset} | d={params['d']} np={params['num_powers']} "
            f"λ={params['lambda_hol']} lr={params['lr']} wd={params['weight_decay']}")


def fmt_time(seconds):
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# ---------------------------------------------------------------------------
# Sweep principal
# ---------------------------------------------------------------------------

def sweep(out_dir="runs/sweep", dry_run=False):
    configs = build_configs()
    total = len(configs)

    print(f"\n{'='*60}")
    print(f"  PiOneGNN Sweep — {total} runs")
    print(f"  Datasets : {DATASETS}")
    print(f"  Grid     : { {k: v for k, v in GRID.items()} }")
    print(f"  Epochs   : {FIXED['epochs']}")
    print(f"  Out dir  : {out_dir}")
    print(f"  Modo     : subprocesso isolado por run (CUDA-safe)")
    print(f"{'='*60}\n")

    if dry_run:
        print("[dry-run] Nenhum treino executado.")
        for i, (run_id, dataset, params, _) in enumerate(configs, 1):
            print(f"  {i:3d}/{total}  {run_label(dataset, params)}")
        return

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "summary.csv")

    # Retomada: lê runs já concluídas
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                if row["status"] == "ok":
                    done.add(f"{row['dataset']}_{row['d']}_{row['num_powers']}"
                             f"_{row['lambda_hol']}_{row['lr']}_{row['weight_decay']}")

    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    csv_file = open(csv_path, "a", newline='')
    writer = csv.DictWriter(csv_file, fieldnames=CSV_COLS)
    if write_header:
        writer.writeheader()
        csv_file.flush()

    sweep_start = time.time()
    completed = 0

    for i, (run_id, dataset, params, cfg_override) in enumerate(configs, 1):
        done_key = (f"{dataset}_{params['d']}_{params['num_powers']}"
                    f"_{params['lambda_hol']}_{params['lr']}_{params['weight_decay']}")

        if done_key in done:
            print(f"[{i:3d}/{total}] SKIP  {run_label(dataset, params)}")
            completed += 1
            continue

        print(f"[{i:3d}/{total}] START  {run_label(dataset, params)}")
        run_dir = os.path.join(out_dir, run_id)
        t0 = time.time()

        status, payload = run_in_subprocess(cfg_override, run_dir)
        elapsed = time.time() - t0

        if status == "ok":
            result = payload
            row = {
                "dataset":      dataset,
                "d":            params["d"],
                "num_powers":   params["num_powers"],
                "lambda_hol":   params["lambda_hol"],
                "lr":           params["lr"],
                "weight_decay": params["weight_decay"],
                "mean_acc":     f"{result['mean']:.4f}",
                "std_acc":      f"{result['std']:.4f}",
                "status":       "ok",
            }
            print(f"         DONE   mean={result['mean']:.4f} ± {result['std']:.4f}"
                  f"  ({fmt_time(elapsed)})")
        else:
            row = {
                "dataset":      dataset,
                "d":            params["d"],
                "num_powers":   params["num_powers"],
                "lambda_hol":   params["lambda_hol"],
                "lr":           params["lr"],
                "weight_decay": params["weight_decay"],
                "mean_acc":     "",
                "std_acc":      "",
                "status":       f"error: {payload[:120]}",
            }
            print(f"         ERROR  {payload[:200]}")

        writer.writerow(row)
        csv_file.flush()
        completed += 1

        elapsed_total = time.time() - sweep_start
        avg = elapsed_total / completed
        remaining = avg * (total - completed)
        print(f"         Progresso: {completed}/{total}  |  "
              f"Decorrido: {fmt_time(elapsed_total)}  |  "
              f"Estimado restante: {fmt_time(remaining)}\n")

    csv_file.close()
    print(f"\n{'='*60}")
    print("  RESUMO FINAL")
    print(f"{'='*60}")
    _print_summary(csv_path)


def _print_summary(csv_path):
    rows = []
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            if row["status"] == "ok" and row["mean_acc"]:
                row["_mean"] = float(row["mean_acc"])
                rows.append(row)

    for ds in sorted(set(r["dataset"] for r in rows)):
        ds_rows = sorted([r for r in rows if r["dataset"] == ds], key=lambda r: -r["_mean"])
        print(f"\n  {ds} — Top 5:")
        print(f"  {'mean':>6}  {'std':>6}  {'d':>3}  {'np':>3}  {'λ':>5}  {'lr':>8}  {'wd':>6}")
        print(f"  {'-'*50}")
        for r in ds_rows[:5]:
            print(f"  {r['mean_acc']:>6}  {r['std_acc']:>6}  "
                  f"{r['d']:>3}  {r['num_powers']:>3}  "
                  f"{r['lambda_hol']:>5}  {r['lr']:>8}  {r['weight_decay']:>6}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="runs/sweep")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    sweep(out_dir=args.out_dir, dry_run=args.dry_run)
