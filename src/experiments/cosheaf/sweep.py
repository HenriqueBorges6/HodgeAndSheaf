"""
Bateria de experimentos: grid search de hiperparâmetros x múltiplos datasets.

Cada run é executada em um subprocesso isolado. Erros em uma run não
contaminam as seguintes — o processo principal continua independentemente.
Runs concluídas com sucesso são puladas ao retomar um sweep interrompido.

Uso:
    python -m src.experiments.cosheaf.sweep
    python -m src.experiments.cosheaf.sweep --out-dir runs/cosheaf_sweep1 --dry-run
    python -m src.experiments.cosheaf.sweep --datasets MUTAG PTC_MR
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

DATASETS = ["MUTAG", "PTC_MR", "ENZYMES", "PROTEINS"]

GRID = {
    "d":               [2, 3, 4],
    "hidden_dims":     [[64, 64], [128, 128]],
    "learner_hidden":  [[32], [64]],
    "mlp_dims":        [[64], []],
    "lr":              [0.01, 0.001],
    "weight_decay":    [0.0, 1e-4],
    "dropout":         [0.0, 0.25],
    "edge_feat_mode":  ["none"],
    "backbone":        ["sage"],
}

FIXED = {
    "epochs":     200,
    "batch_size": 64,
    "seed":       1,
}

CSV_COLS = [
    "dataset", "d", "hidden_dims", "learner_hidden", "mlp_dims",
    "lr", "weight_decay", "dropout", "edge_feat_mode", "backbone",
    "mean_acc", "std_acc", "status",
]


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker(cfg_override, save_dir, result_queue):
    try:
        from src.experiments.cosheaf.train import main as train_main
        result = train_main(save_dir=save_dir, cfg_override=cfg_override)
        result_queue.put(("ok", result))
    except Exception as e:
        result_queue.put(("error", str(e) + "\n" + traceback.format_exc()))


def run_in_subprocess(cfg_override, run_dir):
    ctx = mp.get_context("spawn")
    q   = ctx.Queue()
    p   = ctx.Process(target=_worker, args=(cfg_override, run_dir, q))
    p.start()
    p.join()
    if not q.empty():
        return q.get()
    return ("error", f"processo filho encerrou com código {p.exitcode}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def hd_str(dims):
    return "x".join(str(h) for h in dims) if dims else "none"


def build_configs(datasets):
    keys = list(GRID.keys())
    configs = []
    for values in itertools.product(*GRID.values()):
        params = dict(zip(keys, values))
        for dataset in datasets:
            cfg_override = {
                "dataset":        dataset,
                "epochs":         FIXED["epochs"],
                "batch_size":     FIXED["batch_size"],
                "seed":           FIXED["seed"],
                "lr":             params["lr"],
                "weight_decay":   params["weight_decay"],
                "edge_feat_mode": params["edge_feat_mode"],
                "model": {
                    "d":              params["d"],
                    "hidden_dims":    params["hidden_dims"],
                    "learner_hidden": params["learner_hidden"],
                    "mlp_dims":       params["mlp_dims"],
                    "dropout":        params["dropout"],
                    "backbone":       params["backbone"],
                },
            }
            run_id = (
                f"{dataset}"
                f"_d{params['d']}"
                f"_hd{hd_str(params['hidden_dims'])}"
                f"_lh{hd_str(params['learner_hidden'])}"
                f"_mlp{hd_str(params['mlp_dims'])}"
                f"_lr{params['lr']}"
                f"_wd{params['weight_decay']}"
                f"_do{params['dropout']}"
                f"_ef{params['edge_feat_mode']}"
                f"_bb{params['backbone']}"
            )
            configs.append((run_id, dataset, params, cfg_override))
    return configs


def run_label(dataset, params):
    return (
        f"{dataset} | d={params['d']}  hd={hd_str(params['hidden_dims'])}"
        f"  lh={hd_str(params['learner_hidden'])}  mlp={hd_str(params['mlp_dims'])}"
        f"  lr={params['lr']}  wd={params['weight_decay']}  drop={params['dropout']}"
        f"  ef={params['edge_feat_mode']}  bb={params['backbone']}"
    )


def done_key(dataset, params):
    return (
        f"{dataset}_{params['d']}_{hd_str(params['hidden_dims'])}"
        f"_{hd_str(params['learner_hidden'])}_{hd_str(params['mlp_dims'])}"
        f"_{params['lr']}_{params['weight_decay']}_{params['dropout']}"
        f"_{params['edge_feat_mode']}_{params['backbone']}"
    )


def fmt_time(seconds):
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Sweep principal
# ---------------------------------------------------------------------------

def sweep(out_dir="runs/cosheaf_sweep", datasets=None, dry_run=False):
    datasets = datasets or DATASETS
    configs  = build_configs(datasets)
    total    = len(configs)

    print(f"\n{'='*65}")
    print(f"  CoSheafNN Sweep — {total} runs")
    print(f"  Datasets : {datasets}")
    print(f"  Grid     : { {k: v for k, v in GRID.items()} }")
    print(f"  Epochs   : {FIXED['epochs']}")
    print(f"  Out dir  : {out_dir}")
    print(f"  Modo     : subprocesso isolado por run")
    print(f"{'='*65}\n")

    if dry_run:
        print("[dry-run] Nenhum treino executado.")
        for i, (run_id, dataset, params, _) in enumerate(configs, 1):
            print(f"  {i:3d}/{total}  {run_label(dataset, params)}")
        return

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "summary.csv")

    # Retomada: carrega runs já concluídas
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                if row["status"] == "ok":
                    done.add(
                        f"{row['dataset']}_{row['d']}_{row['hidden_dims']}"
                        f"_{row['learner_hidden']}_{row['mlp_dims']}"
                        f"_{row['lr']}_{row['weight_decay']}_{row['dropout']}"
                        f"_{row.get('edge_feat_mode', 'none')}_{row.get('backbone', 'sage')}"
                    )

    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    csv_file = open(csv_path, "a", newline='')
    writer   = csv.DictWriter(csv_file, fieldnames=CSV_COLS)
    if write_header:
        writer.writeheader()
        csv_file.flush()

    sweep_start = time.time()
    completed   = 0

    for i, (run_id, dataset, params, cfg_override) in enumerate(configs, 1):
        key = done_key(dataset, params)

        if key in done:
            print(f"[{i:3d}/{total}] SKIP   {run_label(dataset, params)}")
            completed += 1
            continue

        print(f"[{i:3d}/{total}] START  {run_label(dataset, params)}")
        run_dir = os.path.join(out_dir, run_id)
        t0      = time.time()

        status, payload = run_in_subprocess(cfg_override, run_dir)
        elapsed = time.time() - t0

        base_row = {
            "dataset":        dataset,
            "d":              params["d"],
            "hidden_dims":    hd_str(params["hidden_dims"]),
            "learner_hidden": hd_str(params["learner_hidden"]),
            "mlp_dims":       hd_str(params["mlp_dims"]),
            "lr":             params["lr"],
            "weight_decay":   params["weight_decay"],
            "dropout":        params["dropout"],
            "edge_feat_mode": params["edge_feat_mode"],
            "backbone":       params["backbone"],
        }
        if status == "ok":
            result = payload
            row = {**base_row,
                   "mean_acc": f"{result['mean']:.4f}",
                   "std_acc":  f"{result['std']:.4f}",
                   "status":   "ok"}
            print(f"         DONE   mean={result['mean']:.4f} ± {result['std']:.4f}"
                  f"  ({fmt_time(elapsed)})")
        else:
            row = {**base_row,
                   "mean_acc": "",
                   "std_acc":  "",
                   "status":   f"error: {payload[:120]}"}
            print(f"         ERROR  {payload[:200]}")

        writer.writerow(row)
        csv_file.flush()
        completed += 1

        elapsed_total = time.time() - sweep_start
        avg           = elapsed_total / completed
        remaining     = avg * (total - completed)
        print(f"         Progresso: {completed}/{total}  |  "
              f"Decorrido: {fmt_time(elapsed_total)}  |  "
              f"Estimado restante: {fmt_time(remaining)}\n")

    csv_file.close()
    print(f"\n{'='*65}")
    print("  RESUMO FINAL")
    print(f"{'='*65}")
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
        print(f"  {'mean':>6}  {'std':>6}  {'d':>3}  {'hd':>8}  {'lh':>5}  {'mlp':>5}  {'lr':>8}  {'wd':>6}  {'drop':>5}")
        print(f"  {'-'*65}")
        for r in ds_rows[:5]:
            print(f"  {r['mean_acc']:>6}  {r['std_acc']:>6}  "
                  f"{r['d']:>3}  {r['hidden_dims']:>8}  "
                  f"{r['learner_hidden']:>5}  {r['mlp_dims']:>5}  "
                  f"{r['lr']:>8}  {r['weight_decay']:>6}  {r['dropout']:>5}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir",  type=str, default="runs/cosheaf_sweep")
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Subconjunto de datasets (ex: MUTAG PTC_MR)")
    args = parser.parse_args()
    sweep(out_dir=args.out_dir, datasets=args.datasets, dry_run=args.dry_run)