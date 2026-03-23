"""
Grid search da CoSheafNetwork no ENZYMES.

Cada combinacao de hiperparametros e executada em subprocesso isolado
(erros em uma run nao contaminam as demais). Runs concluidas com sucesso
sao puladas ao retomar um sweep interrompido.

Uso:
    python -m src.experiments.cosheaf.training_datasets.enzymes.train
    python -m src.experiments.cosheaf.training_datasets.enzymes.train --dry-run
    python -m src.experiments.cosheaf.training_datasets.enzymes.train --out-dir runs/enzymes_v2
"""

import argparse
import csv
import itertools
import os
import time
import traceback

import torch.multiprocessing as mp

from src.experiments.cosheaf.training_datasets.enzymes.config import (
    DATASET, SEED, EPOCHS, BATCH_SIZE, N_FOLDS, GRID,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def hd_str(dims):
    return "x".join(str(h) for h in dims) if dims else "none"


def run_label(params):
    return (
        f"d={params['d']}  hd={hd_str(params['hidden_dims'])}"
        f"  lh={hd_str(params['learner_hidden'])}  mlp={hd_str(params['mlp_dims'])}"
        f"  lr={params['lr']}  wd={params['weight_decay']}  drop={params['dropout']}"
        f"  bb={params['backbone']}  orth={params['orth_method']}"
        f"  ef={params['edge_feat_mode']}"
    )


def run_id(params):
    return (
        f"d{params['d']}"
        f"_hd{hd_str(params['hidden_dims'])}"
        f"_lh{hd_str(params['learner_hidden'])}"
        f"_mlp{hd_str(params['mlp_dims'])}"
        f"_lr{params['lr']}"
        f"_wd{params['weight_decay']}"
        f"_do{params['dropout']}"
        f"_bb{params['backbone']}"
        f"_orth{params['orth_method']}"
        f"_ef{params['edge_feat_mode']}"
    )


def done_key(params):
    return run_id(params)


def fmt_time(seconds):
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


CSV_COLS = [
    "d", "hidden_dims", "learner_hidden", "mlp_dims",
    "lr", "weight_decay", "dropout", "backbone", "orth_method",
    "edge_feat_mode", "mean_acc", "std_acc",
    "total_time_s", "avg_fold_time_s", "avg_epoch_time_s",
    "fold_accs", "status",
]


# ---------------------------------------------------------------------------
# Worker — roda em subprocesso isolado
# ---------------------------------------------------------------------------

def _worker(cfg_override, save_dir, result_queue):
    try:
        from src.experiments.cosheaf.train import main as cosheaf_train
        result = cosheaf_train(save_dir=save_dir, cfg_override=cfg_override)
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
    return ("error", f"processo filho encerrou com codigo {p.exitcode}")


# ---------------------------------------------------------------------------
# Construcao das configs
# ---------------------------------------------------------------------------

def build_configs():
    keys = list(GRID.keys())
    configs = []
    for values in itertools.product(*GRID.values()):
        params = dict(zip(keys, values))
        cfg_override = {
            "dataset":        DATASET,
            "epochs":         EPOCHS,
            "batch_size":     BATCH_SIZE,
            "seed":           SEED,
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
                "orth_method":    params["orth_method"],
            },
        }
        configs.append((params, cfg_override))
    return configs


# ---------------------------------------------------------------------------
# Sweep principal
# ---------------------------------------------------------------------------

def sweep(out_dir="runs/cosheaf_enzymes", dry_run=False):
    configs = build_configs()
    total   = len(configs)

    print(f"\n{'='*70}")
    print(f"  CoSheafNN Grid Search — {DATASET}")
    print(f"  {total} combinacoes | {N_FOLDS}-fold CV | {EPOCHS} epochs")
    print(f"  Out dir: {out_dir}")
    print(f"{'='*70}\n")

    if dry_run:
        print("[dry-run] Nenhum treino executado.\n")
        for i, (params, _) in enumerate(configs, 1):
            print(f"  {i:3d}/{total}  {run_label(params)}")
        print(f"\nTotal: {total} runs")
        return

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "summary.csv")

    # Retomada: carrega runs ja concluidas
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline='') as f:
            for row in csv.DictReader(f):
                if row["status"] == "ok":
                    done.add(
                        f"d{row['d']}"
                        f"_hd{row['hidden_dims']}"
                        f"_lh{row['learner_hidden']}"
                        f"_mlp{row['mlp_dims']}"
                        f"_lr{row['lr']}"
                        f"_wd{row['weight_decay']}"
                        f"_do{row['dropout']}"
                        f"_bb{row['backbone']}"
                        f"_orth{row.get('orth_method', 'cayley')}"
                        f"_ef{row.get('edge_feat_mode', 'none')}"
                    )

    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    csv_file = open(csv_path, "a", newline='')
    writer   = csv.DictWriter(csv_file, fieldnames=CSV_COLS)
    if write_header:
        writer.writeheader()
        csv_file.flush()

    sweep_start = time.time()
    completed   = 0

    for i, (params, cfg_override) in enumerate(configs, 1):
        key = done_key(params)

        if key in done:
            print(f"[{i:3d}/{total}] SKIP   {run_label(params)}")
            completed += 1
            continue

        print(f"[{i:3d}/{total}] START  {run_label(params)}")
        r_dir = os.path.join(out_dir, run_id(params))
        t0    = time.time()

        status, payload = run_in_subprocess(cfg_override, r_dir)
        elapsed = time.time() - t0

        base_row = {
            "d":              params["d"],
            "hidden_dims":    hd_str(params["hidden_dims"]),
            "learner_hidden": hd_str(params["learner_hidden"]),
            "mlp_dims":       hd_str(params["mlp_dims"]),
            "lr":             params["lr"],
            "weight_decay":   params["weight_decay"],
            "dropout":        params["dropout"],
            "backbone":       params["backbone"],
            "orth_method":    params["orth_method"],
            "edge_feat_mode": params["edge_feat_mode"],
        }

        if status == "ok":
            result = payload
            row = {
                **base_row,
                "mean_acc":          f"{result['mean']:.4f}",
                "std_acc":           f"{result['std']:.4f}",
                "total_time_s":      result.get("total_time_s", ""),
                "avg_fold_time_s":   result.get("avg_fold_time_s", ""),
                "avg_epoch_time_s":  result.get("avg_epoch_time_s", ""),
                "fold_accs":         str([f"{a:.4f}" for a in result['fold_accs']]),
                "status":            "ok",
            }
            t_total = result.get('total_time_s', elapsed)
            t_epoch = result.get('avg_epoch_time_s', 0)
            print(f"         DONE   mean={result['mean']:.4f} +/- {result['std']:.4f}"
                  f"  | {t_total}s total | {t_epoch}s/epoch"
                  f"  ({fmt_time(elapsed)})")
        else:
            row = {
                **base_row,
                "mean_acc":         "",
                "std_acc":          "",
                "total_time_s":     "",
                "avg_fold_time_s":  "",
                "avg_epoch_time_s": "",
                "fold_accs":        "",
                "status":           f"error: {payload[:120]}",
            }
            print(f"         ERROR  {payload[:200]}")

        writer.writerow(row)
        csv_file.flush()
        completed += 1

        elapsed_total = time.time() - sweep_start
        avg           = elapsed_total / completed
        remaining     = avg * (total - completed)
        print(f"         Progresso: {completed}/{total}  |  "
              f"Decorrido: {fmt_time(elapsed_total)}  |  "
              f"Restante: ~{fmt_time(remaining)}\n")

    csv_file.close()

    print(f"\n{'='*70}")
    print(f"  RESULTADOS — {DATASET} (Top 10)")
    print(f"{'='*70}")
    print_summary(csv_path)


def print_summary(csv_path):
    rows = []
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            if row["status"] == "ok" and row["mean_acc"]:
                row["_mean"] = float(row["mean_acc"])
                rows.append(row)

    rows.sort(key=lambda r: -r["_mean"])

    print(f"\n  {'#':>3}  {'mean':>6}  {'std':>6}  {'d':>2}  {'hd':>8}  {'lh':>5}"
          f"  {'mlp':>5}  {'lr':>6}  {'wd':>6}  {'drop':>5}  {'bb':>4}  {'orth':>10}"
          f"  {'ef':>12}  {'total':>7}  {'s/epoch':>7}")
    print(f"  {'-'*115}")

    for i, r in enumerate(rows[:10], 1):
        t_total = r.get('total_time_s', '')
        t_epoch = r.get('avg_epoch_time_s', '')
        t_total_str = f"{float(t_total):.0f}s" if t_total else "?"
        t_epoch_str = f"{float(t_epoch):.3f}" if t_epoch else "?"
        print(f"  {i:3d}  {r['mean_acc']:>6}  {r['std_acc']:>6}  "
              f"{r['d']:>2}  {r['hidden_dims']:>8}  "
              f"{r['learner_hidden']:>5}  {r['mlp_dims']:>5}  "
              f"{r['lr']:>6}  {r['weight_decay']:>6}  {r['dropout']:>5}  "
              f"{r['backbone']:>4}  {r.get('orth_method', 'cayley'):>10}  "
              f"{r.get('edge_feat_mode', 'none'):>12}  "
              f"{t_total_str:>7}  {t_epoch_str:>7}")

    if rows:
        best = rows[0]
        print(f"\n  Melhor: {best['mean_acc']} +/- {best['std_acc']}")
        print(f"    d={best['d']}  hd={best['hidden_dims']}  lh={best['learner_hidden']}"
              f"  mlp={best['mlp_dims']}  lr={best['lr']}  wd={best['weight_decay']}"
              f"  drop={best['dropout']}  bb={best['backbone']}  orth={best.get('orth_method', 'cayley')}"
              f"  ef={best.get('edge_feat_mode', 'none')}"
              f"  | {best.get('total_time_s', '?')}s total")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Grid search CoSheafNN no {DATASET}")
    parser.add_argument("--out-dir", type=str, default="runs/cosheaf_enzymes",
                        help="Diretorio para salvar pesos e resultados")
    parser.add_argument("--dry-run", action="store_true",
                        help="Lista as combinacoes sem executar")
    args = parser.parse_args()
    sweep(out_dir=args.out_dir, dry_run=args.dry_run)